"""Memory pool management for GPU allocations."""

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from .memory_pool_strategy import MemoryPoolStrategy

logger = logging.getLogger(__name__)


@dataclass
class MemoryBlock:
    """Represents a memory block in the pool."""

    size_bytes: int
    tensor: torch.Tensor | None = None
    allocated_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    is_free: bool = True

    def mark_used(self):
        """Mark block as used."""
        self.is_free = False
        self.last_used = time.time()
        self.use_count += 1

    def mark_free(self):
        """Mark block as free."""
        self.is_free = True
        self.last_used = time.time()

    @property
    def age_seconds(self) -> float:
        """Get age of block in seconds."""
        return time.time() - self.allocated_at

    @property
    def idle_seconds(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_used


@dataclass
class MemoryPoolStats:
    """Statistics for memory pool."""

    total_blocks: int
    free_blocks: int
    allocated_blocks: int
    total_size_gb: float
    free_size_gb: float
    allocated_size_gb: float
    hit_rate: float
    miss_rate: float
    fragmentation_ratio: float
    avg_block_age_seconds: float

    @property
    def utilization_percent(self) -> float:
        """Calculate pool utilization percentage."""
        if self.total_size_gb == 0:
            return 0.0
        return (self.allocated_size_gb / self.total_size_gb) * 100.0


class MemoryPoolManager:
    """Manages memory pool for GPU allocations with smart reuse.

    Features:
    - Pre-allocated memory blocks for common sizes
    - Block reuse to reduce allocation overhead
    - Automatic pool growth and shrinkage
    - Fragmentation management
    - Usage statistics and monitoring
    """

    def __init__(
        self,
        device: torch.device,
        initial_pool_size_gb: float = 1.0,
        max_pool_size_gb: float = 4.0,
        strategy: MemoryPoolStrategy = MemoryPoolStrategy.ADAPTIVE,
        enable_stats: bool = True,
    ):
        """Initialize memory pool manager.

        Args:
            device: Target device for allocations
            initial_pool_size_gb: Initial pool size in GB
            max_pool_size_gb: Maximum pool size in GB
            strategy: Pool allocation strategy
            enable_stats: Enable statistics tracking
        """
        self.device = device
        self.initial_pool_size_gb = initial_pool_size_gb
        self.max_pool_size_gb = max_pool_size_gb
        self.strategy = strategy
        self.enable_stats = enable_stats

        # Memory blocks organized by size
        self.blocks: Dict[int, List[MemoryBlock]] = {}
        self.lock = threading.Lock()

        # Statistics
        self.total_allocations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_size_bytes = 0

        # Common block sizes (in bytes) for typical patch processing
        # Based on common tensor sizes: batch_size * channels * height * width * dtype_size
        self.common_sizes = self._calculate_common_sizes()

        # Pre-allocate initial pool
        self._preallocate_pool()

        logger.info(
            f"MemoryPoolManager initialized: {initial_pool_size_gb:.2f}GB initial, "
            f"{max_pool_size_gb:.2f}GB max, strategy={strategy.value}"
        )

    def _calculate_common_sizes(self) -> List[int]:
        """Calculate common memory block sizes based on typical usage."""
        common_sizes = []

        # Common batch sizes
        batch_sizes = [1, 4, 8, 16, 32, 64]

        # Common tensor shapes for patches
        # Format: (channels, height, width)
        tensor_shapes = [
            (3, 96, 96),  # Small patches
            (3, 224, 224),  # Standard patches
            (3, 512, 512),  # Large patches
        ]

        # Common feature dimensions
        feature_dims = [128, 256, 512, 1024, 2048]

        # Calculate sizes for patch tensors (float32)
        for batch_size in batch_sizes:
            for c, h, w in tensor_shapes:
                size_bytes = batch_size * c * h * w * 4  # float32 = 4 bytes
                common_sizes.append(size_bytes)

        # Calculate sizes for feature tensors (float32)
        for batch_size in batch_sizes:
            for feat_dim in feature_dims:
                size_bytes = batch_size * feat_dim * 4
                common_sizes.append(size_bytes)

        # Remove duplicates and sort
        common_sizes = sorted(list(set(common_sizes)))

        return common_sizes

    def _preallocate_pool(self):
        """Pre-allocate memory pool with common sizes."""
        if self.device.type != "cuda":
            logger.info("Skipping pool pre-allocation for non-CUDA device")
            return

        # Calculate how many blocks to pre-allocate
        available_bytes = int(self.initial_pool_size_gb * 1024**3)

        with self.lock:
            for size_bytes in self.common_sizes:
                if available_bytes <= 0:
                    break

                # Allocate 2-3 blocks of each common size
                num_blocks = min(3, available_bytes // size_bytes)

                if num_blocks > 0:
                    self.blocks[size_bytes] = []

                    for _ in range(num_blocks):
                        try:
                            tensor = torch.empty(
                                size_bytes // 4,  # float32 elements
                                dtype=torch.float32,
                                device=self.device,
                            )

                            block = MemoryBlock(size_bytes=size_bytes, tensor=tensor, is_free=True)

                            self.blocks[size_bytes].append(block)
                            self.total_size_bytes += size_bytes
                            available_bytes -= size_bytes

                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                logger.warning(f"OOM during pre-allocation at {size_bytes} bytes")
                                break
                            raise

        logger.info(
            f"Pre-allocated {len(self.blocks)} size classes, "
            f"total: {self.total_size_bytes / 1024**3:.2f}GB"
        )

    def allocate(self, size_bytes: int) -> torch.Tensor:
        """Allocate memory from pool or create new block.

        Args:
            size_bytes: Size in bytes to allocate

        Returns:
            Allocated tensor
        """
        self.total_allocations += 1

        with self.lock:
            # Try to find exact size match
            if size_bytes in self.blocks:
                for block in self.blocks[size_bytes]:
                    if block.is_free:
                        block.mark_used()
                        self.cache_hits += 1
                        return block.tensor

            # Try to find larger block that can be reused
            for block_size in sorted(self.blocks.keys()):
                if block_size >= size_bytes:
                    for block in self.blocks[block_size]:
                        if block.is_free:
                            block.mark_used()
                            self.cache_hits += 1
                            # Return view of appropriate size
                            return block.tensor.view(-1)[: size_bytes // 4]

            # Cache miss - allocate new block
            self.cache_misses += 1

            # Check if we can grow the pool
            current_size_gb = self.total_size_bytes / (1024**3)
            new_size_gb = (self.total_size_bytes + size_bytes) / (1024**3)

            if new_size_gb > self.max_pool_size_gb:
                # Try to free some blocks first
                self._cleanup_idle_blocks()

            # Allocate new block
            try:
                tensor = torch.empty(size_bytes // 4, dtype=torch.float32, device=self.device)

                block = MemoryBlock(size_bytes=size_bytes, tensor=tensor, is_free=False)

                if size_bytes not in self.blocks:
                    self.blocks[size_bytes] = []

                self.blocks[size_bytes].append(block)
                self.total_size_bytes += size_bytes

                return tensor

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Emergency cleanup
                    self._emergency_cleanup()

                    # Retry once
                    tensor = torch.empty(size_bytes // 4, dtype=torch.float32, device=self.device)
                    return tensor
                raise

    def deallocate(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse.

        Args:
            tensor: Tensor to deallocate
        """
        size_bytes = tensor.numel() * tensor.element_size()

        with self.lock:
            # Find matching block
            if size_bytes in self.blocks:
                for block in self.blocks[size_bytes]:
                    if block.tensor is tensor:
                        block.mark_free()
                        return

            # Check other sizes (for views)
            for blocks_list in self.blocks.values():
                for block in blocks_list:
                    if block.tensor is tensor:
                        block.mark_free()
                        return

    def _cleanup_idle_blocks(self, max_idle_seconds: float = 60.0):
        """Clean up blocks that have been idle for too long.

        Args:
            max_idle_seconds: Maximum idle time before cleanup
        """
        with self.lock:
            blocks_removed = 0
            bytes_freed = 0

            for size_bytes, blocks_list in list(self.blocks.items()):
                # Keep at least 1 block of each size
                free_blocks = [b for b in blocks_list if b.is_free]

                if len(free_blocks) > 1:
                    for block in free_blocks[1:]:  # Keep first free block
                        if block.idle_seconds > max_idle_seconds:
                            blocks_list.remove(block)
                            blocks_removed += 1
                            bytes_freed += block.size_bytes
                            self.total_size_bytes -= block.size_bytes

                            # Delete tensor
                            del block.tensor

                # Remove empty size classes
                if not blocks_list:
                    del self.blocks[size_bytes]

            if blocks_removed > 0:
                logger.info(
                    f"Cleaned up {blocks_removed} idle blocks, "
                    f"freed {bytes_freed / 1024**3:.2f}GB"
                )

                # Trigger CUDA cache cleanup
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

    def _emergency_cleanup(self):
        """Emergency cleanup when OOM occurs."""
        logger.warning("Emergency memory cleanup triggered")

        with self.lock:
            # Free all idle blocks immediately
            for blocks_list in self.blocks.values():
                for block in blocks_list:
                    if block.is_free:
                        del block.tensor
                        block.tensor = None

            # Clear blocks
            self.blocks.clear()
            self.total_size_bytes = 0

        # Aggressive cleanup
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_stats(self) -> MemoryPoolStats:
        """Get memory pool statistics."""
        with self.lock:
            total_blocks = sum(len(blocks) for blocks in self.blocks.values())
            free_blocks = sum(1 for blocks in self.blocks.values() for b in blocks if b.is_free)
            allocated_blocks = total_blocks - free_blocks

            free_size_bytes = sum(
                b.size_bytes for blocks in self.blocks.values() for b in blocks if b.is_free
            )
            allocated_size_bytes = self.total_size_bytes - free_size_bytes

            # Calculate hit rate
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
            miss_rate = 1.0 - hit_rate

            # Calculate fragmentation (simplified)
            num_size_classes = len(self.blocks)
            fragmentation = num_size_classes / max(1, total_blocks)

            # Calculate average block age
            all_blocks = [b for blocks in self.blocks.values() for b in blocks]
            avg_age = np.mean([b.age_seconds for b in all_blocks]) if all_blocks else 0.0

            return MemoryPoolStats(
                total_blocks=total_blocks,
                free_blocks=free_blocks,
                allocated_blocks=allocated_blocks,
                total_size_gb=self.total_size_bytes / (1024**3),
                free_size_gb=free_size_bytes / (1024**3),
                allocated_size_gb=allocated_size_bytes / (1024**3),
                hit_rate=hit_rate,
                miss_rate=miss_rate,
                fragmentation_ratio=fragmentation,
                avg_block_age_seconds=avg_age,
            )

    def cleanup(self):
        """Clean up all memory pool resources."""
        with self.lock:
            for blocks_list in self.blocks.values():
                for block in blocks_list:
                    if block.tensor is not None:
                        del block.tensor

            self.blocks.clear()
            self.total_size_bytes = 0

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info("Memory pool cleaned up")
