"""Advanced memory optimization for real-time WSI streaming.

This module implements:
- Memory pool management for GPU allocations
- Smart garbage collection strategies
- Memory usage prediction and preallocation
"""

import gc
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


class MemoryPoolStrategy(Enum):
    """Memory pool allocation strategies."""

    FIXED = "fixed"  # Fixed-size pool
    DYNAMIC = "dynamic"  # Dynamic pool that grows/shrinks
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


@dataclass
class MemoryBlock:
    """Represents a memory block in the pool."""

    size_bytes: int
    tensor: Optional[torch.Tensor] = None
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


@dataclass
class GCStats:
    """Garbage collection statistics."""

    collections_triggered: int
    memory_freed_gb: float
    avg_collection_time_ms: float
    last_collection_time: float
    blocks_collected: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "collections_triggered": self.collections_triggered,
            "memory_freed_gb": self.memory_freed_gb,
            "avg_collection_time_ms": self.avg_collection_time_ms,
            "last_collection_time": self.last_collection_time,
            "blocks_collected": self.blocks_collected,
        }


@dataclass
class MemoryPrediction:
    """Memory usage prediction."""

    predicted_peak_gb: float
    predicted_avg_gb: float
    confidence: float
    based_on_samples: int
    slide_characteristics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predicted_peak_gb": self.predicted_peak_gb,
            "predicted_avg_gb": self.predicted_avg_gb,
            "confidence": self.confidence,
            "based_on_samples": self.based_on_samples,
            "slide_characteristics": self.slide_characteristics,
        }


# ============================================================================
# Memory Pool Manager
# ============================================================================


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


# ============================================================================
# Smart Garbage Collector
# ============================================================================


class SmartGarbageCollector:
    """Smart garbage collection with adaptive strategies.

    Features:
    - Pressure-based collection triggers
    - Generational collection strategies
    - Adaptive collection thresholds
    - Performance-aware scheduling
    """

    def __init__(
        self,
        device: torch.device,
        memory_pressure_threshold: float = 0.8,
        collection_interval_seconds: float = 10.0,
        enable_adaptive: bool = True,
    ):
        """Initialize smart garbage collector.

        Args:
            device: Target device
            memory_pressure_threshold: Trigger collection above this memory usage
            collection_interval_seconds: Minimum time between collections
            enable_adaptive: Enable adaptive threshold adjustment
        """
        self.device = device
        self.memory_pressure_threshold = memory_pressure_threshold
        self.collection_interval_seconds = collection_interval_seconds
        self.enable_adaptive = enable_adaptive

        # Statistics
        self.collections_triggered = 0
        self.total_memory_freed_gb = 0.0
        self.collection_times = deque(maxlen=100)
        self.last_collection_time = 0.0

        # Adaptive thresholds
        self.min_threshold = 0.6
        self.max_threshold = 0.95
        self.threshold_adjustment_rate = 0.05

        logger.info(
            f"SmartGarbageCollector initialized: threshold={memory_pressure_threshold:.2f}, "
            f"interval={collection_interval_seconds}s"
        )

    def should_collect(self, current_memory_gb: float, total_memory_gb: float) -> bool:
        """Determine if garbage collection should be triggered.

        Args:
            current_memory_gb: Current memory usage in GB
            total_memory_gb: Total available memory in GB

        Returns:
            True if collection should be triggered
        """
        # Check time since last collection
        time_since_last = time.time() - self.last_collection_time
        if time_since_last < self.collection_interval_seconds:
            return False

        # Check memory pressure
        if total_memory_gb == 0:
            return False

        memory_pressure = current_memory_gb / total_memory_gb

        return memory_pressure >= self.memory_pressure_threshold

    def collect(self, aggressive: bool = False) -> float:
        """Perform garbage collection.

        Args:
            aggressive: If True, perform aggressive collection

        Returns:
            Memory freed in GB
        """
        start_time = time.time()

        # Get memory before collection
        if self.device.type == "cuda":
            memory_before = torch.cuda.memory_allocated(self.device) / (1024**3)
        else:
            memory_before = 0.0

        # Python garbage collection
        if aggressive:
            # Aggressive: collect all generations
            gc.collect(2)
        else:
            # Normal: collect generation 0
            gc.collect(0)

        # CUDA cache cleanup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()

        # Get memory after collection
        if self.device.type == "cuda":
            memory_after = torch.cuda.memory_allocated(self.device) / (1024**3)
        else:
            memory_after = 0.0

        memory_freed = max(0.0, memory_before - memory_after)

        # Update statistics
        collection_time = time.time() - start_time
        self.collections_triggered += 1
        self.total_memory_freed_gb += memory_freed
        self.collection_times.append(collection_time * 1000)  # Convert to ms
        self.last_collection_time = time.time()

        # Adaptive threshold adjustment
        if self.enable_adaptive:
            self._adjust_threshold(memory_freed, collection_time)

        logger.debug(f"GC completed: freed {memory_freed:.3f}GB in {collection_time*1000:.1f}ms")

        return memory_freed

    def _adjust_threshold(self, memory_freed: float, collection_time: float):
        """Adjust collection threshold based on effectiveness.

        Args:
            memory_freed: Memory freed by last collection
            collection_time: Time taken for collection
        """
        # If collection freed significant memory, we can be more conservative
        if memory_freed > 0.5:  # Freed >500MB
            self.memory_pressure_threshold = min(
                self.max_threshold, self.memory_pressure_threshold + self.threshold_adjustment_rate
            )

        # If collection freed little memory, be more aggressive
        elif memory_freed < 0.1:  # Freed <100MB
            self.memory_pressure_threshold = max(
                self.min_threshold, self.memory_pressure_threshold - self.threshold_adjustment_rate
            )

        # If collection took too long, be more conservative
        if collection_time > 0.5:  # >500ms
            self.memory_pressure_threshold = min(
                self.max_threshold, self.memory_pressure_threshold + self.threshold_adjustment_rate
            )

    def get_stats(self) -> GCStats:
        """Get garbage collection statistics."""
        avg_time_ms = np.mean(self.collection_times) if self.collection_times else 0.0

        return GCStats(
            collections_triggered=self.collections_triggered,
            memory_freed_gb=self.total_memory_freed_gb,
            avg_collection_time_ms=avg_time_ms,
            last_collection_time=self.last_collection_time,
            blocks_collected=self.collections_triggered,  # Simplified
        )


# ============================================================================
# Memory Usage Predictor
# ============================================================================


class MemoryUsagePredictor:
    """Predicts memory usage based on slide characteristics.

    Features:
    - Historical usage pattern analysis
    - Slide characteristic-based prediction
    - Confidence estimation
    - Preallocation recommendations
    """

    def __init__(self, enable_learning: bool = True):
        """Initialize memory usage predictor.

        Args:
            enable_learning: Enable learning from historical data
        """
        self.enable_learning = enable_learning

        # Historical data
        self.usage_history: List[Tuple[Dict[str, Any], float, float]] = []
        # Format: (slide_characteristics, peak_memory_gb, avg_memory_gb)

        # Prediction models (simple heuristics for now)
        self.base_memory_gb = 0.5  # Base memory overhead
        self.memory_per_patch_mb = 2.0  # Memory per patch in MB
        self.memory_per_feature_mb = 0.5  # Memory per feature in MB

        logger.info("MemoryUsagePredictor initialized")

    def predict(self, slide_characteristics: Dict[str, Any]) -> MemoryPrediction:
        """Predict memory usage for a slide.

        Args:
            slide_characteristics: Dictionary with slide properties
                - dimensions: (width, height)
                - estimated_patches: int
                - tile_size: int
                - batch_size: int
                - feature_dim: int

        Returns:
            Memory prediction with confidence
        """
        # Extract characteristics
        dimensions = slide_characteristics.get("dimensions", (10000, 10000))
        estimated_patches = slide_characteristics.get("estimated_patches", 1000)
        tile_size = slide_characteristics.get("tile_size", 224)
        batch_size = slide_characteristics.get("batch_size", 32)
        feature_dim = slide_characteristics.get("feature_dim", 512)

        # Base prediction using heuristics
        # Memory = base + (batch_size * tile_memory) + (patches * feature_memory)

        # Tile memory: batch_size * channels * tile_size^2 * bytes_per_element
        tile_memory_gb = (batch_size * 3 * tile_size * tile_size * 4) / (1024**3)

        # Feature memory: estimated_patches * feature_dim * bytes_per_element
        feature_memory_gb = (estimated_patches * feature_dim * 4) / (1024**3)

        # Peak memory (during processing)
        predicted_peak_gb = self.base_memory_gb + tile_memory_gb + feature_memory_gb

        # Average memory (steady state)
        predicted_avg_gb = self.base_memory_gb + (feature_memory_gb * 0.7)

        # Adjust based on historical data if available
        if self.enable_learning and self.usage_history:
            predicted_peak_gb, predicted_avg_gb = self._adjust_with_history(
                slide_characteristics, predicted_peak_gb, predicted_avg_gb
            )

        # Calculate confidence based on historical data
        confidence = self._calculate_confidence(slide_characteristics)

        return MemoryPrediction(
            predicted_peak_gb=predicted_peak_gb,
            predicted_avg_gb=predicted_avg_gb,
            confidence=confidence,
            based_on_samples=len(self.usage_history),
            slide_characteristics=slide_characteristics,
        )

    def _adjust_with_history(
        self, slide_characteristics: Dict[str, Any], predicted_peak: float, predicted_avg: float
    ) -> Tuple[float, float]:
        """Adjust prediction using historical data.

        Args:
            slide_characteristics: Current slide characteristics
            predicted_peak: Initial peak prediction
            predicted_avg: Initial average prediction

        Returns:
            Adjusted (peak, avg) predictions
        """
        # Find similar slides in history
        similar_samples = []

        current_patches = slide_characteristics.get("estimated_patches", 0)
        current_tile_size = slide_characteristics.get("tile_size", 224)

        for hist_chars, hist_peak, hist_avg in self.usage_history:
            hist_patches = hist_chars.get("estimated_patches", 0)
            hist_tile_size = hist_chars.get("tile_size", 224)

            # Simple similarity: within 20% of patches and same tile size
            if (
                abs(hist_patches - current_patches) / max(1, current_patches) < 0.2
                and hist_tile_size == current_tile_size
            ):
                similar_samples.append((hist_peak, hist_avg))

        # If we have similar samples, blend predictions
        if similar_samples:
            hist_peak_avg = np.mean([s[0] for s in similar_samples])
            hist_avg_avg = np.mean([s[1] for s in similar_samples])

            # Blend: 70% historical, 30% heuristic
            adjusted_peak = 0.7 * hist_peak_avg + 0.3 * predicted_peak
            adjusted_avg = 0.7 * hist_avg_avg + 0.3 * predicted_avg

            return adjusted_peak, adjusted_avg

        return predicted_peak, predicted_avg

    def _calculate_confidence(self, slide_characteristics: Dict[str, Any]) -> float:
        """Calculate prediction confidence.

        Args:
            slide_characteristics: Slide characteristics

        Returns:
            Confidence score [0, 1]
        """
        if not self.usage_history:
            return 0.5  # Medium confidence with no history

        # Find similar samples
        current_patches = slide_characteristics.get("estimated_patches", 0)
        similar_count = 0

        for hist_chars, _, _ in self.usage_history:
            hist_patches = hist_chars.get("estimated_patches", 0)
            if abs(hist_patches - current_patches) / max(1, current_patches) < 0.2:
                similar_count += 1

        # Confidence based on number of similar samples
        # 0 similar: 0.5, 5+ similar: 0.9
        confidence = 0.5 + min(0.4, similar_count * 0.08)

        return confidence

    def record_usage(
        self, slide_characteristics: Dict[str, Any], peak_memory_gb: float, avg_memory_gb: float
    ):
        """Record actual memory usage for learning.

        Args:
            slide_characteristics: Slide characteristics
            peak_memory_gb: Actual peak memory usage
            avg_memory_gb: Actual average memory usage
        """
        if not self.enable_learning:
            return

        self.usage_history.append((slide_characteristics.copy(), peak_memory_gb, avg_memory_gb))

        # Keep only recent history (last 100 slides)
        if len(self.usage_history) > 100:
            self.usage_history = self.usage_history[-100:]

        logger.debug(f"Recorded usage: peak={peak_memory_gb:.2f}GB, avg={avg_memory_gb:.2f}GB")

    def get_preallocation_recommendation(self, slide_characteristics: Dict[str, Any]) -> float:
        """Get recommended preallocation size.

        Args:
            slide_characteristics: Slide characteristics

        Returns:
            Recommended preallocation size in GB
        """
        prediction = self.predict(slide_characteristics)

        # Preallocate based on predicted peak with safety margin
        # Higher confidence = smaller margin
        safety_margin = 1.0 + (0.5 * (1.0 - prediction.confidence))

        recommended_gb = prediction.predicted_peak_gb * safety_margin

        return recommended_gb


# ============================================================================
# Memory Monitor
# ============================================================================


class MemoryPressureLevel(Enum):
    """Memory pressure levels."""

    NORMAL = "normal"  # < 60% usage
    MODERATE = "moderate"  # 60-75% usage
    HIGH = "high"  # 75-90% usage
    CRITICAL = "critical"  # > 90% usage


@dataclass
class MemorySnapshot:
    """Point-in-time memory usage snapshot."""

    timestamp: float
    allocated_gb: float
    reserved_gb: float
    total_gb: float
    pressure_level: MemoryPressureLevel

    @property
    def utilization_percent(self) -> float:
        """Calculate memory utilization percentage."""
        if self.total_gb == 0:
            return 0.0
        return (self.allocated_gb / self.total_gb) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "allocated_gb": self.allocated_gb,
            "reserved_gb": self.reserved_gb,
            "total_gb": self.total_gb,
            "utilization_percent": self.utilization_percent,
            "pressure_level": self.pressure_level.value,
        }


@dataclass
class MemoryAlert:
    """Memory alert notification."""

    timestamp: float
    alert_type: str  # 'pressure', 'threshold', 'oom_risk'
    severity: str  # 'warning', 'error', 'critical'
    message: str
    current_usage_gb: float
    threshold_gb: float
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "current_usage_gb": self.current_usage_gb,
            "threshold_gb": self.threshold_gb,
            "recommended_action": self.recommended_action,
        }


@dataclass
class MemoryAnalytics:
    """Memory usage analytics and statistics."""

    monitoring_duration_seconds: float
    total_snapshots: int
    peak_usage_gb: float
    avg_usage_gb: float
    min_usage_gb: float
    pressure_distribution: Dict[str, float]  # Percentage time in each pressure level
    alerts_triggered: int
    oom_events: int
    gc_collections: int
    memory_freed_gb: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "monitoring_duration_seconds": self.monitoring_duration_seconds,
            "total_snapshots": self.total_snapshots,
            "peak_usage_gb": self.peak_usage_gb,
            "avg_usage_gb": self.avg_usage_gb,
            "min_usage_gb": self.min_usage_gb,
            "pressure_distribution": self.pressure_distribution,
            "alerts_triggered": self.alerts_triggered,
            "oom_events": self.oom_events,
            "gc_collections": self.gc_collections,
            "memory_freed_gb": self.memory_freed_gb,
        }


class MemoryMonitor:
    """Real-time memory monitoring and alerting system.

    Features:
    - Real-time memory usage tracking with <100ms latency
    - Memory pressure detection with configurable thresholds
    - Alert generation for memory issues
    - Analytics and reporting capabilities
    - Integration with memory optimizer components
    """

    def __init__(
        self,
        device: torch.device,
        memory_limit_gb: float = 2.0,
        sampling_interval_ms: float = 100.0,
        enable_alerts: bool = True,
        alert_callback: Optional[callable] = None,
    ):
        """Initialize memory monitor.

        Args:
            device: Target device to monitor
            memory_limit_gb: Memory limit in GB for pressure calculation
            sampling_interval_ms: Sampling interval in milliseconds
            enable_alerts: Enable alert generation
            alert_callback: Optional callback function for alerts
        """
        self.device = device
        self.memory_limit_gb = memory_limit_gb
        self.sampling_interval_ms = sampling_interval_ms
        self.enable_alerts = enable_alerts
        self.alert_callback = alert_callback

        # Get total device memory
        if device.type == "cuda":
            self.total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        else:
            self.total_memory_gb = memory_limit_gb

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.start_time = None

        # Memory snapshots (circular buffer)
        self.snapshots = deque(maxlen=1000)  # Keep last 1000 snapshots

        # Alerts
        self.alerts = deque(maxlen=100)  # Keep last 100 alerts

        # Pressure thresholds
        self.pressure_thresholds = {
            MemoryPressureLevel.NORMAL: 0.60,
            MemoryPressureLevel.MODERATE: 0.75,
            MemoryPressureLevel.HIGH: 0.90,
            MemoryPressureLevel.CRITICAL: 0.95,
        }

        # Statistics
        self.peak_usage_gb = 0.0
        self.oom_events = 0
        self.alerts_triggered = 0

        # Lock for thread safety
        self.lock = threading.Lock()

        logger.info(
            f"MemoryMonitor initialized: device={device}, limit={memory_limit_gb:.2f}GB, "
            f"sampling={sampling_interval_ms}ms"
        )

    def _get_current_memory_usage(self) -> Tuple[float, float]:
        """Get current memory usage.

        Returns:
            Tuple of (allocated_gb, reserved_gb)
        """
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
            return allocated, reserved
        else:
            # For CPU, use a simple estimate
            return 0.0, 0.0

    def _calculate_pressure_level(self, allocated_gb: float) -> MemoryPressureLevel:
        """Calculate memory pressure level.

        Args:
            allocated_gb: Current allocated memory in GB

        Returns:
            Memory pressure level
        """
        if self.memory_limit_gb == 0:
            return MemoryPressureLevel.NORMAL

        utilization = allocated_gb / self.memory_limit_gb

        # Check thresholds from highest to lowest
        if utilization >= self.pressure_thresholds[MemoryPressureLevel.CRITICAL]:
            return MemoryPressureLevel.CRITICAL
        elif utilization >= self.pressure_thresholds[MemoryPressureLevel.HIGH]:
            return MemoryPressureLevel.HIGH
        elif utilization >= self.pressure_thresholds[MemoryPressureLevel.MODERATE]:
            return MemoryPressureLevel.MODERATE
        else:
            return MemoryPressureLevel.NORMAL

    def _create_snapshot(self) -> MemorySnapshot:
        """Create memory snapshot.

        Returns:
            Memory snapshot
        """
        allocated_gb, reserved_gb = self._get_current_memory_usage()
        pressure_level = self._calculate_pressure_level(allocated_gb)

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_gb=allocated_gb,
            reserved_gb=reserved_gb,
            total_gb=self.total_memory_gb,
            pressure_level=pressure_level,
        )

        # Update peak usage
        if allocated_gb > self.peak_usage_gb:
            self.peak_usage_gb = allocated_gb

        return snapshot

    def _check_and_generate_alerts(self, snapshot: MemorySnapshot):
        """Check conditions and generate alerts if needed.

        Args:
            snapshot: Current memory snapshot
        """
        if not self.enable_alerts:
            return

        # Check for critical pressure
        if snapshot.pressure_level == MemoryPressureLevel.CRITICAL:
            alert = MemoryAlert(
                timestamp=snapshot.timestamp,
                alert_type="pressure",
                severity="critical",
                message=f"Critical memory pressure: {snapshot.utilization_percent:.1f}% usage",
                current_usage_gb=snapshot.allocated_gb,
                threshold_gb=self.memory_limit_gb
                * self.pressure_thresholds[MemoryPressureLevel.CRITICAL],
                recommended_action="Reduce batch size or trigger garbage collection immediately",
            )
            self._trigger_alert(alert)

        # Check for high pressure
        elif snapshot.pressure_level == MemoryPressureLevel.HIGH:
            # Only alert if sustained high pressure (check last 3 snapshots)
            if len(self.snapshots) >= 3:
                recent_high = all(
                    s.pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]
                    for s in list(self.snapshots)[-3:]
                )

                if recent_high:
                    alert = MemoryAlert(
                        timestamp=snapshot.timestamp,
                        alert_type="pressure",
                        severity="warning",
                        message=f"Sustained high memory pressure: {snapshot.utilization_percent:.1f}% usage",
                        current_usage_gb=snapshot.allocated_gb,
                        threshold_gb=self.memory_limit_gb
                        * self.pressure_thresholds[MemoryPressureLevel.HIGH],
                        recommended_action="Consider reducing batch size or triggering garbage collection",
                    )
                    self._trigger_alert(alert)

        # Check for approaching limit
        if snapshot.allocated_gb > self.memory_limit_gb * 0.95:
            alert = MemoryAlert(
                timestamp=snapshot.timestamp,
                alert_type="threshold",
                severity="error",
                message=f"Memory usage approaching limit: {snapshot.allocated_gb:.2f}GB / {self.memory_limit_gb:.2f}GB",
                current_usage_gb=snapshot.allocated_gb,
                threshold_gb=self.memory_limit_gb,
                recommended_action="Immediate action required: reduce memory usage or risk OOM",
            )
            self._trigger_alert(alert)

    def _trigger_alert(self, alert: MemoryAlert):
        """Trigger an alert.

        Args:
            alert: Alert to trigger
        """
        with self.lock:
            self.alerts.append(alert)
            self.alerts_triggered += 1

        # Log alert
        if alert.severity == "critical":
            logger.error(f"Memory Alert [{alert.severity}]: {alert.message}")
        elif alert.severity == "error":
            logger.error(f"Memory Alert [{alert.severity}]: {alert.message}")
        else:
            logger.warning(f"Memory Alert [{alert.severity}]: {alert.message}")

        # Call callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _monitoring_loop(self):
        """Main monitoring loop (runs in separate thread)."""
        logger.info("Memory monitoring started")

        while self.is_monitoring:
            try:
                # Create snapshot
                snapshot = self._create_snapshot()

                with self.lock:
                    self.snapshots.append(snapshot)

                # Check for alerts
                self._check_and_generate_alerts(snapshot)

                # Sleep for sampling interval
                time.sleep(self.sampling_interval_ms / 1000.0)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)  # Back off on error

        logger.info("Memory monitoring stopped")

    def start_monitoring(self):
        """Start real-time memory monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return

        self.is_monitoring = True
        self.start_time = time.time()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("Memory monitoring thread started")

    def stop_monitoring(self):
        """Stop memory monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        # Wait for thread to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)

        logger.info("Memory monitoring stopped")

    def get_current_snapshot(self) -> MemorySnapshot:
        """Get current memory snapshot.

        Returns:
            Current memory snapshot
        """
        return self._create_snapshot()

    def get_recent_snapshots(self, count: int = 10) -> List[MemorySnapshot]:
        """Get recent memory snapshots.

        Args:
            count: Number of recent snapshots to return

        Returns:
            List of recent snapshots
        """
        with self.lock:
            return list(self.snapshots)[-count:]

    def get_recent_alerts(self, count: int = 10) -> List[MemoryAlert]:
        """Get recent alerts.

        Args:
            count: Number of recent alerts to return

        Returns:
            List of recent alerts
        """
        with self.lock:
            return list(self.alerts)[-count:]

    def get_analytics(self) -> MemoryAnalytics:
        """Get memory usage analytics.

        Returns:
            Memory analytics
        """
        with self.lock:
            if not self.snapshots:
                return MemoryAnalytics(
                    monitoring_duration_seconds=0.0,
                    total_snapshots=0,
                    peak_usage_gb=0.0,
                    avg_usage_gb=0.0,
                    min_usage_gb=0.0,
                    pressure_distribution={},
                    alerts_triggered=0,
                    oom_events=0,
                    gc_collections=0,
                    memory_freed_gb=0.0,
                )

            # Calculate statistics
            snapshots_list = list(self.snapshots)
            allocated_values = [s.allocated_gb for s in snapshots_list]

            avg_usage = np.mean(allocated_values)
            min_usage = np.min(allocated_values)

            # Calculate pressure distribution
            pressure_counts = {}
            for snapshot in snapshots_list:
                level = snapshot.pressure_level.value
                pressure_counts[level] = pressure_counts.get(level, 0) + 1

            total_snapshots = len(snapshots_list)
            pressure_distribution = {
                level: (count / total_snapshots) * 100.0 for level, count in pressure_counts.items()
            }

            # Calculate monitoring duration
            if self.start_time:
                duration = time.time() - self.start_time
            else:
                duration = 0.0

            return MemoryAnalytics(
                monitoring_duration_seconds=duration,
                total_snapshots=total_snapshots,
                peak_usage_gb=self.peak_usage_gb,
                avg_usage_gb=avg_usage,
                min_usage_gb=min_usage,
                pressure_distribution=pressure_distribution,
                alerts_triggered=self.alerts_triggered,
                oom_events=self.oom_events,
                gc_collections=0,  # Would need integration with SmartGC
                memory_freed_gb=0.0,  # Would need integration with SmartGC
            )

    def record_oom_event(self):
        """Record an out-of-memory event."""
        with self.lock:
            self.oom_events += 1

        # Generate critical alert
        snapshot = self._create_snapshot()
        alert = MemoryAlert(
            timestamp=time.time(),
            alert_type="oom_risk",
            severity="critical",
            message="Out of memory event detected",
            current_usage_gb=snapshot.allocated_gb,
            threshold_gb=self.memory_limit_gb,
            recommended_action="Emergency cleanup required: reduce batch size and trigger aggressive GC",
        )
        self._trigger_alert(alert)

    def set_pressure_threshold(self, level: MemoryPressureLevel, threshold: float):
        """Set custom pressure threshold.

        Args:
            level: Pressure level to configure
            threshold: Threshold value (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        with self.lock:
            self.pressure_thresholds[level] = threshold

        logger.info(f"Updated pressure threshold: {level.value} = {threshold:.2f}")

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory monitoring report.

        Returns:
            Dictionary with monitoring report
        """
        analytics = self.get_analytics()
        current_snapshot = self.get_current_snapshot()
        recent_alerts = self.get_recent_alerts(count=5)

        return {
            "current_status": current_snapshot.to_dict(),
            "analytics": analytics.to_dict(),
            "recent_alerts": [alert.to_dict() for alert in recent_alerts],
            "pressure_thresholds": {
                level.value: threshold for level, threshold in self.pressure_thresholds.items()
            },
            "monitoring_config": {
                "device": str(self.device),
                "memory_limit_gb": self.memory_limit_gb,
                "sampling_interval_ms": self.sampling_interval_ms,
                "alerts_enabled": self.enable_alerts,
            },
        }

    def cleanup(self):
        """Clean up monitoring resources."""
        self.stop_monitoring()

        with self.lock:
            self.snapshots.clear()
            self.alerts.clear()

        logger.info("Memory monitor cleaned up")

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
