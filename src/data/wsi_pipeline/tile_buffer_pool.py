"""
Tile Buffer Pool for Real-Time WSI Streaming.

This module implements a configurable memory-limited buffer pool for WSI tiles
that enables efficient tile caching and reuse during streaming operations.
Part of the WSIStreamReader class implementation for progressive tile loading.

Key Features:
- Configurable memory limits (0.5-32GB range)
- Efficient tile allocation and deallocation
- Memory pressure detection and response
- Thread-safe operations for concurrent access
- LRU eviction policy for optimal cache utilization
- Adaptive sizing based on available memory

Requirements Addressed:
- REQ-2.2.1: Memory usage below 2GB during processing
- REQ-1.1.2: Progressive tile streaming with configurable buffer sizes
- REQ-1.1.3: Adaptive tile sizing based on available memory
"""

import gc
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
from PIL import Image

from .exceptions import ProcessingError, ResourceError

logger = logging.getLogger(__name__)


@dataclass
class TileBufferConfig:
    """Configuration for tile buffer pool."""

    # Memory limits
    max_memory_gb: float = 2.0  # Maximum memory usage in GB
    min_memory_gb: float = 0.5  # Minimum memory allocation in GB

    # Buffer settings
    initial_buffer_size: int = 16  # Initial number of tiles to buffer
    max_buffer_size: int = 128  # Maximum number of tiles in buffer
    tile_size: int = 1024  # Default tile size in pixels

    # Adaptive tile sizing
    adaptive_sizing_enabled: bool = True  # Enable adaptive tile sizing
    min_tile_size: int = 64  # Minimum tile size in pixels
    max_tile_size: int = 4096  # Maximum tile size in pixels
    memory_pressure_tile_reduction: float = 0.75  # Reduce tile size by 25% under pressure

    # Memory management
    memory_pressure_threshold: float = 0.8  # Trigger cleanup at 80% usage
    cleanup_threshold: float = 0.9  # Aggressive cleanup at 90% usage
    eviction_batch_size: int = 8  # Number of tiles to evict at once

    # Performance tuning
    preload_enabled: bool = True  # Enable tile preloading
    compression_enabled: bool = False  # Enable tile compression (slower but saves memory)
    thread_safe: bool = True  # Enable thread-safe operations

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not (0.5 <= self.min_memory_gb <= 32.0):
            raise ValueError(
                f"min_memory_gb must be between 0.5 and 32.0, got {self.min_memory_gb}"
            )

        if not (0.5 <= self.max_memory_gb <= 32.0):
            raise ValueError(
                f"max_memory_gb must be between 0.5 and 32.0, got {self.max_memory_gb}"
            )

        if self.min_memory_gb > self.max_memory_gb:
            raise ValueError(
                f"min_memory_gb ({self.min_memory_gb}) cannot exceed max_memory_gb ({self.max_memory_gb})"
            )

        if not (1 <= self.initial_buffer_size <= self.max_buffer_size):
            raise ValueError(
                f"initial_buffer_size must be between 1 and max_buffer_size ({self.max_buffer_size})"
            )

        if not (64 <= self.tile_size <= 4096):
            raise ValueError(f"tile_size must be between 64 and 4096, got {self.tile_size}")

        # Validate adaptive sizing parameters
        if not (64 <= self.min_tile_size <= 4096):
            raise ValueError(f"min_tile_size must be between 64 and 4096, got {self.min_tile_size}")

        if not (64 <= self.max_tile_size <= 4096):
            raise ValueError(f"max_tile_size must be between 64 and 4096, got {self.max_tile_size}")

        if self.min_tile_size > self.max_tile_size:
            raise ValueError(
                f"min_tile_size ({self.min_tile_size}) cannot exceed max_tile_size ({self.max_tile_size})"
            )

        if not (self.min_tile_size <= self.tile_size <= self.max_tile_size):
            raise ValueError(
                f"tile_size ({self.tile_size}) must be between min_tile_size ({self.min_tile_size}) and max_tile_size ({self.max_tile_size})"
            )

        if not (0.1 <= self.memory_pressure_tile_reduction <= 1.0):
            raise ValueError(
                f"memory_pressure_tile_reduction must be between 0.1 and 1.0, got {self.memory_pressure_tile_reduction}"
            )

        if not (0.1 <= self.memory_pressure_threshold <= 1.0):
            raise ValueError(
                f"memory_pressure_threshold must be between 0.1 and 1.0, got {self.memory_pressure_threshold}"
            )

        if not (0.1 <= self.cleanup_threshold <= 1.0):
            raise ValueError(
                f"cleanup_threshold must be between 0.1 and 1.0, got {self.cleanup_threshold}"
            )

        if self.memory_pressure_threshold >= self.cleanup_threshold:
            raise ValueError("memory_pressure_threshold must be less than cleanup_threshold")


@dataclass
class TileMetadata:
    """Metadata for a cached tile."""

    coordinate: Tuple[int, int]  # (x, y) coordinate in slide space
    level: int  # Pyramid level
    size: Tuple[int, int]  # (width, height) in pixels
    memory_size: int  # Memory usage in bytes
    access_time: float  # Last access timestamp
    access_count: int  # Number of times accessed
    compressed: bool = False  # Whether tile data is compressed


class TileBufferPool:
    """
    Memory-efficient tile buffer pool for WSI streaming.

    Manages a configurable pool of WSI tiles with LRU eviction policy,
    memory pressure detection, and adaptive sizing capabilities.

    Features:
    - Configurable memory limits with automatic enforcement
    - LRU eviction policy for optimal cache utilization
    - Memory pressure detection and proactive cleanup
    - Thread-safe operations for concurrent access
    - Adaptive buffer sizing based on available memory
    - Optional tile compression for memory savings

    Example:
        >>> config = TileBufferConfig(max_memory_gb=2.0, tile_size=1024)
        >>> pool = TileBufferPool(config)
        >>>
        >>> # Store a tile
        >>> tile_data = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        >>> pool.store_tile((0, 0), 0, tile_data)
        >>>
        >>> # Retrieve a tile
        >>> retrieved_tile = pool.get_tile((0, 0), 0)
        >>>
        >>> # Check memory usage
        >>> usage = pool.get_memory_usage()
        >>> print(f"Memory usage: {usage:.2f} GB")
    """

    def __init__(self, config: TileBufferConfig):
        """
        Initialize tile buffer pool.

        Args:
            config: Buffer pool configuration

        Raises:
            ValueError: If configuration is invalid
            ResourceError: If insufficient system memory available
        """
        config.validate()
        self.config = config

        # Thread safety
        self._lock = threading.RLock() if config.thread_safe else None

        # Tile storage: OrderedDict for LRU behavior
        self._tiles: OrderedDict[Tuple[Tuple[int, int], int], np.ndarray] = OrderedDict()
        self._metadata: Dict[Tuple[Tuple[int, int], int], TileMetadata] = {}

        # Memory tracking
        self._current_memory_bytes = 0
        self._max_memory_bytes = int(config.max_memory_gb * 1024**3)
        self._min_memory_bytes = int(config.min_memory_gb * 1024**3)

        # Adaptive tile sizing
        self._current_tile_size = config.tile_size
        self._adaptive_sizing_history = []  # Track tile size changes

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "compressions": 0,
            "decompressions": 0,
            "memory_cleanups": 0,
            "tile_size_adjustments": 0,
        }

        # System memory monitoring
        self._system_memory = psutil.virtual_memory()

        # Validate system has sufficient memory
        available_gb = self._system_memory.available / 1024**3
        if available_gb < config.min_memory_gb:
            raise ResourceError(
                f"Insufficient system memory: need {config.min_memory_gb:.1f}GB, "
                f"available {available_gb:.1f}GB"
            )

        logger.info(
            f"Initialized TileBufferPool: max_memory={config.max_memory_gb:.1f}GB, "
            f"buffer_size={config.initial_buffer_size}-{config.max_buffer_size}, "
            f"tile_size={config.tile_size}px"
        )

    def _with_lock(self, func):
        """Execute function with thread lock if enabled."""
        if self._lock is not None:
            with self._lock:
                return func()
        else:
            return func()

    def _calculate_tile_memory_size(self, tile_data: np.ndarray) -> int:
        """Calculate memory size of tile data in bytes."""
        return tile_data.nbytes

    def _compress_tile(self, tile_data: np.ndarray) -> np.ndarray:
        """
        Compress tile data to save memory.

        Uses PNG compression for RGB tiles, which provides good compression
        ratios while maintaining lossless quality.

        Args:
            tile_data: Tile data as numpy array

        Returns:
            Compressed tile data as bytes
        """
        if not self.config.compression_enabled:
            return tile_data

        try:
            # Convert to PIL Image and compress as PNG
            if tile_data.dtype != np.uint8:
                # Normalize to uint8 if needed
                tile_data = (
                    (tile_data - tile_data.min()) / (tile_data.max() - tile_data.min()) * 255
                ).astype(np.uint8)

            if len(tile_data.shape) == 3 and tile_data.shape[2] == 3:
                # RGB image
                image = Image.fromarray(tile_data, mode="RGB")
            elif len(tile_data.shape) == 2:
                # Grayscale image
                image = Image.fromarray(tile_data, mode="L")
            else:
                # Unsupported format, return original
                return tile_data

            # Compress to bytes
            import io

            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            compressed_data = buffer.getvalue()

            # Convert back to numpy array for storage
            compressed_array = np.frombuffer(compressed_data, dtype=np.uint8)

            self._stats["compressions"] += 1

            logger.debug(
                f"Compressed tile: {tile_data.nbytes} -> {compressed_array.nbytes} bytes "
                f"({compressed_array.nbytes / tile_data.nbytes:.2f}x)"
            )

            return compressed_array

        except Exception as e:
            logger.warning(f"Tile compression failed: {e}, using uncompressed data")
            return tile_data

    def _decompress_tile(
        self, compressed_data: np.ndarray, original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Decompress tile data.

        Args:
            compressed_data: Compressed tile data as numpy array
            original_shape: Original shape of the tile

        Returns:
            Decompressed tile data
        """
        if not self.config.compression_enabled:
            return compressed_data

        try:
            # Convert bytes back to PIL Image
            import io

            buffer = io.BytesIO(compressed_data.tobytes())
            image = Image.open(buffer)

            # Convert back to numpy array
            tile_data = np.array(image)

            # Ensure correct shape
            if tile_data.shape != original_shape:
                logger.warning(
                    f"Decompressed tile shape mismatch: expected {original_shape}, "
                    f"got {tile_data.shape}"
                )

            self._stats["decompressions"] += 1

            return tile_data

        except Exception as e:
            logger.error(f"Tile decompression failed: {e}")
            # Return zeros with correct shape as fallback
            return np.zeros(original_shape, dtype=np.uint8)

    def _evict_lru_tiles(self, target_memory_bytes: int) -> int:
        """
        Evict least recently used tiles to free memory.

        Args:
            target_memory_bytes: Target memory usage after eviction

        Returns:
            Number of tiles evicted
        """
        evicted_count = 0

        # Evict tiles in LRU order until we reach target memory
        while self._current_memory_bytes > target_memory_bytes and len(self._tiles) > 0:

            # Get least recently used tile (first item in OrderedDict)
            key = next(iter(self._tiles))
            tile_data = self._tiles.pop(key)
            metadata = self._metadata.pop(key)

            # Update memory usage
            self._current_memory_bytes -= metadata.memory_size
            evicted_count += 1

            logger.debug(
                f"Evicted tile at {metadata.coordinate} level {metadata.level}, "
                f"freed {metadata.memory_size} bytes"
            )

            # Stop if we've evicted enough tiles in this batch
            if evicted_count >= self.config.eviction_batch_size:
                break

        if evicted_count > 0:
            self._stats["evictions"] += evicted_count
            logger.debug(
                f"Evicted {evicted_count} tiles, "
                f"memory usage: {self._current_memory_bytes / 1024**3:.3f}GB"
            )

        return evicted_count

    def _check_memory_pressure(self) -> bool:
        """
        Check if memory usage is approaching limits.

        Returns:
            True if memory pressure detected
        """
        memory_ratio = self._current_memory_bytes / self._max_memory_bytes
        return memory_ratio >= self.config.memory_pressure_threshold

    def calculate_adaptive_tile_size(self) -> int:
        """
        Calculate optimal tile size based on available memory and system conditions.

        This method implements adaptive tile sizing that adjusts tile dimensions based on:
        - Available system memory
        - Current memory usage
        - GPU memory availability
        - Processing performance metrics

        Returns:
            Optimal tile size in pixels (between min_tile_size and max_tile_size)
        """
        if not self.config.adaptive_sizing_enabled:
            return self.config.tile_size

        # Get current system memory status
        system_memory = psutil.virtual_memory()
        available_memory_gb = system_memory.available / (1024**3)
        memory_pressure = system_memory.percent > 80.0

        # Get GPU memory status if available
        gpu_memory_pressure = False
        if torch.cuda.is_available():
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_memory_pressure = (gpu_memory_used / gpu_memory_total) > 0.8
            except Exception:
                pass  # Ignore GPU memory check errors

        # Calculate base tile size based on available memory
        if available_memory_gb >= 8.0:
            # High memory: use larger tiles for better performance
            base_tile_size = min(self.config.max_tile_size, int(self.config.tile_size * 1.5))
        elif available_memory_gb >= 4.0:
            # Medium memory: use default tile size
            base_tile_size = self.config.tile_size
        elif available_memory_gb >= 2.0:
            # Low memory: reduce tile size
            base_tile_size = max(self.config.min_tile_size, int(self.config.tile_size * 0.75))
        else:
            # Very low memory: use minimum tile size
            base_tile_size = self.config.min_tile_size

        # Apply memory pressure adjustments
        if memory_pressure or gpu_memory_pressure:
            pressure_reduction = self.config.memory_pressure_tile_reduction
            base_tile_size = max(
                self.config.min_tile_size, int(base_tile_size * pressure_reduction)
            )

        # Apply buffer pool memory pressure adjustments
        if self._check_memory_pressure():
            buffer_pressure_reduction = 0.8  # Reduce by 20% under buffer pressure
            base_tile_size = max(
                self.config.min_tile_size, int(base_tile_size * buffer_pressure_reduction)
            )

        # Apply additional GPU memory pressure reduction if detected
        if gpu_memory_pressure:
            gpu_pressure_reduction = 0.6  # More aggressive reduction for GPU pressure
            base_tile_size = max(
                self.config.min_tile_size, int(base_tile_size * gpu_pressure_reduction)
            )

        # Ensure tile size is within bounds and is a power of 2 for efficiency
        adaptive_tile_size = max(
            self.config.min_tile_size, min(self.config.max_tile_size, base_tile_size)
        )

        # Round to nearest power of 2 for memory alignment efficiency
        adaptive_tile_size = 2 ** round(np.log2(adaptive_tile_size))

        # Ensure final size is within bounds after rounding
        adaptive_tile_size = max(
            self.config.min_tile_size, min(self.config.max_tile_size, adaptive_tile_size)
        )

        return adaptive_tile_size

    def update_adaptive_tile_size(self) -> bool:
        """
        Update the current tile size based on adaptive sizing algorithm.

        Returns:
            True if tile size was changed, False otherwise
        """
        if not self.config.adaptive_sizing_enabled:
            return False

        new_tile_size = self.calculate_adaptive_tile_size()

        if new_tile_size != self._current_tile_size:
            old_size = self._current_tile_size
            self._current_tile_size = new_tile_size

            # Record the change in history
            self._adaptive_sizing_history.append(
                {
                    "timestamp": time.time(),
                    "old_size": old_size,
                    "new_size": new_tile_size,
                    "memory_usage_gb": self.get_memory_usage(),
                    "system_memory_percent": psutil.virtual_memory().percent,
                }
            )

            # Keep only recent history (last 100 changes)
            if len(self._adaptive_sizing_history) > 100:
                self._adaptive_sizing_history = self._adaptive_sizing_history[-100:]

            self._stats["tile_size_adjustments"] += 1

            logger.info(
                f"Adaptive tile size changed: {old_size}px -> {new_tile_size}px "
                f"(memory usage: {self.get_memory_usage():.2f}GB, "
                f"system memory: {psutil.virtual_memory().percent:.1f}%)"
            )

            return True

        return False

    def get_current_tile_size(self) -> int:
        """
        Get the current adaptive tile size.

        Returns:
            Current tile size in pixels
        """
        return self._current_tile_size

    def get_adaptive_sizing_stats(self) -> Dict[str, Union[int, float, List]]:
        """
        Get statistics about adaptive tile sizing.

        Returns:
            Dictionary with adaptive sizing statistics
        """
        return {
            "adaptive_sizing_enabled": self.config.adaptive_sizing_enabled,
            "current_tile_size": self._current_tile_size,
            "default_tile_size": self.config.tile_size,
            "min_tile_size": self.config.min_tile_size,
            "max_tile_size": self.config.max_tile_size,
            "total_adjustments": self._stats["tile_size_adjustments"],
            "recent_changes": (
                self._adaptive_sizing_history[-10:] if self._adaptive_sizing_history else []
            ),
            "recommended_tile_size": self.calculate_adaptive_tile_size(),
        }

    def estimate_tile_memory_usage(
        self, tile_size: int, channels: int = 3, dtype_bytes: int = 1
    ) -> int:
        """
        Estimate memory usage for a tile of given size.

        Args:
            tile_size: Tile size in pixels (assumes square tiles)
            channels: Number of channels (default: 3 for RGB)
            dtype_bytes: Bytes per pixel value (default: 1 for uint8)

        Returns:
            Estimated memory usage in bytes
        """
        return tile_size * tile_size * channels * dtype_bytes

    def get_optimal_batch_size_for_tile_size(
        self, tile_size: int, target_memory_gb: float = 1.0
    ) -> int:
        """
        Calculate optimal batch size for given tile size and memory target.

        Args:
            tile_size: Tile size in pixels
            target_memory_gb: Target memory usage in GB

        Returns:
            Optimal batch size
        """
        tile_memory_bytes = self.estimate_tile_memory_usage(tile_size)
        target_memory_bytes = target_memory_gb * 1024**3

        # Account for processing overhead (2x memory for input + intermediate results)
        effective_memory_per_tile = tile_memory_bytes * 2

        optimal_batch_size = max(1, int(target_memory_bytes / effective_memory_per_tile))

        return optimal_batch_size

    def _cleanup_memory(self, aggressive: bool = False) -> None:
        """
        Perform memory cleanup operations.

        Args:
            aggressive: Whether to perform aggressive cleanup
        """

        def _cleanup():
            initial_memory = self._current_memory_bytes

            if aggressive:
                # Aggressive cleanup: reduce to 50% of max memory
                target_memory = int(self._max_memory_bytes * 0.5)
            else:
                # Normal cleanup: reduce to 70% of max memory
                target_memory = int(self._max_memory_bytes * 0.7)

            # Evict LRU tiles
            evicted = self._evict_lru_tiles(target_memory)

            # Force garbage collection
            gc.collect()

            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            freed_memory = initial_memory - self._current_memory_bytes
            self._stats["memory_cleanups"] += 1

            logger.info(
                f"Memory cleanup: evicted {evicted} tiles, "
                f"freed {freed_memory / 1024**2:.1f}MB, "
                f"usage: {self._current_memory_bytes / 1024**3:.3f}GB"
            )

        if self._lock is not None:
            with self._lock:
                _cleanup()
        else:
            _cleanup()

    def store_tile(self, coordinate: Tuple[int, int], level: int, tile_data: np.ndarray) -> bool:
        """
        Store a tile in the buffer pool.

        Args:
            coordinate: (x, y) coordinate in slide space
            level: Pyramid level
            tile_data: Tile data as numpy array

        Returns:
            True if tile was stored successfully

        Raises:
            ProcessingError: If tile storage fails
        """

        def _store():
            try:
                key = (coordinate, level)

                # Check if tile already exists
                if key in self._tiles:
                    # Update existing tile (move to end for LRU)
                    self._tiles.move_to_end(key)
                    self._metadata[key].access_time = time.time()
                    self._metadata[key].access_count += 1
                    return True

                # Calculate memory requirements
                original_size = self._calculate_tile_memory_size(tile_data)

                # Compress tile if enabled
                stored_data = self._compress_tile(tile_data)
                compressed_size = self._calculate_tile_memory_size(stored_data)

                # Check if we need to free memory or enforce buffer size limits
                required_memory = self._current_memory_bytes + compressed_size
                needs_eviction = (
                    required_memory > self._max_memory_bytes
                    or len(self._tiles) >= self.config.max_buffer_size
                )

                if needs_eviction:
                    # Calculate target memory and tile count
                    target_memory = min(
                        self._max_memory_bytes - compressed_size,
                        int(self._max_memory_bytes * 0.8),  # Leave 20% headroom
                    )
                    target_tile_count = self.config.max_buffer_size - 1  # Leave room for new tile

                    # Evict tiles until we meet both memory and count constraints
                    while (
                        self._current_memory_bytes > target_memory
                        or len(self._tiles) >= target_tile_count
                    ) and len(self._tiles) > 0:

                        # Get least recently used tile (first item in OrderedDict)
                        evict_key = next(iter(self._tiles))
                        evicted_tile_data = self._tiles.pop(evict_key)
                        evicted_metadata = self._metadata.pop(evict_key)

                        # Update memory usage
                        self._current_memory_bytes -= evicted_metadata.memory_size
                        self._stats["evictions"] += 1

                        logger.debug(
                            f"Evicted tile at {evicted_metadata.coordinate} level {evicted_metadata.level}, "
                            f"freed {evicted_metadata.memory_size} bytes"
                        )

                    # Check if we still don't have enough memory after eviction
                    if self._current_memory_bytes + compressed_size > self._max_memory_bytes:
                        logger.warning(
                            f"Cannot store tile at {coordinate} level {level}: "
                            f"insufficient memory ({compressed_size / 1024**2:.1f}MB needed)"
                        )
                        return False

                # Store tile and metadata
                self._tiles[key] = stored_data
                self._metadata[key] = TileMetadata(
                    coordinate=coordinate,
                    level=level,
                    size=tile_data.shape[:2] if len(tile_data.shape) >= 2 else (0, 0),
                    memory_size=compressed_size,
                    access_time=time.time(),
                    access_count=1,
                    compressed=self.config.compression_enabled
                    and (compressed_size < original_size),
                )

                # Update memory usage
                self._current_memory_bytes += compressed_size

                logger.debug(
                    f"Stored tile at {coordinate} level {level}: "
                    f"{compressed_size / 1024:.1f}KB, "
                    f"total memory: {self._current_memory_bytes / 1024**3:.3f}GB"
                )

                # Check for memory pressure and update adaptive tile size
                if self._check_memory_pressure():
                    # Update adaptive tile size before cleanup
                    self.update_adaptive_tile_size()
                    # Trigger cleanup in background if possible
                    self._cleanup_memory(aggressive=False)

                return True

            except Exception as e:
                raise ProcessingError(f"Failed to store tile at {coordinate} level {level}: {e}")

        return self._with_lock(_store)

    def get_tile(self, coordinate: Tuple[int, int], level: int) -> Optional[np.ndarray]:
        """
        Retrieve a tile from the buffer pool.

        Args:
            coordinate: (x, y) coordinate in slide space
            level: Pyramid level

        Returns:
            Tile data as numpy array, or None if not found
        """

        def _get():
            key = (coordinate, level)

            if key not in self._tiles:
                self._stats["misses"] += 1
                return None

            # Update access statistics
            self._stats["hits"] += 1
            metadata = self._metadata[key]
            metadata.access_time = time.time()
            metadata.access_count += 1

            # Move to end for LRU (most recently used)
            self._tiles.move_to_end(key)

            # Get tile data
            stored_data = self._tiles[key]

            # Decompress if needed
            if metadata.compressed:
                original_shape = metadata.size + (3,)  # Assume RGB for now
                tile_data = self._decompress_tile(stored_data, original_shape)
            else:
                tile_data = stored_data

            logger.debug(
                f"Retrieved tile at {coordinate} level {level} "
                f"(access count: {metadata.access_count})"
            )

            return tile_data.copy()  # Return copy to prevent modification

        return self._with_lock(_get)

    def has_tile(self, coordinate: Tuple[int, int], level: int) -> bool:
        """
        Check if a tile exists in the buffer pool.

        Args:
            coordinate: (x, y) coordinate in slide space
            level: Pyramid level

        Returns:
            True if tile exists in buffer
        """

        def _has():
            key = (coordinate, level)
            return key in self._tiles

        return self._with_lock(_has)

    def remove_tile(self, coordinate: Tuple[int, int], level: int) -> bool:
        """
        Remove a tile from the buffer pool.

        Args:
            coordinate: (x, y) coordinate in slide space
            level: Pyramid level

        Returns:
            True if tile was removed, False if not found
        """

        def _remove():
            key = (coordinate, level)

            if key not in self._tiles:
                return False

            # Remove tile and metadata
            self._tiles.pop(key)
            metadata = self._metadata.pop(key)

            # Update memory usage
            self._current_memory_bytes -= metadata.memory_size

            logger.debug(
                f"Removed tile at {coordinate} level {level}, "
                f"freed {metadata.memory_size} bytes"
            )

            return True

        return self._with_lock(_remove)

    def clear(self) -> None:
        """Clear all tiles from the buffer pool."""

        def _clear():
            tile_count = len(self._tiles)
            memory_freed = self._current_memory_bytes

            self._tiles.clear()
            self._metadata.clear()
            self._current_memory_bytes = 0

            # Force garbage collection
            gc.collect()

            logger.info(
                f"Cleared buffer pool: removed {tile_count} tiles, "
                f"freed {memory_freed / 1024**3:.3f}GB"
            )

        self._with_lock(_clear)

    def get_memory_usage(self) -> float:
        """
        Get current memory usage in GB.

        Returns:
            Memory usage in GB
        """
        return self._current_memory_bytes / 1024**3

    def get_memory_limit(self) -> float:
        """
        Get memory limit in GB.

        Returns:
            Memory limit in GB
        """
        return self._max_memory_bytes / 1024**3

    def get_buffer_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get buffer pool statistics.

        Returns:
            Dictionary with buffer statistics
        """

        def _stats():
            total_accesses = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_accesses if total_accesses > 0 else 0.0

            stats = {
                "tile_count": len(self._tiles),
                "memory_usage_gb": self.get_memory_usage(),
                "memory_limit_gb": self.get_memory_limit(),
                "memory_utilization": self.get_memory_usage() / self.get_memory_limit(),
                "hit_rate": hit_rate,
                "total_hits": self._stats["hits"],
                "total_misses": self._stats["misses"],
                "total_evictions": self._stats["evictions"],
                "total_compressions": self._stats["compressions"],
                "total_decompressions": self._stats["decompressions"],
                "memory_cleanups": self._stats["memory_cleanups"],
                "tile_size_adjustments": self._stats["tile_size_adjustments"],
                "current_tile_size": self._current_tile_size,
                "adaptive_sizing_enabled": self.config.adaptive_sizing_enabled,
            }

            return stats

        return self._with_lock(_stats)

    def optimize_memory_usage(self) -> Dict[str, Union[int, float]]:
        """
        Optimize memory usage by cleaning up and adjusting buffer size and tile size.

        Returns:
            Dictionary with optimization results
        """

        def _optimize():
            initial_memory = self._current_memory_bytes
            initial_tiles = len(self._tiles)
            initial_tile_size = self._current_tile_size

            # Check system memory pressure
            system_memory = psutil.virtual_memory()
            system_pressure = system_memory.percent > 80.0

            # Update adaptive tile size first
            tile_size_changed = self.update_adaptive_tile_size()

            if system_pressure or self._check_memory_pressure():
                self._cleanup_memory(aggressive=system_pressure)

            # Adaptive buffer sizing based on hit rate
            stats = self.get_buffer_stats()
            hit_rate = stats["hit_rate"]

            # If hit rate is low, we might be thrashing - reduce buffer size
            if hit_rate < 0.5 and len(self._tiles) > self.config.initial_buffer_size:
                target_tiles = max(self.config.initial_buffer_size, int(len(self._tiles) * 0.8))

                # Evict tiles to reach target size
                while len(self._tiles) > target_tiles:
                    self._evict_lru_tiles(0)  # Evict one batch

            final_memory = self._current_memory_bytes
            final_tiles = len(self._tiles)
            final_tile_size = self._current_tile_size

            return {
                "initial_memory_gb": initial_memory / 1024**3,
                "final_memory_gb": final_memory / 1024**3,
                "memory_freed_gb": (initial_memory - final_memory) / 1024**3,
                "initial_tiles": initial_tiles,
                "final_tiles": final_tiles,
                "tiles_evicted": initial_tiles - final_tiles,
                "initial_tile_size": initial_tile_size,
                "final_tile_size": final_tile_size,
                "tile_size_changed": tile_size_changed,
                "system_memory_pressure": system_pressure,
            }

        return self._with_lock(_optimize)

    def preload_tiles(
        self, coordinates: List[Tuple[int, int]], level: int, tile_loader_func
    ) -> int:
        """
        Preload tiles into the buffer pool.

        Args:
            coordinates: List of (x, y) coordinates to preload
            level: Pyramid level
            tile_loader_func: Function to load tile data, signature: (x, y, level) -> np.ndarray

        Returns:
            Number of tiles successfully preloaded
        """
        if not self.config.preload_enabled:
            return 0

        def _preload():
            loaded_count = 0

            for coordinate in coordinates:
                # Skip if tile already exists
                if self.has_tile(coordinate, level):
                    continue

                # Check memory pressure before loading
                if self._check_memory_pressure():
                    logger.debug("Stopping preload due to memory pressure")
                    break

                try:
                    # Load tile data
                    tile_data = tile_loader_func(coordinate[0], coordinate[1], level)

                    if tile_data is not None:
                        # Store in buffer
                        if self.store_tile(coordinate, level, tile_data):
                            loaded_count += 1

                except Exception as e:
                    logger.warning(f"Failed to preload tile at {coordinate}: {e}")
                    continue

            logger.debug(f"Preloaded {loaded_count} tiles at level {level}")
            return loaded_count

        return self._with_lock(_preload)

    def get_adaptive_buffer_size(self) -> int:
        """
        Calculate adaptive buffer size based on current performance.

        Returns:
            Recommended buffer size
        """
        stats = self.get_buffer_stats()
        hit_rate = stats["hit_rate"]
        memory_utilization = stats["memory_utilization"]

        current_size = len(self._tiles)

        # Increase buffer size if hit rate is high and memory allows
        if hit_rate > 0.8 and memory_utilization < 0.7:
            recommended_size = min(self.config.max_buffer_size, int(current_size * 1.2))
        # Decrease buffer size if hit rate is low or memory pressure
        elif hit_rate < 0.5 or memory_utilization > 0.9:
            recommended_size = max(self.config.initial_buffer_size, int(current_size * 0.8))
        else:
            recommended_size = current_size

        return recommended_size

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.clear()

    def __del__(self):
        """Destructor - cleanup resources."""
        try:
            self.clear()
        except Exception:
            pass  # Ignore errors during cleanup
