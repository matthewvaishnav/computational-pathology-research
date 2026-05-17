"""LRU cache management for WSI streaming.

This module provides LRU (Least Recently Used) cache management for tiles
and features during whole slide image (WSI) processing.

Components:
    - CacheEntry: Represents a cached item with metadata
    - CacheManager: LRU cache with configurable size and eviction

Usage:
    from streaming.memory.cache_manager import CacheManager

    cache = CacheManager(max_size_mb=1000)
    cache.put("tile_123", tile_data)
    tile = cache.get("tile_123")

    # Check cache stats
    stats = cache.get_stats()
    print(f"Hit rate: {stats['hit_rate']:.2%}")
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""

    key: str
    value: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0

    def mark_accessed(self):
        """Mark entry as accessed."""
        self.last_accessed = time.time()
        self.access_count += 1

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_accessed


# ============================================================================
# Cache Manager
# ============================================================================


class CacheManager:
    """LRU cache manager for tiles and features.

    Features:
    - LRU (Least Recently Used) eviction policy
    - Configurable size limit
    - Thread-safe operations
    - Cache statistics tracking
    - Automatic eviction when size limit exceeded

    Example:
        >>> cache = CacheManager(max_size_mb=1000)
        >>> cache.put("tile_1", tile_data)
        >>> tile = cache.get("tile_1")
        >>> stats = cache.get_stats()
    """

    def __init__(
        self,
        max_size_mb: float = 1000.0,
        enable_stats: bool = True,
    ):
        """Initialize cache manager.

        Args:
            max_size_mb: Maximum cache size in MB
            enable_stats: Enable statistics tracking
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.enable_stats = enable_stats

        # LRU cache using OrderedDict
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0

        # Thread safety
        self.lock = threading.Lock()

        # Statistics
        self.total_gets = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_puts = 0
        self.total_evictions = 0
        self.total_bytes_evicted = 0

        logger.info(f"CacheManager initialized: max_size={max_size_mb:.2f}MB")

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found, None otherwise
        """
        with self.lock:
            self.total_gets += 1

            if key in self.cache:
                # Cache hit - move to end (most recently used)
                entry = self.cache.pop(key)
                entry.mark_accessed()
                self.cache[key] = entry

                self.cache_hits += 1
                return entry.value
            else:
                # Cache miss
                self.cache_misses += 1
                return None

    def put(self, key: str, value: Any, size_bytes: Optional[int] = None):
        """Put item into cache.

        Args:
            key: Cache key
            value: Value to cache
            size_bytes: Size of value in bytes (estimated if not provided)
        """
        with self.lock:
            self.total_puts += 1

            # Estimate size if not provided
            if size_bytes is None:
                size_bytes = self._estimate_size(value)

            # Check if key already exists
            if key in self.cache:
                # Update existing entry
                old_entry = self.cache.pop(key)
                self.current_size_bytes -= old_entry.size_bytes

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=0,
            )

            # Evict entries if necessary to make room
            while self.current_size_bytes + size_bytes > self.max_size_bytes and self.cache:
                self._evict_lru()

            # Add new entry (most recently used)
            self.cache[key] = entry
            self.current_size_bytes += size_bytes

    def evict(self, key: str) -> bool:
        """Manually evict a specific key from cache.

        Args:
            key: Cache key to evict

        Returns:
            True if key was evicted, False if not found
        """
        with self.lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.current_size_bytes -= entry.size_bytes
                self.total_evictions += 1
                self.total_bytes_evicted += entry.size_bytes
                return True
            return False

    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.cache:
            return

        # Pop first item (least recently used)
        key, entry = self.cache.popitem(last=False)
        self.current_size_bytes -= entry.size_bytes
        self.total_evictions += 1
        self.total_bytes_evicted += entry.size_bytes

        logger.debug(
            f"Evicted LRU entry: key={key}, size={entry.size_bytes / 1024:.2f}KB, "
            f"age={entry.age_seconds:.1f}s, accesses={entry.access_count}"
        )

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes.

        Args:
            value: Value to estimate

        Returns:
            Estimated size in bytes
        """
        try:
            # Try to get size from common types
            if hasattr(value, "nbytes"):  # NumPy arrays, PyTorch tensors
                return int(value.nbytes)
            elif hasattr(value, "__sizeof__"):
                return value.__sizeof__()
            else:
                # Fallback: assume 1KB
                return 1024
        except Exception:
            return 1024

    def contains(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        with self.lock:
            return key in self.cache

    def clear(self):
        """Clear all entries from cache."""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            hit_rate = self.cache_hits / self.total_gets if self.total_gets > 0 else 0.0
            miss_rate = 1.0 - hit_rate

            return {
                "total_entries": len(self.cache),
                "current_size_mb": self.current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization_percent": (
                    (self.current_size_bytes / self.max_size_bytes) * 100.0
                    if self.max_size_bytes > 0
                    else 0.0
                ),
                "total_gets": self.total_gets,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": hit_rate,
                "miss_rate": miss_rate,
                "total_puts": self.total_puts,
                "total_evictions": self.total_evictions,
                "total_bytes_evicted_mb": self.total_bytes_evicted / (1024 * 1024),
            }

    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get information about a cached entry.

        Args:
            key: Cache key

        Returns:
            Dictionary with entry information, or None if not found
        """
        with self.lock:
            if key not in self.cache:
                return None

            entry = self.cache[key]
            return {
                "key": entry.key,
                "size_bytes": entry.size_bytes,
                "age_seconds": entry.age_seconds,
                "idle_seconds": entry.idle_seconds,
                "access_count": entry.access_count,
                "created_at": entry.created_at,
                "last_accessed": entry.last_accessed,
            }

    def get_all_keys(self) -> list[str]:
        """Get all cache keys.

        Returns:
            List of cache keys (ordered from LRU to MRU)
        """
        with self.lock:
            return list(self.cache.keys())

    def resize(self, new_max_size_mb: float):
        """Resize cache to new maximum size.

        Args:
            new_max_size_mb: New maximum size in MB
        """
        with self.lock:
            old_max = self.max_size_bytes
            self.max_size_bytes = int(new_max_size_mb * 1024 * 1024)

            # Evict entries if new size is smaller
            while self.current_size_bytes > self.max_size_bytes and self.cache:
                self._evict_lru()

            logger.info(
                f"Cache resized: {old_max / (1024 * 1024):.2f}MB -> " f"{new_max_size_mb:.2f}MB"
            )

    def __len__(self) -> int:
        """Get number of entries in cache."""
        with self.lock:
            return len(self.cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.contains(key)

    def __repr__(self) -> str:
        """String representation of cache manager."""
        with self.lock:
            return (
                f"CacheManager(entries={len(self.cache)}, "
                f"size={self.current_size_bytes / (1024 * 1024):.2f}MB, "
                f"max={self.max_size_bytes / (1024 * 1024):.2f}MB)"
            )
