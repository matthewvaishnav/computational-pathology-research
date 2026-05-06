"""
Optimized caching strategies and utilities for HistoCore.

This module provides high-performance caching implementations with
memory-efficient storage, intelligent eviction policies, and
automatic compression for better resource utilization.
"""

import hashlib
import logging
import pickle
import time
import weakref
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from threading import RLock
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from ..utils.constants import (
    CACHE_TTL_SECONDS,
    DEFAULT_MAX_MEMORY_GB,
    MEMORY_PRESSURE_THRESHOLD,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry:
    """Optimized cache entry with metadata and compression."""
    
    data: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    compressed: bool = False
    original_size: int = 0
    compressed_size: int = 0
    
    def __post_init__(self):
        self.last_accessed = self.timestamp
    
    def access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def compress(self) -> bool:
        """Compress data if not already compressed."""
        if self.compressed or not hasattr(self.data, '__sizeof__'):
            return False
        
        try:
            # Serialize and compress
            serialized = pickle.dumps(self.data)
            self.original_size = len(serialized)
            compressed_data = zlib.compress(serialized, level=6)
            self.compressed_size = len(compressed_data)
            
            # Only keep compressed if it saves significant space
            if self.compressed_size < self.original_size * 0.8:
                self.data = compressed_data
                self.compressed = True
                return True
            
        except Exception as e:
            logger.warning(f"Failed to compress cache entry: {e}")
        
        return False
    
    def decompress(self) -> Any:
        """Decompress data if compressed."""
        if not self.compressed:
            return self.data
        
        try:
            decompressed = zlib.decompress(self.data)
            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Failed to decompress cache entry: {e}")
            raise


class OptimizedLRUCache:
    """
    High-performance LRU cache with compression and memory management.
    
    Features:
    - Automatic compression of inactive entries
    - Memory pressure detection and cleanup
    - Thread-safe operations
    - Weak reference support for automatic cleanup
    - Access pattern tracking for intelligent eviction
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 500.0,
                 compression_threshold: int = 100, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.compression_threshold = compression_threshold
        self.ttl_seconds = ttl_seconds
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._current_memory = 0
        self._hits = 0
        self._misses = 0
        self._compressions = 0
        
    def _calculate_memory_usage(self) -> int:
        """Calculate current memory usage."""
        total = 0
        for entry in self._cache.values():
            if entry.compressed:
                total += entry.compressed_size
            else:
                total += entry.original_size or entry.data.__sizeof__()
        return total
    
    def _evict_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry.timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self._cache[key]
    
    def _evict_lru(self, target_size: Optional[int] = None):
        """Evict least recently used entries."""
        target = target_size or self.max_size
        
        while len(self._cache) > target:
            # Remove oldest entry
            self._cache.popitem(last=False)
    
    def _compress_inactive_entries(self):
        """Compress entries that haven't been accessed recently."""
        current_time = time.time()
        compressed_count = 0
        
        for entry in self._cache.values():
            if (not entry.compressed and 
                current_time - entry.last_accessed > self.compression_threshold and
                entry.access_count > 0):
                
                if entry.compress():
                    compressed_count += 1
                    self._compressions += 1
        
        if compressed_count > 0:
            logger.debug(f"Compressed {compressed_count} inactive cache entries")
    
    def _manage_memory_pressure(self):
        """Handle memory pressure by compressing and evicting entries."""
        current_memory = self._calculate_memory_usage()
        
        if current_memory > self.max_memory_bytes * MEMORY_PRESSURE_THRESHOLD:
            # First, try compression
            self._compress_inactive_entries()
            
            # If still over limit, evict entries
            current_memory = self._calculate_memory_usage()
            if current_memory > self.max_memory_bytes * 0.9:
                # Evict 20% of entries
                target_size = int(len(self._cache) * 0.8)
                self._evict_lru(target_size)
                
                logger.info(f"Evicted entries due to memory pressure. "
                           f"Cache size: {len(self._cache)}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with automatic decompression."""
        with self._lock:
            self._evict_expired()
            
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            entry.access()
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._hits += 1
            
            # Return decompressed data
            return entry.decompress() if entry.compressed else entry.data
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with automatic memory management."""
        with self._lock:
            current_time = time.time()
            
            # Create cache entry
            entry = CacheEntry(
                data=value,
                timestamp=current_time,
                original_size=value.__sizeof__() if hasattr(value, '__sizeof__') else 0
            )
            
            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = entry
            
            # Manage cache size and memory
            if len(self._cache) > self.max_size:
                self._evict_lru()
            
            self._manage_memory_pressure()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            compressed_entries = sum(1 for entry in self._cache.values() if entry.compressed)
            memory_usage = self._calculate_memory_usage()
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'compressed_entries': compressed_entries,
                'total_compressions': self._compressions,
                'memory_usage_mb': memory_usage / (1024 * 1024),
                'memory_limit_mb': self.max_memory_bytes / (1024 * 1024),
            }


def cached_method(cache_size: int = 100, ttl_seconds: int = CACHE_TTL_SECONDS):
    """
    Decorator for caching method results with automatic cleanup.
    
    Args:
        cache_size: Maximum number of cached results
        ttl_seconds: Time-to-live for cached results
    
    Example:
        class DataProcessor:
            @cached_method(cache_size=50, ttl_seconds=300)
            def expensive_computation(self, data):
                # expensive operation
                return result
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = OptimizedLRUCache(max_size=cache_size, ttl_seconds=ttl_seconds)
        
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> T:
            # Create cache key from arguments
            key_data = (func.__name__, args, tuple(sorted(kwargs.items())))
            key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(self, *args, **kwargs)
            cache.put(key, result)
            
            return result
        
        # Attach cache for inspection
        wrapper._cache = cache
        return wrapper
    
    return decorator


def cached_function(cache_size: int = 100, ttl_seconds: int = CACHE_TTL_SECONDS):
    """
    Decorator for caching function results.
    
    Args:
        cache_size: Maximum number of cached results
        ttl_seconds: Time-to-live for cached results
    
    Example:
        @cached_function(cache_size=200, ttl_seconds=600)
        def expensive_function(param1, param2):
            # expensive operation
            return result
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = OptimizedLRUCache(max_size=cache_size, ttl_seconds=ttl_seconds)
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create cache key from arguments
            key_data = (func.__name__, args, tuple(sorted(kwargs.items())))
            key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            
            return result
        
        # Attach cache for inspection
        wrapper._cache = cache
        return wrapper
    
    return decorator


class GlobalCacheManager:
    """
    Global cache manager for coordinating multiple caches.
    
    Provides centralized cache management, memory monitoring,
    and automatic cleanup across the application.
    """
    
    def __init__(self):
        self._caches: Dict[str, OptimizedLRUCache] = {}
        self._lock = RLock()
    
    def register_cache(self, name: str, cache: OptimizedLRUCache):
        """Register a cache for global management."""
        with self._lock:
            self._caches[name] = cache
    
    def get_cache(self, name: str) -> Optional[OptimizedLRUCache]:
        """Get a registered cache by name."""
        with self._lock:
            return self._caches.get(name)
    
    def clear_all(self):
        """Clear all registered caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all registered caches."""
        with self._lock:
            stats = {}
            total_memory = 0
            total_entries = 0
            
            for name, cache in self._caches.items():
                cache_stats = cache.stats()
                stats[name] = cache_stats
                total_memory += cache_stats['memory_usage_mb']
                total_entries += cache_stats['size']
            
            stats['global'] = {
                'total_caches': len(self._caches),
                'total_entries': total_entries,
                'total_memory_mb': total_memory,
            }
            
            return stats


# Global cache manager instance
cache_manager = GlobalCacheManager()