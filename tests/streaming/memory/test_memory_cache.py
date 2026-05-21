"""Tests for cache manager."""

import time

import numpy as np
import pytest

from src.streaming.memory.cache_manager import CacheEntry, CacheManager


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            size_bytes=1024,
            created_at=time.time(),
            last_accessed=time.time(),
        )

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.size_bytes == 1024
        assert entry.access_count == 0

    def test_mark_accessed(self):
        """Test marking entry as accessed."""
        entry = CacheEntry(
            key="test",
            value="data",
            size_bytes=100,
            created_at=time.time(),
            last_accessed=time.time(),
        )

        initial_access_count = entry.access_count
        initial_last_accessed = entry.last_accessed

        time.sleep(0.01)
        entry.mark_accessed()

        assert entry.access_count == initial_access_count + 1
        assert entry.last_accessed > initial_last_accessed

    def test_age_seconds(self):
        """Test age calculation."""
        entry = CacheEntry(
            key="test",
            value="data",
            size_bytes=100,
            created_at=time.time() - 5.0,  # Created 5 seconds ago
            last_accessed=time.time(),
        )

        assert entry.age_seconds >= 5.0
        assert entry.age_seconds < 6.0

    def test_idle_seconds(self):
        """Test idle time calculation."""
        entry = CacheEntry(
            key="test",
            value="data",
            size_bytes=100,
            created_at=time.time(),
            last_accessed=time.time() - 3.0,  # Last accessed 3 seconds ago
        )

        assert entry.idle_seconds >= 3.0
        assert entry.idle_seconds < 4.0


class TestCacheManager:
    """Tests for CacheManager."""

    def test_cache_manager_initialization(self):
        """Test cache manager initialization."""
        cache = CacheManager(max_size_mb=100.0)

        assert cache.max_size_bytes == 100 * 1024 * 1024
        assert len(cache) == 0
        assert cache.current_size_bytes == 0

    def test_put_and_get(self):
        """Test putting and getting items."""
        cache = CacheManager(max_size_mb=10.0)

        # Put item
        cache.put("key1", "value1", size_bytes=1024)

        # Get item
        value = cache.get("key1")
        assert value == "value1"

        # Check stats
        stats = cache.get_stats()
        assert stats["total_puts"] == 1
        assert stats["total_gets"] == 1
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 0

    def test_cache_miss(self):
        """Test cache miss."""
        cache = CacheManager(max_size_mb=10.0)

        # Get non-existent item
        value = cache.get("nonexistent")
        assert value is None

        # Check stats
        stats = cache.get_stats()
        assert stats["cache_misses"] == 1
        assert stats["cache_hits"] == 0

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = CacheManager(max_size_mb=0.01)  # 10KB cache

        # Fill cache
        cache.put("key1", "value1", size_bytes=4096)  # 4KB
        cache.put("key2", "value2", size_bytes=4096)  # 4KB
        cache.put("key3", "value3", size_bytes=4096)  # 4KB - should evict key1

        # key1 should be evicted (LRU)
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

        # Check eviction stats
        stats = cache.get_stats()
        assert stats["total_evictions"] >= 1

    def test_lru_ordering(self):
        """Test that LRU ordering is maintained."""
        cache = CacheManager(max_size_mb=0.01)  # 10KB cache

        # Add items
        cache.put("key1", "value1", size_bytes=3000)
        cache.put("key2", "value2", size_bytes=3000)
        cache.put("key3", "value3", size_bytes=3000)

        # Access key1 (makes it most recently used)
        cache.get("key1")

        # Add key4 (should evict key2, not key1)
        cache.put("key4", "value4", size_bytes=3000)

        # key2 should be evicted, key1 should still be there
        assert cache.get("key2") is None
        assert cache.get("key1") == "value1"
        assert cache.get("key4") == "value4"

    def test_update_existing_key(self):
        """Test updating an existing key."""
        cache = CacheManager(max_size_mb=10.0)

        # Put initial value
        cache.put("key1", "value1", size_bytes=1024)
        assert cache.get("key1") == "value1"

        # Update value
        cache.put("key1", "value2", size_bytes=2048)
        assert cache.get("key1") == "value2"

        # Size should be updated
        stats = cache.get_stats()
        assert stats["current_size_mb"] == pytest.approx(2048 / (1024 * 1024), rel=0.01)

    def test_manual_evict(self):
        """Test manual eviction."""
        cache = CacheManager(max_size_mb=10.0)

        cache.put("key1", "value1", size_bytes=1024)
        assert cache.contains("key1")

        # Manually evict
        result = cache.evict("key1")
        assert result is True
        assert not cache.contains("key1")

        # Try to evict non-existent key
        result = cache.evict("nonexistent")
        assert result is False

    def test_contains(self):
        """Test contains method."""
        cache = CacheManager(max_size_mb=10.0)

        cache.put("key1", "value1", size_bytes=1024)

        assert cache.contains("key1")
        assert "key1" in cache
        assert not cache.contains("key2")
        assert "key2" not in cache

    def test_clear(self):
        """Test clearing cache."""
        cache = CacheManager(max_size_mb=10.0)

        # Add items
        cache.put("key1", "value1", size_bytes=1024)
        cache.put("key2", "value2", size_bytes=1024)

        assert len(cache) == 2

        # Clear
        cache.clear()

        assert len(cache) == 0
        assert cache.current_size_bytes == 0
        assert cache.get("key1") is None

    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = CacheManager(max_size_mb=10.0)

        # Perform operations
        cache.put("key1", "value1", size_bytes=1024)
        cache.put("key2", "value2", size_bytes=2048)
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss

        stats = cache.get_stats()

        assert stats["total_entries"] == 2
        assert stats["total_puts"] == 2
        assert stats["total_gets"] == 2
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["miss_rate"] == 0.5
        assert stats["current_size_mb"] > 0

    def test_get_entry_info(self):
        """Test getting entry information."""
        cache = CacheManager(max_size_mb=10.0)

        cache.put("key1", "value1", size_bytes=1024)

        info = cache.get_entry_info("key1")
        assert info is not None
        assert info["key"] == "key1"
        assert info["size_bytes"] == 1024
        assert info["access_count"] == 0
        assert "age_seconds" in info
        assert "idle_seconds" in info

        # Non-existent key
        info = cache.get_entry_info("nonexistent")
        assert info is None

    def test_get_all_keys(self):
        """Test getting all cache keys."""
        cache = CacheManager(max_size_mb=10.0)

        cache.put("key1", "value1", size_bytes=1024)
        cache.put("key2", "value2", size_bytes=1024)
        cache.put("key3", "value3", size_bytes=1024)

        keys = cache.get_all_keys()
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys

    def test_resize(self):
        """Test resizing cache."""
        cache = CacheManager(max_size_mb=10.0)

        # Fill cache
        cache.put("key1", "value1", size_bytes=3 * 1024 * 1024)  # 3MB
        cache.put("key2", "value2", size_bytes=3 * 1024 * 1024)  # 3MB
        cache.put("key3", "value3", size_bytes=3 * 1024 * 1024)  # 3MB

        assert len(cache) == 3

        # Resize to smaller size
        cache.resize(5.0)  # 5MB

        # Should evict entries to fit new size
        assert len(cache) <= 2  # At most 2 entries can fit in 5MB

    def test_size_estimation_numpy(self):
        """Test size estimation for NumPy arrays."""
        cache = CacheManager(max_size_mb=10.0)

        # Create NumPy array
        arr = np.zeros((100, 100), dtype=np.float32)
        expected_size = arr.nbytes

        cache.put("array", arr)  # Size auto-estimated

        info = cache.get_entry_info("array")
        assert info["size_bytes"] == expected_size

    def test_thread_safety(self):
        """Test thread-safe operations."""
        import threading

        cache = CacheManager(max_size_mb=10.0)
        errors = []

        def worker(thread_id):
            try:
                for i in range(100):
                    key = f"key_{thread_id}_{i}"
                    cache.put(key, f"value_{i}", size_bytes=1024)
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0

    def test_repr(self):
        """Test string representation."""
        cache = CacheManager(max_size_mb=10.0)
        cache.put("key1", "value1", size_bytes=1024)

        repr_str = repr(cache)
        assert "CacheManager" in repr_str
        assert "entries=1" in repr_str
        assert "MB" in repr_str
