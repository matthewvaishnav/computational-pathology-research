"""
Tests for streaming cache functionality.

Tests cache operations, serialization, and error handling.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.exceptions import CacheError, CacheSerializationError

pytestmark = pytest.mark.skip(reason="Requires OpenSlide DLL")


class TestCacheConfiguration:
    """Test cache configuration and initialization."""

    def test_cache_config_defaults(self):
        """Test CacheConfig has sensible defaults."""
        from src.streaming.cache import CacheConfig
        
        config = CacheConfig()
        
        assert config.redis_host == "localhost"
        assert config.redis_port == 6379
        assert config.max_memory_mb > 0
        assert config.ttl_seconds > 0

    def test_cache_config_custom_values(self):
        """Test CacheConfig accepts custom values."""
        from src.streaming.cache import CacheConfig
        
        config = CacheConfig(
            redis_host="custom_host",
            redis_port=9999,
            max_memory_mb=2048,
            ttl_seconds=7200
        )
        
        assert config.redis_host == "custom_host"
        assert config.redis_port == 9999
        assert config.max_memory_mb == 2048
        assert config.ttl_seconds == 7200


class TestCacheSerialization:
    """Test cache serialization and deserialization."""

    def test_serialize_simple_data(self):
        """Test serialization of simple data types."""
        from src.streaming.cache import CacheSerializer
        
        serializer = CacheSerializer()
        
        # Test dict
        data = {"key": "value", "number": 42}
        serialized = serializer.serialize(data)
        assert isinstance(serialized, bytes)
        
        deserialized = serializer.deserialize(serialized)
        assert deserialized == data

    def test_serialize_numpy_array(self):
        """Test serialization of numpy arrays."""
        import numpy as np
        from src.streaming.cache import CacheSerializer
        
        serializer = CacheSerializer()
        
        # Test numpy array
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        serialized = serializer.serialize(arr)
        assert isinstance(serialized, bytes)
        
        deserialized = serializer.deserialize(serialized)
        assert isinstance(deserialized, np.ndarray)
        assert np.array_equal(deserialized, arr)

    def test_deserialize_corrupted_data_raises_error(self):
        """Test deserialization of corrupted data raises CacheSerializationError."""
        from src.streaming.cache import CacheSerializer
        
        serializer = CacheSerializer()
        
        # Corrupted pickle data
        corrupted = b"corrupted_pickle_data\x00\x01\x02"
        
        with pytest.raises(CacheSerializationError):
            serializer.deserialize(corrupted)


class TestCacheOperations:
    """Test cache get/set/delete operations."""

    @pytest.fixture
    def mock_redis_cache(self):
        """Create a mock Redis cache for testing."""
        from src.streaming.cache import RedisCache, CacheConfig
        
        with patch('redis.Redis') as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client
            
            config = CacheConfig()
            cache = RedisCache(config)
            cache._client = mock_client
            
            yield cache, mock_client

    def test_cache_set_and_get(self, mock_redis_cache):
        """Test cache set and get operations."""
        cache, mock_client = mock_redis_cache
        
        # Mock get to return serialized data
        test_data = {"key": "value"}
        import pickle
        mock_client.get.return_value = pickle.dumps(test_data)
        
        # Set data
        cache.set("test_key", test_data)
        mock_client.set.assert_called_once()
        
        # Get data
        result = cache.get("test_key")
        mock_client.get.assert_called_once_with("test_key")
        assert result == test_data

    def test_cache_delete(self, mock_redis_cache):
        """Test cache delete operation."""
        cache, mock_client = mock_redis_cache
        
        cache.delete("test_key")
        mock_client.delete.assert_called_once_with("test_key")

    def test_cache_exists(self, mock_redis_cache):
        """Test cache exists check."""
        cache, mock_client = mock_redis_cache
        
        mock_client.exists.return_value = True
        
        result = cache.exists("test_key")
        mock_client.exists.assert_called_once_with("test_key")
        assert result is True

    def test_cache_clear(self, mock_redis_cache):
        """Test cache clear operation."""
        cache, mock_client = mock_redis_cache
        
        cache.clear()
        mock_client.flushdb.assert_called_once()


class TestCacheErrorHandling:
    """Test cache error handling."""

    def test_cache_get_nonexistent_key_returns_none(self):
        """Test getting nonexistent key returns None."""
        from src.streaming.cache import RedisCache, CacheConfig
        
        with patch('redis.Redis') as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_client.get.return_value = None
            mock_redis.return_value = mock_client
            
            config = CacheConfig()
            cache = RedisCache(config)
            
            result = cache.get("nonexistent_key")
            assert result is None

    def test_cache_operation_failure_raises_cache_error(self):
        """Test cache operation failure raises CacheError."""
        from src.streaming.cache import RedisCache, CacheConfig
        
        with patch('redis.Redis') as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_client.set.side_effect = Exception("Redis error")
            mock_redis.return_value = mock_client
            
            config = CacheConfig()
            cache = RedisCache(config)
            
            with pytest.raises(CacheError):
                cache.set("test_key", {"data": "value"})


class TestCacheMetrics:
    """Test cache metrics and statistics."""

    def test_cache_hit_rate_tracking(self):
        """Test cache tracks hit rate."""
        from src.streaming.cache import RedisCache, CacheConfig
        
        with patch('redis.Redis') as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client
            
            config = CacheConfig()
            cache = RedisCache(config)
            
            # Simulate hits and misses
            import pickle
            mock_client.get.side_effect = [
                pickle.dumps({"data": "value"}),  # Hit
                None,  # Miss
                pickle.dumps({"data": "value"}),  # Hit
            ]
            
            cache.get("key1")  # Hit
            cache.get("key2")  # Miss
            cache.get("key3")  # Hit
            
            stats = cache.get_stats()
            assert stats['hits'] == 2
            assert stats['misses'] == 1
            assert stats['hit_rate'] == pytest.approx(0.666, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
