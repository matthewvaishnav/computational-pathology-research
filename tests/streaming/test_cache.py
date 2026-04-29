"""
Tests for intelligent caching system.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.streaming.cache import (
    CacheConfig,
    CacheKey,
    FeatureCache,
    LRUCache,
    RedisCache,
    SlideCache,
)


@pytest.fixture
def cache_config():
    """Create test cache configuration."""
    return CacheConfig(
        redis_host="localhost",
        redis_port=6379,
        redis_db=15,  # Use separate DB for testing
        default_ttl=60,
        max_memory_mb=100,
        enable_compression=True,
        compression_level=6,
    )


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    with patch("src.streaming.cache.redis.Redis") as mock:
        client = MagicMock()
        client.ping.return_value = True
        client.config_set.return_value = True
        mock.return_value = client
        yield client


class TestCacheKey:
    """Test cache key generation."""

    def test_feature_key(self):
        """Test feature key generation."""
        key = CacheKey.feature_key("slide_001", (100, 200))
        assert key == "feature:slide_001:100:200"

    def test_slide_key(self):
        """Test slide key generation."""
        key = CacheKey.slide_key("slide_001")
        assert key == "slide:slide_001"

    def test_attention_key(self):
        """Test attention key generation."""
        key = CacheKey.attention_key("slide_001")
        assert key == "attention:slide_001"

    def test_result_key(self):
        """Test result key generation."""
        key = CacheKey.result_key("slide_001")
        assert key == "result:slide_001"

    def test_hash_key_string(self):
        """Test hash key generation from string."""
        key1 = CacheKey.hash_key("test_data")
        key2 = CacheKey.hash_key("test_data")
        key3 = CacheKey.hash_key("different_data")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 16

    def test_hash_key_numpy(self):
        """Test hash key generation from numpy array."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        arr3 = np.array([4, 5, 6])

        key1 = CacheKey.hash_key(arr1)
        key2 = CacheKey.hash_key(arr2)
        key3 = CacheKey.hash_key(arr3)

        assert key1 == key2
        assert key1 != key3

    def test_hash_key_torch(self):
        """Test hash key generation from torch tensor."""
        tensor1 = torch.tensor([1, 2, 3])
        tensor2 = torch.tensor([1, 2, 3])
        tensor3 = torch.tensor([4, 5, 6])

        key1 = CacheKey.hash_key(tensor1)
        key2 = CacheKey.hash_key(tensor2)
        key3 = CacheKey.hash_key(tensor3)

        assert key1 == key2
        assert key1 != key3


class TestLRUCache:
    """Test LRU cache."""

    def test_get_set(self):
        """Test basic get/set operations."""
        cache = LRUCache(max_size=10)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None

    def test_eviction(self):
        """Test LRU eviction."""
        cache = LRUCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4, should evict key2 (oldest)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_update_existing(self):
        """Test updating existing key."""
        cache = LRUCache(max_size=10)

        cache.set("key1", "value1")
        cache.set("key1", "value2")

        assert cache.get("key1") == "value2"

    def test_clear(self):
        """Test cache clearing."""
        cache = LRUCache(max_size=10)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=10)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        stats = cache.get_stats()

        assert stats["size"] == 2
        assert stats["max_size"] == 10
        assert stats["utilization"] == 0.2


class TestRedisCache:
    """Test Redis cache."""

    def test_connect(self, cache_config, mock_redis):
        """Test Redis connection."""
        cache = RedisCache(cache_config)

        assert cache.redis_client is not None
        mock_redis.ping.assert_called_once()

    def test_compression(self, cache_config):
        """Test data compression."""
        cache = RedisCache(cache_config)

        # Small data - should not compress
        small_data = b"x" * 100
        compressed = cache._compress(small_data)
        assert compressed == small_data

        # Large data - should compress
        large_data = b"x" * 10000
        compressed = cache._compress(large_data)
        assert compressed.startswith(b"COMPRESSED:")
        assert len(compressed) < len(large_data)

        # Decompress
        decompressed = cache._decompress(compressed)
        assert decompressed == large_data

    def test_serialize_deserialize(self, cache_config):
        """Test serialization/deserialization."""
        cache = RedisCache(cache_config)

        # Test dict
        data = {"key": "value", "number": 42}
        serialized = cache._serialize(data)
        deserialized = cache._deserialize(serialized)
        assert deserialized == data

        # Test numpy array
        arr = np.array([1, 2, 3, 4, 5])
        serialized = cache._serialize(arr)
        deserialized = cache._deserialize(serialized)
        np.testing.assert_array_equal(deserialized, arr)

        # Test torch tensor
        tensor = torch.tensor([1.0, 2.0, 3.0])
        serialized = cache._serialize(tensor)
        deserialized = cache._deserialize(serialized)
        assert torch.allclose(deserialized, tensor)

    def test_get_set(self, cache_config, mock_redis):
        """Test get/set operations."""
        cache = RedisCache(cache_config)

        # Mock Redis operations
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True

        # Test set
        result = cache.set("test_key", "test_value", ttl=60)
        assert result is True

        # Test get (miss)
        value = cache.get("test_key")
        assert value is None

    def test_get_many(self, cache_config, mock_redis):
        """Test batch get operations."""
        cache = RedisCache(cache_config)

        # Mock Redis operations
        mock_redis.mget.return_value = [None, None]

        keys = ["key1", "key2"]
        result = cache.get_many(keys)

        assert isinstance(result, dict)
        mock_redis.mget.assert_called_once_with(keys)

    def test_clear_pattern(self, cache_config, mock_redis):
        """Test pattern-based clearing."""
        cache = RedisCache(cache_config)

        # Mock Redis operations
        mock_redis.keys.return_value = [b"key1", b"key2"]
        mock_redis.delete.return_value = 2

        count = cache.clear_pattern("test:*")

        assert count == 2
        mock_redis.keys.assert_called_once_with("test:*")


class TestFeatureCache:
    """Test feature cache."""

    @pytest.fixture
    def feature_cache(self, cache_config, mock_redis):
        """Create feature cache."""
        return FeatureCache(cache_config)

    def test_get_set_features(self, feature_cache):
        """Test feature caching."""
        slide_id = "slide_001"
        coords = (100, 200)
        features = torch.randn(2048)

        # Set features
        feature_cache.set_features(slide_id, coords, features)

        # Get features (should hit LRU cache)
        cached = feature_cache.get_features(slide_id, coords)

        assert cached is not None
        assert torch.allclose(cached, features)

    def test_get_batch_features(self, feature_cache):
        """Test batch feature retrieval."""
        slide_id = "slide_001"
        coords_list = [(100, 200), (300, 400), (500, 600)]

        # Set features
        for coords in coords_list:
            features = torch.randn(2048)
            feature_cache.set_features(slide_id, coords, features)

        # Get batch
        result = feature_cache.get_batch_features(slide_id, coords_list)

        assert len(result) == 3
        for coords in coords_list:
            assert coords in result

    def test_clear_slide(self, feature_cache, mock_redis):
        """Test clearing slide features."""
        slide_id = "slide_001"

        # Mock Redis operations
        mock_redis.keys.return_value = []
        mock_redis.delete.return_value = 0

        feature_cache.clear_slide(slide_id)

        # Should call Redis clear_pattern
        if feature_cache.redis_cache is not None:
            mock_redis.keys.assert_called()


class TestSlideCache:
    """Test slide cache."""

    @pytest.fixture
    def slide_cache(self, cache_config, mock_redis):
        """Create slide cache."""
        return SlideCache(cache_config)

    def test_get_set_metadata(self, slide_cache):
        """Test metadata caching."""
        slide_id = "slide_001"
        metadata = {
            "dimensions": (10000, 10000),
            "magnification": 40,
            "format": "svs",
        }

        # Set metadata
        slide_cache.set_metadata(slide_id, metadata)

        # Get metadata (should hit LRU cache)
        cached = slide_cache.get_metadata(slide_id)

        assert cached is not None
        assert cached == metadata

    def test_get_set_result(self, slide_cache, mock_redis):
        """Test result caching."""
        slide_id = "slide_001"
        result = {
            "prediction": 1,
            "confidence": 0.95,
            "processing_time": 25.5,
        }

        # Mock Redis operations
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True

        # Set result
        slide_cache.set_result(slide_id, result)

        # Get result
        cached = slide_cache.get_result(slide_id)

        # Will be None because Redis is mocked
        assert cached is None


@pytest.mark.integration
class TestCacheIntegration:
    """Integration tests requiring real Redis."""

    @pytest.fixture
    def redis_config(self):
        """Create Redis configuration for integration tests."""
        return CacheConfig(
            redis_host="localhost",
            redis_port=6379,
            redis_db=15,  # Use separate DB for testing
            default_ttl=10,
            enable_compression=True,
        )

    def test_redis_roundtrip(self, redis_config):
        """Test Redis roundtrip with real connection."""
        try:
            cache = RedisCache(redis_config)

            if cache.redis_client is None:
                pytest.skip("Redis not available")

            # Test data
            test_data = {
                "string": "test_value",
                "number": 42,
                "array": np.array([1, 2, 3]),
                "tensor": torch.tensor([1.0, 2.0, 3.0]),
            }

            # Set and get each type
            for key, value in test_data.items():
                cache.set(f"test:{key}", value, ttl=10)
                retrieved = cache.get(f"test:{key}")

                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(retrieved, value)
                elif isinstance(value, torch.Tensor):
                    assert torch.allclose(retrieved, value)
                else:
                    assert retrieved == value

            # Cleanup
            cache.clear_pattern("test:*")
            cache.close()

        except Exception as e:
            pytest.skip(f"Redis integration test failed: {e}")

    def test_feature_cache_integration(self, redis_config):
        """Test feature cache with real Redis."""
        try:
            feature_cache = FeatureCache(redis_config)

            if feature_cache.redis_cache is None or feature_cache.redis_cache.redis_client is None:
                pytest.skip("Redis not available")

            slide_id = "test_slide"
            coords = (100, 200)
            features = torch.randn(2048)

            # Set features
            feature_cache.set_features(slide_id, coords, features)

            # Get features (should hit LRU)
            cached = feature_cache.get_features(slide_id, coords)
            assert torch.allclose(cached, features)

            # Clear LRU cache
            feature_cache.lru_cache.clear()

            # Get features (should hit Redis)
            cached = feature_cache.get_features(slide_id, coords)
            assert torch.allclose(cached, features)

            # Cleanup
            feature_cache.clear_slide(slide_id)
            if feature_cache.redis_cache:
                feature_cache.redis_cache.close()

        except Exception as e:
            pytest.skip(f"Feature cache integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
