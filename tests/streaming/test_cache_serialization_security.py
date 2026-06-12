"""Security regression tests for streaming cache serialization."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.streaming.cache import CacheConfig, RedisCache


@pytest.fixture
def cache() -> RedisCache:
    """Create a cache instance without connecting to Redis."""
    with patch.object(RedisCache, "_connect", return_value=None):
        return RedisCache(CacheConfig(enable_compression=True, compression_threshold=1))


def test_json_round_trip(cache: RedisCache) -> None:
    value = {"slide": "case-1", "scores": [0.1, 0.9], "accepted": True}
    assert cache._deserialize(cache._serialize(value)) == value


def test_numpy_round_trip(cache: RedisCache) -> None:
    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    restored = cache._deserialize(cache._serialize(value))
    assert isinstance(restored, np.ndarray)
    assert restored.dtype == value.dtype
    assert np.array_equal(restored, value)


def test_torch_round_trip(cache: RedisCache) -> None:
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    restored = cache._deserialize(cache._serialize(value))
    assert isinstance(restored, torch.Tensor)
    assert restored.dtype == value.dtype
    assert torch.equal(restored, value)


def test_rejects_unknown_version(cache: RedisCache) -> None:
    payload = b'{"version":0,"value":{"type":"json","data":{}}}'
    with pytest.raises(ValueError, match="legacy"):
        cache._deserialize(payload)
