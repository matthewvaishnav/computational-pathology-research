"""
Tests for caching module with safe_pickle integration.
"""

import os
import pickle
import time
from unittest.mock import patch

import pytest

from src.core.utils.caching import CacheEntry, OptimizedLRUCache


def test_cache_entry_compress_decompress():
    """Test basic compression and decompression."""
    # Use larger data that will actually compress
    data = {"key": "value" * 100, "numbers": list(range(1000))}
    entry = CacheEntry(data=data, timestamp=time.time())

    # Compress
    entry.compress()
    assert entry.compressed

    # Decompress
    decompressed = entry.decompress()
    assert decompressed == data


def test_cache_entry_safe_pickle_integration():
    """Test that decompress uses safe_pickle.loads()."""
    # Use larger data for compression
    data = {"test": "data" * 100, "list": list(range(500))}
    entry = CacheEntry(data=data, timestamp=time.time())

    # Set HMAC key for test
    os.environ["CACHE_SECRET_KEY"] = "test-secret-key"

    try:
        # Compress entry
        entry.compress()
        assert entry.compressed

        # Decompress should use safe_pickle
        with patch("src.utils.caching.safe_pickle.loads") as mock_loads:
            mock_loads.return_value = data

            result = entry.decompress()

            # Verify safe_pickle.loads was called with trusted=True
            assert mock_loads.called
            assert mock_loads.call_args[1]["trusted"] is True
            assert result == data
    finally:
        # Cleanup
        if "CACHE_SECRET_KEY" in os.environ:
            del os.environ["CACHE_SECRET_KEY"]


def test_optimized_lru_cache_basic():
    """Test basic LRU cache operations."""
    cache = OptimizedLRUCache(max_size=3, ttl_seconds=60)

    # Put items
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")

    # Get items
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"

    # Stats
    stats = cache.stats()
    assert stats["size"] == 3
    assert stats["hits"] == 3
    assert stats["misses"] == 0


def test_optimized_lru_cache_eviction():
    """Test LRU eviction."""
    cache = OptimizedLRUCache(max_size=2, ttl_seconds=60)

    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")  # Should evict key1

    assert cache.get("key1") is None  # Evicted
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"


def test_optimized_lru_cache_ttl():
    """Test TTL expiration."""
    cache = OptimizedLRUCache(max_size=10, ttl_seconds=1)

    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"

    # Wait for expiration
    time.sleep(1.1)

    # Should be expired
    assert cache.get("key1") is None


def test_cache_with_safe_pickle():
    """Integration test: cache with compression uses safe_pickle."""
    cache = OptimizedLRUCache(max_size=10, ttl_seconds=60)

    os.environ["CACHE_SECRET_KEY"] = "integration-test-key"

    try:
        # Put large data that will be compressed
        large_data = {"data": list(range(1000))}
        cache.put("large_key", large_data)

        # Force compression
        for entry in cache._cache.values():
            entry.compress()

        # Get should decompress using safe_pickle
        result = cache.get("large_key")
        assert result == large_data
    finally:
        if "CACHE_SECRET_KEY" in os.environ:
            del os.environ["CACHE_SECRET_KEY"]


def test_cache_hmac_validation():
    """Test that HMAC validation detects tampered cache data."""
    data = {"test": "data" * 100, "list": list(range(500))}
    entry = CacheEntry(data=data, timestamp=time.time())

    os.environ["CACHE_SECRET_KEY"] = "test-secret-key"

    try:
        # Compress entry
        entry.compress()
        assert entry.compressed

        # Decompress to get signed data, then tamper with HMAC signature
        import zlib

        decompressed = zlib.decompress(entry.data)

        # Tamper with HMAC signature (first 32 bytes)
        tampered_sig = bytearray(decompressed)
        tampered_sig[0] ^= 0xFF  # Flip bits in signature

        # Re-compress tampered data
        entry.data = zlib.compress(bytes(tampered_sig))

        # Decompress should raise ValueError due to HMAC mismatch
        with pytest.raises(ValueError, match="integrity check failed"):
            entry.decompress()
    finally:
        if "CACHE_SECRET_KEY" in os.environ:
            del os.environ["CACHE_SECRET_KEY"]


def test_malicious_pickle_rejected():
    """Test that safe_pickle rejects malicious pickle payloads."""
    from src.platform.security.exceptions import PickleSecurityError
    from src.platform.security.pickle_security_control import safe_pickle

    # Create malicious pickle that tries to execute code
    class MaliciousClass:
        def __reduce__(self):
            import os

            return (os.system, ("echo pwned",))

    malicious_obj = MaliciousClass()
    malicious_pickle = pickle.dumps(malicious_obj)

    # safe_pickle should reject this (not in whitelist)
    with pytest.raises((pickle.UnpicklingError, TypeError, AttributeError, PickleSecurityError)):
        safe_pickle.loads(malicious_pickle, trusted=False)
