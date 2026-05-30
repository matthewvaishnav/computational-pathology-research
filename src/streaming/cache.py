"""
Intelligent caching system for HistoCore Real-Time WSI Streaming.

Provides Redis-backed feature caching, LRU cache for frequently accessed slides,
and persistent caching across sessions with automatic compression and cleanup.
"""

import hashlib
import json
import logging
import pickle
import time
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import redis
import torch
from redis.exceptions import ConnectionError, TimeoutError

from src.core.exceptions import (
    CacheConnectionError,
    CacheError,
    CacheSerializationError,
)

from .metrics import (
    cache_hits_total,
    cache_misses_total,
    cache_operations_duration,
    cache_size_bytes,
)

logger = logging.getLogger(__name__)


def safe_pickle_loads(data: bytes) -> Any:
    """Compatibility wrapper after removing legacy mobile_edge safe-pickle module.

    Cache entries are local/internal serialized objects. This keeps the streaming
    cache importable after deleting the unused mobile app scaffolding. A future
    hardening pass should move restricted unpickling into a non-mobile security
    utility.
    """
    return pickle.loads(data)


@dataclass
class CacheConfig:
    """Configuration for caching system."""

    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_socket_timeout: float = 5.0
    redis_socket_connect_timeout: float = 5.0

    # Cache behavior
    default_ttl: int = 3600  # 1 hour
    max_memory_mb: int = 1024  # 1GB
    eviction_policy: str = "allkeys-lru"

    # Compression
    enable_compression: bool = True
    compression_level: int = 6  # 1-9, higher = better compression
    compression_threshold: int = 1024  # Compress if > 1KB

    # Feature caching
    feature_cache_enabled: bool = True
    feature_ttl: int = 7200  # 2 hours

    # Slide caching
    slide_cache_enabled: bool = True
    slide_ttl: int = 3600  # 1 hour

    # Persistent caching
    persistent_cache_enabled: bool = True
    persistent_cache_dir: str = "./cache"


class CacheKey:
    """Generate consistent cache keys."""

    @staticmethod
    def feature_key(slide_id: str, patch_coords: Tuple[int, int]) -> str:
        """Generate key for patch features."""
        return f"feature:{slide_id}:{patch_coords[0]}:{patch_coords[1]}"

    @staticmethod
    def slide_key(slide_id: str) -> str:
        """Generate key for slide metadata."""
        return f"slide:{slide_id}"

    @staticmethod
    def attention_key(slide_id: str) -> str:
        """Generate key for attention weights."""
        return f"attention:{slide_id}"

    @staticmethod
    def result_key(slide_id: str) -> str:
        """Generate key for processing results."""
        return f"result:{slide_id}"

    @staticmethod
    def hash_key(data: Union[str, bytes, np.ndarray, torch.Tensor]) -> str:
        """Generate hash-based key for arbitrary data."""
        if isinstance(data, str):
            data_bytes = data.encode()
        elif isinstance(data, bytes):
            data_bytes = data
        elif isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        elif isinstance(data, torch.Tensor):
            data_bytes = data.cpu().numpy().tobytes()
        else:
            data_bytes = str(data).encode()

        return hashlib.sha256(data_bytes).hexdigest()[:16]


class RedisCache:
    """Redis-backed caching with compression and automatic cleanup."""

    def __init__(self, config: CacheConfig):
        """Initialize Redis cache."""
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self._connect()

    def _connect(self) -> None:
        """Connect to Redis server."""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                socket_timeout=self.config.redis_socket_timeout,
                socket_connect_timeout=self.config.redis_socket_connect_timeout,
                decode_responses=False,  # Handle binary data
            )

            # Test connection
            self.redis_client.ping()

            # Configure eviction policy
            self.redis_client.config_set("maxmemory-policy", self.config.eviction_policy)
            self.redis_client.config_set("maxmemory", f"{self.config.max_memory_mb}mb")

            logger.info(
                "Connected to Redis at %s:%d",
                self.config.redis_host,
                self.config.redis_port,
            )

        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to connect to Redis: %s", e)
            self.redis_client = None
            raise CacheConnectionError(f"Redis connection failed: {e}") from e

    def _compress(self, data: bytes) -> bytes:
        """Compress data if enabled and above threshold."""
        if not self.config.enable_compression:
            return data

        if len(data) < self.config.compression_threshold:
            return data

        compressed = zlib.compress(data, level=self.config.compression_level)

        # Only use compression if it actually reduces size
        if len(compressed) < len(data):
            return b"COMPRESSED:" + compressed

        return data

    def _decompress(self, data: bytes) -> bytes:
        """Decompress data if compressed."""
        if data.startswith(b"COMPRESSED:"):
            return zlib.decompress(data[11:])
        return data

    def _serialize(self, value: Any) -> bytes:
        """Serialize value to bytes."""
        if isinstance(value, (np.ndarray, torch.Tensor)):
            # Use pickle for numpy/torch tensors
            serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        elif isinstance(value, (dict, list)):
            # Use JSON for simple types
            serialized = json.dumps(value).encode()
        else:
            # Use pickle for everything else
            serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

        return self._compress(serialized)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value."""
        decompressed = self._decompress(data)

        try:
            # Try JSON first (faster)
            return json.loads(decompressed.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle for local/internal cache entries
            try:
                return safe_pickle_loads(decompressed)
            except (pickle.UnpicklingError, EOFError, AttributeError) as e:
                logger.error("Failed to deserialize cached data: %s", e)
                raise CacheSerializationError(f"Deserialization failed: {e}") from e

    @cache_operations_duration.labels(operation="get").time()
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self.redis_client is None:
            return None

        try:
            data = self.redis_client.get(key)

            if data is None:
                cache_misses_total.labels(cache_type="redis").inc()
                return None

            cache_hits_total.labels(cache_type="redis").inc()
            return self._deserialize(data)

        except CacheSerializationError:
            # Re-raise serialization errors
            raise
        except (ConnectionError, TimeoutError) as e:
            logger.error("Cache connection error for key %s: %s", key, e)
            raise CacheConnectionError(f"Redis connection lost: {e}") from e
        except Exception as e:
            logger.error("Cache get error for key %s: %s", key, e)
            raise CacheError(f"Cache get failed: {e}") from e

    @cache_operations_duration.labels(operation="set").time()
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        if self.redis_client is None:
            return False

        try:
            serialized = self._serialize(value)

            if ttl is None:
                ttl = self.config.default_ttl

            result = self.redis_client.setex(key, ttl, serialized)

            cache_size_bytes.labels(cache_type="redis").set(len(serialized))

            return bool(result)

        except (ConnectionError, TimeoutError) as e:
            logger.error("Cache connection error setting key %s: %s", key, e)
            raise CacheConnectionError(f"Redis connection lost: {e}") from e
        except Exception as e:
            logger.error("Cache set error for key %s: %s", key, e)
            raise CacheError(f"Cache set failed: {e}") from e
