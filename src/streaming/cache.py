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

from .metrics import (
    cache_hits_total,
    cache_misses_total,
    cache_operations_duration,
    cache_size_bytes,
)

logger = logging.getLogger(__name__)


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
            # Fall back to pickle
            return pickle.loads(decompressed)

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

        except Exception as e:
            logger.error("Cache get error for key %s: %s", key, e)
            return None

    @cache_operations_duration.labels(operation="set").time()
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        if self.redis_client is None:
            return False

        try:
            serialized = self._serialize(value)

            if ttl is None:
                ttl = self.config.default_ttl

            self.redis_client.setex(key, ttl, serialized)

            # Update metrics
            cache_size_bytes.labels(cache_type="redis").set(len(serialized))

            return True

        except Exception as e:
            logger.error("Cache set error for key %s: %s", key, e)
            return False

    @cache_operations_duration.labels(operation="delete").time()
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if self.redis_client is None:
            return False

        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error("Cache delete error for key %s: %s", key, e)
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if self.redis_client is None:
            return False

        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error("Cache exists error for key %s: %s", key, e)
            return False

    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        if self.redis_client is None:
            return {}

        try:
            values = self.redis_client.mget(keys)

            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value)
                    cache_hits_total.labels(cache_type="redis").inc()
                else:
                    cache_misses_total.labels(cache_type="redis").inc()

            return result

        except Exception as e:
            logger.error("Cache get_many error: %s", e)
            return {}

    def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache."""
        if self.redis_client is None:
            return False

        try:
            if ttl is None:
                ttl = self.config.default_ttl

            pipe = self.redis_client.pipeline()

            for key, value in mapping.items():
                serialized = self._serialize(value)
                pipe.setex(key, ttl, serialized)

            pipe.execute()
            return True

        except Exception as e:
            logger.error("Cache set_many error: %s", e)
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        if self.redis_client is None:
            return 0

        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error("Cache clear_pattern error for pattern %s: %s", pattern, e)
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.redis_client is None:
            return {"connected": False}

        try:
            info = self.redis_client.info("stats")
            memory = self.redis_client.info("memory")

            return {
                "connected": True,
                "total_keys": self.redis_client.dbsize(),
                "used_memory_mb": memory.get("used_memory", 0) / (1024 * 1024),
                "max_memory_mb": self.config.max_memory_mb,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0), info.get("keyspace_misses", 0)
                ),
                "evicted_keys": info.get("evicted_keys", 0),
            }
        except Exception as e:
            logger.error("Cache get_stats error: %s", e)
            return {"connected": False, "error": str(e)}

    @staticmethod
    def _calculate_hit_rate(hits: int, misses: int) -> float:
        """Calculate cache hit rate."""
        total = hits + misses
        if total == 0:
            return 0.0
        return hits / total

    def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client is not None:
            self.redis_client.close()
            logger.info("Redis connection closed")


class LRUCache:
    """In-memory LRU cache for frequently accessed data."""

    def __init__(self, max_size: int = 1000):
        """Initialize LRU cache."""
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_times: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            self.access_times[key] = time.time()
            cache_hits_total.labels(cache_type="lru").inc()
            return self.cache[key][0]

        cache_misses_total.labels(cache_type="lru").inc()
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        current_time = time.time()

        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[key] = (value, current_time)
        self.access_times[key] = current_time

    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        self.access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0,
        }


class FeatureCache:
    """High-level feature caching with Redis and LRU fallback."""

    def __init__(self, config: CacheConfig):
        """Initialize feature cache."""
        self.config = config
        self.redis_cache = RedisCache(config) if config.feature_cache_enabled else None
        self.lru_cache = LRUCache(max_size=10000)

        logger.info(
            "Feature cache initialized: redis_enabled=%s lru_size=%d",
            config.feature_cache_enabled,
            10000,
        )

    def get_features(self, slide_id: str, patch_coords: Tuple[int, int]) -> Optional[torch.Tensor]:
        """Get cached features for a patch."""
        key = CacheKey.feature_key(slide_id, patch_coords)

        # Try LRU cache first (fastest)
        features = self.lru_cache.get(key)
        if features is not None:
            return features

        # Try Redis cache
        if self.redis_cache is not None:
            features = self.redis_cache.get(key)
            if features is not None:
                # Populate LRU cache
                self.lru_cache.set(key, features)
                return features

        return None

    def set_features(
        self, slide_id: str, patch_coords: Tuple[int, int], features: torch.Tensor
    ) -> None:
        """Cache features for a patch."""
        key = CacheKey.feature_key(slide_id, patch_coords)

        # Store in LRU cache
        self.lru_cache.set(key, features)

        # Store in Redis cache
        if self.redis_cache is not None:
            self.redis_cache.set(key, features, ttl=self.config.feature_ttl)

    def get_batch_features(
        self, slide_id: str, patch_coords_list: List[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """Get cached features for multiple patches."""
        keys = [CacheKey.feature_key(slide_id, coords) for coords in patch_coords_list]

        # Try LRU cache first
        result = {}
        missing_keys = []
        missing_coords = []

        for coords, key in zip(patch_coords_list, keys):
            features = self.lru_cache.get(key)
            if features is not None:
                result[coords] = features
            else:
                missing_keys.append(key)
                missing_coords.append(coords)

        # Try Redis for missing keys
        if missing_keys and self.redis_cache is not None:
            redis_results = self.redis_cache.get_many(missing_keys)

            for coords, key in zip(missing_coords, missing_keys):
                if key in redis_results:
                    features = redis_results[key]
                    result[coords] = features
                    # Populate LRU cache
                    self.lru_cache.set(key, features)

        return result

    def clear_slide(self, slide_id: str) -> None:
        """Clear all cached features for a slide."""
        if self.redis_cache is not None:
            pattern = f"feature:{slide_id}:*"
            self.redis_cache.clear_pattern(pattern)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "lru": self.lru_cache.get_stats(),
        }

        if self.redis_cache is not None:
            stats["redis"] = self.redis_cache.get_stats()

        return stats


class SlideCache:
    """Cache for slide metadata and results."""

    def __init__(self, config: CacheConfig):
        """Initialize slide cache."""
        self.config = config
        self.redis_cache = RedisCache(config) if config.slide_cache_enabled else None
        self.lru_cache = LRUCache(max_size=1000)

    def get_metadata(self, slide_id: str) -> Optional[Dict[str, Any]]:
        """Get cached slide metadata."""
        key = CacheKey.slide_key(slide_id)

        # Try LRU first
        metadata = self.lru_cache.get(key)
        if metadata is not None:
            return metadata

        # Try Redis
        if self.redis_cache is not None:
            metadata = self.redis_cache.get(key)
            if metadata is not None:
                self.lru_cache.set(key, metadata)
                return metadata

        return None

    def set_metadata(self, slide_id: str, metadata: Dict[str, Any]) -> None:
        """Cache slide metadata."""
        key = CacheKey.slide_key(slide_id)

        self.lru_cache.set(key, metadata)

        if self.redis_cache is not None:
            self.redis_cache.set(key, metadata, ttl=self.config.slide_ttl)

    def get_result(self, slide_id: str) -> Optional[Dict[str, Any]]:
        """Get cached processing result."""
        key = CacheKey.result_key(slide_id)

        if self.redis_cache is not None:
            return self.redis_cache.get(key)

        return None

    def set_result(self, slide_id: str, result: Dict[str, Any]) -> None:
        """Cache processing result."""
        key = CacheKey.result_key(slide_id)

        if self.redis_cache is not None:
            self.redis_cache.set(key, result, ttl=self.config.slide_ttl)


# Global cache instances
_feature_cache: Optional[FeatureCache] = None
_slide_cache: Optional[SlideCache] = None


def initialize_caches(config: CacheConfig) -> None:
    """Initialize global cache instances."""
    global _feature_cache, _slide_cache

    _feature_cache = FeatureCache(config)
    _slide_cache = SlideCache(config)

    logger.info("Global caches initialized")


def get_feature_cache() -> FeatureCache:
    """Get global feature cache instance."""
    if _feature_cache is None:
        raise RuntimeError("Feature cache not initialized. Call initialize_caches() first.")
    return _feature_cache


def get_slide_cache() -> SlideCache:
    """Get global slide cache instance."""
    if _slide_cache is None:
        raise RuntimeError("Slide cache not initialized. Call initialize_caches() first.")
    return _slide_cache
