"""Caching primitives for real-time WSI streaming.

Cached values use a versioned JSON envelope. Unknown or legacy values are
rejected; no executable deserialization format is accepted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import redis
import torch
from redis.exceptions import RedisError

from .metrics import (
    cache_hits_total,
    cache_misses_total,
    cache_operations_duration,
    cache_size_bytes,
)

logger = logging.getLogger(__name__)
_FORMAT_VERSION = 1


@dataclass
class CacheConfig:
    """Configuration for the caching system."""

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_socket_timeout: float = 5.0
    redis_socket_connect_timeout: float = 5.0
    default_ttl: int = 3600
    max_memory_mb: int = 1024
    eviction_policy: str = "allkeys-lru"
    enable_compression: bool = True
    compression_level: int = 6
    compression_threshold: int = 1024
    feature_cache_enabled: bool = True
    feature_ttl: int = 7200
    slide_cache_enabled: bool = True
    slide_ttl: int = 3600
    persistent_cache_enabled: bool = True
    persistent_cache_dir: str = "./cache"


class CacheKey:
    """Generate stable cache keys."""

    @staticmethod
    def feature_key(slide_id: str, patch_coords: Tuple[int, int]) -> str:
        return f"feature:{slide_id}:{patch_coords[0]}:{patch_coords[1]}"

    @staticmethod
    def slide_key(slide_id: str) -> str:
        return f"slide:{slide_id}"

    @staticmethod
    def attention_key(slide_id: str) -> str:
        return f"attention:{slide_id}"

    @staticmethod
    def result_key(slide_id: str) -> str:
        return f"result:{slide_id}"

    @staticmethod
    def hash_key(data: Union[str, bytes, np.ndarray, torch.Tensor]) -> str:
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
        elif isinstance(data, bytes):
            data_bytes = data
        elif isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        elif isinstance(data, torch.Tensor):
            data_bytes = data.detach().cpu().contiguous().numpy().tobytes()
        else:
            data_bytes = str(data).encode("utf-8")
        return hashlib.sha256(data_bytes).hexdigest()[:16]


def _encode_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return {
            "type": "torch",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": array.tolist(),
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "numpy",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    return {"type": "json", "data": value}


def _decode_value(envelope: Dict[str, Any]) -> Any:
    value_type = envelope.get("type")
    if value_type == "json":
        return envelope.get("data")
    if value_type in {"numpy", "torch"}:
        dtype = envelope.get("dtype")
        shape = envelope.get("shape")
        data = envelope.get("data")
        if not isinstance(dtype, str) or not isinstance(shape, list):
            raise ValueError("Invalid array cache envelope")
        array = np.asarray(data, dtype=np.dtype(dtype))
        if list(array.shape) != shape:
            raise ValueError("Cached array shape does not match its envelope")
        if value_type == "torch":
            return torch.from_numpy(array.copy())
        return array
    raise ValueError("Unsupported cache value type")


class RedisCache:
    """Redis-backed cache using a non-executable JSON format."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self._connect()

    def _connect(self) -> None:
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                socket_timeout=self.config.redis_socket_timeout,
                socket_connect_timeout=self.config.redis_socket_connect_timeout,
                decode_responses=False,
            )
            self.redis_client.ping()
            self.redis_client.config_set("maxmemory-policy", self.config.eviction_policy)
            self.redis_client.config_set("maxmemory", f"{self.config.max_memory_mb}mb")
        except RedisError as exc:
            logger.warning("Redis cache unavailable: %s", exc)
            self.redis_client = None

    def _compress(self, data: bytes) -> bytes:
        if not self.config.enable_compression or len(data) < self.config.compression_threshold:
            return data
        compressed = zlib.compress(data, level=self.config.compression_level)
        return b"COMPRESSED:" + compressed if len(compressed) < len(data) else data

    @staticmethod
    def _decompress(data: bytes) -> bytes:
        marker = b"COMPRESSED:"
        if data.startswith(marker):
            return zlib.decompress(data[len(marker) :])
        return data

    def _serialize(self, value: Any) -> bytes:
        payload = {
            "version": _FORMAT_VERSION,
            "value": _encode_value(value),
        }
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        return self._compress(encoded)

    def _deserialize(self, data: bytes) -> Any:
        decoded = json.loads(self._decompress(data).decode("utf-8"))
        if not isinstance(decoded, dict) or decoded.get("version") != _FORMAT_VERSION:
            raise ValueError("Unknown or legacy cache serialization format")
        value = decoded.get("value")
        if not isinstance(value, dict):
            raise ValueError("Invalid cache envelope")
        return _decode_value(value)

    @cache_operations_duration.labels(operation="get").time()
    def get(self, key: str) -> Optional[Any]:
        if self.redis_client is None:
            return None
        try:
            data = self.redis_client.get(key)
            if data is None:
                cache_misses_total.labels(cache_type="redis").inc()
                return None
            result = self._deserialize(data)
            cache_hits_total.labels(cache_type="redis").inc()
            return result
        except (RedisError, ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
            logger.error("Cache get error for key %s: %s", key, exc)
            return None

    @cache_operations_duration.labels(operation="set").time()
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if self.redis_client is None:
            return False
        try:
            serialized = self._serialize(value)
            self.redis_client.setex(key, ttl or self.config.default_ttl, serialized)
            cache_size_bytes.labels(cache_type="redis").set(len(serialized))
            return True
        except (RedisError, TypeError, ValueError) as exc:
            logger.error("Cache set error for key %s: %s", key, exc)
            return False

    def delete(self, key: str) -> bool:
        if self.redis_client is None:
            return False
        try:
            self.redis_client.delete(key)
            return True
        except RedisError as exc:
            logger.error("Cache delete error for key %s: %s", key, exc)
            return False

    def exists(self, key: str) -> bool:
        if self.redis_client is None:
            return False
        try:
            return bool(self.redis_client.exists(key))
        except RedisError as exc:
            logger.error("Cache exists error for key %s: %s", key, exc)
            return False

    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        if self.redis_client is None:
            return {}
        try:
            values = self.redis_client.mget(keys)
            result: Dict[str, Any] = {}
            for key, value in zip(keys, values):
                if value is None:
                    cache_misses_total.labels(cache_type="redis").inc()
                    continue
                try:
                    result[key] = self._deserialize(value)
                    cache_hits_total.labels(cache_type="redis").inc()
                except (ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
                    logger.error("Ignoring invalid cached value for key %s: %s", key, exc)
            return result
        except RedisError as exc:
            logger.error("Cache get_many error: %s", exc)
            return {}

    def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        if self.redis_client is None:
            return False
        try:
            pipe = self.redis_client.pipeline()
            effective_ttl = ttl or self.config.default_ttl
            for key, value in mapping.items():
                pipe.setex(key, effective_ttl, self._serialize(value))
            pipe.execute()
            return True
        except (RedisError, TypeError, ValueError) as exc:
            logger.error("Cache set_many error: %s", exc)
            return False

    def clear_pattern(self, pattern: str) -> int:
        if self.redis_client is None:
            return 0
        try:
            keys = self.redis_client.keys(pattern)
            return int(self.redis_client.delete(*keys)) if keys else 0
        except RedisError as exc:
            logger.error("Cache clear_pattern error for pattern %s: %s", pattern, exc)
            return 0

    def get_stats(self) -> Dict[str, Any]:
        if self.redis_client is None:
            return {"connected": False}
        try:
            info = self.redis_client.info("stats")
            memory = self.redis_client.info("memory")
            hits = int(info.get("keyspace_hits", 0))
            misses = int(info.get("keyspace_misses", 0))
            return {
                "connected": True,
                "total_keys": self.redis_client.dbsize(),
                "used_memory_mb": memory.get("used_memory", 0) / (1024 * 1024),
                "max_memory_mb": self.config.max_memory_mb,
                "hits": hits,
                "misses": misses,
                "hit_rate": self._calculate_hit_rate(hits, misses),
                "evicted_keys": info.get("evicted_keys", 0),
            }
        except RedisError as exc:
            logger.error("Cache get_stats error: %s", exc)
            return {"connected": False, "error": str(exc)}

    @staticmethod
    def _calculate_hit_rate(hits: int, misses: int) -> float:
        total = hits + misses
        return hits / total if total else 0.0

    def close(self) -> None:
        if self.redis_client is not None:
            self.redis_client.close()


class LRUCache:
    """Small in-memory least-recently-used cache."""

    def __init__(self, max_size: int = 1000):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_times: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.access_times[key] = time.monotonic()
            cache_hits_total.labels(cache_type="lru").inc()
            return self.cache[key][0]
        cache_misses_total.labels(cache_type="lru").inc()
        return None

    def set(self, key: str, value: Any) -> None:
        now = time.monotonic()
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(self.access_times, key=self.access_times.__getitem__)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        self.cache[key] = (value, now)
        self.access_times[key] = now

    def delete_prefix(self, prefix: str) -> None:
        for key in [key for key in self.cache if key.startswith(prefix)]:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)

    def clear(self) -> None:
        self.cache.clear()
        self.access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size,
        }


class FeatureCache:
    """High-level feature cache with in-memory and Redis tiers."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_cache = RedisCache(config) if config.feature_cache_enabled else None
        self.lru_cache = LRUCache(max_size=10000)

    def get_features(self, slide_id: str, patch_coords: Tuple[int, int]) -> Optional[torch.Tensor]:
        key = CacheKey.feature_key(slide_id, patch_coords)
        features = self.lru_cache.get(key)
        if features is not None:
            return features
        if self.redis_cache is not None:
            features = self.redis_cache.get(key)
            if isinstance(features, torch.Tensor):
                self.lru_cache.set(key, features)
                return features
        return None

    def set_features(
        self,
        slide_id: str,
        patch_coords: Tuple[int, int],
        features: torch.Tensor,
    ) -> None:
        key = CacheKey.feature_key(slide_id, patch_coords)
        self.lru_cache.set(key, features)
        if self.redis_cache is not None:
            self.redis_cache.set(key, features, ttl=self.config.feature_ttl)

    def get_batch_features(
        self,
        slide_id: str,
        patch_coords_list: List[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        result: Dict[Tuple[int, int], torch.Tensor] = {}
        missing: List[Tuple[Tuple[int, int], str]] = []
        for coords in patch_coords_list:
            key = CacheKey.feature_key(slide_id, coords)
            features = self.lru_cache.get(key)
            if isinstance(features, torch.Tensor):
                result[coords] = features
            else:
                missing.append((coords, key))

        if missing and self.redis_cache is not None:
            redis_results = self.redis_cache.get_many([key for _, key in missing])
            for coords, key in missing:
                features = redis_results.get(key)
                if isinstance(features, torch.Tensor):
                    result[coords] = features
                    self.lru_cache.set(key, features)
        return result

    def clear_slide(self, slide_id: str) -> None:
        prefix = f"feature:{slide_id}:"
        self.lru_cache.delete_prefix(prefix)
        if self.redis_cache is not None:
            self.redis_cache.clear_pattern(f"{prefix}*")

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"lru": self.lru_cache.get_stats()}
        if self.redis_cache is not None:
            stats["redis"] = self.redis_cache.get_stats()
        return stats


class SlideCache:
    """Cache for slide metadata and processing results."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_cache = RedisCache(config) if config.slide_cache_enabled else None
        self.lru_cache = LRUCache(max_size=1000)

    def get_metadata(self, slide_id: str) -> Optional[Dict[str, Any]]:
        key = CacheKey.slide_key(slide_id)
        metadata = self.lru_cache.get(key)
        if isinstance(metadata, dict):
            return metadata
        if self.redis_cache is not None:
            metadata = self.redis_cache.get(key)
            if isinstance(metadata, dict):
                self.lru_cache.set(key, metadata)
                return metadata
        return None

    def set_metadata(self, slide_id: str, metadata: Dict[str, Any]) -> None:
        key = CacheKey.slide_key(slide_id)
        self.lru_cache.set(key, metadata)
        if self.redis_cache is not None:
            self.redis_cache.set(key, metadata, ttl=self.config.slide_ttl)

    def get_result(self, slide_id: str) -> Optional[Dict[str, Any]]:
        if self.redis_cache is None:
            return None
        result = self.redis_cache.get(CacheKey.result_key(slide_id))
        return result if isinstance(result, dict) else None

    def set_result(self, slide_id: str, result: Dict[str, Any]) -> None:
        if self.redis_cache is not None:
            self.redis_cache.set(CacheKey.result_key(slide_id), result, ttl=self.config.slide_ttl)


_feature_cache: Optional[FeatureCache] = None
_slide_cache: Optional[SlideCache] = None


def initialize_caches(config: CacheConfig) -> None:
    global _feature_cache, _slide_cache
    _feature_cache = FeatureCache(config)
    _slide_cache = SlideCache(config)


def get_feature_cache() -> FeatureCache:
    if _feature_cache is None:
        raise RuntimeError("Feature cache not initialized. Call initialize_caches() first.")
    return _feature_cache


def get_slide_cache() -> SlideCache:
    if _slide_cache is None:
        raise RuntimeError("Slide cache not initialized. Call initialize_caches() first.")
    return _slide_cache
