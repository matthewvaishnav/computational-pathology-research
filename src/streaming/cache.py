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

    Security note:
        This function must only deserialize trusted cache entries produced by
        this application. It is intentionally not a general-purpose loader for
        user uploads, network payloads, request bodies, or other untrusted data.
    """
    return pickle.loads(data)  # nosec B301 - trusted internal cache compatibility only; never use for untrusted input.


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