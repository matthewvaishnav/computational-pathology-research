"""
Edge Inference Caching System

Provides intelligent caching for mobile and edge inference to improve performance
and reduce computational overhead through result caching, model caching, and
feature caching strategies.
"""

from .feature_cache import FeatureCacheConfig, FeatureCacheManager, FeatureEntry
from .inference_cache import (
    CacheConfig,
    CacheEntry,
    CacheEvictionPolicy,
    CacheStrategy,
    InferenceCacheManager,
)

__all__ = [
    "InferenceCacheManager",
    "CacheConfig",
    "CacheEntry",
    "CacheStrategy",
    "CacheEvictionPolicy",
    "FeatureCacheManager",
    "FeatureCacheConfig",
    "FeatureEntry",
]
