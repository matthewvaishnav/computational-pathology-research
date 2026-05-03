"""
Edge Inference Caching System

Provides intelligent caching for mobile and edge inference to improve performance
and reduce computational overhead through result caching, model caching, and
feature caching strategies.
"""

from .inference_cache import (
    InferenceCacheManager,
    CacheConfig,
    CacheEntry,
    CacheStrategy,
    CacheEvictionPolicy
)
from .feature_cache import (
    FeatureCacheManager,
    FeatureCacheConfig,
    FeatureEntry
)

__all__ = [
    "InferenceCacheManager",
    "CacheConfig", 
    "CacheEntry",
    "CacheStrategy",
    "CacheEvictionPolicy",
    "FeatureCacheManager",
    "FeatureCacheConfig",
    "FeatureEntry"
]