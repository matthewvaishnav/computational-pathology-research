"""Advanced memory optimization for real-time WSI streaming.

This module implements:
- Memory pool management for GPU allocations
- Smart garbage collection strategies
- Memory usage prediction and preallocation
"""

# Re-export all components for backward compatibility
from .memory_gc import GCStats, SmartGarbageCollector
from .memory_monitoring import (
    MemoryAlert,
    MemoryAnalytics,
    MemoryMonitor,
    MemoryPressureLevel,
    MemorySnapshot,
)
from .memory_pool import MemoryBlock, MemoryPoolManager, MemoryPoolStats
from .memory_pool_strategy import MemoryPoolStrategy
from .memory_prediction import MemoryPrediction, MemoryUsagePredictor

__all__ = [
    # Strategy
    "MemoryPoolStrategy",
    # Pool management
    "MemoryBlock",
    "MemoryPoolStats",
    "MemoryPoolManager",
    # Garbage collection
    "GCStats",
    "SmartGarbageCollector",
    # Prediction
    "MemoryPrediction",
    "MemoryUsagePredictor",
    # Monitoring
    "MemoryPressureLevel",
    "MemorySnapshot",
    "MemoryAlert",
    "MemoryAnalytics",
    "MemoryMonitor",
]
