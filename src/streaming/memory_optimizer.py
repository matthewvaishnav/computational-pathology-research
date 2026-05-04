"""Advanced memory optimization for real-time WSI streaming.

This module implements:
- Memory pool management for GPU allocations
- Smart garbage collection strategies
- Memory usage prediction and preallocation
"""

# Re-export all components for backward compatibility
from src.streaming.memory_gc import GCStats, SmartGarbageCollector
from src.streaming.memory_monitoring import (
    MemoryAlert,
    MemoryAnalytics,
    MemoryMonitor,
    MemoryPressureLevel,
    MemorySnapshot,
)
from src.streaming.memory_pool import MemoryBlock, MemoryPoolManager, MemoryPoolStats
from src.streaming.memory_pool_strategy import MemoryPoolStrategy
from src.streaming.memory_prediction import MemoryPrediction, MemoryUsagePredictor

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
