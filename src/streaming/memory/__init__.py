"""Memory management module for WSI streaming.

This module provides focused components for memory optimization during
whole slide image (WSI) streaming and processing. It replaces the monolithic
MemoryOptimizer class with a set of specialized classes, each with a single
responsibility.

Components:
    - MemoryProfiler: Profile and track memory usage (CPU and GPU)
    - CacheManager: LRU cache management for tiles and features
    - BatchOptimizer: Optimize batch and tile sizes based on available memory
    - MemoryMonitor: Monitor memory usage and send alerts
    - OptimizerConfig: Configuration management for memory optimization
    - MemoryCoordinator: Facade that coordinates all memory management components

Usage:
    from streaming.memory import MemoryCoordinator, OptimizerConfig
    
    config = OptimizerConfig(cache_size_mb=1000, alert_threshold_mb=8000)
    coordinator = MemoryCoordinator(config)
    
    # Optimize for workload
    optimal_sizes = coordinator.optimize_for_workload(workload_size=1000)

This refactoring improves:
    - Testability: Each component can be tested in isolation
    - Maintainability: Clear separation of concerns
    - Readability: Each file <300 lines vs original 1097 lines
    - Reusability: Components can be used independently

See Also:
    - Original implementation: src/streaming/memory_optimizer.py
    - Design document: .kiro/specs/clean-code-refactoring/design.md
"""

__version__ = "1.0.0"

# Import profiler components
from .profiler import MemoryProfiler, MemoryPressureLevel, MemorySnapshot

# Import cache manager
from .cache_manager import CacheManager, CacheEntry

# Import batch optimizer
from .batch_optimizer import BatchOptimizer, OptimalSizes

# Import monitor
from .monitor import MemoryAlert, MemoryAnalytics, MemoryMonitor

__all__ = [
    "MemoryProfiler",
    "MemorySnapshot",
    "MemoryPressureLevel",
    "CacheManager",
    "CacheEntry",
    "BatchOptimizer",
    "OptimalSizes",
    "MemoryMonitor",
    "MemoryAlert",
    "MemoryAnalytics",
]

# Components will be imported here as they are created:
# from .config import OptimizerConfig
# from .coordinator import MemoryCoordinator
