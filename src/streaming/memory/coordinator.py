"""Memory coordinator facade for unified memory management.

This module provides a facade that coordinates all memory management components:
profiler, cache, batch optimizer, and monitor.

Classes:
    MemoryCoordinator: Facade coordinating all memory components

Example:
    >>> from streaming.memory import MemoryCoordinator, OptimizerConfig
    >>>
    >>> config = OptimizerConfig(memory_limit_gb=8.0, cache_size_mb=1000)
    >>> coordinator = MemoryCoordinator(config)
    >>>
    >>> # Optimize batch size
    >>> optimal = coordinator.optimize_batch_size(available_memory_gb=6.0)
    >>>
    >>> # Get current status
    >>> status = coordinator.get_status()
"""

import logging
from typing import Any, Dict, Optional


from .batch_optimizer import BatchOptimizer, OptimalSizes
from .cache_manager import CacheManager
from .config import OptimizerConfig
from .monitor import MemoryAnalytics, MemoryMonitor
from .profiler import MemoryPressureLevel, MemoryProfiler, MemorySnapshot

logger = logging.getLogger(__name__)


class MemoryCoordinator:
    """Facade coordinating all memory management components.

    This class provides a unified interface to:
    - Memory profiling (MemoryProfiler)
    - Cache management (CacheManager)
    - Batch optimization (BatchOptimizer)
    - Real-time monitoring (MemoryMonitor)

    Features:
    - Simplified API for common operations
    - Automatic component coordination
    - Backward-compatible with MemoryOptimizer
    - Comprehensive status reporting
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """Initialize memory coordinator.

        Args:
            config: Configuration for memory optimization. If None, uses defaults.
        """
        self.config = config or OptimizerConfig()

        # Initialize components
        self.profiler = (
            MemoryProfiler(device=self.config.device, memory_limit_gb=self.config.memory_limit_gb)
            if self.config.enable_profiling
            else None
        )

        self.cache = CacheManager(max_size_mb=self.config.cache_size_mb)

        self.batch_optimizer = BatchOptimizer(
            available_memory_gb=self.config.memory_limit_gb,
            min_batch_size=self.config.batch_size_range[0],
            max_batch_size=self.config.batch_size_range[1],
        )

        self.monitor = (
            MemoryMonitor(
                device=self.config.device,
                memory_limit_gb=self.config.memory_limit_gb,
                sampling_interval_ms=self.config.sampling_interval_ms,
                enable_alerts=self.config.enable_monitoring,
            )
            if self.config.enable_monitoring
            else None
        )

        # Start monitoring if enabled
        if self.monitor:
            self.monitor.start_monitoring()

        logger.info(
            f"MemoryCoordinator initialized: device={self.config.device}, "
            f"limit={self.config.memory_limit_gb:.2f}GB, "
            f"cache={self.config.cache_size_mb:.0f}MB"
        )

    # ========================================================================
    # Profiling Operations
    # ========================================================================

    def get_current_snapshot(self) -> Optional[MemorySnapshot]:
        """Get current memory snapshot.

        Returns:
            Current memory snapshot or None if profiling disabled
        """
        if self.profiler:
            return self.profiler.get_snapshot()
        return None

    def get_pressure_level(self) -> Optional[MemoryPressureLevel]:
        """Get current memory pressure level.

        Returns:
            Current pressure level or None if profiling disabled
        """
        snapshot = self.get_current_snapshot()
        return snapshot.pressure_level if snapshot else None

    # ========================================================================
    # Cache Operations
    # ========================================================================

    def cache_get(self, key: str) -> Optional[Any]:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        return self.cache.get(key)

    def cache_put(self, key: str, value: Any, size_mb: Optional[float] = None):
        """Put item in cache.

        Args:
            key: Cache key
            value: Value to cache
            size_mb: Size in MB (estimated if None)
        """
        self.cache.put(key, value, size_mb)

    def cache_clear(self):
        """Clear all cache entries."""
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return self.cache.get_stats()

    # ========================================================================
    # Batch Optimization Operations
    # ========================================================================

    def optimize_batch_size(
        self,
        available_memory_gb: Optional[float] = None,
        tile_size: int = 224,
        feature_dim: int = 512,
    ) -> int:
        """Optimize batch size for available memory.

        Args:
            available_memory_gb: Available memory (uses current if None)
            tile_size: Tile size in pixels
            feature_dim: Feature dimension

        Returns:
            Optimal batch size
        """
        if available_memory_gb is None and self.profiler:
            snapshot = self.profiler.get_snapshot()
            available_memory_gb = snapshot.total_gb - snapshot.allocated_gb
        elif available_memory_gb is None:
            available_memory_gb = self.config.memory_limit_gb * 0.7

        return self.batch_optimizer.optimize_batch_size(
            available_memory_gb=available_memory_gb, tile_size=tile_size, feature_dim=feature_dim
        )

    def optimize_for_workload(
        self, workload_size: int, tile_size: int = 224, feature_dim: int = 512
    ) -> OptimalSizes:
        """Optimize batch and tile sizes for workload.

        Args:
            workload_size: Total number of items to process
            tile_size: Current tile size
            feature_dim: Feature dimension

        Returns:
            Optimal sizes (batch_size, tile_size, num_batches)
        """
        return self.batch_optimizer.optimize_for_workload(
            workload_size=workload_size, tile_size=tile_size, feature_dim=feature_dim
        )

    # ========================================================================
    # Monitoring Operations
    # ========================================================================

    def get_recent_alerts(self, count: int = 10) -> list:
        """Get recent memory alerts.

        Args:
            count: Number of alerts to return

        Returns:
            List of recent alerts (empty if monitoring disabled)
        """
        if self.monitor:
            return self.monitor.get_recent_alerts(count)
        return []

    def get_analytics(self) -> Optional[MemoryAnalytics]:
        """Get memory usage analytics.

        Returns:
            Memory analytics or None if monitoring disabled
        """
        if self.monitor:
            return self.monitor.get_analytics()
        return None

    def record_oom_event(self):
        """Record an out-of-memory event."""
        if self.monitor:
            self.monitor.record_oom_event()

    # ========================================================================
    # Unified Status and Reporting
    # ========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive memory management status.

        Returns:
            Dictionary with status from all components
        """
        status = {
            "config": self.config.to_dict(),
            "profiling_enabled": self.profiler is not None,
            "monitoring_enabled": self.monitor is not None,
        }

        # Add profiler status
        if self.profiler:
            snapshot = self.profiler.get_snapshot()
            status["current_memory"] = snapshot.to_dict()

        # Add cache status
        status["cache"] = self.get_cache_stats()

        # Add monitoring status
        if self.monitor:
            analytics = self.monitor.get_analytics()
            status["analytics"] = analytics.to_dict()
            status["recent_alerts"] = [
                alert.to_dict() for alert in self.monitor.get_recent_alerts(5)
            ]

        return status

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory management report.

        Returns:
            Detailed report dictionary
        """
        report = self.get_status()

        # Add recommendations
        recommendations = []

        if self.profiler:
            snapshot = self.profiler.get_snapshot()

            if snapshot.pressure_level == MemoryPressureLevel.HIGH:
                recommendations.append("Memory pressure is HIGH - consider reducing batch size")
            elif snapshot.pressure_level == MemoryPressureLevel.CRITICAL:
                recommendations.append("Memory pressure is CRITICAL - immediate action required")

        cache_stats = self.get_cache_stats()
        if cache_stats["hit_rate"] < 0.5:
            recommendations.append("Cache hit rate is low - consider increasing cache size")

        report["recommendations"] = recommendations

        return report

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    def cleanup(self):
        """Clean up all resources."""
        if self.monitor:
            self.monitor.stop_monitoring()
            self.monitor.cleanup()

        if self.cache:
            self.cache.clear()

        logger.info("MemoryCoordinator cleaned up")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
