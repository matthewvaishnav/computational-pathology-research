"""Real-time memory monitoring and alerting system.

This module provides real-time memory monitoring with configurable thresholds,
alert generation, and analytics capabilities.

Classes:
    MemoryAlert: Memory alert notification dataclass
    MemoryAnalytics: Memory usage analytics and statistics dataclass
    MemoryMonitor: Real-time memory monitoring and alerting system

Features:
    - Real-time memory usage tracking with <100ms latency
    - Memory pressure detection with configurable thresholds
    - Alert generation for memory issues
    - Analytics and reporting capabilities
    - Thread-safe monitoring with background thread
    - Context manager support for automatic start/stop

Example:
    >>> monitor = MemoryMonitor(device=torch.device('cuda'), memory_limit_gb=8.0)
    >>> monitor.start_monitoring()
    >>> snapshot = monitor.get_current_snapshot()
    >>> print(f"Usage: {snapshot.utilization_percent:.1f}%")
    >>> monitor.stop_monitoring()

    # Or use as context manager:
    >>> with MemoryMonitor(device=torch.device('cuda')) as monitor:
    ...     # Monitoring runs automatically
    ...     analytics = monitor.get_analytics()
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .profiler import MemoryPressureLevel, MemorySnapshot

logger = logging.getLogger(__name__)


@dataclass
class MemoryAlert:
    """Memory alert notification.

    Attributes:
        timestamp: Unix timestamp when alert was generated
        alert_type: Type of alert ('pressure', 'threshold', 'oom_risk')
        severity: Alert severity ('warning', 'error', 'critical')
        message: Human-readable alert message
        current_usage_gb: Current memory usage in GB
        threshold_gb: Threshold that triggered the alert in GB
        recommended_action: Suggested action to resolve the issue
    """

    timestamp: float
    alert_type: str  # 'pressure', 'threshold', 'oom_risk'
    severity: str  # 'warning', 'error', 'critical'
    message: str
    current_usage_gb: float
    threshold_gb: float
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "current_usage_gb": self.current_usage_gb,
            "threshold_gb": self.threshold_gb,
            "recommended_action": self.recommended_action,
        }


@dataclass
class MemoryAnalytics:
    """Memory usage analytics and statistics.

    Attributes:
        monitoring_duration_seconds: Total monitoring duration
        total_snapshots: Number of snapshots collected
        peak_usage_gb: Peak memory usage observed
        avg_usage_gb: Average memory usage
        min_usage_gb: Minimum memory usage
        pressure_distribution: Percentage time in each pressure level
        alerts_triggered: Total number of alerts triggered
        oom_events: Number of out-of-memory events
        gc_collections: Number of garbage collections
        memory_freed_gb: Total memory freed by GC
    """

    monitoring_duration_seconds: float
    total_snapshots: int
    peak_usage_gb: float
    avg_usage_gb: float
    min_usage_gb: float
    pressure_distribution: Dict[str, float]  # Percentage time in each pressure level
    alerts_triggered: int
    oom_events: int
    gc_collections: int
    memory_freed_gb: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "monitoring_duration_seconds": self.monitoring_duration_seconds,
            "total_snapshots": self.total_snapshots,
            "peak_usage_gb": self.peak_usage_gb,
            "avg_usage_gb": self.avg_usage_gb,
            "min_usage_gb": self.min_usage_gb,
            "pressure_distribution": self.pressure_distribution,
            "alerts_triggered": self.alerts_triggered,
            "oom_events": self.oom_events,
            "gc_collections": self.gc_collections,
            "memory_freed_gb": self.memory_freed_gb,
        }


class MemoryMonitor:
    """Real-time memory monitoring and alerting system.

    Features:
    - Real-time memory usage tracking with <100ms latency
    - Memory pressure detection with configurable thresholds
    - Alert generation for memory issues
    - Analytics and reporting capabilities
    - Integration with memory optimizer components

    Thread Safety:
        All public methods are thread-safe. Internal state is protected
        by a lock to allow safe concurrent access from monitoring thread
        and application threads.
    """

    def __init__(
        self,
        device: torch.device,
        memory_limit_gb: float = 2.0,
        sampling_interval_ms: float = 100.0,
        enable_alerts: bool = True,
        alert_callback: Optional[Callable[[MemoryAlert], None]] = None,
    ):
        """Initialize memory monitor.

        Args:
            device: Target device to monitor
            memory_limit_gb: Memory limit in GB for pressure calculation
            sampling_interval_ms: Sampling interval in milliseconds
            enable_alerts: Enable alert generation
            alert_callback: Optional callback function for alerts
        """
        self.device = device
        self.memory_limit_gb = memory_limit_gb
        self.sampling_interval_ms = sampling_interval_ms
        self.enable_alerts = enable_alerts
        self.alert_callback = alert_callback

        # Get total device memory
        if device.type == "cuda":
            self.total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        else:
            self.total_memory_gb = memory_limit_gb

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None

        # Memory snapshots (circular buffer)
        self.snapshots: deque = deque(maxlen=1000)  # Keep last 1000 snapshots

        # Alerts
        self.alerts: deque = deque(maxlen=100)  # Keep last 100 alerts

        # Pressure thresholds
        self.pressure_thresholds = {
            MemoryPressureLevel.NORMAL: 0.60,
            MemoryPressureLevel.MODERATE: 0.75,
            MemoryPressureLevel.HIGH: 0.90,
            MemoryPressureLevel.CRITICAL: 0.95,
        }

        # Statistics
        self.peak_usage_gb = 0.0
        self.oom_events = 0
        self.alerts_triggered = 0

        # Lock for thread safety
        self.lock = threading.Lock()

        logger.info(
            f"MemoryMonitor initialized: device={device}, limit={memory_limit_gb:.2f}GB, "
            f"sampling={sampling_interval_ms}ms"
        )

    def _get_current_memory_usage(self) -> Tuple[float, float]:
        """Get current memory usage.

        Returns:
            Tuple of (allocated_gb, reserved_gb)
        """
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
            return allocated, reserved
        else:
            # For CPU, use a simple estimate
            return 0.0, 0.0

    def _calculate_pressure_level(self, allocated_gb: float) -> MemoryPressureLevel:
        """Calculate memory pressure level.

        Args:
            allocated_gb: Current allocated memory in GB

        Returns:
            Memory pressure level
        """
        if self.memory_limit_gb == 0:
            return MemoryPressureLevel.NORMAL

        utilization = allocated_gb / self.memory_limit_gb

        # Check thresholds from highest to lowest
        if utilization >= self.pressure_thresholds[MemoryPressureLevel.CRITICAL]:
            return MemoryPressureLevel.CRITICAL
        elif utilization >= self.pressure_thresholds[MemoryPressureLevel.HIGH]:
            return MemoryPressureLevel.HIGH
        elif utilization >= self.pressure_thresholds[MemoryPressureLevel.MODERATE]:
            return MemoryPressureLevel.MODERATE
        else:
            return MemoryPressureLevel.NORMAL

    def _create_snapshot(self) -> MemorySnapshot:
        """Create memory snapshot.

        Returns:
            Memory snapshot
        """
        allocated_gb, reserved_gb = self._get_current_memory_usage()
        pressure_level = self._calculate_pressure_level(allocated_gb)

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_gb=allocated_gb,
            reserved_gb=reserved_gb,
            total_gb=self.total_memory_gb,
            pressure_level=pressure_level,
        )

        # Update peak usage
        if allocated_gb > self.peak_usage_gb:
            self.peak_usage_gb = allocated_gb

        return snapshot

    def _check_and_generate_alerts(self, snapshot: MemorySnapshot):
        """Check conditions and generate alerts if needed.

        Args:
            snapshot: Current memory snapshot
        """
        if not self.enable_alerts:
            return

        # Check for critical pressure
        if snapshot.pressure_level == MemoryPressureLevel.CRITICAL:
            alert = MemoryAlert(
                timestamp=snapshot.timestamp,
                alert_type="pressure",
                severity="critical",
                message=f"Critical memory pressure: {snapshot.utilization_percent:.1f}% usage",
                current_usage_gb=snapshot.allocated_gb,
                threshold_gb=self.memory_limit_gb
                * self.pressure_thresholds[MemoryPressureLevel.CRITICAL],
                recommended_action="Reduce batch size or trigger garbage collection immediately",
            )
            self._trigger_alert(alert)

        # Check for high pressure
        elif snapshot.pressure_level == MemoryPressureLevel.HIGH:
            # Only alert if sustained high pressure (check last 3 snapshots)
            if len(self.snapshots) >= 3:
                recent_high = all(
                    s.pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]
                    for s in list(self.snapshots)[-3:]
                )

                if recent_high:
                    alert = MemoryAlert(
                        timestamp=snapshot.timestamp,
                        alert_type="pressure",
                        severity="warning",
                        message=f"Sustained high memory pressure: {snapshot.utilization_percent:.1f}% usage",
                        current_usage_gb=snapshot.allocated_gb,
                        threshold_gb=self.memory_limit_gb
                        * self.pressure_thresholds[MemoryPressureLevel.HIGH],
                        recommended_action="Consider reducing batch size or triggering garbage collection",
                    )
                    self._trigger_alert(alert)

        # Check for approaching limit
        if snapshot.allocated_gb > self.memory_limit_gb * 0.95:
            alert = MemoryAlert(
                timestamp=snapshot.timestamp,
                alert_type="threshold",
                severity="error",
                message=f"Memory usage approaching limit: {snapshot.allocated_gb:.2f}GB / {self.memory_limit_gb:.2f}GB",
                current_usage_gb=snapshot.allocated_gb,
                threshold_gb=self.memory_limit_gb,
                recommended_action="Immediate action required: reduce memory usage or risk OOM",
            )
            self._trigger_alert(alert)

    def _trigger_alert(self, alert: MemoryAlert):
        """Trigger an alert.

        Args:
            alert: Alert to trigger
        """
        with self.lock:
            self.alerts.append(alert)
            self.alerts_triggered += 1

        # Log alert
        if alert.severity == "critical":
            logger.error(f"Memory Alert [{alert.severity}]: {alert.message}")
        elif alert.severity == "error":
            logger.error(f"Memory Alert [{alert.severity}]: {alert.message}")
        else:
            logger.warning(f"Memory Alert [{alert.severity}]: {alert.message}")

        # Call callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _monitoring_loop(self):
        """Main monitoring loop (runs in separate thread)."""
        logger.info("Memory monitoring started")

        while self.is_monitoring:
            try:
                # Create snapshot
                snapshot = self._create_snapshot()

                with self.lock:
                    self.snapshots.append(snapshot)

                # Check for alerts
                self._check_and_generate_alerts(snapshot)

                # Sleep for sampling interval
                time.sleep(self.sampling_interval_ms / 1000.0)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)  # Back off on error

        logger.info("Memory monitoring stopped")

    def start_monitoring(self):
        """Start real-time memory monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return

        self.is_monitoring = True
        self.start_time = time.time()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("Memory monitoring thread started")

    def stop_monitoring(self):
        """Stop memory monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        # Wait for thread to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)

        logger.info("Memory monitoring stopped")

    def get_current_snapshot(self) -> MemorySnapshot:
        """Get current memory snapshot.

        Returns:
            Current memory snapshot
        """
        return self._create_snapshot()

    def get_recent_snapshots(self, count: int = 10) -> List[MemorySnapshot]:
        """Get recent memory snapshots.

        Args:
            count: Number of recent snapshots to return

        Returns:
            List of recent snapshots
        """
        with self.lock:
            return list(self.snapshots)[-count:]

    def get_recent_alerts(self, count: int = 10) -> List[MemoryAlert]:
        """Get recent alerts.

        Args:
            count: Number of recent alerts to return

        Returns:
            List of recent alerts
        """
        with self.lock:
            return list(self.alerts)[-count:]

    def get_analytics(self) -> MemoryAnalytics:
        """Get memory usage analytics.

        Returns:
            Memory analytics
        """
        with self.lock:
            if not self.snapshots:
                return MemoryAnalytics(
                    monitoring_duration_seconds=0.0,
                    total_snapshots=0,
                    peak_usage_gb=0.0,
                    avg_usage_gb=0.0,
                    min_usage_gb=0.0,
                    pressure_distribution={},
                    alerts_triggered=0,
                    oom_events=0,
                    gc_collections=0,
                    memory_freed_gb=0.0,
                )

            # Calculate statistics
            snapshots_list = list(self.snapshots)
            allocated_values = [s.allocated_gb for s in snapshots_list]

            avg_usage = np.mean(allocated_values)
            min_usage = np.min(allocated_values)

            # Calculate pressure distribution
            pressure_counts = {}
            for snapshot in snapshots_list:
                level = snapshot.pressure_level.value
                pressure_counts[level] = pressure_counts.get(level, 0) + 1

            total_snapshots = len(snapshots_list)
            pressure_distribution = {
                level: (count / total_snapshots) * 100.0 for level, count in pressure_counts.items()
            }

            # Calculate monitoring duration
            if self.start_time:
                duration = time.time() - self.start_time
            else:
                duration = 0.0

            return MemoryAnalytics(
                monitoring_duration_seconds=duration,
                total_snapshots=total_snapshots,
                peak_usage_gb=self.peak_usage_gb,
                avg_usage_gb=avg_usage,
                min_usage_gb=min_usage,
                pressure_distribution=pressure_distribution,
                alerts_triggered=self.alerts_triggered,
                oom_events=self.oom_events,
                gc_collections=0,  # Would need integration with SmartGC
                memory_freed_gb=0.0,  # Would need integration with SmartGC
            )

    def record_oom_event(self):
        """Record an out-of-memory event."""
        with self.lock:
            self.oom_events += 1

        # Generate critical alert
        snapshot = self._create_snapshot()
        alert = MemoryAlert(
            timestamp=time.time(),
            alert_type="oom_risk",
            severity="critical",
            message="Out of memory event detected",
            current_usage_gb=snapshot.allocated_gb,
            threshold_gb=self.memory_limit_gb,
            recommended_action="Emergency cleanup required: reduce batch size and trigger aggressive GC",
        )
        self._trigger_alert(alert)

    def set_pressure_threshold(self, level: MemoryPressureLevel, threshold: float):
        """Set custom pressure threshold.

        Args:
            level: Pressure level to configure
            threshold: Threshold value (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        with self.lock:
            self.pressure_thresholds[level] = threshold

        logger.info(f"Updated pressure threshold: {level.value} = {threshold:.2f}")

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory monitoring report.

        Returns:
            Dictionary with monitoring report
        """
        analytics = self.get_analytics()
        current_snapshot = self.get_current_snapshot()
        recent_alerts = self.get_recent_alerts(count=5)

        return {
            "current_status": current_snapshot.to_dict(),
            "analytics": analytics.to_dict(),
            "recent_alerts": [alert.to_dict() for alert in recent_alerts],
            "pressure_thresholds": {
                level.value: threshold for level, threshold in self.pressure_thresholds.items()
            },
            "monitoring_config": {
                "device": str(self.device),
                "memory_limit_gb": self.memory_limit_gb,
                "sampling_interval_ms": self.sampling_interval_ms,
                "alerts_enabled": self.enable_alerts,
            },
        }

    def cleanup(self):
        """Clean up monitoring resources."""
        self.stop_monitoring()

        with self.lock:
            self.snapshots.clear()
            self.alerts.clear()

        logger.info("Memory monitor cleaned up")

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
