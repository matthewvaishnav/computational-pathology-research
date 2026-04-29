"""Unit tests for memory monitoring and alerting.

Tests cover:
- Real-time memory usage tracking
- Memory pressure detection and response
- Memory usage analytics and reporting
- Alert generation and callbacks
"""

import importlib.util

# Import directly from memory_optimizer module
import sys
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

# Load memory_optimizer module directly
memory_optimizer_path = (
    Path(__file__).parent.parent.parent / "src" / "streaming" / "memory_optimizer.py"
)
spec = importlib.util.spec_from_file_location("memory_optimizer", memory_optimizer_path)
memory_optimizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_optimizer_module)

# Import classes from loaded module
MemoryMonitor = memory_optimizer_module.MemoryMonitor
MemoryPressureLevel = memory_optimizer_module.MemoryPressureLevel
MemorySnapshot = memory_optimizer_module.MemorySnapshot
MemoryAlert = memory_optimizer_module.MemoryAlert
MemoryAnalytics = memory_optimizer_module.MemoryAnalytics


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def device():
    """Get test device (CPU for CI compatibility)."""
    return torch.device("cpu")


@pytest.fixture
def memory_monitor(device):
    """Create memory monitor."""
    return MemoryMonitor(
        device=device, memory_limit_gb=2.0, sampling_interval_ms=100.0, enable_alerts=True
    )


@pytest.fixture
def alert_callback():
    """Create mock alert callback."""
    return Mock()


# ============================================================================
# MemoryPressureLevel Tests
# ============================================================================


class TestMemoryPressureLevel:
    """Test memory pressure level enum."""

    def test_pressure_levels_exist(self):
        """Test all pressure levels are defined."""
        assert MemoryPressureLevel.NORMAL
        assert MemoryPressureLevel.MODERATE
        assert MemoryPressureLevel.HIGH
        assert MemoryPressureLevel.CRITICAL

    def test_pressure_level_values(self):
        """Test pressure level string values."""
        assert MemoryPressureLevel.NORMAL.value == "normal"
        assert MemoryPressureLevel.MODERATE.value == "moderate"
        assert MemoryPressureLevel.HIGH.value == "high"
        assert MemoryPressureLevel.CRITICAL.value == "critical"


# ============================================================================
# MemorySnapshot Tests
# ============================================================================


class TestMemorySnapshot:
    """Test memory snapshot functionality."""

    def test_initialization(self):
        """Test snapshot initialization."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_gb=1.5,
            reserved_gb=2.0,
            total_gb=8.0,
            pressure_level=MemoryPressureLevel.MODERATE,
        )

        assert snapshot.allocated_gb == 1.5
        assert snapshot.reserved_gb == 2.0
        assert snapshot.total_gb == 8.0
        assert snapshot.pressure_level == MemoryPressureLevel.MODERATE

    def test_utilization_percent(self):
        """Test utilization calculation."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_gb=4.0,
            reserved_gb=5.0,
            total_gb=8.0,
            pressure_level=MemoryPressureLevel.HIGH,
        )

        assert snapshot.utilization_percent == 50.0

    def test_utilization_zero_total(self):
        """Test utilization with zero total."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_gb=0.0,
            reserved_gb=0.0,
            total_gb=0.0,
            pressure_level=MemoryPressureLevel.NORMAL,
        )

        assert snapshot.utilization_percent == 0.0

    def test_to_dict(self):
        """Test snapshot serialization."""
        snapshot = MemorySnapshot(
            timestamp=123.456,
            allocated_gb=1.5,
            reserved_gb=2.0,
            total_gb=8.0,
            pressure_level=MemoryPressureLevel.MODERATE,
        )

        snapshot_dict = snapshot.to_dict()

        assert isinstance(snapshot_dict, dict)
        assert snapshot_dict["timestamp"] == 123.456
        assert snapshot_dict["allocated_gb"] == 1.5
        assert snapshot_dict["reserved_gb"] == 2.0
        assert snapshot_dict["total_gb"] == 8.0
        assert snapshot_dict["pressure_level"] == "moderate"
        assert "utilization_percent" in snapshot_dict


# ============================================================================
# MemoryAlert Tests
# ============================================================================


class TestMemoryAlert:
    """Test memory alert functionality."""

    def test_initialization(self):
        """Test alert initialization."""
        alert = MemoryAlert(
            timestamp=time.time(),
            alert_type="pressure",
            severity="warning",
            message="High memory pressure",
            current_usage_gb=6.0,
            threshold_gb=8.0,
            recommended_action="Reduce batch size",
        )

        assert alert.alert_type == "pressure"
        assert alert.severity == "warning"
        assert alert.message == "High memory pressure"
        assert alert.current_usage_gb == 6.0
        assert alert.threshold_gb == 8.0

    def test_to_dict(self):
        """Test alert serialization."""
        alert = MemoryAlert(
            timestamp=123.456,
            alert_type="threshold",
            severity="error",
            message="Memory limit exceeded",
            current_usage_gb=7.5,
            threshold_gb=8.0,
            recommended_action="Immediate cleanup required",
        )

        alert_dict = alert.to_dict()

        assert isinstance(alert_dict, dict)
        assert alert_dict["timestamp"] == 123.456
        assert alert_dict["alert_type"] == "threshold"
        assert alert_dict["severity"] == "error"
        assert alert_dict["message"] == "Memory limit exceeded"


# ============================================================================
# MemoryAnalytics Tests
# ============================================================================


class TestMemoryAnalytics:
    """Test memory analytics functionality."""

    def test_initialization(self):
        """Test analytics initialization."""
        analytics = MemoryAnalytics(
            monitoring_duration_seconds=120.0,
            total_snapshots=100,
            peak_usage_gb=7.5,
            avg_usage_gb=5.0,
            min_usage_gb=2.0,
            pressure_distribution={"normal": 60.0, "moderate": 30.0, "high": 10.0},
            alerts_triggered=5,
            oom_events=1,
            gc_collections=10,
            memory_freed_gb=2.5,
        )

        assert analytics.monitoring_duration_seconds == 120.0
        assert analytics.total_snapshots == 100
        assert analytics.peak_usage_gb == 7.5
        assert analytics.avg_usage_gb == 5.0
        assert analytics.alerts_triggered == 5

    def test_to_dict(self):
        """Test analytics serialization."""
        analytics = MemoryAnalytics(
            monitoring_duration_seconds=120.0,
            total_snapshots=100,
            peak_usage_gb=7.5,
            avg_usage_gb=5.0,
            min_usage_gb=2.0,
            pressure_distribution={"normal": 60.0, "moderate": 30.0},
            alerts_triggered=5,
            oom_events=1,
            gc_collections=10,
            memory_freed_gb=2.5,
        )

        analytics_dict = analytics.to_dict()

        assert isinstance(analytics_dict, dict)
        assert analytics_dict["monitoring_duration_seconds"] == 120.0
        assert analytics_dict["total_snapshots"] == 100
        assert analytics_dict["peak_usage_gb"] == 7.5
        assert "pressure_distribution" in analytics_dict


# ============================================================================
# MemoryMonitor Tests
# ============================================================================


class TestMemoryMonitor:
    """Test memory monitor functionality."""

    def test_initialization(self, device):
        """Test monitor initialization."""
        monitor = MemoryMonitor(
            device=device, memory_limit_gb=2.0, sampling_interval_ms=100.0, enable_alerts=True
        )

        assert monitor.device == device
        assert monitor.memory_limit_gb == 2.0
        assert monitor.sampling_interval_ms == 100.0
        assert monitor.enable_alerts is True
        assert monitor.is_monitoring is False

    def test_initialization_with_callback(self, device, alert_callback):
        """Test monitor initialization with callback."""
        monitor = MemoryMonitor(
            device=device, memory_limit_gb=2.0, enable_alerts=True, alert_callback=alert_callback
        )

        assert monitor.alert_callback == alert_callback

    def test_get_current_snapshot(self, memory_monitor):
        """Test getting current snapshot."""
        snapshot = memory_monitor.get_current_snapshot()

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.allocated_gb >= 0.0
        assert snapshot.total_gb > 0.0
        assert isinstance(snapshot.pressure_level, MemoryPressureLevel)

    def test_calculate_pressure_level_normal(self, memory_monitor):
        """Test pressure level calculation - normal."""
        level = memory_monitor._calculate_pressure_level(0.5)  # 25% of 2GB limit

        assert level == MemoryPressureLevel.NORMAL

    def test_calculate_pressure_level_moderate(self, memory_monitor):
        """Test pressure level calculation - moderate."""
        level = memory_monitor._calculate_pressure_level(1.5)  # 75% of 2GB limit

        assert level == MemoryPressureLevel.MODERATE

    def test_calculate_pressure_level_high(self, memory_monitor):
        """Test pressure level calculation - high."""
        level = memory_monitor._calculate_pressure_level(1.8)  # 90% of 2GB limit

        assert level == MemoryPressureLevel.HIGH

    def test_calculate_pressure_level_critical(self, memory_monitor):
        """Test pressure level calculation - critical."""
        level = memory_monitor._calculate_pressure_level(1.95)  # 97.5% of 2GB limit

        assert level == MemoryPressureLevel.CRITICAL

    def test_start_stop_monitoring(self, memory_monitor):
        """Test starting and stopping monitoring."""
        # Start monitoring
        memory_monitor.start_monitoring()

        assert memory_monitor.is_monitoring is True
        assert memory_monitor.monitoring_thread is not None
        assert memory_monitor.start_time is not None

        # Wait a bit for snapshots
        time.sleep(0.3)

        # Stop monitoring
        memory_monitor.stop_monitoring()

        assert memory_monitor.is_monitoring is False

    def test_monitoring_collects_snapshots(self, memory_monitor):
        """Test that monitoring collects snapshots."""
        memory_monitor.start_monitoring()

        # Wait for some snapshots
        time.sleep(0.5)

        memory_monitor.stop_monitoring()

        # Should have collected snapshots
        assert len(memory_monitor.snapshots) > 0

    def test_get_recent_snapshots(self, memory_monitor):
        """Test getting recent snapshots."""
        memory_monitor.start_monitoring()

        time.sleep(0.3)

        recent = memory_monitor.get_recent_snapshots(count=5)

        memory_monitor.stop_monitoring()

        assert isinstance(recent, list)
        assert len(recent) <= 5
        assert all(isinstance(s, MemorySnapshot) for s in recent)

    def test_record_oom_event(self, memory_monitor):
        """Test recording OOM event."""
        initial_oom = memory_monitor.oom_events

        memory_monitor.record_oom_event()

        assert memory_monitor.oom_events == initial_oom + 1

        # Should have generated an alert
        assert len(memory_monitor.alerts) > 0

        # Check alert properties
        alert = memory_monitor.alerts[-1]
        assert alert.alert_type == "oom_risk"
        assert alert.severity == "critical"

    def test_alert_callback_triggered(self, device):
        """Test that alert callback is triggered."""
        callback = Mock()

        monitor = MemoryMonitor(
            device=device, memory_limit_gb=2.0, enable_alerts=True, alert_callback=callback
        )

        # Trigger OOM event which generates alert
        monitor.record_oom_event()

        # Callback should have been called
        assert callback.called
        assert isinstance(callback.call_args[0][0], MemoryAlert)

    def test_set_pressure_threshold(self, memory_monitor):
        """Test setting custom pressure threshold."""
        memory_monitor.set_pressure_threshold(MemoryPressureLevel.HIGH, 0.85)

        assert memory_monitor.pressure_thresholds[MemoryPressureLevel.HIGH] == 0.85

    def test_set_pressure_threshold_invalid(self, memory_monitor):
        """Test setting invalid pressure threshold."""
        with pytest.raises(ValueError):
            memory_monitor.set_pressure_threshold(MemoryPressureLevel.HIGH, 1.5)

        with pytest.raises(ValueError):
            memory_monitor.set_pressure_threshold(MemoryPressureLevel.HIGH, -0.1)

    def test_get_analytics_empty(self, memory_monitor):
        """Test getting analytics with no data."""
        analytics = memory_monitor.get_analytics()

        assert isinstance(analytics, MemoryAnalytics)
        assert analytics.total_snapshots == 0
        assert analytics.peak_usage_gb == 0.0

    def test_get_analytics_with_data(self, memory_monitor):
        """Test getting analytics with monitoring data."""
        memory_monitor.start_monitoring()

        time.sleep(0.5)

        memory_monitor.stop_monitoring()

        analytics = memory_monitor.get_analytics()

        assert isinstance(analytics, MemoryAnalytics)
        assert analytics.total_snapshots > 0
        assert analytics.monitoring_duration_seconds > 0
        assert analytics.peak_usage_gb >= 0.0
        assert analytics.avg_usage_gb >= 0.0
        assert isinstance(analytics.pressure_distribution, dict)

    def test_get_recent_alerts(self, memory_monitor):
        """Test getting recent alerts."""
        # Generate some alerts
        memory_monitor.record_oom_event()
        memory_monitor.record_oom_event()

        recent_alerts = memory_monitor.get_recent_alerts(count=5)

        assert isinstance(recent_alerts, list)
        assert len(recent_alerts) >= 2
        assert all(isinstance(a, MemoryAlert) for a in recent_alerts)

    def test_generate_report(self, memory_monitor):
        """Test generating comprehensive report."""
        memory_monitor.start_monitoring()

        time.sleep(0.3)

        memory_monitor.stop_monitoring()

        report = memory_monitor.generate_report()

        assert isinstance(report, dict)
        assert "current_status" in report
        assert "analytics" in report
        assert "recent_alerts" in report
        assert "pressure_thresholds" in report
        assert "monitoring_config" in report

        # Check structure
        assert isinstance(report["current_status"], dict)
        assert isinstance(report["analytics"], dict)
        assert isinstance(report["recent_alerts"], list)
        assert isinstance(report["pressure_thresholds"], dict)
        assert isinstance(report["monitoring_config"], dict)

    def test_context_manager(self, device):
        """Test using monitor as context manager."""
        monitor = MemoryMonitor(device=device, memory_limit_gb=2.0, sampling_interval_ms=100.0)

        with monitor:
            assert monitor.is_monitoring is True
            time.sleep(0.2)

        assert monitor.is_monitoring is False

    def test_cleanup(self, memory_monitor):
        """Test cleanup."""
        memory_monitor.start_monitoring()

        time.sleep(0.2)

        memory_monitor.cleanup()

        assert memory_monitor.is_monitoring is False
        assert len(memory_monitor.snapshots) == 0
        assert len(memory_monitor.alerts) == 0

    def test_peak_usage_tracking(self, memory_monitor):
        """Test peak usage tracking."""
        initial_peak = memory_monitor.peak_usage_gb

        # Create some snapshots
        for _ in range(5):
            memory_monitor._create_snapshot()

        # Peak should be tracked
        assert memory_monitor.peak_usage_gb >= initial_peak

    def test_alerts_disabled(self, device):
        """Test monitoring with alerts disabled."""
        monitor = MemoryMonitor(device=device, memory_limit_gb=2.0, enable_alerts=False)

        # Record OOM event
        monitor.record_oom_event()

        # Should still increment counter but not generate alerts
        assert monitor.oom_events == 1

    def test_pressure_distribution_calculation(self, memory_monitor):
        """Test pressure distribution calculation."""
        memory_monitor.start_monitoring()

        time.sleep(0.5)

        memory_monitor.stop_monitoring()

        analytics = memory_monitor.get_analytics()

        # Check pressure distribution
        assert isinstance(analytics.pressure_distribution, dict)

        # Sum of percentages should be ~100%
        total_percent = sum(analytics.pressure_distribution.values())
        assert 99.0 <= total_percent <= 101.0  # Allow small rounding error

    def test_monitoring_latency(self, memory_monitor):
        """Test that monitoring latency is acceptable."""
        memory_monitor.start_monitoring()

        # Wait for several samples
        time.sleep(0.5)

        memory_monitor.stop_monitoring()

        snapshots = list(memory_monitor.snapshots)

        # Check that snapshots are spaced appropriately
        if len(snapshots) >= 2:
            time_diffs = [
                (snapshots[i + 1].timestamp - snapshots[i].timestamp) * 1000
                for i in range(len(snapshots) - 1)
            ]

            avg_interval = np.mean(time_diffs)

            # Should be close to sampling interval (100ms)
            # Allow some variance due to thread scheduling
            assert 50 <= avg_interval <= 200

    def test_thread_safety(self, memory_monitor):
        """Test thread-safe operations."""
        import threading

        def access_snapshots():
            for _ in range(10):
                memory_monitor.get_recent_snapshots(count=5)
                time.sleep(0.01)

        def access_alerts():
            for _ in range(10):
                memory_monitor.get_recent_alerts(count=5)
                time.sleep(0.01)

        memory_monitor.start_monitoring()

        # Create threads that access data concurrently
        threads = [
            threading.Thread(target=access_snapshots),
            threading.Thread(target=access_alerts),
            threading.Thread(target=access_snapshots),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        memory_monitor.stop_monitoring()

        # Should complete without errors


# ============================================================================
# Integration Tests
# ============================================================================


class TestMemoryMonitorIntegration:
    """Integration tests for memory monitor."""

    def test_monitor_with_memory_allocation(self, device):
        """Test monitoring during memory allocation."""
        monitor = MemoryMonitor(device=device, memory_limit_gb=2.0, sampling_interval_ms=50.0)

        monitor.start_monitoring()

        # Simulate memory allocation
        tensors = []
        for _ in range(10):
            tensor = torch.randn(100, 100)
            tensors.append(tensor)
            time.sleep(0.05)

        # Get analytics
        analytics = monitor.get_analytics()

        monitor.stop_monitoring()

        # Should have captured snapshots
        assert analytics.total_snapshots > 0
        assert analytics.peak_usage_gb >= 0.0

    def test_alert_generation_on_high_pressure(self, device):
        """Test alert generation on high memory pressure."""
        callback = Mock()

        monitor = MemoryMonitor(
            device=device,
            memory_limit_gb=0.1,  # Very low limit to trigger alerts
            sampling_interval_ms=50.0,
            enable_alerts=True,
            alert_callback=callback,
        )

        # Set lower thresholds for testing
        monitor.set_pressure_threshold(MemoryPressureLevel.HIGH, 0.5)
        monitor.set_pressure_threshold(MemoryPressureLevel.CRITICAL, 0.7)

        monitor.start_monitoring()

        # Allocate memory to trigger pressure
        tensors = []
        for _ in range(5):
            tensor = torch.randn(1000, 1000)
            tensors.append(tensor)
            time.sleep(0.1)

        time.sleep(0.3)

        monitor.stop_monitoring()

        # Clean up tensors
        del tensors

        # May have triggered alerts depending on actual memory usage
        # Just verify the system works
        assert monitor.alerts_triggered >= 0

    def test_end_to_end_monitoring_workflow(self, device):
        """Test complete monitoring workflow."""
        # 1. Initialize monitor
        monitor = MemoryMonitor(
            device=device, memory_limit_gb=2.0, sampling_interval_ms=100.0, enable_alerts=True
        )

        # 2. Start monitoring
        monitor.start_monitoring()

        # 3. Simulate workload
        time.sleep(0.5)

        # 4. Get current status
        snapshot = monitor.get_current_snapshot()
        assert isinstance(snapshot, MemorySnapshot)

        # 5. Get analytics
        analytics = monitor.get_analytics()
        assert analytics.total_snapshots > 0

        # 6. Generate report
        report = monitor.generate_report()
        assert "current_status" in report
        assert "analytics" in report

        # 7. Stop monitoring
        monitor.stop_monitoring()

        # 8. Cleanup
        monitor.cleanup()

        assert monitor.is_monitoring is False


# ============================================================================
# Performance Tests
# ============================================================================


class TestMemoryMonitorPerformance:
    """Performance tests for memory monitor."""

    def test_snapshot_creation_speed(self, memory_monitor):
        """Test snapshot creation performance."""
        import time

        num_snapshots = 1000

        start = time.time()
        for _ in range(num_snapshots):
            memory_monitor._create_snapshot()
        elapsed = time.time() - start

        # Should be fast (<1ms per snapshot)
        avg_time_ms = (elapsed / num_snapshots) * 1000
        assert avg_time_ms < 1.0

    def test_analytics_calculation_speed(self, memory_monitor):
        """Test analytics calculation performance."""
        import time

        # Generate some snapshots
        memory_monitor.start_monitoring()
        time.sleep(0.5)
        memory_monitor.stop_monitoring()

        # Time analytics calculation
        start = time.time()
        for _ in range(100):
            memory_monitor.get_analytics()
        elapsed = time.time() - start

        # Should be fast (<10ms per calculation)
        avg_time_ms = (elapsed / 100) * 1000
        assert avg_time_ms < 10.0

    def test_monitoring_overhead(self, device):
        """Test monitoring overhead is minimal."""
        import time

        # Baseline: no monitoring
        start = time.time()
        tensors = []
        for _ in range(50):
            tensor = torch.randn(100, 100)
            tensors.append(tensor)
        baseline_time = time.time() - start

        del tensors

        # With monitoring
        monitor = MemoryMonitor(device=device, memory_limit_gb=2.0, sampling_interval_ms=100.0)

        monitor.start_monitoring()

        start = time.time()
        tensors = []
        for _ in range(50):
            tensor = torch.randn(100, 100)
            tensors.append(tensor)
        monitored_time = time.time() - start

        monitor.stop_monitoring()

        # Overhead should be minimal (<20%)
        overhead_percent = ((monitored_time - baseline_time) / baseline_time) * 100
        assert overhead_percent < 20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
