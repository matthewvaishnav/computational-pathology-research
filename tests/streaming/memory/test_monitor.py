"""Tests for MemoryMonitor class.

Tests cover:
- Monitor initialization and configuration
- Snapshot creation and retrieval
- Alert generation and thresholds
- Analytics calculation
- Thread safety
- Context manager usage
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.streaming.memory.monitor import MemoryAlert, MemoryAnalytics, MemoryMonitor
from src.streaming.memory.profiler import MemoryPressureLevel, MemorySnapshot


@pytest.fixture
def device():
    """Provide test device."""
    return torch.device("cpu")


@pytest.fixture
def monitor(device):
    """Provide MemoryMonitor instance."""
    return MemoryMonitor(
        device=device, memory_limit_gb=8.0, sampling_interval_ms=50.0, enable_alerts=True
    )


def test_monitor_initialization(device):
    """Test monitor initialization."""
    monitor = MemoryMonitor(device=device, memory_limit_gb=8.0)

    assert monitor.device == device
    assert monitor.memory_limit_gb == 8.0
    assert monitor.sampling_interval_ms == 100.0
    assert monitor.enable_alerts is True
    assert not monitor.is_monitoring
    assert len(monitor.snapshots) == 0
    assert len(monitor.alerts) == 0


def test_get_current_snapshot(monitor):
    """Test snapshot creation."""
    snapshot = monitor.get_current_snapshot()

    assert isinstance(snapshot, MemorySnapshot)
    assert snapshot.timestamp > 0
    assert snapshot.allocated_gb >= 0
    assert snapshot.reserved_gb >= 0
    assert snapshot.total_gb > 0
    assert isinstance(snapshot.pressure_level, MemoryPressureLevel)


def test_pressure_level_calculation(monitor):
    """Test pressure level calculation."""
    # Normal pressure (<60%)
    level = monitor._calculate_pressure_level(4.0)  # 50% of 8GB
    assert level == MemoryPressureLevel.NORMAL

    # Moderate pressure (>=60%, <75%)
    level = monitor._calculate_pressure_level(4.81)  # 60.125% of 8GB
    assert level == MemoryPressureLevel.MODERATE

    # High pressure (>=75%, <90%)
    level = monitor._calculate_pressure_level(6.1)  # 76.25% of 8GB
    assert level == MemoryPressureLevel.HIGH

    # Critical pressure (>=95%)
    level = monitor._calculate_pressure_level(7.7)  # 96.25% of 8GB
    assert level == MemoryPressureLevel.CRITICAL


def test_start_stop_monitoring(monitor):
    """Test starting and stopping monitoring."""
    assert not monitor.is_monitoring

    monitor.start_monitoring()
    assert monitor.is_monitoring
    assert monitor.monitoring_thread is not None
    assert monitor.start_time is not None

    # Let it collect some snapshots
    time.sleep(0.2)

    monitor.stop_monitoring()
    assert not monitor.is_monitoring


def test_context_manager(device):
    """Test context manager usage."""
    with MemoryMonitor(device=device, memory_limit_gb=8.0, sampling_interval_ms=50.0) as monitor:
        assert monitor.is_monitoring
        time.sleep(0.15)  # Let it collect snapshots

    # Should be stopped after exiting context
    assert not monitor.is_monitoring


def test_get_recent_snapshots(monitor):
    """Test retrieving recent snapshots."""
    monitor.start_monitoring()
    time.sleep(0.2)  # Collect snapshots
    monitor.stop_monitoring()

    snapshots = monitor.get_recent_snapshots(count=5)
    assert len(snapshots) > 0
    assert all(isinstance(s, MemorySnapshot) for s in snapshots)


def test_alert_generation_critical(monitor):
    """Test critical pressure alert generation."""
    # Mock high memory usage
    with patch.object(monitor, "_get_current_memory_usage", return_value=(7.8, 8.0)):
        snapshot = monitor._create_snapshot()
        monitor._check_and_generate_alerts(snapshot)

    alerts = monitor.get_recent_alerts()
    assert len(alerts) > 0
    assert any(a.severity == "critical" for a in alerts)
    assert any(a.alert_type == "pressure" for a in alerts)


def test_alert_generation_threshold(monitor):
    """Test threshold alert generation."""
    # Mock memory approaching limit
    with patch.object(monitor, "_get_current_memory_usage", return_value=(7.7, 8.0)):
        snapshot = monitor._create_snapshot()
        monitor._check_and_generate_alerts(snapshot)

    alerts = monitor.get_recent_alerts()
    assert len(alerts) > 0
    assert any(a.alert_type == "threshold" for a in alerts)


def test_alert_callback(device):
    """Test alert callback invocation."""
    callback = MagicMock()
    monitor = MemoryMonitor(device=device, memory_limit_gb=8.0, alert_callback=callback)

    # Trigger alert
    with patch.object(monitor, "_get_current_memory_usage", return_value=(7.8, 8.0)):
        snapshot = monitor._create_snapshot()
        monitor._check_and_generate_alerts(snapshot)

    # Callback should be called
    callback.assert_called()
    assert isinstance(callback.call_args[0][0], MemoryAlert)


def test_record_oom_event(monitor):
    """Test OOM event recording."""
    assert monitor.oom_events == 0

    monitor.record_oom_event()

    assert monitor.oom_events == 1
    alerts = monitor.get_recent_alerts()
    assert len(alerts) > 0
    assert any(a.alert_type == "oom_risk" for a in alerts)


def test_set_pressure_threshold(monitor):
    """Test setting custom pressure thresholds."""
    monitor.set_pressure_threshold(MemoryPressureLevel.HIGH, 0.85)

    assert monitor.pressure_thresholds[MemoryPressureLevel.HIGH] == 0.85

    # Test invalid threshold
    with pytest.raises(ValueError):
        monitor.set_pressure_threshold(MemoryPressureLevel.HIGH, 1.5)


def test_get_analytics_empty(monitor):
    """Test analytics with no snapshots."""
    analytics = monitor.get_analytics()

    assert isinstance(analytics, MemoryAnalytics)
    assert analytics.total_snapshots == 0
    assert analytics.peak_usage_gb == 0.0
    assert analytics.avg_usage_gb == 0.0


def test_get_analytics_with_data(monitor):
    """Test analytics calculation with data."""
    monitor.start_monitoring()
    time.sleep(0.2)  # Collect snapshots
    monitor.stop_monitoring()

    analytics = monitor.get_analytics()

    assert analytics.total_snapshots > 0
    assert analytics.monitoring_duration_seconds > 0
    assert analytics.peak_usage_gb >= 0
    assert analytics.avg_usage_gb >= 0
    assert analytics.min_usage_gb >= 0
    assert isinstance(analytics.pressure_distribution, dict)


def test_generate_report(monitor):
    """Test report generation."""
    monitor.start_monitoring()
    time.sleep(0.15)
    monitor.stop_monitoring()

    report = monitor.generate_report()

    assert "current_status" in report
    assert "analytics" in report
    assert "recent_alerts" in report
    assert "pressure_thresholds" in report
    assert "monitoring_config" in report


def test_cleanup(monitor):
    """Test cleanup."""
    monitor.start_monitoring()
    time.sleep(0.1)
    monitor.stop_monitoring()

    # Add some data
    assert len(monitor.snapshots) > 0

    monitor.cleanup()

    assert len(monitor.snapshots) == 0
    assert len(monitor.alerts) == 0
    assert not monitor.is_monitoring


def test_thread_safety(monitor):
    """Test thread-safe access to snapshots and alerts."""
    import threading

    def access_data():
        for _ in range(10):
            monitor.get_recent_snapshots()
            monitor.get_recent_alerts()
            time.sleep(0.01)

    monitor.start_monitoring()

    # Start multiple threads accessing data
    threads = [threading.Thread(target=access_data) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    monitor.stop_monitoring()

    # Should not crash


def test_memory_alert_to_dict():
    """Test MemoryAlert serialization."""
    alert = MemoryAlert(
        timestamp=time.time(),
        alert_type="pressure",
        severity="warning",
        message="Test alert",
        current_usage_gb=5.0,
        threshold_gb=8.0,
        recommended_action="Reduce batch size",
    )

    data = alert.to_dict()

    assert data["alert_type"] == "pressure"
    assert data["severity"] == "warning"
    assert data["message"] == "Test alert"
    assert data["current_usage_gb"] == 5.0


def test_memory_analytics_to_dict():
    """Test MemoryAnalytics serialization."""
    analytics = MemoryAnalytics(
        monitoring_duration_seconds=10.0,
        total_snapshots=100,
        peak_usage_gb=6.0,
        avg_usage_gb=4.5,
        min_usage_gb=3.0,
        pressure_distribution={"normal": 80.0, "moderate": 20.0},
        alerts_triggered=2,
        oom_events=0,
        gc_collections=5,
        memory_freed_gb=1.5,
    )

    data = analytics.to_dict()

    assert data["total_snapshots"] == 100
    assert data["peak_usage_gb"] == 6.0
    assert data["avg_usage_gb"] == 4.5
    assert data["pressure_distribution"]["normal"] == 80.0
