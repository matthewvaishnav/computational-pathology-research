"""Tests for MemoryCoordinator class."""

import time

import pytest
import torch

from src.streaming.memory.config import OptimizerConfig
from src.streaming.memory.coordinator import MemoryCoordinator


@pytest.fixture
def config():
    """Provide test configuration."""
    return OptimizerConfig(
        device=torch.device("cpu"),
        memory_limit_gb=8.0,
        cache_size_mb=500.0,
        enable_monitoring=True,
        enable_profiling=True,
        sampling_interval_ms=50.0,
    )


@pytest.fixture
def coordinator(config):
    """Provide MemoryCoordinator instance."""
    coord = MemoryCoordinator(config)
    yield coord
    coord.cleanup()


def test_coordinator_initialization(config):
    """Test coordinator initialization."""
    coordinator = MemoryCoordinator(config)

    assert coordinator.config == config
    assert coordinator.profiler is not None
    assert coordinator.cache is not None
    assert coordinator.batch_optimizer is not None
    assert coordinator.monitor is not None

    coordinator.cleanup()


def test_coordinator_default_config():
    """Test coordinator with default config."""
    coordinator = MemoryCoordinator()

    assert coordinator.config is not None
    assert coordinator.profiler is not None
    assert coordinator.cache is not None

    coordinator.cleanup()


def test_get_current_snapshot(coordinator):
    """Test getting current memory snapshot."""
    snapshot = coordinator.get_current_snapshot()

    assert snapshot is not None
    assert snapshot.allocated_gb >= 0
    assert snapshot.total_gb > 0


def test_get_pressure_level(coordinator):
    """Test getting memory pressure level."""
    level = coordinator.get_pressure_level()

    assert level is not None


def test_cache_operations(coordinator):
    """Test cache get/put operations."""
    # Put item
    coordinator.cache_put("test_key", "test_value", size_mb=0.1)

    # Get item
    value = coordinator.cache_get("test_key")
    assert value == "test_value"

    # Get non-existent item
    value = coordinator.cache_get("nonexistent")
    assert value is None


def test_cache_clear(coordinator):
    """Test cache clearing."""
    coordinator.cache_put("key1", "value1")
    coordinator.cache_put("key2", "value2")

    coordinator.cache_clear()

    assert coordinator.cache_get("key1") is None
    assert coordinator.cache_get("key2") is None


def test_get_cache_stats(coordinator):
    """Test getting cache statistics."""
    stats = coordinator.get_cache_stats()

    assert "size_mb" in stats
    assert "max_size_mb" in stats
    assert "num_entries" in stats
    assert "hit_rate" in stats


def test_optimize_batch_size(coordinator):
    """Test batch size optimization."""
    batch_size = coordinator.optimize_batch_size(
        available_memory_gb=4.0, tile_size=224, feature_dim=512
    )

    assert batch_size > 0
    assert batch_size <= 64


def test_optimize_batch_size_auto_memory(coordinator):
    """Test batch size optimization with automatic memory detection."""
    batch_size = coordinator.optimize_batch_size(tile_size=224, feature_dim=512)

    assert batch_size > 0


def test_optimize_for_workload(coordinator):
    """Test workload optimization."""
    optimal = coordinator.optimize_for_workload(workload_size=1000, tile_size=224, feature_dim=512)

    assert optimal.batch_size > 0
    assert optimal.tile_size > 0
    assert optimal.num_batches > 0


def test_get_recent_alerts(coordinator):
    """Test getting recent alerts."""
    # Let monitoring run briefly
    time.sleep(0.15)

    alerts = coordinator.get_recent_alerts(count=5)

    assert isinstance(alerts, list)


def test_get_analytics(coordinator):
    """Test getting analytics."""
    time.sleep(0.15)

    analytics = coordinator.get_analytics()

    assert analytics is not None
    assert analytics.total_snapshots >= 0


def test_record_oom_event(coordinator):
    """Test recording OOM event."""
    coordinator.record_oom_event()

    analytics = coordinator.get_analytics()
    assert analytics.oom_events == 1


def test_get_status(coordinator):
    """Test getting comprehensive status."""
    time.sleep(0.1)

    status = coordinator.get_status()

    assert "config" in status
    assert "profiling_enabled" in status
    assert "monitoring_enabled" in status
    assert "current_memory" in status
    assert "cache" in status
    assert "analytics" in status


def test_generate_report(coordinator):
    """Test generating report."""
    time.sleep(0.1)

    report = coordinator.generate_report()

    assert "config" in report
    assert "recommendations" in report
    assert isinstance(report["recommendations"], list)


def test_context_manager():
    """Test context manager usage."""
    config = OptimizerConfig(memory_limit_gb=8.0, sampling_interval_ms=50.0)

    with MemoryCoordinator(config) as coordinator:
        assert coordinator.monitor.is_monitoring
        snapshot = coordinator.get_current_snapshot()
        assert snapshot is not None

    # Should be cleaned up after exiting context
    assert not coordinator.monitor.is_monitoring


def test_coordinator_without_monitoring():
    """Test coordinator with monitoring disabled."""
    config = OptimizerConfig(enable_monitoring=False, enable_profiling=False)

    coordinator = MemoryCoordinator(config)

    assert coordinator.monitor is None
    assert coordinator.profiler is None

    # These should return None/empty
    assert coordinator.get_current_snapshot() is None
    assert coordinator.get_pressure_level() is None
    assert coordinator.get_recent_alerts() == []
    assert coordinator.get_analytics() is None

    coordinator.cleanup()


def test_cleanup(coordinator):
    """Test cleanup."""
    coordinator.cache_put("test", "value")

    coordinator.cleanup()

    # Cache should be cleared
    assert coordinator.cache_get("test") is None

    # Monitor should be stopped
    assert not coordinator.monitor.is_monitoring
