"""Unit tests for memory profiler.

Tests cover:
- MemorySnapshot dataclass functionality
- MemoryProfiler snapshot creation
- Peak usage tracking
- Pressure level calculation
- Historical snapshot management
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.streaming.memory.profiler import (
    MemoryPressureLevel,
    MemoryProfiler,
    MemorySnapshot,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def device():
    """Get test device (CPU for CI compatibility)."""
    return torch.device("cpu")


@pytest.fixture
def cuda_device():
    """Get CUDA device if available, skip otherwise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def profiler(device):
    """Create memory profiler."""
    return MemoryProfiler(device=device, memory_limit_gb=2.0, max_snapshots=100)


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
# MemoryProfiler Tests
# ============================================================================


class TestMemoryProfiler:
    """Test memory profiler functionality."""

    def test_initialization(self, device):
        """Test profiler initialization."""
        profiler = MemoryProfiler(device=device, memory_limit_gb=4.0, max_snapshots=50)

        assert profiler.device == device
        assert profiler.memory_limit_gb == 4.0
        assert profiler.max_snapshots == 50
        assert len(profiler.snapshots) == 0
        assert profiler.peak_allocated_gb == 0.0
        assert profiler.peak_reserved_gb == 0.0

    def test_take_snapshot_cpu(self, device):
        """Test taking snapshot on CPU device."""
        profiler = MemoryProfiler(device=device, memory_limit_gb=2.0)
        snapshot = profiler.take_snapshot()

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.allocated_gb >= 0.0
        assert snapshot.reserved_gb >= 0.0
        assert snapshot.total_gb == 2.0
        assert isinstance(snapshot.pressure_level, MemoryPressureLevel)
        assert len(profiler.snapshots) == 1

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024**3)  # 1GB
    @patch("torch.cuda.memory_reserved", return_value=2 * 1024**3)  # 2GB
    @patch("torch.cuda.get_device_properties")
    def test_take_snapshot_cuda(self, mock_props, mock_reserved, mock_allocated, mock_available):
        """Test taking snapshot on CUDA device."""
        # Mock device properties
        mock_device_props = MagicMock()
        mock_device_props.total_memory = 8 * 1024**3  # 8GB
        mock_props.return_value = mock_device_props

        device = torch.device("cuda")
        profiler = MemoryProfiler(device=device, memory_limit_gb=8.0)
        snapshot = profiler.take_snapshot()

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.allocated_gb == 1.0
        assert snapshot.reserved_gb == 2.0
        assert snapshot.total_gb == 8.0

    def test_peak_usage_tracking(self, profiler):
        """Test peak usage tracking."""
        # Mock memory usage to simulate increasing usage
        with patch.object(profiler, "_get_current_memory_usage") as mock_usage:
            # First snapshot: 1GB allocated, 1.5GB reserved
            mock_usage.return_value = (1.0, 1.5)
            profiler.take_snapshot()

            # Second snapshot: 2GB allocated, 2.5GB reserved (new peak)
            mock_usage.return_value = (2.0, 2.5)
            profiler.take_snapshot()

            # Third snapshot: 1.5GB allocated, 2GB reserved (lower than peak)
            mock_usage.return_value = (1.5, 2.0)
            profiler.take_snapshot()

            peak = profiler.get_peak_usage()
            assert peak["allocated_gb"] == 2.0
            assert peak["reserved_gb"] == 2.5

    def test_pressure_level_calculation(self, profiler):
        """Test pressure level calculation."""
        # Test NORMAL (< 60%)
        assert profiler._calculate_pressure_level(1.0) == MemoryPressureLevel.NORMAL

        # Test MODERATE (60-75%)
        assert profiler._calculate_pressure_level(1.3) == MemoryPressureLevel.MODERATE

        # Test HIGH (75-90%)
        assert profiler._calculate_pressure_level(1.6) == MemoryPressureLevel.HIGH

        # Test CRITICAL (> 90%)
        assert profiler._calculate_pressure_level(1.95) == MemoryPressureLevel.CRITICAL

    def test_pressure_level_zero_limit(self, device):
        """Test pressure level with zero memory limit."""
        profiler = MemoryProfiler(device=device, memory_limit_gb=0.0)
        level = profiler._calculate_pressure_level(1.0)
        assert level == MemoryPressureLevel.NORMAL

    def test_get_recent_snapshots(self, profiler):
        """Test getting recent snapshots."""
        # Take 5 snapshots
        with patch.object(profiler, "_get_current_memory_usage", return_value=(1.0, 1.5)):
            for _ in range(5):
                profiler.take_snapshot()

        # Get last 3 snapshots
        recent = profiler.get_recent_snapshots(count=3)
        assert len(recent) == 3
        assert all(isinstance(s, MemorySnapshot) for s in recent)

    def test_get_recent_snapshots_empty(self, profiler):
        """Test getting recent snapshots when none exist."""
        recent = profiler.get_recent_snapshots(count=5)
        assert len(recent) == 0

    def test_max_snapshots_limit(self, device):
        """Test that snapshot history respects max limit."""
        profiler = MemoryProfiler(device=device, memory_limit_gb=2.0, max_snapshots=10)

        # Take 15 snapshots (more than max)
        with patch.object(profiler, "_get_current_memory_usage", return_value=(1.0, 1.5)):
            for _ in range(15):
                profiler.take_snapshot()

        # Should only keep last 10
        assert len(profiler.snapshots) == 10

    def test_get_average_usage(self, profiler):
        """Test average usage calculation."""
        # Take snapshots with known values
        with patch.object(profiler, "_get_current_memory_usage") as mock_usage:
            mock_usage.return_value = (1.0, 1.5)
            profiler.take_snapshot()

            mock_usage.return_value = (2.0, 2.5)
            profiler.take_snapshot()

            mock_usage.return_value = (3.0, 3.5)
            profiler.take_snapshot()

            avg = profiler.get_average_usage()
            assert avg["allocated_gb"] == 2.0  # (1+2+3)/3
            assert avg["reserved_gb"] == 2.5  # (1.5+2.5+3.5)/3

    def test_get_average_usage_empty(self, profiler):
        """Test average usage with no snapshots."""
        avg = profiler.get_average_usage()
        assert avg["allocated_gb"] == 0.0
        assert avg["reserved_gb"] == 0.0

    def test_get_pressure_distribution(self, profiler):
        """Test pressure distribution calculation."""
        # Take snapshots with different pressure levels
        with patch.object(profiler, "_get_current_memory_usage") as mock_usage:
            # 2 NORMAL snapshots
            mock_usage.return_value = (0.5, 0.5)
            profiler.take_snapshot()
            profiler.take_snapshot()

            # 3 MODERATE snapshots
            mock_usage.return_value = (1.3, 1.3)
            profiler.take_snapshot()
            profiler.take_snapshot()
            profiler.take_snapshot()

            # 1 HIGH snapshot
            mock_usage.return_value = (1.6, 1.6)
            profiler.take_snapshot()

            distribution = profiler.get_pressure_distribution()

            # 2/6 = 33.33% NORMAL, 3/6 = 50% MODERATE, 1/6 = 16.67% HIGH
            assert abs(distribution["normal"] - 33.33) < 0.1
            assert abs(distribution["moderate"] - 50.0) < 0.1
            assert abs(distribution["high"] - 16.67) < 0.1

    def test_get_pressure_distribution_empty(self, profiler):
        """Test pressure distribution with no snapshots."""
        distribution = profiler.get_pressure_distribution()
        assert distribution == {}

    def test_reset(self, profiler):
        """Test profiler reset."""
        # Take some snapshots
        with patch.object(profiler, "_get_current_memory_usage", return_value=(2.0, 2.5)):
            for _ in range(5):
                profiler.take_snapshot()

        # Verify state before reset
        assert len(profiler.snapshots) > 0
        assert profiler.peak_allocated_gb > 0.0

        # Reset
        profiler.reset()

        # Verify state after reset
        assert len(profiler.snapshots) == 0
        assert profiler.peak_allocated_gb == 0.0
        assert profiler.peak_reserved_gb == 0.0

    def test_set_pressure_threshold(self, profiler):
        """Test setting custom pressure threshold."""
        profiler.set_pressure_threshold(MemoryPressureLevel.HIGH, 0.85)
        assert profiler.pressure_thresholds[MemoryPressureLevel.HIGH] == 0.85

    def test_set_pressure_threshold_invalid(self, profiler):
        """Test setting invalid pressure threshold."""
        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            profiler.set_pressure_threshold(MemoryPressureLevel.HIGH, 1.5)

        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            profiler.set_pressure_threshold(MemoryPressureLevel.HIGH, -0.1)

    def test_repr(self, profiler):
        """Test string representation."""
        repr_str = repr(profiler)
        assert "MemoryProfiler" in repr_str
        assert "device=" in repr_str
        assert "total=" in repr_str
        assert "snapshots=" in repr_str


# ============================================================================
# Integration Tests
# ============================================================================


class TestMemoryProfilerIntegration:
    """Integration tests for memory profiler."""

    def test_profiling_workflow(self, device):
        """Test complete profiling workflow."""
        # 1. Create profiler
        profiler = MemoryProfiler(device=device, memory_limit_gb=4.0)

        # 2. Take multiple snapshots
        with patch.object(profiler, "_get_current_memory_usage") as mock_usage:
            # Simulate increasing memory usage
            for i in range(10):
                mock_usage.return_value = (0.5 + i * 0.2, 1.0 + i * 0.2)
                profiler.take_snapshot()
                time.sleep(0.01)  # Small delay

        # 3. Verify snapshots were recorded
        assert len(profiler.snapshots) == 10

        # 4. Check peak usage
        peak = profiler.get_peak_usage()
        assert peak["allocated_gb"] > 0.0

        # 5. Check average usage
        avg = profiler.get_average_usage()
        assert avg["allocated_gb"] > 0.0

        # 6. Check pressure distribution
        distribution = profiler.get_pressure_distribution()
        assert len(distribution) > 0

        # 7. Get recent snapshots
        recent = profiler.get_recent_snapshots(count=5)
        assert len(recent) == 5

    def test_pressure_level_transitions(self, profiler):
        """Test pressure level transitions."""
        with patch.object(profiler, "_get_current_memory_usage") as mock_usage:
            # Start NORMAL
            mock_usage.return_value = (0.5, 0.5)
            snapshot1 = profiler.take_snapshot()
            assert snapshot1.pressure_level == MemoryPressureLevel.NORMAL

            # Transition to MODERATE
            mock_usage.return_value = (1.3, 1.3)
            snapshot2 = profiler.take_snapshot()
            assert snapshot2.pressure_level == MemoryPressureLevel.MODERATE

            # Transition to HIGH
            mock_usage.return_value = (1.6, 1.6)
            snapshot3 = profiler.take_snapshot()
            assert snapshot3.pressure_level == MemoryPressureLevel.HIGH

            # Transition to CRITICAL
            mock_usage.return_value = (1.95, 1.95)
            snapshot4 = profiler.take_snapshot()
            assert snapshot4.pressure_level == MemoryPressureLevel.CRITICAL

            # Back to NORMAL
            mock_usage.return_value = (0.5, 0.5)
            snapshot5 = profiler.take_snapshot()
            assert snapshot5.pressure_level == MemoryPressureLevel.NORMAL
