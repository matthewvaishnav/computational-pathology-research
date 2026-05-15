"""
Property-Based Tests for Resource Manager.

Tests Task 13: Resource manager
- 13.1 GPU memory monitoring
- 13.2 CPU usage monitoring
- 13.3 Disk space monitoring
- 13.4 Resource limit enforcement
- 13.5 Scheduled training windows

**Validates: Requirement 16 - Resource Management**
"""

import shutil
import tempfile
import time
from datetime import datetime
from datetime import time as dt_time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import psutil
import pytest
import torch

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from src.federated.client.resource_manager import (
    ResourceLimits,
    ResourceManager,
    ResourceUsage,
    TrainingWindow,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary checkpoint directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def resource_limits():
    """Create default resource limits."""
    return ResourceLimits(
        gpu_memory_gb=8.0,
        cpu_cores=4,
        disk_space_gb=100.0,
        pause_threshold=0.9,
    )


@pytest.fixture
def resource_manager(resource_limits, temp_checkpoint_dir):
    """Create resource manager instance."""
    return ResourceManager(
        limits=resource_limits,
        checkpoint_dir=temp_checkpoint_dir,
    )


# ============================================================================
# Unit Tests - Task 13.1: GPU Memory Monitoring
# ============================================================================


class TestGPUMemoryMonitoring:
    """
    Unit tests for GPU memory monitoring.

    **Validates: Requirement 16.1, 16.4**
    """

    def test_get_gpu_memory_usage_with_gpu(self, resource_manager):
        """Test GPU memory usage retrieval with GPU available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        used_gb, total_gb, percent = resource_manager.get_gpu_memory_usage()

        # Validate return types
        assert isinstance(used_gb, float)
        assert isinstance(total_gb, float)
        assert isinstance(percent, float)

        # Validate ranges
        assert used_gb >= 0.0
        assert total_gb > 0.0
        assert 0.0 <= percent <= 100.0

        # Validate consistency
        assert used_gb <= total_gb

    def test_get_gpu_memory_usage_without_gpu(self, resource_limits, temp_checkpoint_dir):
        """Test GPU memory usage retrieval without GPU."""
        # Force GPU unavailable
        with patch("torch.cuda.is_available", return_value=False):
            manager = ResourceManager(
                limits=resource_limits,
                checkpoint_dir=temp_checkpoint_dir,
            )

            used_gb, total_gb, percent = manager.get_gpu_memory_usage()

            # Should return zeros when no GPU
            assert used_gb == 0.0
            assert total_gb == 0.0
            assert percent == 0.0

    def test_check_gpu_memory_limit_within_bounds(self, resource_manager):
        """Test GPU memory limit check when within bounds."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Set high limit
        resource_manager.limits.gpu_memory_gb = 100.0

        within_limit = resource_manager.check_gpu_memory_limit()

        # Should be within limit with high threshold
        assert within_limit is True

    def test_check_gpu_memory_limit_exceeded(self, resource_manager):
        """Test GPU memory limit check when exceeded."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Set very low limit
        resource_manager.limits.gpu_memory_gb = 0.001  # 1MB

        # Allocate some GPU memory
        tensor = torch.randn(1000, 1000, device="cuda")

        within_limit = resource_manager.check_gpu_memory_limit()

        # Should exceed limit
        assert within_limit is False

        # Cleanup
        del tensor
        torch.cuda.empty_cache()


# ============================================================================
# Unit Tests - Task 13.2: CPU Usage Monitoring
# ============================================================================


class TestCPUUsageMonitoring:
    """
    Unit tests for CPU usage monitoring.

    **Validates: Requirement 16.2, 16.4**
    """

    def test_get_cpu_usage(self, resource_manager):
        """Test CPU usage retrieval."""
        cpu_percent, cores_used = resource_manager.get_cpu_usage()

        # Validate return types
        assert isinstance(cpu_percent, float)
        assert isinstance(cores_used, float)

        # Validate ranges
        assert 0.0 <= cpu_percent <= 100.0
        assert 0.0 <= cores_used <= resource_manager.cpu_count

    def test_check_cpu_limit_within_bounds(self, resource_manager):
        """Test CPU limit check when within bounds."""
        # Set high limit
        resource_manager.limits.cpu_cores = resource_manager.cpu_count

        within_limit = resource_manager.check_cpu_limit()

        # Should be within limit
        assert within_limit is True

    def test_check_cpu_limit_exceeded(self, resource_manager):
        """Test CPU limit check when exceeded."""
        # Mock high CPU usage
        with patch.object(resource_manager, "get_cpu_usage", return_value=(100.0, 8.0)):
            # Set low limit
            resource_manager.limits.cpu_cores = 2

            within_limit = resource_manager.check_cpu_limit()

            # Should exceed limit
            assert within_limit is False


# ============================================================================
# Unit Tests - Task 13.3: Disk Space Monitoring
# ============================================================================


class TestDiskSpaceMonitoring:
    """
    Unit tests for disk space monitoring.

    **Validates: Requirement 16.3, 16.4**
    """

    def test_get_disk_usage(self, resource_manager):
        """Test disk usage retrieval."""
        used_gb, total_gb, percent = resource_manager.get_disk_usage()

        # Validate return types
        assert isinstance(used_gb, float)
        assert isinstance(total_gb, float)
        assert isinstance(percent, float)

        # Validate ranges
        assert used_gb >= 0.0
        assert total_gb > 0.0
        assert 0.0 <= percent <= 100.0

        # Validate consistency
        assert used_gb <= total_gb

    def test_check_disk_limit_within_bounds(self, resource_manager):
        """Test disk limit check when within bounds."""
        # Set low limit (should have plenty of space)
        resource_manager.limits.disk_space_gb = 1.0

        within_limit = resource_manager.check_disk_limit()

        # Should be within limit
        assert within_limit is True

    def test_check_disk_limit_exceeded(self, resource_manager):
        """Test disk limit check when exceeded."""
        # Mock low disk space
        mock_usage = Mock()
        mock_usage.used = 900 * (1024**3)  # 900GB used
        mock_usage.total = 1000 * (1024**3)  # 1000GB total

        with patch("shutil.disk_usage", return_value=mock_usage):
            # Set high limit (requires more than available)
            resource_manager.limits.disk_space_gb = 200.0

            within_limit = resource_manager.check_disk_limit()

            # Should exceed limit (only 100GB available, need 200GB)
            assert within_limit is False


# ============================================================================
# Unit Tests - Task 13.4: Resource Limit Enforcement
# ============================================================================


class TestResourceLimitEnforcement:
    """
    Unit tests for resource limit enforcement.

    **Validates: Requirement 16.5, 16.7**
    """

    def test_monitor_resources(self, resource_manager):
        """Test resource monitoring returns valid snapshot."""
        usage = resource_manager.monitor_resources()

        # Validate type
        assert isinstance(usage, ResourceUsage)

        # Validate fields
        assert usage.timestamp > 0
        assert usage.gpu_memory_used_gb >= 0.0
        assert usage.gpu_memory_total_gb >= 0.0
        assert 0.0 <= usage.gpu_memory_percent <= 100.0
        assert 0.0 <= usage.cpu_percent <= 100.0
        assert usage.cpu_cores_used >= 0.0
        assert usage.disk_used_gb >= 0.0
        assert usage.disk_total_gb > 0.0
        assert 0.0 <= usage.disk_percent <= 100.0

        # Validate history updated
        assert len(resource_manager.usage_history) == 1
        assert resource_manager.usage_history[0] == usage

    def test_check_resource_limits_all_within(self, resource_manager):
        """Test resource limit check when all within limits."""
        # Set high limits
        resource_manager.limits.gpu_memory_gb = 100.0
        resource_manager.limits.cpu_cores = 100
        resource_manager.limits.disk_space_gb = 1.0

        within_limits, reason = resource_manager.check_resource_limits()

        assert within_limits is True
        assert reason is None

    def test_check_resource_limits_gpu_exceeded(self, resource_manager):
        """Test resource limit check when GPU limit exceeded."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Set very low GPU limit
        resource_manager.limits.gpu_memory_gb = 0.001  # 1MB
        resource_manager.limits.pause_threshold = 0.5  # 50% threshold

        # Allocate GPU memory
        tensor = torch.randn(1000, 1000, device="cuda")

        within_limits, reason = resource_manager.check_resource_limits()

        assert within_limits is False
        assert reason is not None
        assert "GPU memory" in reason

        # Cleanup
        del tensor
        torch.cuda.empty_cache()

    def test_check_resource_limits_cpu_exceeded(self, resource_manager):
        """Test resource limit check when CPU limit exceeded."""
        # Mock high CPU usage
        with patch.object(resource_manager, "get_cpu_usage", return_value=(100.0, 8.0)):
            # Set low CPU limit
            resource_manager.limits.cpu_cores = 4
            resource_manager.limits.pause_threshold = 0.5  # 50% threshold

            within_limits, reason = resource_manager.check_resource_limits()

            assert within_limits is False
            assert reason is not None
            assert "CPU" in reason

    def test_check_resource_limits_disk_exceeded(self, resource_manager):
        """Test resource limit check when disk limit exceeded."""
        # Mock low disk space
        mock_usage = Mock()
        mock_usage.used = 950 * (1024**3)  # 950GB used
        mock_usage.total = 1000 * (1024**3)  # 1000GB total (50GB available)

        with patch("shutil.disk_usage", return_value=mock_usage):
            # Mock low CPU usage to avoid CPU limit triggering first
            with patch.object(resource_manager, "get_cpu_usage", return_value=(10.0, 0.4)):
                # Set high disk limit
                resource_manager.limits.disk_space_gb = 100.0  # Need 100GB
                resource_manager.limits.pause_threshold = 0.5  # 50% threshold

                within_limits, reason = resource_manager.check_resource_limits()

                assert within_limits is False
                assert reason is not None
                assert "Disk" in reason

    def test_enforce_limits_pauses_training(self, resource_manager):
        """Test enforce_limits pauses training when limits exceeded."""
        # Mock resource limit exceeded
        with patch.object(
            resource_manager,
            "check_resource_limits",
            return_value=(False, "Test limit exceeded"),
        ):
            can_continue = resource_manager.enforce_limits()

            assert can_continue is False
            assert resource_manager.is_paused is True
            assert resource_manager.pause_reason == "Test limit exceeded"

    def test_enforce_limits_resumes_training(self, resource_manager):
        """Test enforce_limits resumes training when limits OK."""
        # Pause training first
        resource_manager.pause_training("Test pause")

        # Mock resource limits OK
        with patch.object(
            resource_manager,
            "check_resource_limits",
            return_value=(True, None),
        ):
            can_continue = resource_manager.enforce_limits()

            assert can_continue is True
            assert resource_manager.is_paused is False
            assert resource_manager.pause_reason is None

    def test_pause_and_resume_training(self, resource_manager):
        """Test pause and resume training methods."""
        # Initially not paused
        assert resource_manager.is_paused is False

        # Pause training
        resource_manager.pause_training("Test reason")
        assert resource_manager.is_paused is True
        assert resource_manager.pause_reason == "Test reason"

        # Resume training
        resource_manager.resume_training()
        assert resource_manager.is_paused is False
        assert resource_manager.pause_reason is None


# ============================================================================
# Unit Tests - Task 13.5: Scheduled Training Windows
# ============================================================================


class TestScheduledTrainingWindows:
    """
    Unit tests for scheduled training windows.

    **Validates: Requirement 16.6**
    """

    def test_add_training_window(self, resource_manager):
        """Test adding training window."""
        # Add window: Monday 22:00-06:00
        resource_manager.add_training_window(
            day_of_week=0,  # Monday
            start_time=dt_time(22, 0),
            end_time=dt_time(6, 0),
            enabled=True,
        )

        assert len(resource_manager.training_windows) == 1
        window = resource_manager.training_windows[0]
        assert window.day_of_week == 0
        assert window.start_time == dt_time(22, 0)
        assert window.end_time == dt_time(6, 0)
        assert window.enabled is True

    def test_is_within_training_window_no_windows(self, resource_manager):
        """Test training allowed when no windows configured."""
        is_allowed, reason = resource_manager.is_within_training_window()

        # Should always be allowed with no windows
        assert is_allowed is True
        assert reason is None

    def test_is_within_training_window_inside(self, resource_manager):
        """Test training allowed when inside window."""
        # Get current day and time
        now = datetime.now()
        current_day = now.weekday()
        current_time = now.time()

        # Create window around current time (±1 hour)
        start_time = dt_time(
            (current_time.hour - 1) % 24,
            current_time.minute,
        )
        end_time = dt_time(
            (current_time.hour + 1) % 24,
            current_time.minute,
        )

        resource_manager.add_training_window(
            day_of_week=current_day,
            start_time=start_time,
            end_time=end_time,
            enabled=True,
        )

        is_allowed, reason = resource_manager.is_within_training_window()

        # Should be allowed (inside window)
        assert is_allowed is True
        assert reason is None

    def test_is_within_training_window_outside(self, resource_manager):
        """Test training blocked when outside window."""
        # Get current day
        now = datetime.now()
        current_day = now.weekday()

        # Create window that doesn't include current time
        # Use a narrow window in the past
        resource_manager.add_training_window(
            day_of_week=current_day,
            start_time=dt_time(0, 0),
            end_time=dt_time(0, 1),  # 1 minute window at midnight
            enabled=True,
        )

        is_allowed, reason = resource_manager.is_within_training_window()

        # Should be blocked (outside window) unless we're exactly at midnight
        if now.time() < dt_time(0, 0) or now.time() > dt_time(0, 1):
            assert is_allowed is False
            assert reason is not None
            assert "Outside training windows" in reason

    def test_is_within_training_window_crosses_midnight(self, resource_manager):
        """Test training window that crosses midnight."""
        # Get current day and time
        now = datetime.now()
        current_day = now.weekday()

        # Create window that crosses midnight (22:00 - 06:00)
        resource_manager.add_training_window(
            day_of_week=current_day,
            start_time=dt_time(22, 0),
            end_time=dt_time(6, 0),
            enabled=True,
        )

        # Test at 23:00 (should be inside)
        with patch("src.federated.client.resource_manager.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 1, 23, 0)  # Monday 23:00
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            # Need to recreate manager to use mocked datetime
            manager = ResourceManager(
                limits=resource_manager.limits,
                checkpoint_dir=resource_manager.checkpoint_dir,
            )
            manager.add_training_window(
                day_of_week=0,  # Monday
                start_time=dt_time(22, 0),
                end_time=dt_time(6, 0),
                enabled=True,
            )

            is_allowed, reason = manager.is_within_training_window()

            # Should be allowed (23:00 is after 22:00)
            assert is_allowed is True

    def test_is_within_training_window_disabled(self, resource_manager):
        """Test disabled training window is ignored."""
        # Get current day and time
        now = datetime.now()
        current_day = now.weekday()
        current_time = now.time()

        # Create disabled window around current time
        start_time = dt_time(
            (current_time.hour - 1) % 24,
            current_time.minute,
        )
        end_time = dt_time(
            (current_time.hour + 1) % 24,
            current_time.minute,
        )

        resource_manager.add_training_window(
            day_of_week=current_day,
            start_time=start_time,
            end_time=end_time,
            enabled=False,  # Disabled
        )

        is_allowed, reason = resource_manager.is_within_training_window()

        # Should be blocked (window disabled, no other windows)
        assert is_allowed is False
        assert reason is not None

    def test_check_training_allowed_combines_checks(self, resource_manager):
        """Test check_training_allowed combines resource and schedule checks."""
        # Set high limits (resources OK)
        resource_manager.limits.gpu_memory_gb = 100.0
        resource_manager.limits.cpu_cores = 100
        resource_manager.limits.disk_space_gb = 1.0

        # No training windows (schedule OK)
        is_allowed, reason = resource_manager.check_training_allowed()

        # Should be allowed
        assert is_allowed is True
        assert reason is None


# ============================================================================
# Property-Based Tests - Requirement 16.7 (Invariant Properties)
# ============================================================================


class TestResourceManagerProperties:
    """
    Property-based tests for resource manager invariants.

    **Validates: Requirement 16.7**
    """

    @given(
        gpu_memory_gb=st.floats(min_value=0.1, max_value=100.0),
        cpu_cores=st.integers(min_value=1, max_value=64),
        disk_space_gb=st.floats(min_value=1.0, max_value=1000.0),
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_resource_limits_always_positive(
        self, gpu_memory_gb, cpu_cores, disk_space_gb, temp_checkpoint_dir
    ):
        """
        Property: Resource limits are always positive.

        **Validates: Requirement 16.7**
        """
        limits = ResourceLimits(
            gpu_memory_gb=gpu_memory_gb,
            cpu_cores=cpu_cores,
            disk_space_gb=disk_space_gb,
        )

        manager = ResourceManager(
            limits=limits,
            checkpoint_dir=temp_checkpoint_dir,
        )

        # Property: All limits are positive
        assert manager.limits.gpu_memory_gb > 0
        assert manager.limits.cpu_cores > 0
        assert manager.limits.disk_space_gb > 0

    @given(
        num_monitors=st.integers(min_value=1, max_value=100),
    )
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_usage_history_bounded(self, num_monitors, resource_manager):
        """
        Property: Usage history is bounded by max_history_size.

        **Validates: Requirement 16.4**
        """
        # Set small history size for testing
        resource_manager.max_history_size = 10

        # Monitor resources multiple times
        for _ in range(num_monitors):
            resource_manager.monitor_resources()

        # Property: History never exceeds max size
        assert len(resource_manager.usage_history) <= resource_manager.max_history_size

    def test_property_gpu_memory_within_total(self, resource_manager):
        """
        Property: GPU memory used ≤ GPU memory total.

        **Validates: Requirement 16.1, 16.7**
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        used_gb, total_gb, percent = resource_manager.get_gpu_memory_usage()

        # Property: Used memory never exceeds total
        assert used_gb <= total_gb

    def test_property_cpu_cores_within_total(self, resource_manager):
        """
        Property: CPU cores used ≤ CPU cores total.

        **Validates: Requirement 16.2, 16.7**
        """
        cpu_percent, cores_used = resource_manager.get_cpu_usage()

        # Property: Cores used never exceeds total
        assert cores_used <= resource_manager.cpu_count

    def test_property_disk_used_within_total(self, resource_manager):
        """
        Property: Disk used ≤ Disk total.

        **Validates: Requirement 16.3, 16.7**
        """
        used_gb, total_gb, percent = resource_manager.get_disk_usage()

        # Property: Used disk never exceeds total
        assert used_gb <= total_gb

    def test_property_pause_threshold_enforced(self, resource_manager):
        """
        Property: Training pauses when utilization ≥ pause_threshold.

        **Validates: Requirement 16.5, 16.7**
        """
        # Mock high utilization (95%)
        with patch.object(resource_manager, "monitor_resources") as mock_monitor:
            mock_usage = ResourceUsage(
                timestamp=time.time(),
                gpu_memory_used_gb=9.5,  # 95% of 10GB
                gpu_memory_total_gb=10.0,
                gpu_memory_percent=95.0,
                cpu_percent=95.0,
                cpu_cores_used=3.8,  # 95% of 4 cores
                disk_used_gb=900.0,
                disk_total_gb=1000.0,
                disk_percent=90.0,
            )
            mock_monitor.return_value = mock_usage

            # Set limits
            resource_manager.limits.gpu_memory_gb = 10.0
            resource_manager.limits.cpu_cores = 4
            resource_manager.limits.disk_space_gb = 100.0
            resource_manager.limits.pause_threshold = 0.9  # 90%

            within_limits, reason = resource_manager.check_resource_limits()

            # Property: Should pause when utilization ≥ threshold
            assert within_limits is False
            assert reason is not None


# ============================================================================
# Integration Tests
# ============================================================================


class TestResourceManagerIntegration:
    """Integration tests for resource manager."""

    def test_full_monitoring_cycle(self, resource_manager):
        """Test complete monitoring cycle."""
        # Clear any existing history
        resource_manager.clear_usage_history()

        # Monitor resources
        usage1 = resource_manager.monitor_resources()
        time.sleep(0.1)
        usage2 = resource_manager.monitor_resources()

        # Validate history (should have 2 entries)
        assert len(resource_manager.usage_history) == 2
        assert usage2.timestamp > usage1.timestamp

        # Get summary (this will add one more entry to history)
        summary = resource_manager.get_resource_summary()
        assert "gpu" in summary
        assert "cpu" in summary
        assert "disk" in summary
        assert "is_paused" in summary

        # Get history (should now have 3 entries due to get_resource_summary)
        history = resource_manager.get_usage_history()
        assert len(history) == 3

    def test_training_workflow_with_limits(self, resource_manager):
        """Test training workflow with resource limits."""
        # Set reasonable limits
        resource_manager.limits.gpu_memory_gb = 100.0
        resource_manager.limits.cpu_cores = 100
        resource_manager.limits.disk_space_gb = 1.0

        # Check if training allowed
        is_allowed, reason = resource_manager.check_training_allowed()
        assert is_allowed is True

        # Enforce limits
        can_continue = resource_manager.enforce_limits()
        assert can_continue is True
        assert resource_manager.is_paused is False

    def test_training_workflow_with_windows(self, resource_manager):
        """Test training workflow with scheduled windows."""
        # Set high resource limits to avoid resource limit failures
        resource_manager.limits.gpu_memory_gb = 100.0
        resource_manager.limits.cpu_cores = 100
        resource_manager.limits.disk_space_gb = 1.0

        # Add window for current time
        now = datetime.now()
        current_day = now.weekday()
        current_time = now.time()

        start_time = dt_time(
            (current_time.hour - 1) % 24,
            current_time.minute,
        )
        end_time = dt_time(
            (current_time.hour + 1) % 24,
            current_time.minute,
        )

        resource_manager.add_training_window(
            day_of_week=current_day,
            start_time=start_time,
            end_time=end_time,
            enabled=True,
        )

        # Check if training allowed
        is_allowed, reason = resource_manager.check_training_allowed()
        assert is_allowed is True

        # Get summary
        summary = resource_manager.get_resource_summary()
        assert len(summary["training_windows"]) == 1


# ============================================================================
# Edge Cases
# ============================================================================


class TestResourceManagerEdgeCases:
    """Test edge cases for resource manager."""

    def test_zero_limits(self, temp_checkpoint_dir):
        """Test behavior with zero limits."""
        limits = ResourceLimits(
            gpu_memory_gb=0.001,
            cpu_cores=1,
            disk_space_gb=0.001,
        )

        manager = ResourceManager(
            limits=limits,
            checkpoint_dir=temp_checkpoint_dir,
        )

        # Should not crash
        usage = manager.monitor_resources()
        assert usage is not None

    def test_very_high_limits(self, temp_checkpoint_dir):
        """Test behavior with very high limits."""
        limits = ResourceLimits(
            gpu_memory_gb=1000.0,
            cpu_cores=1000,
            disk_space_gb=1.0,  # Use reasonable disk limit
        )

        manager = ResourceManager(
            limits=limits,
            checkpoint_dir=temp_checkpoint_dir,
        )

        # Mock low resource usage to ensure within limits
        with patch.object(manager, "get_cpu_usage", return_value=(10.0, 1.0)):
            # Should always be within limits
            within_limits, reason = manager.check_resource_limits()
            assert within_limits is True

    def test_nonexistent_checkpoint_dir(self):
        """Test with nonexistent checkpoint directory."""
        limits = ResourceLimits()
        nonexistent_dir = Path("/tmp/nonexistent_dir_12345")

        # Should create directory
        manager = ResourceManager(
            limits=limits,
            checkpoint_dir=nonexistent_dir,
        )

        assert nonexistent_dir.exists()

        # Cleanup
        shutil.rmtree(nonexistent_dir, ignore_errors=True)

    def test_multiple_training_windows_same_day(self, resource_manager):
        """Test multiple training windows on same day."""
        # Add two windows for Monday
        resource_manager.add_training_window(
            day_of_week=0,
            start_time=dt_time(9, 0),
            end_time=dt_time(12, 0),
            enabled=True,
        )
        resource_manager.add_training_window(
            day_of_week=0,
            start_time=dt_time(14, 0),
            end_time=dt_time(17, 0),
            enabled=True,
        )

        assert len(resource_manager.training_windows) == 2

        # Get summary
        summary = resource_manager.get_resource_summary()
        assert len(summary["training_windows"]) == 2

    def test_clear_usage_history(self, resource_manager):
        """Test clearing usage history."""
        # Monitor resources
        resource_manager.monitor_resources()
        resource_manager.monitor_resources()

        assert len(resource_manager.usage_history) == 2

        # Clear history
        resource_manager.clear_usage_history()

        assert len(resource_manager.usage_history) == 0
