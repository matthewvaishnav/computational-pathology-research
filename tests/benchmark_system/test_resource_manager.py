"""
Unit tests for Resource Manager component.

Tests GPU detection, exclusive execution enforcement, GPU memory cleanup,
memory warning threshold, temperature throttling, and OOM error handling.

Requirements: 3.1, 3.2, 3.3, 3.5, 3.8
"""

import subprocess
import time
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch

from experiments.benchmark_system.resource_manager import (
    ResourceManager,
    GPUInfo,
    GPUAllocation,
    ResourceMetrics,
    ResourceLimits,
)


class TestResourceManager:
    """Test suite for ResourceManager."""

    @pytest.fixture
    def manager(self):
        """Create ResourceManager instance for testing."""
        return ResourceManager(
            target_gpu_name="RTX 4070",
            memory_warning_threshold=0.9,
            temperature_threshold=85.0,
        )

    def test_init(self):
        """Test ResourceManager initialization."""
        manager = ResourceManager(
            target_gpu_name="RTX 4070",
            memory_warning_threshold=0.85,
            temperature_threshold=80.0,
        )

        assert manager.target_gpu_name == "RTX 4070"
        assert manager.memory_warning_threshold == 0.85
        assert manager.temperature_threshold == 80.0
        assert manager.current_allocation is None

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.get_device_properties")
    @patch.object(ResourceManager, "_query_nvidia_smi")
    def test_verify_gpu_availability_success(
        self,
        mock_nvidia_smi,
        mock_device_props,
        mock_device_name,
        mock_cuda_available,
        manager,
    ):
        """
        Test successful GPU detection.

        Requirement 3.1: GPU detection and verification
        """
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        mock_device_name.return_value = "NVIDIA GeForce RTX 4070"

        # Mock device properties
        mock_props = Mock()
        mock_props.total_memory = 12 * 1024**3  # 12GB in bytes
        mock_device_props.return_value = mock_props

        # Mock nvidia-smi output
        mock_nvidia_smi.return_value = {
            "temperature": 45.0,
            "utilization": 10.0,
            "memory_free_mb": 11000.0,
            "driver_version": "535.104.05",
        }

        gpu_info = manager.verify_gpu_availability()

        assert gpu_info.available is True
        assert "RTX 4070" in gpu_info.name
        assert gpu_info.memory_total_mb > 0
        assert gpu_info.temperature == 45.0
        assert gpu_info.utilization == 10.0
        assert gpu_info.driver_version == "535.104.05"
        assert gpu_info.error_message is None

    @patch("torch.cuda.is_available")
    def test_verify_gpu_availability_cuda_not_available(
        self,
        mock_cuda_available,
        manager,
    ):
        """
        Test GPU detection when CUDA is not available.

        Requirement 3.1: GPU detection and verification
        """
        mock_cuda_available.return_value = False

        gpu_info = manager.verify_gpu_availability()

        assert gpu_info.available is False
        assert gpu_info.error_message == "CUDA is not available"
        assert gpu_info.name == "Unknown"

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.get_device_properties")
    @patch.object(ResourceManager, "_query_nvidia_smi")
    def test_verify_gpu_availability_wrong_gpu(
        self,
        mock_nvidia_smi,
        mock_device_props,
        mock_device_name,
        mock_cuda_available,
        manager,
    ):
        """
        Test GPU detection with wrong GPU model.

        Requirement 3.1: GPU detection and verification
        """
        mock_cuda_available.return_value = True
        mock_device_name.return_value = "NVIDIA GeForce RTX 3080"

        mock_props = Mock()
        mock_props.total_memory = 10 * 1024**3
        mock_device_props.return_value = mock_props

        mock_nvidia_smi.return_value = {
            "temperature": 50.0,
            "utilization": 5.0,
            "memory_free_mb": 9000.0,
            "driver_version": "535.104.05",
        }

        gpu_info = manager.verify_gpu_availability()

        # Should still be available but with warning
        assert gpu_info.available is True
        assert "RTX 3080" in gpu_info.name
        assert gpu_info.error_message is not None
        assert "Expected RTX 4070" in gpu_info.error_message

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    def test_verify_gpu_availability_exception(
        self,
        mock_device_name,
        mock_cuda_available,
        manager,
    ):
        """Test GPU detection with exception during query."""
        mock_cuda_available.return_value = True
        mock_device_name.side_effect = RuntimeError("GPU query failed")

        gpu_info = manager.verify_gpu_availability()

        assert gpu_info.available is False
        assert "GPU query failed" in gpu_info.error_message

    def test_allocate_gpu_success(self, manager):
        """
        Test successful GPU allocation.

        Requirement 3.2: Exclusive GPU access enforcement
        """
        allocation = manager.allocate_gpu("HistoCore")

        assert allocation.framework_name == "HistoCore"
        assert allocation.gpu_id == 0
        assert allocation.allocated_at > 0
        assert manager.current_allocation == allocation

    def test_allocate_gpu_already_allocated(self, manager):
        """
        Test GPU allocation when already allocated (exclusive execution).

        Requirement 3.2: Exclusive GPU access enforcement
        """
        # First allocation succeeds
        manager.allocate_gpu("HistoCore")

        # Second allocation should fail
        with pytest.raises(RuntimeError) as exc_info:
            manager.allocate_gpu("PathML")

        assert "already allocated" in str(exc_info.value).lower()
        assert "HistoCore" in str(exc_info.value)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.synchronize")
    def test_clear_gpu_memory(
        self,
        mock_synchronize,
        mock_empty_cache,
        mock_cuda_available,
        manager,
    ):
        """
        Test GPU memory cleanup.

        Requirement 3.3: GPU memory cleanup between runs
        """
        mock_cuda_available.return_value = True

        # Allocate GPU first
        manager.allocate_gpu("HistoCore")
        assert manager.current_allocation is not None

        # Clear GPU memory
        manager.clear_gpu_memory()

        # Verify PyTorch cache was cleared
        mock_empty_cache.assert_called_once()
        mock_synchronize.assert_called_once()

        # Verify allocation was released
        assert manager.current_allocation is None

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.synchronize")
    def test_clear_gpu_memory_no_allocation(
        self,
        mock_synchronize,
        mock_empty_cache,
        mock_cuda_available,
        manager,
    ):
        """Test GPU memory cleanup when no allocation exists."""
        mock_cuda_available.return_value = True

        # Clear without prior allocation
        manager.clear_gpu_memory()

        # Should still clear cache
        mock_empty_cache.assert_called_once()
        mock_synchronize.assert_called_once()

    @patch("torch.cuda.is_available")
    def test_clear_gpu_memory_cuda_not_available(
        self,
        mock_cuda_available,
        manager,
    ):
        """Test GPU memory cleanup when CUDA is not available."""
        mock_cuda_available.return_value = False

        # Should not raise exception
        manager.clear_gpu_memory()

        assert manager.current_allocation is None

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    @patch.object(ResourceManager, "_query_nvidia_smi")
    def test_monitor_resources(
        self,
        mock_nvidia_smi,
        mock_device_props,
        mock_memory_reserved,
        mock_memory_allocated,
        mock_cuda_available,
        manager,
    ):
        """
        Test resource monitoring.

        Requirement 3.4: Resource monitoring (GPU memory, temperature, utilization)
        """
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 2 * 1024**3  # 2GB in bytes
        mock_memory_reserved.return_value = 3 * 1024**3  # 3GB in bytes

        mock_props = Mock()
        mock_props.total_memory = 12 * 1024**3  # 12GB in bytes
        mock_device_props.return_value = mock_props

        mock_nvidia_smi.return_value = {
            "temperature": 65.0,
            "utilization": 85.0,
        }

        metrics = manager.monitor_resources()

        assert metrics.gpu_memory_used_mb > 0
        assert metrics.gpu_memory_total_mb > 0
        assert metrics.gpu_memory_percent > 0
        assert metrics.gpu_temperature == 65.0
        assert metrics.gpu_utilization == 85.0
        assert metrics.timestamp > 0

    @patch("torch.cuda.is_available")
    def test_monitor_resources_cuda_not_available(
        self,
        mock_cuda_available,
        manager,
    ):
        """Test resource monitoring when CUDA is not available."""
        mock_cuda_available.return_value = False

        metrics = manager.monitor_resources()

        assert metrics.gpu_memory_used_mb == 0.0
        assert metrics.gpu_memory_total_mb == 0.0
        assert metrics.gpu_memory_percent == 0.0
        assert metrics.gpu_temperature == 0.0
        assert metrics.gpu_utilization == 0.0

    @patch.object(ResourceManager, "monitor_resources")
    def test_enforce_limits_memory_warning(
        self,
        mock_monitor,
        manager,
        caplog,
    ):
        """
        Test memory warning at 90% threshold.

        Requirement 3.5: Memory limit enforcement with warnings
        """
        # Mock high memory usage (95%)
        mock_monitor.return_value = ResourceMetrics(
            gpu_memory_used_mb=11400.0,
            gpu_memory_total_mb=12000.0,
            gpu_memory_percent=95.0,
            gpu_temperature=70.0,
            gpu_utilization=80.0,
            timestamp=time.time(),
        )

        limits = ResourceLimits(
            memory_warning_threshold=0.9,
            max_temperature=85.0,
        )

        with caplog.at_level("WARNING"):
            manager.enforce_limits(limits)

        # Should log memory warning
        assert any("memory usage" in record.message.lower() for record in caplog.records)
        assert any("95.0%" in record.message for record in caplog.records)

    @patch.object(ResourceManager, "monitor_resources")
    def test_enforce_limits_no_memory_warning(
        self,
        mock_monitor,
        manager,
        caplog,
    ):
        """Test no memory warning when below threshold."""
        # Mock normal memory usage (50%)
        mock_monitor.return_value = ResourceMetrics(
            gpu_memory_used_mb=6000.0,
            gpu_memory_total_mb=12000.0,
            gpu_memory_percent=50.0,
            gpu_temperature=60.0,
            gpu_utilization=70.0,
            timestamp=time.time(),
        )

        limits = ResourceLimits(
            memory_warning_threshold=0.9,
            max_temperature=85.0,
        )

        with caplog.at_level("WARNING"):
            manager.enforce_limits(limits)

        # Should not log memory warning
        memory_warnings = [r for r in caplog.records if "memory usage" in r.message.lower()]
        assert len(memory_warnings) == 0

    @patch.object(ResourceManager, "monitor_resources")
    def test_enforce_limits_temperature_throttling(
        self,
        mock_monitor,
        manager,
        caplog,
    ):
        """
        Test temperature throttling at 85°C.

        Requirement 3.6: Temperature monitoring and throttling
        """
        # Mock high temperature (87°C)
        mock_monitor.return_value = ResourceMetrics(
            gpu_memory_used_mb=8000.0,
            gpu_memory_total_mb=12000.0,
            gpu_memory_percent=66.7,
            gpu_temperature=87.0,
            gpu_utilization=90.0,
            timestamp=time.time(),
        )

        limits = ResourceLimits(
            memory_warning_threshold=0.9,
            max_temperature=85.0,
        )

        with caplog.at_level("WARNING"):
            manager.enforce_limits(limits)

        # Should log temperature warning
        assert any("temperature" in record.message.lower() for record in caplog.records)
        assert any("87" in record.message for record in caplog.records)
        assert any("throttling" in record.message.lower() for record in caplog.records)

    @patch.object(ResourceManager, "monitor_resources")
    def test_enforce_limits_no_temperature_warning(
        self,
        mock_monitor,
        manager,
        caplog,
    ):
        """Test no temperature warning when below threshold."""
        # Mock normal temperature (70°C)
        mock_monitor.return_value = ResourceMetrics(
            gpu_memory_used_mb=8000.0,
            gpu_memory_total_mb=12000.0,
            gpu_memory_percent=66.7,
            gpu_temperature=70.0,
            gpu_utilization=80.0,
            timestamp=time.time(),
        )

        limits = ResourceLimits(
            memory_warning_threshold=0.9,
            max_temperature=85.0,
        )

        with caplog.at_level("WARNING"):
            manager.enforce_limits(limits)

        # Should not log temperature warning
        temp_warnings = [r for r in caplog.records if "temperature" in r.message.lower()]
        assert len(temp_warnings) == 0

    @patch.object(ResourceManager, "monitor_resources")
    def test_enforce_limits_both_warnings(
        self,
        mock_monitor,
        manager,
        caplog,
    ):
        """Test both memory and temperature warnings triggered."""
        # Mock high memory and temperature
        mock_monitor.return_value = ResourceMetrics(
            gpu_memory_used_mb=11000.0,
            gpu_memory_total_mb=12000.0,
            gpu_memory_percent=91.7,
            gpu_temperature=88.0,
            gpu_utilization=95.0,
            timestamp=time.time(),
        )

        limits = ResourceLimits(
            memory_warning_threshold=0.9,
            max_temperature=85.0,
        )

        with caplog.at_level("WARNING"):
            manager.enforce_limits(limits)

        # Should log both warnings
        assert any("memory usage" in record.message.lower() for record in caplog.records)
        assert any("temperature" in record.message.lower() for record in caplog.records)

    @patch("subprocess.run")
    def test_query_nvidia_smi_success(self, mock_run, manager):
        """Test successful nvidia-smi query."""
        # Mock nvidia-smi output
        mock_result = Mock()
        mock_result.stdout = "65, 85, 10240, 535.104.05"
        mock_run.return_value = mock_result

        info = manager._query_nvidia_smi()

        assert info["temperature"] == 65.0
        assert info["utilization"] == 85.0
        assert info["memory_free_mb"] == 10240.0
        assert info["driver_version"] == "535.104.05"

    @patch("subprocess.run")
    def test_query_nvidia_smi_failure(self, mock_run, manager):
        """Test nvidia-smi query failure."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "nvidia-smi")

        info = manager._query_nvidia_smi()

        assert info == {}

    @patch("subprocess.run")
    def test_query_nvidia_smi_timeout(self, mock_run, manager):
        """Test nvidia-smi query timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 5)

        info = manager._query_nvidia_smi()

        assert info == {}

    @patch("subprocess.run")
    def test_query_nvidia_smi_not_found(self, mock_run, manager):
        """Test nvidia-smi not found."""
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

        info = manager._query_nvidia_smi()

        assert info == {}

    @patch("subprocess.run")
    def test_query_nvidia_smi_invalid_output(self, mock_run, manager):
        """Test nvidia-smi with invalid output format."""
        mock_result = Mock()
        mock_result.stdout = "invalid output"
        mock_run.return_value = mock_result

        info = manager._query_nvidia_smi()

        assert info == {}

    def test_resource_limits_defaults(self):
        """Test ResourceLimits default values."""
        limits = ResourceLimits()

        assert limits.max_memory_mb is None
        assert limits.max_temperature == 85.0
        assert limits.memory_warning_threshold == 0.9

    def test_gpu_info_dataclass(self):
        """Test GPUInfo dataclass."""
        info = GPUInfo(
            name="RTX 4070",
            memory_total_mb=12000.0,
            memory_free_mb=11000.0,
            temperature=65.0,
            utilization=50.0,
            driver_version="535.104.05",
            cuda_version="12.1",
            available=True,
        )

        assert info.name == "RTX 4070"
        assert info.available is True
        assert info.error_message is None

    def test_gpu_allocation_dataclass(self):
        """Test GPUAllocation dataclass."""
        allocation = GPUAllocation(
            framework_name="HistoCore",
            gpu_id=0,
            allocated_at=time.time(),
        )

        assert allocation.framework_name == "HistoCore"
        assert allocation.gpu_id == 0
        assert allocation.memory_limit_mb is None

    def test_resource_metrics_dataclass(self):
        """Test ResourceMetrics dataclass."""
        metrics = ResourceMetrics(
            gpu_memory_used_mb=8000.0,
            gpu_memory_total_mb=12000.0,
            gpu_memory_percent=66.7,
            gpu_temperature=70.0,
            gpu_utilization=80.0,
            timestamp=time.time(),
        )

        assert metrics.gpu_memory_used_mb == 8000.0
        assert metrics.gpu_memory_percent == 66.7


class TestResourceManagerIntegration:
    """Integration tests for ResourceManager (require actual GPU)."""

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_verify_gpu_availability_real(self):
        """Test GPU detection with real GPU."""
        manager = ResourceManager()

        gpu_info = manager.verify_gpu_availability()

        assert gpu_info.available is True
        assert gpu_info.memory_total_mb > 0
        assert len(gpu_info.name) > 0

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_monitor_resources_real(self):
        """Test resource monitoring with real GPU."""
        manager = ResourceManager()

        metrics = manager.monitor_resources()

        assert metrics.gpu_memory_total_mb > 0
        assert metrics.gpu_memory_used_mb >= 0
        assert 0 <= metrics.gpu_memory_percent <= 100
        assert metrics.timestamp > 0

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_clear_gpu_memory_real(self):
        """Test GPU memory cleanup with real GPU."""
        manager = ResourceManager()

        # Allocate some GPU memory
        tensor = torch.randn(1000, 1000, device="cuda")

        # Clear GPU memory
        manager.clear_gpu_memory()

        # Memory should be cleared (though not necessarily zero due to PyTorch caching)
        metrics = manager.monitor_resources()
        assert metrics.gpu_memory_total_mb > 0
