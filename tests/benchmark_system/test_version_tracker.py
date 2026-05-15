"""
Unit tests for VersionTracker.

Tests version recording, requirements.txt generation, config export/import,
and config validation.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9
"""

import json
import platform
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import yaml

from experiments.benchmark_system.models import FrameworkEnvironment
from experiments.benchmark_system.version_tracker import (
    EnvironmentInfo,
    VersionTracker,
)


class TestEnvironmentInfo:
    """Test EnvironmentInfo data class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        env_info = EnvironmentInfo(
            framework_versions={"HistoCore": "1.0.0"},
            pytorch_version="2.0.0",
            cuda_version="11.8",
            cudnn_version="8.6.0",
            gpu_model="RTX 4070",
            gpu_driver_version="535.54.03",
            gpu_memory_mb=12288.0,
            gpu_count=1,
            os_name="Linux",
            os_version="5.15.0",
            os_release="Ubuntu 22.04",
            python_version="3.10.0",
            python_implementation="CPython",
            processor="x86_64",
            machine="x86_64",
            recorded_at="2024-01-01T00:00:00",
        )

        result = env_info.to_dict()

        assert isinstance(result, dict)
        assert result["pytorch_version"] == "2.0.0"
        assert result["gpu_model"] == "RTX 4070"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "framework_versions": {"HistoCore": "1.0.0"},
            "pytorch_version": "2.0.0",
            "cuda_version": "11.8",
            "cudnn_version": "8.6.0",
            "gpu_model": "RTX 4070",
            "gpu_driver_version": "535.54.03",
            "gpu_memory_mb": 12288.0,
            "gpu_count": 1,
            "os_name": "Linux",
            "os_version": "5.15.0",
            "os_release": "Ubuntu 22.04",
            "python_version": "3.10.0",
            "python_implementation": "CPython",
            "processor": "x86_64",
            "machine": "x86_64",
            "recorded_at": "2024-01-01T00:00:00",
        }

        env_info = EnvironmentInfo.from_dict(data)

        assert env_info.pytorch_version == "2.0.0"
        assert env_info.gpu_model == "RTX 4070"


class TestVersionTracker:
    """Test VersionTracker class."""

    def test_initialization(self):
        """Test VersionTracker initialization."""
        tracker = VersionTracker()

        assert tracker.environment_info is None

    def test_record_environment_basic(self):
        """Test basic environment recording (Requirements 9.1, 9.2, 9.3, 9.4)."""
        tracker = VersionTracker()

        env_info = tracker.record_environment()

        # Verify all fields are populated
        assert env_info.pytorch_version != ""
        assert env_info.python_version != ""
        assert env_info.os_name != ""
        assert env_info.recorded_at != ""

        # Verify tracker stores the info
        assert tracker.environment_info is not None
        assert tracker.environment_info == env_info

    def test_record_environment_pytorch_version(self):
        """Test PyTorch version recording (Requirement 9.2)."""
        tracker = VersionTracker()

        env_info = tracker.record_environment()

        # Should match actual PyTorch version
        assert env_info.pytorch_version == torch.__version__

    def test_record_environment_cuda_version(self):
        """Test CUDA version recording (Requirement 9.2)."""
        tracker = VersionTracker()

        env_info = tracker.record_environment()

        # Should record CUDA version or "not_available"
        if torch.cuda.is_available():
            assert env_info.cuda_version != ""
            assert env_info.cuda_version != "unknown"
        else:
            assert env_info.cuda_version == "not_available"

    def test_record_environment_cudnn_version(self):
        """Test cuDNN version recording (Requirement 9.2)."""
        tracker = VersionTracker()

        env_info = tracker.record_environment()

        # Should record cuDNN version or "not_available"
        if torch.cuda.is_available() and torch.backends.cudnn.is_available():
            assert env_info.cudnn_version != ""
            assert env_info.cudnn_version != "unknown"
        else:
            assert env_info.cudnn_version == "not_available"

    def test_record_environment_gpu_info(self):
        """Test GPU hardware recording (Requirement 9.3)."""
        tracker = VersionTracker()

        env_info = tracker.record_environment()

        if torch.cuda.is_available():
            # Should have GPU info
            assert env_info.gpu_model != "not_available"
            assert env_info.gpu_count > 0
            assert env_info.gpu_memory_mb > 0
        else:
            # Should indicate no GPU
            assert env_info.gpu_model == "not_available"
            assert env_info.gpu_count == 0

    def test_record_environment_os_info(self):
        """Test OS version recording (Requirement 9.4)."""
        tracker = VersionTracker()

        env_info = tracker.record_environment()

        # Should match platform info
        assert env_info.os_name == platform.system()
        assert env_info.os_version == platform.version()
        assert env_info.os_release == platform.release()

    def test_record_environment_python_version(self):
        """Test Python version recording (Requirement 9.4)."""
        tracker = VersionTracker()

        env_info = tracker.record_environment()

        # Should match platform info
        assert env_info.python_version == platform.python_version()
        assert env_info.python_implementation == platform.python_implementation()

    def test_record_environment_with_frameworks(self):
        """Test framework version recording (Requirement 9.1)."""
        tracker = VersionTracker()

        # Create mock framework environments
        mock_env1 = Mock()
        mock_env1.framework_version = "1.0.0"

        mock_env2 = Mock()
        mock_env2.framework_version = "2.0.0"

        framework_envs = {
            "HistoCore": mock_env1,
            "PathML": mock_env2,
        }

        env_info = tracker.record_environment(framework_environments=framework_envs)

        # Should record framework versions
        assert "HistoCore" in env_info.framework_versions
        assert env_info.framework_versions["HistoCore"] == "1.0.0"
        assert "PathML" in env_info.framework_versions
        assert env_info.framework_versions["PathML"] == "2.0.0"

        # Should always include PyTorch
        assert "PyTorch" in env_info.framework_versions

    def test_generate_requirements_txt(self):
        """Test requirements.txt generation (Requirement 9.6)."""
        tracker = VersionTracker()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "requirements.txt"

            result_path = tracker.generate_requirements_txt(output_path)

            # Should create file
            assert result_path.exists()
            assert result_path == output_path

            # Should contain packages
            content = result_path.read_text()
            assert "# Generated by VersionTracker" in content
            assert len(content.strip().split("\n")) > 3  # Header + packages

    def test_generate_requirements_txt_filtered(self):
        """Test filtered requirements.txt generation (Requirement 9.6)."""
        tracker = VersionTracker()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "requirements.txt"

            result_path = tracker.generate_requirements_txt(output_path, include_all_packages=False)

            # Should create file with key packages only
            assert result_path.exists()

            content = result_path.read_text()

            # Should contain key packages
            # At least one of these should be present
            key_packages = ["torch", "numpy", "pillow", "pyyaml"]
            has_key_package = any(pkg in content.lower() for pkg in key_packages)
            assert has_key_package

    def test_generate_requirements_txt_all_packages(self):
        """Test full requirements.txt generation (Requirement 9.6)."""
        tracker = VersionTracker()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "requirements.txt"

            result_path = tracker.generate_requirements_txt(output_path, include_all_packages=True)

            # Should create file
            assert result_path.exists()

            content = result_path.read_text()
            lines = [line for line in content.split("\n") if line and not line.startswith("#")]

            # Should have many packages
            assert len(lines) > 10

    def test_export_config_yaml(self):
        """Test config YAML export (Requirement 9.8)."""
        tracker = VersionTracker()
        tracker.record_environment()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "config.yaml"

            result_path = tracker.export_config_yaml(output_path)

            # Should create file
            assert result_path.exists()
            assert result_path == output_path

            # Should be valid YAML
            with open(result_path, "r") as f:
                config = yaml.safe_load(f)

            assert "metadata" in config
            assert "environment" in config
            assert config["environment"]["pytorch_version"] != ""

    def test_export_config_yaml_with_additional_config(self):
        """Test config YAML export with additional config (Requirement 9.8)."""
        tracker = VersionTracker()
        tracker.record_environment()

        additional_config = {
            "benchmark_mode": "full",
            "frameworks": ["HistoCore", "PathML"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "config.yaml"

            result_path = tracker.export_config_yaml(
                output_path, additional_config=additional_config
            )

            # Should include additional config
            with open(result_path, "r") as f:
                config = yaml.safe_load(f)

            assert "additional_config" in config
            assert config["additional_config"]["benchmark_mode"] == "full"

    def test_export_config_yaml_auto_record(self):
        """Test config export auto-records environment if not done (Requirement 9.8)."""
        tracker = VersionTracker()

        # Don't call record_environment()
        assert tracker.environment_info is None

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "config.yaml"

            result_path = tracker.export_config_yaml(output_path)

            # Should auto-record and create file
            assert result_path.exists()
            assert tracker.environment_info is not None

    def test_validate_config_matching(self):
        """Test config validation with matching environment (Requirement 9.9)."""
        tracker = VersionTracker()

        # Record current environment
        env_info = tracker.record_environment()

        # Export to file
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            tracker.export_config_yaml(config_path)

            # Validate against same environment
            is_valid, warnings = tracker.validate_config(config_path)

            # Should be valid with no warnings
            assert is_valid
            assert len(warnings) == 0

    def test_validate_config_python_version_mismatch(self):
        """Test config validation with Python version mismatch (Requirement 9.9)."""
        tracker = VersionTracker()

        # Create config with different Python version
        env_info = EnvironmentInfo(
            framework_versions={},
            pytorch_version=str(torch.__version__),
            cuda_version="11.8",
            cudnn_version="8.6.0",
            gpu_model="RTX 4070",
            gpu_driver_version="535.54.03",
            gpu_memory_mb=12288.0,
            gpu_count=1,
            os_name=platform.system(),
            os_version=platform.version(),
            os_release=platform.release(),
            python_version="3.8.0",  # Different version
            python_implementation="CPython",
            processor="x86_64",
            machine="x86_64",
            recorded_at="2024-01-01T00:00:00",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Write config
            config = {
                "metadata": {"generated_by": "test"},
                "environment": env_info.to_dict(),
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Validate
            is_valid, warnings = tracker.validate_config(config_path)

            # Should have warning about Python version
            assert len(warnings) > 0
            assert any("Python version mismatch" in w for w in warnings)

            # Should be invalid if major.minor differs
            current_version = platform.python_version()
            if current_version.split(".")[:2] != ["3", "8"]:
                assert not is_valid

    def test_validate_config_pytorch_version_mismatch(self):
        """Test config validation with PyTorch version mismatch (Requirement 9.9)."""
        tracker = VersionTracker()

        # Create config with different PyTorch version
        env_info = EnvironmentInfo(
            framework_versions={},
            pytorch_version="1.0.0",  # Different version
            cuda_version="11.8",
            cudnn_version="8.6.0",
            gpu_model="RTX 4070",
            gpu_driver_version="535.54.03",
            gpu_memory_mb=12288.0,
            gpu_count=1,
            os_name=platform.system(),
            os_version=platform.version(),
            os_release=platform.release(),
            python_version=platform.python_version(),
            python_implementation="CPython",
            processor="x86_64",
            machine="x86_64",
            recorded_at="2024-01-01T00:00:00",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Write config
            config = {
                "metadata": {"generated_by": "test"},
                "environment": env_info.to_dict(),
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Validate
            is_valid, warnings = tracker.validate_config(config_path)

            # Should have warning about PyTorch version
            assert len(warnings) > 0
            assert any("PyTorch version mismatch" in w for w in warnings)

            # Should be invalid (major version differs)
            assert not is_valid

    def test_validate_config_gpu_mismatch(self):
        """Test config validation with GPU mismatch (Requirement 9.9)."""
        tracker = VersionTracker()

        # Create config with different GPU
        env_info = EnvironmentInfo(
            framework_versions={},
            pytorch_version=str(torch.__version__),
            cuda_version="11.8",
            cudnn_version="8.6.0",
            gpu_model="RTX 3090",  # Different GPU
            gpu_driver_version="535.54.03",
            gpu_memory_mb=24576.0,
            gpu_count=1,
            os_name=platform.system(),
            os_version=platform.version(),
            os_release=platform.release(),
            python_version=platform.python_version(),
            python_implementation="CPython",
            processor="x86_64",
            machine="x86_64",
            recorded_at="2024-01-01T00:00:00",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Write config
            config = {
                "metadata": {"generated_by": "test"},
                "environment": env_info.to_dict(),
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Validate
            is_valid, warnings = tracker.validate_config(config_path)

            # Should have warning about GPU
            assert len(warnings) > 0
            assert any("GPU model mismatch" in w for w in warnings)

            # GPU mismatch is critical
            assert not is_valid

    def test_validate_config_missing_file(self):
        """Test config validation with missing file (Requirement 9.9)."""
        tracker = VersionTracker()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent.yaml"

            is_valid, warnings = tracker.validate_config(config_path)

            # Should be invalid
            assert not is_valid
            assert len(warnings) > 0
            assert "not found" in warnings[0]

    def test_validate_config_strict_mode(self):
        """Test config validation in strict mode (Requirement 9.9)."""
        tracker = VersionTracker()

        # Create config with minor CUDA version difference
        env_info = EnvironmentInfo(
            framework_versions={},
            pytorch_version=torch.__version__,
            cuda_version="11.7",  # Slightly different
            cudnn_version="8.6.0",
            gpu_model=(
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not_available"
            ),
            gpu_driver_version="535.54.03",
            gpu_memory_mb=12288.0,
            gpu_count=1,
            os_name=platform.system(),
            os_version=platform.version(),
            os_release=platform.release(),
            python_version=platform.python_version(),
            python_implementation="CPython",
            processor="x86_64",
            machine="x86_64",
            recorded_at="2024-01-01T00:00:00",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Write config
            config = {
                "metadata": {"generated_by": "test"},
                "environment": env_info.to_dict(),
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Validate in strict mode
            is_valid_strict, warnings_strict = tracker.validate_config(config_path, strict=True)

            # Validate in non-strict mode
            is_valid_normal, warnings_normal = tracker.validate_config(config_path, strict=False)

            # Strict mode should be more restrictive
            if torch.cuda.is_available() and torch.version.cuda != "11.7":
                # CUDA mismatch
                assert len(warnings_strict) > 0 or len(warnings_normal) > 0

    def test_validate_config_framework_version_mismatch(self):
        """Test config validation with framework version mismatch (Requirement 9.9)."""
        tracker = VersionTracker()

        # Create config with framework versions
        env_info = EnvironmentInfo(
            framework_versions={"HistoCore": "1.0.0", "PathML": "2.0.0"},
            pytorch_version=str(torch.__version__),
            cuda_version="11.8",
            cudnn_version="8.6.0",
            gpu_model=(
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not_available"
            ),
            gpu_driver_version="535.54.03",
            gpu_memory_mb=12288.0,
            gpu_count=1,
            os_name=platform.system(),
            os_version=platform.version(),
            os_release=platform.release(),
            python_version=platform.python_version(),
            python_implementation="CPython",
            processor="x86_64",
            machine="x86_64",
            recorded_at="2024-01-01T00:00:00",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Write config
            config = {
                "metadata": {"generated_by": "test"},
                "environment": env_info.to_dict(),
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Validate (current environment won't have these frameworks)
            is_valid, warnings = tracker.validate_config(config_path)

            # Should have warnings about missing frameworks
            assert len(warnings) > 0
            assert any("not found" in w for w in warnings)
            assert not is_valid


class TestVersionTrackerIntegration:
    """Integration tests for VersionTracker."""

    def test_full_workflow(self):
        """Test complete workflow: record, export, validate."""
        tracker = VersionTracker()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Record environment
            env_info = tracker.record_environment()
            assert env_info is not None

            # Generate requirements.txt
            req_path = tmpdir / "requirements.txt"
            tracker.generate_requirements_txt(req_path)
            assert req_path.exists()

            # Export config
            config_path = tmpdir / "config.yaml"
            tracker.export_config_yaml(config_path)
            assert config_path.exists()

            # Validate config
            is_valid, warnings = tracker.validate_config(config_path)
            assert is_valid
            assert len(warnings) == 0

    def test_config_roundtrip(self):
        """Test config export and import roundtrip."""
        tracker1 = VersionTracker()
        env_info1 = tracker1.record_environment()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Export
            tracker1.export_config_yaml(config_path)

            # Load and validate
            tracker2 = VersionTracker()
            is_valid, warnings = tracker2.validate_config(config_path)

            # Should be valid (same environment)
            assert is_valid
            assert len(warnings) == 0
