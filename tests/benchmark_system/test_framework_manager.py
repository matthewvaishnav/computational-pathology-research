"""
Unit tests for Framework Manager component.

Tests framework installation, validation, compatibility patches, and version extraction.

Requirements: 1.1, 1.2, 1.5, 1.7
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from experiments.benchmark_system.framework_manager import FrameworkManager
from experiments.benchmark_system.models import FrameworkEnvironment


class TestFrameworkManager:
    """Test suite for FrameworkManager."""
    
    @pytest.fixture
    def temp_env_dir(self, tmp_path):
        """Create temporary directory for test environments."""
        env_dir = tmp_path / "test_envs"
        env_dir.mkdir()
        return env_dir
    
    @pytest.fixture
    def manager(self, temp_env_dir):
        """Create FrameworkManager instance for testing."""
        return FrameworkManager(base_env_dir=temp_env_dir)
    
    def test_init_creates_base_directory(self, tmp_path):
        """Test that FrameworkManager creates base environment directory."""
        env_dir = tmp_path / "new_envs"
        assert not env_dir.exists()
        
        manager = FrameworkManager(base_env_dir=env_dir)
        
        assert env_dir.exists()
        assert manager.base_env_dir == env_dir
    
    def test_detect_python_version(self, manager):
        """
        Test Python version detection.
        
        Requirement 1.1: Detect Python version
        """
        version = manager._detect_python_version()
        
        # Should return version in format "X.Y.Z"
        parts = version.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)
        
        # Should match current Python version
        expected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        assert version == expected
    
    def test_parse_python_version(self, manager):
        """Test parsing Python version strings."""
        # Standard version
        major, minor = manager._parse_python_version("3.14.0")
        assert major == 3
        assert minor == 14
        
        # Different version
        major, minor = manager._parse_python_version("3.9.7")
        assert major == 3
        assert minor == 9
        
        # Edge case: only major.minor
        major, minor = manager._parse_python_version("3.10")
        assert major == 3
        assert minor == 10
    
    def test_apply_compatibility_patches_python_314_pathml(self, manager):
        """
        Test that PathML patches are applied for Python 3.14.
        
        Requirement 1.2: Apply compatibility patches for PathML numpy/pandas Python 3.14 issues
        """
        patches = manager.apply_compatibility_patches("PathML", "3.14.0")
        
        assert len(patches) > 0
        assert any("PathML" in patch and "3.14" in patch for patch in patches)
        assert any("numpy" in patch or "pandas" in patch for patch in patches)
    
    def test_apply_compatibility_patches_python_39_pathml(self, manager):
        """Test that no patches are applied for PathML on Python 3.9."""
        patches = manager.apply_compatibility_patches("PathML", "3.9.0")
        
        assert len(patches) == 0
    
    def test_apply_compatibility_patches_python_314_other_frameworks(self, manager):
        """Test that no patches are applied for non-PathML frameworks on Python 3.14."""
        for framework in ["HistoCore", "CLAM", "PyTorch"]:
            patches = manager.apply_compatibility_patches(framework, "3.14.0")
            assert len(patches) == 0
    
    def test_get_python_executable_windows(self, manager, temp_env_dir):
        """Test Python executable path on Windows."""
        venv_path = temp_env_dir / "test_env"
        
        with patch("sys.platform", "win32"):
            exe_path = manager._get_python_executable(venv_path)
            
            assert exe_path == str(venv_path / "Scripts" / "python.exe")
    
    def test_get_python_executable_unix(self, manager, temp_env_dir):
        """Test Python executable path on Unix-like systems."""
        venv_path = temp_env_dir / "test_env"
        
        with patch("sys.platform", "linux"):
            exe_path = manager._get_python_executable(venv_path)
            
            assert exe_path == str(venv_path / "bin" / "python")
    
    @patch("venv.create")
    def test_create_venv(self, mock_venv_create, manager, temp_env_dir):
        """
        Test virtual environment creation.
        
        Requirement 1.8: Create separate virtual environments
        """
        venv_path = temp_env_dir / "new_venv"
        
        manager._create_venv(venv_path)
        
        mock_venv_create.assert_called_once_with(venv_path, with_pip=True, clear=False)
    
    @patch("venv.create")
    def test_create_venv_already_exists(self, mock_venv_create, manager, temp_env_dir):
        """Test that existing venv is not recreated."""
        venv_path = temp_env_dir / "existing_venv"
        venv_path.mkdir()
        
        manager._create_venv(venv_path)
        
        # Should not call venv.create if directory exists
        mock_venv_create.assert_not_called()
    
    @patch("subprocess.run")
    def test_get_framework_version_success(self, mock_run, manager, temp_env_dir):
        """
        Test framework version extraction.
        
        Requirement 1.6: Extract version information
        """
        venv_path = temp_env_dir / "test_env"
        
        # Mock pip show output
        mock_result = Mock()
        mock_result.stdout = "Name: pathml\nVersion: 2.1.0\nSummary: PathML library\n"
        mock_run.return_value = mock_result
        
        version = manager.get_framework_version("PathML", venv_path)
        
        assert version == "2.1.0"
        mock_run.assert_called_once()
    
    @patch("subprocess.run")
    def test_get_framework_version_failure(self, mock_run, manager, temp_env_dir):
        """Test version extraction when pip show fails."""
        venv_path = temp_env_dir / "test_env"
        
        # Mock pip show failure
        mock_run.side_effect = subprocess.CalledProcessError(1, "pip show")
        
        version = manager.get_framework_version("PathML", venv_path)
        
        assert version == "unknown"
    
    @patch("subprocess.run")
    def test_get_installed_dependencies(self, mock_run, manager, temp_env_dir):
        """Test getting list of installed packages."""
        venv_path = temp_env_dir / "test_env"
        
        # Mock pip list output
        mock_result = Mock()
        mock_result.stdout = json.dumps([
            {"name": "numpy", "version": "1.24.0"},
            {"name": "pandas", "version": "2.0.0"},
            {"name": "torch", "version": "2.0.1"},
        ])
        mock_run.return_value = mock_result
        
        deps = manager._get_installed_dependencies(venv_path)
        
        assert deps == {
            "numpy": "1.24.0",
            "pandas": "2.0.0",
            "torch": "2.0.1",
        }
    
    @patch("subprocess.run")
    def test_get_installed_dependencies_failure(self, mock_run, manager, temp_env_dir):
        """Test dependency listing when pip list fails."""
        venv_path = temp_env_dir / "test_env"
        
        mock_run.side_effect = subprocess.CalledProcessError(1, "pip list")
        
        deps = manager._get_installed_dependencies(venv_path)
        
        assert deps == {}
    
    @patch("subprocess.run")
    def test_install_framework_packages_success(self, mock_run, manager, temp_env_dir):
        """
        Test successful framework package installation.
        
        Requirements: 1.3, 1.4, 1.5
        """
        venv_path = temp_env_dir / "test_env"
        venv_path.mkdir()
        
        # Mock successful installation
        mock_result = Mock()
        mock_result.stdout = "Successfully installed pathml-2.1.0"
        mock_run.return_value = mock_result
        
        # Should not raise exception
        manager._install_framework_packages("PathML", venv_path)
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "pip" in str(call_args)
        assert "install" in str(call_args)
    
    @patch("subprocess.run")
    def test_install_framework_packages_failure(self, mock_run, manager, temp_env_dir):
        """
        Test framework installation failure with error logging.
        
        Requirement 1.7: Log detailed error messages on installation failure
        """
        venv_path = temp_env_dir / "test_env"
        venv_path.mkdir()
        
        # Mock installation failure
        error = subprocess.CalledProcessError(1, "pip install", stderr="Installation failed")
        mock_run.side_effect = error
        
        with pytest.raises(RuntimeError) as exc_info:
            manager._install_framework_packages("PathML", venv_path)
        
        # Should include detailed error information
        assert "PathML" in str(exc_info.value)
        assert "Installation failed" in str(exc_info.value)
    
    @patch("subprocess.run")
    def test_install_framework_packages_timeout(self, mock_run, manager, temp_env_dir):
        """Test framework installation timeout handling."""
        venv_path = temp_env_dir / "test_env"
        venv_path.mkdir()
        
        # Mock installation timeout
        mock_run.side_effect = subprocess.TimeoutExpired("pip install", 600)
        
        with pytest.raises(RuntimeError) as exc_info:
            manager._install_framework_packages("PathML", venv_path)
        
        assert "timed out" in str(exc_info.value).lower()
    
    @patch("subprocess.run")
    def test_validate_installation_success(self, mock_run, manager):
        """
        Test successful installation validation.
        
        Requirement 1.6: Validate framework imports
        """
        env = FrameworkEnvironment(
            framework_name="PathML",
            venv_path=Path("/fake/path"),
            python_version="3.9.0",
            framework_version="2.1.0",
            dependencies={},
            installed_at=datetime.now(),
            patches_applied=[],
            validation_status="not_validated",
        )
        
        # Mock successful import test
        mock_result = Mock()
        mock_result.stdout = ""
        mock_run.return_value = mock_result
        
        validated_env = manager.validate_installation(env)
        
        assert validated_env.validation_status == "valid"
        assert len(validated_env.validation_errors) == 0
    
    @patch("subprocess.run")
    def test_validate_installation_import_failure(self, mock_run, manager):
        """
        Test installation validation with import failure.
        
        Requirement 1.7: Log detailed error messages
        """
        env = FrameworkEnvironment(
            framework_name="PathML",
            venv_path=Path("/fake/path"),
            python_version="3.9.0",
            framework_version="2.1.0",
            dependencies={},
            installed_at=datetime.now(),
            patches_applied=[],
            validation_status="not_validated",
        )
        
        # Mock import failure
        error = subprocess.CalledProcessError(
            1, 
            "python -c import pathml",
            stderr="ModuleNotFoundError: No module named 'pathml'"
        )
        mock_run.side_effect = error
        
        validated_env = manager.validate_installation(env)
        
        assert validated_env.validation_status == "invalid"
        assert len(validated_env.validation_errors) > 0
        assert "Import test failed" in validated_env.validation_errors[0]
        assert "ModuleNotFoundError" in validated_env.validation_errors[0]
    
    @patch("subprocess.run")
    def test_validate_installation_timeout(self, mock_run, manager):
        """Test installation validation timeout handling."""
        env = FrameworkEnvironment(
            framework_name="PathML",
            venv_path=Path("/fake/path"),
            python_version="3.9.0",
            framework_version="2.1.0",
            dependencies={},
            installed_at=datetime.now(),
            patches_applied=[],
            validation_status="not_validated",
        )
        
        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired("python", 30)
        
        validated_env = manager.validate_installation(env)
        
        assert validated_env.validation_status == "invalid"
        assert len(validated_env.validation_errors) > 0
        assert "timed out" in validated_env.validation_errors[0].lower()
    
    def test_validate_installation_unknown_framework(self, manager):
        """Test validation of unknown framework."""
        env = FrameworkEnvironment(
            framework_name="UnknownFramework",
            venv_path=Path("/fake/path"),
            python_version="3.9.0",
            framework_version="1.0.0",
            dependencies={},
            installed_at=datetime.now(),
            patches_applied=[],
            validation_status="not_validated",
        )
        
        validated_env = manager.validate_installation(env)
        
        assert validated_env.validation_status == "invalid"
        assert len(validated_env.validation_errors) > 0
        assert "Unknown framework" in validated_env.validation_errors[0]
    
    @patch.object(FrameworkManager, "_install_framework_packages")
    @patch.object(FrameworkManager, "_get_installed_dependencies")
    @patch.object(FrameworkManager, "get_framework_version")
    @patch.object(FrameworkManager, "apply_compatibility_patches")
    @patch.object(FrameworkManager, "_create_venv")
    @patch.object(FrameworkManager, "_detect_python_version")
    def test_install_framework_full_workflow(
        self,
        mock_detect_version,
        mock_create_venv,
        mock_apply_patches,
        mock_get_version,
        mock_get_deps,
        mock_install_packages,
        manager,
        temp_env_dir,
    ):
        """
        Test complete framework installation workflow.
        
        Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8
        """
        # Setup mocks
        mock_detect_version.return_value = "3.14.0"
        mock_apply_patches.return_value = ["PathML Python 3.14 patch"]
        mock_get_version.return_value = "2.1.0"
        mock_get_deps.return_value = {"numpy": "1.24.0", "pandas": "2.0.0"}
        
        # Install framework
        env = manager.install_framework("PathML")
        
        # Verify all steps were called
        mock_detect_version.assert_called_once()
        mock_create_venv.assert_called_once()
        mock_apply_patches.assert_called_once_with("PathML", "3.14.0")
        mock_install_packages.assert_called_once()
        mock_get_version.assert_called_once()
        mock_get_deps.assert_called_once()
        
        # Verify environment object
        assert env.framework_name == "PathML"
        assert env.python_version == "3.14.0"
        assert env.framework_version == "2.1.0"
        assert env.patches_applied == ["PathML Python 3.14 patch"]
        assert env.dependencies == {"numpy": "1.24.0", "pandas": "2.0.0"}
        assert env.validation_status == "not_validated"
    
    def test_install_framework_unsupported(self, manager):
        """Test installation of unsupported framework raises error."""
        with pytest.raises(ValueError) as exc_info:
            manager.install_framework("UnsupportedFramework")
        
        assert "Unsupported framework" in str(exc_info.value)
        assert "UnsupportedFramework" in str(exc_info.value)
    
    @patch.object(FrameworkManager, "_install_framework_packages")
    @patch.object(FrameworkManager, "_create_venv")
    def test_install_framework_installation_failure(
        self,
        mock_create_venv,
        mock_install_packages,
        manager,
    ):
        """Test that installation failure is properly handled and logged."""
        # Mock installation failure
        mock_install_packages.side_effect = RuntimeError("Installation failed")
        
        with pytest.raises(RuntimeError) as exc_info:
            manager.install_framework("PathML")
        
        assert "Installation failed" in str(exc_info.value)
    
    def test_framework_specs_completeness(self, manager):
        """Test that all framework specs have required fields."""
        required_fields = ["install_command", "import_test", "version_command"]
        
        for framework_name, spec in manager.FRAMEWORK_SPECS.items():
            for field in required_fields:
                assert field in spec, f"{framework_name} missing {field}"
                assert spec[field], f"{framework_name} has empty {field}"
    
    def test_supported_frameworks(self, manager):
        """Test that expected frameworks are supported."""
        expected_frameworks = ["HistoCore", "PathML", "CLAM", "PyTorch"]
        
        for framework in expected_frameworks:
            assert framework in manager.FRAMEWORK_SPECS


class TestFrameworkManagerIntegration:
    """Integration tests for FrameworkManager (require actual environment)."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_create_real_venv(self, tmp_path):
        """Test creating a real virtual environment."""
        manager = FrameworkManager(base_env_dir=tmp_path)
        venv_path = tmp_path / "test_venv"
        
        manager._create_venv(venv_path)
        
        assert venv_path.exists()
        
        # Check that Python executable exists
        python_exe = manager._get_python_executable(venv_path)
        assert Path(python_exe).exists()
