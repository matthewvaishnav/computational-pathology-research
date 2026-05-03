"""
Version Tracker for the Competitor Benchmark System.

This module captures complete environment details to ensure benchmark results
can be reproduced. Records framework versions, hardware specifications, OS details,
and generates shareable configuration files.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9
"""

import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """Complete environment information for reproducibility."""
    
    # Framework versions (Requirement 9.1)
    framework_versions: Dict[str, str] = field(default_factory=dict)
    
    # PyTorch and CUDA (Requirements 9.2)
    pytorch_version: str = ""
    cuda_version: str = ""
    cudnn_version: str = ""
    
    # Hardware specifications (Requirement 9.3)
    gpu_model: str = ""
    gpu_driver_version: str = ""
    gpu_memory_mb: float = 0.0
    gpu_count: int = 0
    
    # OS and Python (Requirement 9.4)
    os_name: str = ""
    os_version: str = ""
    os_release: str = ""
    python_version: str = ""
    python_implementation: str = ""
    
    # Additional system info
    processor: str = ""
    machine: str = ""
    
    # Timestamp
    recorded_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentInfo":
        """Create from dictionary."""
        return cls(**data)


class VersionTracker:
    """Tracks version information and environment details for reproducibility."""
    
    def __init__(self):
        """Initialize Version Tracker."""
        self.environment_info: Optional[EnvironmentInfo] = None
        
    def record_environment(
        self,
        framework_environments: Optional[Dict[str, Any]] = None
    ) -> EnvironmentInfo:
        """
        Capture all version information.
        
        Records framework versions, PyTorch version, CUDA version, cuDNN version,
        hardware specifications (GPU model, driver version, memory), OS version,
        and Python version.
        
        Args:
            framework_environments: Optional dict mapping framework names to
                                   FrameworkEnvironment objects
        
        Returns:
            EnvironmentInfo with all captured details
            
        Requirements: 9.1, 9.2, 9.3, 9.4
        """
        logger.info("Recording environment information")
        
        env_info = EnvironmentInfo()
        
        # Record timestamp
        env_info.recorded_at = datetime.now().isoformat()
        
        # Record framework versions (Requirement 9.1)
        env_info.framework_versions = self._get_framework_versions(framework_environments)
        
        # Record PyTorch and CUDA versions (Requirement 9.2)
        env_info.pytorch_version = self._get_pytorch_version()
        env_info.cuda_version = self._get_cuda_version()
        env_info.cudnn_version = self._get_cudnn_version()
        
        # Record hardware specifications (Requirement 9.3)
        gpu_info = self._get_gpu_info()
        env_info.gpu_model = gpu_info["model"]
        env_info.gpu_driver_version = gpu_info["driver_version"]
        env_info.gpu_memory_mb = gpu_info["memory_mb"]
        env_info.gpu_count = gpu_info["count"]
        
        # Record OS and Python versions (Requirement 9.4)
        env_info.os_name = platform.system()
        env_info.os_version = platform.version()
        env_info.os_release = platform.release()
        env_info.python_version = platform.python_version()
        env_info.python_implementation = platform.python_implementation()
        
        # Additional system info
        env_info.processor = platform.processor()
        env_info.machine = platform.machine()
        
        self.environment_info = env_info
        
        logger.info(
            f"Environment recorded: Python {env_info.python_version}, "
            f"PyTorch {env_info.pytorch_version}, "
            f"CUDA {env_info.cuda_version}, "
            f"GPU: {env_info.gpu_model}"
        )
        
        return env_info
    
    def generate_requirements_txt(
        self,
        output_path: Path,
        include_all_packages: bool = False
    ) -> Path:
        """
        Create pinned dependencies file.
        
        Generates a requirements.txt file with all Python dependencies and their
        exact versions for reproducibility.
        
        Args:
            output_path: Path where requirements.txt should be saved
            include_all_packages: If True, include all installed packages.
                                 If False, only include key packages.
        
        Returns:
            Path to generated requirements.txt file
            
        Requirements: 9.6
        """
        logger.info(f"Generating requirements.txt at {output_path}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get installed packages using pip freeze
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            
            packages = result.stdout.strip().split("\n")
            
            if not include_all_packages:
                # Filter to key packages for benchmarking
                key_packages = {
                    "torch", "torchvision", "numpy", "pandas", "scipy",
                    "scikit-learn", "matplotlib", "seaborn", "pillow",
                    "openslide-python", "h5py", "pyyaml", "tqdm",
                    "psutil", "hypothesis"
                }
                
                filtered_packages = []
                for pkg in packages:
                    pkg_name = pkg.split("==")[0].lower().replace("_", "-")
                    if any(key in pkg_name for key in key_packages):
                        filtered_packages.append(pkg)
                
                packages = filtered_packages
            
            # Write to file
            with open(output_path, "w") as f:
                f.write("# Generated by VersionTracker\n")
                f.write(f"# Date: {datetime.now().isoformat()}\n")
                f.write("# Python version: {}\n".format(
                    platform.python_version()
                ))
                f.write("\n")
                for pkg in sorted(packages):
                    f.write(f"{pkg}\n")
            
            logger.info(
                f"Requirements file generated with {len(packages)} packages"
            )
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate requirements.txt: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating requirements.txt: {e}")
            raise
    
    def export_config_yaml(
        self,
        output_path: Path,
        additional_config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save shareable configuration.
        
        Exports environment information and optional additional configuration
        to a YAML file that can be shared for reproducibility.
        
        Args:
            output_path: Path where config YAML should be saved
            additional_config: Optional additional configuration to include
        
        Returns:
            Path to generated YAML file
            
        Requirements: 9.8
        """
        logger.info(f"Exporting configuration to {output_path}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure environment is recorded
        if self.environment_info is None:
            logger.warning("Environment not recorded, recording now")
            self.record_environment()
        
        # Build configuration dictionary
        config = {
            "metadata": {
                "generated_by": "VersionTracker",
                "generated_at": datetime.now().isoformat(),
            },
            "environment": self.environment_info.to_dict(),
        }
        
        # Add additional configuration if provided
        if additional_config:
            config["additional_config"] = additional_config
        
        # Write to YAML file
        try:
            with open(output_path, "w") as f:
                yaml.dump(
                    config,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    indent=2,
                )
            
            logger.info(f"Configuration exported to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            raise
    
    def validate_config(
        self,
        config_path: Path,
        strict: bool = False
    ) -> tuple[bool, List[str]]:
        """
        Verify config matches current environment.
        
        Loads a configuration file and compares it against the current environment
        to detect version mismatches or hardware differences.
        
        Args:
            config_path: Path to configuration YAML file
            strict: If True, any mismatch is considered invalid.
                   If False, only critical mismatches (major version differences)
                   are considered invalid.
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
            
        Requirements: 9.9
        """
        logger.info(f"Validating configuration from {config_path}")
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            return False, [f"Configuration file not found: {config_path}"]
        
        try:
            # Load configuration
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            if "environment" not in config:
                return False, ["Configuration file missing 'environment' section"]
            
            saved_env = EnvironmentInfo.from_dict(config["environment"])
            
            # Record current environment
            current_env = self.record_environment()
            
            # Compare environments
            warnings = []
            is_valid = True
            
            # Check Python version
            if saved_env.python_version != current_env.python_version:
                msg = (
                    f"Python version mismatch: "
                    f"config={saved_env.python_version}, "
                    f"current={current_env.python_version}"
                )
                warnings.append(msg)
                
                # Check if major.minor differs (critical)
                saved_major_minor = ".".join(saved_env.python_version.split(".")[:2])
                current_major_minor = ".".join(current_env.python_version.split(".")[:2])
                if saved_major_minor != current_major_minor:
                    is_valid = False
            
            # Check PyTorch version
            if saved_env.pytorch_version != current_env.pytorch_version:
                msg = (
                    f"PyTorch version mismatch: "
                    f"config={saved_env.pytorch_version}, "
                    f"current={current_env.pytorch_version}"
                )
                warnings.append(msg)
                
                # Check if major version differs (critical)
                saved_major = saved_env.pytorch_version.split(".")[0]
                current_major = current_env.pytorch_version.split(".")[0]
                if saved_major != current_major:
                    is_valid = False
            
            # Check CUDA version
            if saved_env.cuda_version != current_env.cuda_version:
                msg = (
                    f"CUDA version mismatch: "
                    f"config={saved_env.cuda_version}, "
                    f"current={current_env.cuda_version}"
                )
                warnings.append(msg)
                
                # CUDA version mismatch is critical
                if strict:
                    is_valid = False
            
            # Check GPU model
            if saved_env.gpu_model != current_env.gpu_model:
                msg = (
                    f"GPU model mismatch: "
                    f"config={saved_env.gpu_model}, "
                    f"current={current_env.gpu_model}"
                )
                warnings.append(msg)
                
                # GPU model mismatch is critical for fair comparison
                is_valid = False
            
            # Check OS
            if saved_env.os_name != current_env.os_name:
                msg = (
                    f"OS mismatch: "
                    f"config={saved_env.os_name}, "
                    f"current={current_env.os_name}"
                )
                warnings.append(msg)
                
                # OS mismatch is critical
                if strict:
                    is_valid = False
            
            # Check framework versions
            for framework, saved_version in saved_env.framework_versions.items():
                current_version = current_env.framework_versions.get(framework)
                if current_version is None:
                    msg = f"Framework {framework} not found in current environment"
                    warnings.append(msg)
                    is_valid = False
                elif saved_version != current_version:
                    msg = (
                        f"Framework {framework} version mismatch: "
                        f"config={saved_version}, "
                        f"current={current_version}"
                    )
                    warnings.append(msg)
                    if strict:
                        is_valid = False
            
            if is_valid:
                logger.info("Configuration validation passed")
            else:
                logger.warning(
                    f"Configuration validation failed with {len(warnings)} issues"
                )
            
            return is_valid, warnings
            
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def _get_framework_versions(
        self,
        framework_environments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Get versions of all frameworks.
        
        Args:
            framework_environments: Optional dict of FrameworkEnvironment objects
        
        Returns:
            Dictionary mapping framework names to versions
        """
        versions = {}
        
        if framework_environments:
            for name, env in framework_environments.items():
                if hasattr(env, "framework_version"):
                    versions[name] = env.framework_version
                else:
                    versions[name] = "unknown"
        
        # Always include PyTorch (baseline)
        versions["PyTorch"] = self._get_pytorch_version()
        
        return versions
    
    def _get_pytorch_version(self) -> str:
        """Get PyTorch version."""
        try:
            return str(torch.__version__)
        except Exception as e:
            logger.warning(f"Could not get PyTorch version: {e}")
            return "unknown"
    
    def _get_cuda_version(self) -> str:
        """Get CUDA version."""
        try:
            if torch.cuda.is_available():
                return torch.version.cuda or "unknown"
            else:
                return "not_available"
        except Exception as e:
            logger.warning(f"Could not get CUDA version: {e}")
            return "unknown"
    
    def _get_cudnn_version(self) -> str:
        """Get cuDNN version."""
        try:
            if torch.cuda.is_available():
                return str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "not_available"
            else:
                return "not_available"
        except Exception as e:
            logger.warning(f"Could not get cuDNN version: {e}")
            return "unknown"
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU hardware information.
        
        Returns:
            Dictionary with GPU model, driver version, memory, and count
        """
        gpu_info = {
            "model": "not_available",
            "driver_version": "unknown",
            "memory_mb": 0.0,
            "count": 0,
        }
        
        try:
            if torch.cuda.is_available():
                gpu_info["count"] = torch.cuda.device_count()
                gpu_info["model"] = torch.cuda.get_device_name(0)
                gpu_info["memory_mb"] = (
                    torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                )
                
                # Try to get driver version from nvidia-smi
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=True,
                    )
                    gpu_info["driver_version"] = result.stdout.strip()
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                    logger.debug("Could not query nvidia-smi for driver version")
                    gpu_info["driver_version"] = "unknown"
        except Exception as e:
            logger.warning(f"Could not get GPU information: {e}")
        
        return gpu_info
