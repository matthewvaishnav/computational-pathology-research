"""Configuration management for WSI processing pipeline.

This module defines the ProcessingConfig dataclass and provides
configuration validation and YAML loading functionality.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class ProcessingConfig:
    """Configuration for WSI processing pipeline.
    
    Attributes:
        # Patch extraction
        patch_size: Size of patches to extract (pixels)
        stride: Stride between patches (None = non-overlapping)
        level: Pyramid level to extract patches from
        target_mpp: Target microns per pixel (None = use native resolution)
        
        # Tissue detection
        tissue_method: Method for tissue detection (otsu, deep_learning, hybrid)
        tissue_threshold: Minimum tissue percentage to keep patch (0.0-1.0)
        
        # Feature extraction
        encoder_name: Name of pretrained encoder (resnet50, densenet121, etc.)
        encoder_pretrained: Whether to use pretrained weights
        batch_size: Batch size for feature extraction
        
        # Caching
        cache_dir: Directory to store cached features
        compression: HDF5 compression method (gzip, lzf, None)
        
        # Batch processing
        num_workers: Number of parallel worker processes
        gpu_ids: List of GPU IDs to use (None = all available)
        max_retries: Maximum number of retry attempts for failed slides
        
        # Quality control
        blur_threshold: Minimum blur score (Laplacian variance)
        min_tissue_coverage: Minimum tissue coverage percentage for slide
    """
    
    # Patch extraction
    patch_size: int = 256
    stride: Optional[int] = None
    level: int = 0
    target_mpp: Optional[float] = None
    
    # Tissue detection
    tissue_method: str = "otsu"
    tissue_threshold: float = 0.5
    
    # Feature extraction
    encoder_name: str = "resnet50"
    encoder_pretrained: bool = True
    batch_size: int = 32
    
    # Caching
    cache_dir: str = "features"
    compression: str = "gzip"
    
    # Batch processing
    num_workers: int = 4
    gpu_ids: Optional[List[int]] = None
    max_retries: int = 3
    
    # Quality control
    blur_threshold: float = 100.0
    min_tissue_coverage: float = 0.1
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        errors = []
        
        # Validate patch_size
        if not (64 <= self.patch_size <= 2048):
            errors.append(
                f"patch_size must be between 64 and 2048, got {self.patch_size}"
            )
        
        # Validate tissue_threshold
        if not (0.0 <= self.tissue_threshold <= 1.0):
            errors.append(
                f"tissue_threshold must be between 0.0 and 1.0, "
                f"got {self.tissue_threshold}"
            )
        
        # Validate num_workers
        if not (1 <= self.num_workers <= 16):
            errors.append(
                f"num_workers must be between 1 and 16, got {self.num_workers}"
            )
        
        # Validate max_retries
        if not (0 <= self.max_retries <= 5):
            errors.append(
                f"max_retries must be between 0 and 5, got {self.max_retries}"
            )
        
        # Validate batch_size
        if not (1 <= self.batch_size <= 1024):
            errors.append(
                f"batch_size must be between 1 and 1024, got {self.batch_size}"
            )
        
        # Validate stride if provided
        if self.stride is not None and self.stride < 1:
            errors.append(f"stride must be at least 1, got {self.stride}")
        
        # Validate level
        if self.level < 0:
            errors.append(f"level must be non-negative, got {self.level}")
        
        # Validate target_mpp if provided
        if self.target_mpp is not None and self.target_mpp <= 0:
            errors.append(
                f"target_mpp must be positive, got {self.target_mpp}"
            )
        
        # Validate tissue_method
        valid_methods = ["otsu", "deep_learning", "hybrid"]
        if self.tissue_method not in valid_methods:
            errors.append(
                f"tissue_method must be one of {valid_methods}, "
                f"got {self.tissue_method}"
            )
        
        # Validate blur_threshold
        if self.blur_threshold < 0:
            errors.append(
                f"blur_threshold must be non-negative, got {self.blur_threshold}"
            )
        
        # Validate min_tissue_coverage
        if not (0.0 <= self.min_tissue_coverage <= 1.0):
            errors.append(
                f"min_tissue_coverage must be between 0.0 and 1.0, "
                f"got {self.min_tissue_coverage}"
            )
        
        # Validate compression
        valid_compression = ["gzip", "lzf", None, "none"]
        compression_check = self.compression.lower() if self.compression else None
        if compression_check not in valid_compression:
            errors.append(
                f"compression must be one of {valid_compression}, "
                f"got {self.compression}"
            )
        
        if errors:
            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ProcessingConfig":
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            ProcessingConfig instance
            
        Raises:
            FileNotFoundError: If YAML file does not exist
            ValueError: If configuration is invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested configuration structure if present
        flat_config = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Handle nested sections (e.g., patch_extraction, tissue_detection)
                flat_config.update(value)
            else:
                flat_config[key] = value
        
        # Create config instance
        config = cls(**flat_config)
        
        # Validate configuration
        config.validate()
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ProcessingConfig":
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary of configuration parameters
            
        Returns:
            ProcessingConfig instance
            
        Raises:
            ValueError: If configuration is invalid
        """
        config = cls(**config_dict)
        config.validate()
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        from dataclasses import asdict
        return asdict(self)
    
    def save_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Set stride to patch_size if not specified (non-overlapping)
        if self.stride is None:
            self.stride = self.patch_size
