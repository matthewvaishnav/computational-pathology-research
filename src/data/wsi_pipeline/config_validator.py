"""
Configuration validation and documentation for WSI processing pipeline.

This module provides comprehensive validation of processing configurations
and generates documentation for available options.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import ProcessingConfig
from .exceptions import ProcessingError

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Validate and document WSI processing configurations.
    
    Provides comprehensive validation of configuration parameters
    and generates human-readable documentation.
    """
    
    # Valid encoder names and their properties
    SUPPORTED_ENCODERS = {
        "resnet50": {
            "feature_dim": 2048,
            "description": "ResNet-50 pretrained on ImageNet",
            "memory_usage": "moderate",
            "speed": "fast",
        },
        "densenet121": {
            "feature_dim": 1024,
            "description": "DenseNet-121 pretrained on ImageNet",
            "memory_usage": "low",
            "speed": "moderate",
        },
        "efficientnet_b0": {
            "feature_dim": 1280,
            "description": "EfficientNet-B0 pretrained on ImageNet",
            "memory_usage": "low",
            "speed": "moderate",
        },
    }
    
    # Valid tissue detection methods
    TISSUE_METHODS = {
        "otsu": {
            "description": "Fast Otsu thresholding",
            "speed": "very fast",
            "accuracy": "good",
        },
        "deep_learning": {
            "description": "CNN-based tissue segmentation",
            "speed": "slow",
            "accuracy": "excellent",
        },
        "hybrid": {
            "description": "Otsu + DL refinement",
            "speed": "moderate",
            "accuracy": "very good",
        },
    }
    
    # Configuration parameter ranges
    PARAMETER_RANGES = {
        "patch_size": (64, 2048),
        "stride": (32, 2048),
        "level": (0, 10),
        "tissue_threshold": (0.0, 1.0),
        "batch_size": (1, 1024),
        "num_workers": (1, 32),
        "max_retries": (0, 10),
        "blur_threshold": (0.0, 1000.0),
        "min_tissue_coverage": (0.0, 1.0),
        "max_memory_gb": (0.5, 128.0),
    }
    
    def __init__(self):
        """Initialize configuration validator."""
        logger.debug("Initialized ConfigValidator")
    
    def validate_config(self, config: ProcessingConfig) -> Tuple[bool, List[str]]:
        """
        Validate a processing configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate encoder
        if config.encoder_name not in self.SUPPORTED_ENCODERS:
            errors.append(
                f"Unsupported encoder '{config.encoder_name}'. "
                f"Supported: {list(self.SUPPORTED_ENCODERS.keys())}"
            )
        
        # Validate tissue method
        if config.tissue_method not in self.TISSUE_METHODS:
            errors.append(
                f"Unsupported tissue method '{config.tissue_method}'. "
                f"Supported: {list(self.TISSUE_METHODS.keys())}"
            )
        
        # Validate parameter ranges
        for param_name, (min_val, max_val) in self.PARAMETER_RANGES.items():
            if hasattr(config, param_name):
                value = getattr(config, param_name)
                if value is not None and not (min_val <= value <= max_val):
                    errors.append(
                        f"Parameter '{param_name}' value {value} outside valid range "
                        f"[{min_val}, {max_val}]"
                    )
        
        # Validate logical constraints
        if config.stride and config.stride > config.patch_size:
            errors.append(
                f"Stride ({config.stride}) cannot be larger than patch_size ({config.patch_size})"
            )
        
        # Validate cache directory
        if config.cache_dir:
            cache_path = Path(config.cache_dir)
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create cache directory '{config.cache_dir}': {e}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_and_raise(self, config: ProcessingConfig) -> None:
        """
        Validate configuration and raise exception if invalid.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ProcessingError: If configuration is invalid
        """
        is_valid, errors = self.validate_config(config)
        if not is_valid:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ProcessingError(error_msg)
    
    def get_recommended_config(
        self,
        use_case: str = "general",
        hardware: str = "auto",
    ) -> ProcessingConfig:
        """
        Get recommended configuration for specific use case.
        
        Args:
            use_case: Use case ("general", "high_throughput", "high_quality", "memory_limited")
            hardware: Hardware type ("auto", "gpu", "cpu", "high_memory", "low_memory")
            
        Returns:
            Recommended configuration
        """
        # Base configuration
        base_config = {
            "patch_size": 256,
            "stride": 256,
            "level": 0,
            "tissue_threshold": 0.5,
            "encoder_name": "resnet50",
            "tissue_method": "otsu",
            "batch_size": 32,
            "num_workers": 4,
        }
        
        # Adjust for use case
        if use_case == "high_throughput":
            base_config.update({
                "patch_size": 224,
                "encoder_name": "efficientnet_b0",
                "batch_size": 64,
                "num_workers": 8,
            })
        elif use_case == "high_quality":
            base_config.update({
                "patch_size": 512,
                "encoder_name": "resnet50",
                "tissue_method": "hybrid",
                "batch_size": 16,
            })
        elif use_case == "memory_limited":
            base_config.update({
                "patch_size": 224,
                "encoder_name": "densenet121",
                "batch_size": 8,
                "num_workers": 2,
            })
        
        # Adjust for hardware
        if hardware == "cpu":
            base_config.update({
                "batch_size": min(base_config["batch_size"], 8),
                "num_workers": min(base_config["num_workers"], 4),
            })
        elif hardware == "gpu":
            base_config.update({
                "batch_size": max(base_config["batch_size"], 32),
            })
        elif hardware == "low_memory":
            base_config.update({
                "batch_size": min(base_config["batch_size"], 16),
                "max_memory_gb": 4.0,
            })
        elif hardware == "high_memory":
            base_config.update({
                "batch_size": max(base_config["batch_size"], 64),
                "max_memory_gb": 16.0,
            })
        
        return ProcessingConfig(**base_config)
    
    def generate_documentation(self) -> str:
        """
        Generate comprehensive configuration documentation.
        
        Returns:
            Formatted documentation string
        """
        doc = []
        doc.append("# WSI Processing Pipeline Configuration Guide")
        doc.append("")
        
        # Encoders section
        doc.append("## Supported Encoders")
        doc.append("")
        for name, info in self.SUPPORTED_ENCODERS.items():
            doc.append(f"### {name}")
            doc.append(f"- **Description**: {info['description']}")
            doc.append(f"- **Feature Dimension**: {info['feature_dim']}")
            doc.append(f"- **Memory Usage**: {info['memory_usage']}")
            doc.append(f"- **Speed**: {info['speed']}")
            doc.append("")
        
        # Tissue methods section
        doc.append("## Tissue Detection Methods")
        doc.append("")
        for name, info in self.TISSUE_METHODS.items():
            doc.append(f"### {name}")
            doc.append(f"- **Description**: {info['description']}")
            doc.append(f"- **Speed**: {info['speed']}")
            doc.append(f"- **Accuracy**: {info['accuracy']}")
            doc.append("")
        
        # Parameters section
        doc.append("## Configuration Parameters")
        doc.append("")
        for param, (min_val, max_val) in self.PARAMETER_RANGES.items():
            doc.append(f"- **{param}**: Range [{min_val}, {max_val}]")
        doc.append("")
        
        # Use cases section
        doc.append("## Recommended Configurations")
        doc.append("")
        
        use_cases = ["general", "high_throughput", "high_quality", "memory_limited"]
        for use_case in use_cases:
            config = self.get_recommended_config(use_case=use_case)
            doc.append(f"### {use_case.replace('_', ' ').title()}")
            doc.append("```python")
            doc.append(f"config = ProcessingConfig(")
            doc.append(f"    patch_size={config.patch_size},")
            doc.append(f"    encoder_name='{config.encoder_name}',")
            doc.append(f"    tissue_method='{config.tissue_method}',")
            doc.append(f"    batch_size={config.batch_size},")
            doc.append(f"    num_workers={config.num_workers},")
            doc.append(")")
            doc.append("```")
            doc.append("")
        
        return "\n".join(doc)
    
    def save_documentation(self, output_path: Path) -> None:
        """
        Save configuration documentation to file.
        
        Args:
            output_path: Path to save documentation
        """
        doc = self.generate_documentation()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(doc)
        
        logger.info(f"Configuration documentation saved to {output_path}")


def validate_config(config: ProcessingConfig) -> None:
    """
    Convenience function to validate configuration.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ProcessingError: If configuration is invalid
    """
    validator = ConfigValidator()
    validator.validate_and_raise(config)


def get_recommended_config(use_case: str = "general", hardware: str = "auto") -> ProcessingConfig:
    """
    Convenience function to get recommended configuration.
    
    Args:
        use_case: Use case type
        hardware: Hardware type
        
    Returns:
        Recommended configuration
    """
    validator = ConfigValidator()
    return validator.get_recommended_config(use_case=use_case, hardware=hardware)


if __name__ == "__main__":
    # Generate and save documentation
    validator = ConfigValidator()
    doc_path = Path("WSI_PIPELINE_CONFIG_GUIDE.md")
    validator.save_documentation(doc_path)
    print(f"Configuration documentation saved to {doc_path}")