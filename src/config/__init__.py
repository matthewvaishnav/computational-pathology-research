"""
Configuration management for nnMIL Multiple Instance Learning.

This package provides configuration utilities including:
- nnMILConfig: Rule-based configuration with dataset fingerprinting
- YAML schema validation and inheritance
"""

from .nnmil_config import nnMILConfig

__all__ = [
    "nnMILConfig",
]