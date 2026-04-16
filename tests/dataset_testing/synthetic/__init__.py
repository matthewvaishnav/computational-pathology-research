"""
Synthetic data generators for dataset testing.

This module provides synthetic data generators for creating realistic
test data without requiring large real datasets.
"""

from .pcam_generator import PCamSyntheticGenerator
from .camelyon_generator import CAMELYONSyntheticGenerator
from .multimodal_generator import MultimodalSyntheticGenerator
from .wsi_generator import WSISyntheticGenerator

__all__ = [
    "PCamSyntheticGenerator",
    "CAMELYONSyntheticGenerator",
    "MultimodalSyntheticGenerator",
    "WSISyntheticGenerator",
]
