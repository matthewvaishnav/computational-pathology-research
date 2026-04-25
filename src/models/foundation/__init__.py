"""Foundation model encoders for computational pathology.

This module provides unified interfaces to histopathology-specific
foundation models (Phikon, UNI, CONCH) as drop-in replacements for
ImageNet-pretrained encoders.
"""

from .encoders import (
    CONCHEncoder,
    FoundationModelEncoder,
    PhikonEncoder,
    UNIEncoder,
    load_foundation_model,
)
from .projector import FeatureProjector

__all__ = [
    "FoundationModelEncoder",
    "PhikonEncoder",
    "UNIEncoder",
    "CONCHEncoder",
    "load_foundation_model",
    "FeatureProjector",
]
