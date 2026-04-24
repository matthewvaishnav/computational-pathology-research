"""
Multi-scale hierarchical feature pyramids for WSI analysis.

Encodes 4x/10x/20x/40x magnification levels independently, then fuses
via cross-scale attention. Low magnification captures global architecture;
high magnification resolves cellular detail.
"""

from .encoder import MagnificationEncoder, MultiScaleFeatureExtractor
from .attention import CrossScaleAttention, HierarchicalAttentionPool
from .model import MultiScaleMIL

__all__ = [
    "MagnificationEncoder",
    "MultiScaleFeatureExtractor",
    "CrossScaleAttention",
    "HierarchicalAttentionPool",
    "MultiScaleMIL",
]
