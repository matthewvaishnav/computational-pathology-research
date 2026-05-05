"""
Inference components for nnMIL Multiple Instance Learning.

This package provides inference utilities including:
- SlidingWindowInference: Process large bags with overlapping windows
- UncertaintyEstimator: Quantify prediction uncertainty
"""

from .sliding_window import SlidingWindowInference
from .uncertainty import UncertaintyEstimator

__all__ = [
    "SlidingWindowInference",
    "UncertaintyEstimator",
]