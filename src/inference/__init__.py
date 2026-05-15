"""
Inference components for nnMIL Multiple Instance Learning.

This package provides inference utilities including:
- InferenceEngine: Main inference engine for pathology analysis
- ModelLoader: Load and manage AI models
- SlidingWindowInference: Process large bags with overlapping windows
- UncertaintyEstimator: Quantify prediction uncertainty
"""

from .inference_engine import InferenceEngine
from .model_loader import ModelLoader, get_model_loader
from .sliding_window import SlidingWindowInference
from .uncertainty import UncertaintyEstimator

__all__ = [
    "InferenceEngine",
    "ModelLoader",
    "get_model_loader",
    "SlidingWindowInference",
    "UncertaintyEstimator",
]
