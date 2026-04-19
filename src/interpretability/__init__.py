"""Model interpretability tools for computational pathology.

This module provides comprehensive interpretability capabilities including:
- Grad-CAM visualization for CNN feature extractors
- Attention weight visualization for MIL models
- Failure case analysis and clustering
- Feature importance computation for clinical data
- Interactive visualization dashboard
"""

__version__ = "0.1.0"

# Core components
from .gradcam import GradCAMGenerator
from .failure_analysis import FailureAnalyzer
from .feature_importance import FeatureImportanceCalculator
from .config import (
    GradCAMConfig,
    AttentionConfig,
    AttentionData,
    GradCAMParser,
    GradCAMPrettyPrinter,
    AttentionParser,
    AttentionPrettyPrinter
)
from .dashboard import InterpretabilityDashboard, InMemoryCache, start_dashboard

__all__ = [
    'GradCAMGenerator',
    'FailureAnalyzer',
    'FeatureImportanceCalculator',
    'GradCAMConfig',
    'AttentionConfig',
    'AttentionData',
    'GradCAMParser',
    'GradCAMPrettyPrinter',
    'AttentionParser',
    'AttentionPrettyPrinter',
    'InterpretabilityDashboard',
    'InMemoryCache',
    'start_dashboard'
]

