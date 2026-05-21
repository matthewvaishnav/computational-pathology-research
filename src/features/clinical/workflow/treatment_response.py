"""
Treatment Response Monitoring Module

This module provides comprehensive treatment response analysis capabilities for clinical
workflow integration. It builds on the existing longitudinal tracking infrastructure
to provide detailed treatment response metrics, biological response kinetics modeling,
and therapeutic regimen comparison.

Requirements addressed:
- 5.4: Treatment response identification
- 19.1: Treatment response metrics computation
- 19.2: Response categorization (complete, partial, stable, progressive)
- 19.3: Treatment timeline and biological response kinetics
- 19.4: Treatment response trajectory visualization
- 19.5: Unexpected response detection
- 19.6: Response correlation with patient factors
- 19.7: Therapeutic regimen comparison

This module has been refactored to use focused components:
- ResponseCalculator: Response magnitude and consistency calculation
- ProgressionAnalyzer: Response kinetics and trajectory analysis
- OutcomePredictor: Durability prediction and unexpected response detection
- TreatmentFacade: Unified interface for backward compatibility
"""

import logging

# Import visualization functions
from . import treatment_response_viz

# Re-export components for backward compatibility
from .outcome_predictor import (
    TreatmentResponseMetrics,
    UnexpectedResponseType,
)
from .progression_analyzer import ResponseKinetics
from .treatment_facade import TreatmentResponseAnalyzer

logger = logging.getLogger(__name__)

# Re-export all public classes for backward compatibility
__all__ = [
    "ResponseKinetics",
    "UnexpectedResponseType",
    "TreatmentResponseMetrics",
    "TreatmentResponseAnalyzer",
    "treatment_response_viz",
]
