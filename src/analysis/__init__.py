"""
HistoCore Project Optimization Analysis System.

Comprehensive static analysis and profiling framework for evaluating
the HistoCore computational pathology codebase across 8 dimensions:
Architecture, Performance, Testing, Code Quality, Dependencies,
Deployment, Security, and Scalability.
"""

__version__ = "0.1.0"

from .models import AnalysisResult, Issue, OptimizationPlan

__all__ = ["AnalysisResult", "Issue", "OptimizationPlan"]
