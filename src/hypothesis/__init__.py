"""
Scientific hypothesis generation, statistical testing, and reporting.

Loops over discovered subtypes / latent factors and generates testable
biological hypotheses via Claude API, then validates them against data.
"""

from .generator import HypothesisGenerator, GeneratedHypothesis
from .tester import HypothesisTester, TestResult
from .reporter import HypothesisReporter

__all__ = [
    "HypothesisGenerator",
    "GeneratedHypothesis",
    "HypothesisTester",
    "TestResult",
    "HypothesisReporter",
]
