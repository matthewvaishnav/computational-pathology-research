"""
Comprehensive Dataset Testing Suite

This module provides comprehensive testing for the computational pathology
research framework's dataset implementations including PCam, CAMELYON,
multimodal datasets, OpenSlide integration, and data preprocessing.

The testing suite includes:
- Unit tests for specific functionality
- Property-based tests for universal correctness
- Synthetic data generators for testing without large datasets
- Performance benchmarking and regression detection
- Error simulation and edge case testing
"""

__version__ = "1.0.0"
__author__ = "Computational Pathology Research Framework"

# Test categories
UNIT_TESTS = "unit"
PROPERTY_TESTS = "property"
INTEGRATION_TESTS = "integration"
PERFORMANCE_TESTS = "performance"
EDGE_CASE_TESTS = "edge_cases"

# Dataset types
PCAM_DATASET = "pcam"
CAMELYON_DATASET = "camelyon"
MULTIMODAL_DATASET = "multimodal"
OPENSLIDE_DATASET = "openslide"

# Test configuration
DEFAULT_HYPOTHESIS_MAX_EXAMPLES = 100
DEFAULT_HYPOTHESIS_DEADLINE_MS = 60000
DEFAULT_PERFORMANCE_TIMEOUT_SECONDS = 300