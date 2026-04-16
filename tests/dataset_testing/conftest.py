"""
Shared fixtures and configuration for dataset testing suite.

This module provides pytest fixtures, configuration, and utilities
shared across all dataset tests.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import pytest
import torch
import numpy as np
from hypothesis import settings, Verbosity

# Configure Hypothesis for pathology data testing
settings.register_profile(
    "pathology_testing",
    max_examples=100,
    deadline=60000,  # 60 seconds
    verbosity=Verbosity.verbose,
    suppress_health_check=[],
)
settings.load_profile("pathology_testing")


@pytest.fixture(scope="session")
def temp_data_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp(prefix="dataset_testing_"))
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration for dataset testing suite."""
    return {
        "synthetic_sample_counts": {
            "pcam": 100,
            "camelyon": 10,
            "multimodal": 50,
        },
        "synthetic_corruption_rates": {
            "file_corruption": 0.1,
            "network_failure": 0.05,
            "memory_constraint": 0.02,
        },
        "performance_thresholds": {
            "max_loading_time_seconds": 5.0,
            "max_memory_usage_mb": 2048.0,
            "min_throughput_samples_per_second": 10.0,
        },
        "hypothesis_settings": {
            "max_examples": 100,
            "deadline_ms": 60000,
        },
        "feature_flags": {
            "enable_performance_tests": True,
            "enable_property_tests": True,
            "enable_integration_tests": True,
        },
    }


@pytest.fixture
def device() -> torch.device:
    """Get appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed() -> int:
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def setup_random_state(random_seed: int):
    """Set up reproducible random state for all libraries."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)


@pytest.fixture
def pcam_sample_config() -> Dict[str, Any]:
    """Configuration for PCam synthetic samples."""
    return {
        "image_shape": (96, 96, 3),
        "label_distribution": {0: 0.5, 1: 0.5},
        "noise_level": 0.1,
        "corruption_probability": 0.0,
    }


@pytest.fixture
def camelyon_sample_config() -> Dict[str, Any]:
    """Configuration for CAMELYON synthetic samples."""
    return {
        "patches_per_slide_range": (50, 500),
        "feature_dim": 2048,
        "coordinate_range": (0, 10000),
        "patient_slide_distribution": {},
    }


@pytest.fixture
def multimodal_sample_config() -> Dict[str, Any]:
    """Configuration for multimodal synthetic samples."""
    return {
        "wsi_feature_dim": 2048,
        "genomic_feature_dim": 1000,
        "clinical_text_length_range": (10, 100),
        "missing_modality_probability": 0.2,
    }


# Performance testing fixtures
@pytest.fixture
def performance_baseline_metrics() -> Dict[str, float]:
    """Baseline performance metrics for regression detection."""
    return {
        "pcam_loading_time": 2.0,  # seconds
        "camelyon_loading_time": 1.5,  # seconds
        "multimodal_loading_time": 3.0,  # seconds
        "memory_usage_mb": 1024.0,  # MB
        "throughput_samples_per_second": 50.0,
    }


# Error simulation fixtures
@pytest.fixture
def corruption_types() -> list[str]:
    """Available corruption types for error simulation."""
    return [
        "file_truncation",
        "random_bytes",
        "header_corruption",
        "permission_denied",
        "disk_full",
        "network_timeout",
    ]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for dataset testing."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "property: marks tests as property-based tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "edge_case: marks tests as edge case tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test file location
        if "property_based" in str(item.fspath):
            item.add_marker(pytest.mark.property)
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "edge_cases" in str(item.fspath):
            item.add_marker(pytest.mark.edge_case)