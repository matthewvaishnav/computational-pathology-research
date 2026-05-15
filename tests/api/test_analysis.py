"""
Tests for Analysis Router

Tests for image upload, analysis results, DICOM processing, and case management endpoints.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_analysis_router_exists():
    """Test that analysis router module exists and is importable."""
    try:
        from src.api.routers import analysis

        assert hasattr(analysis, "router")
        print("✓ Analysis router module imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import analysis router: {e}")


def test_analysis_router_configuration():
    """Test that analysis router has correct configuration."""
    from src.api.routers import analysis

    # Check router configuration
    assert analysis.router.prefix == "/api/v1"
    assert "analysis" in analysis.router.tags
    print("✓ Analysis router configured correctly")


def test_analysis_router_has_endpoints():
    """Test that analysis router has the expected endpoints."""
    from src.api.routers import analysis

    # Get all routes from the router
    routes = [route.path for route in analysis.router.routes]

    # Check for analysis endpoints
    assert "/analyze/upload" in routes, "Missing /analyze/upload endpoint"
    assert "/analyze/{analysis_id}" in routes, "Missing /analyze/{analysis_id} endpoint"

    # Check for DICOM endpoints
    assert "/dicom/upload" in routes, "Missing /dicom/upload endpoint"
    assert "/dicom/study/{study_id}" in routes, "Missing /dicom/study/{study_id} endpoint"

    # Check for case endpoints
    assert "/cases" in routes, "Missing /cases endpoint"
    assert "/cases/{case_id}" in routes, "Missing /cases/{case_id} endpoint"
    assert "/cases/{case_id}/status" in routes, "Missing /cases/{case_id}/status endpoint"

    print(f"✓ Analysis router has {len(routes)} endpoints")


def test_analysis_router_endpoint_count():
    """Test that analysis router has the correct number of endpoints."""
    from src.api.routers import analysis

    routes = list(analysis.router.routes)

    # We expect at least 7 endpoints:
    # 1. POST /analyze/upload
    # 2. GET /analyze/{analysis_id}
    # 3. POST /dicom/upload
    # 4. GET /dicom/study/{study_id}
    # 5. GET /cases
    # 6. POST /cases
    # 7. GET /cases/{case_id}
    # 8. PUT /cases/{case_id}/status

    assert len(routes) >= 7, f"Expected at least 7 endpoints, found {len(routes)}"
    print(f"✓ Analysis router has {len(routes)} endpoints (expected >= 7)")


def test_pydantic_models_defined():
    """Test that required Pydantic models are defined in the router."""
    from src.api.routers import analysis

    # Check that models are defined
    assert hasattr(analysis, "AnalysisRequest")
    assert hasattr(analysis, "CaseData")
    assert hasattr(analysis, "CaseStatusUpdate")
    print("✓ All required Pydantic models are defined")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
