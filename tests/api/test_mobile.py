#!/usr/bin/env python3
"""
Tests for Mobile API Router

Tests mobile device registration, synchronization, offline cases, and model downloads.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_mobile_router_exists():
    """Test that mobile router module exists and is importable."""
    try:
        import src.api.routers.mobile as mobile

        assert hasattr(mobile, "router")
        print("✓ Mobile router module imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import mobile router: {e}")


def test_mobile_router_configuration():
    """Test that mobile router has correct configuration."""
    import src.api.routers.mobile as mobile

    # Check router configuration
    assert mobile.router.prefix == "/api/v1/mobile"
    assert "mobile" in mobile.router.tags
    print("✓ Mobile router configured correctly")


def test_mobile_router_has_endpoints():
    """Test that mobile router has the expected endpoints."""
    import src.api.routers.mobile as mobile

    # Get all routes from the router
    routes = [route.path for route in mobile.router.routes]

    # Check for mobile endpoints
    assert "/register-device" in routes, "Missing /register-device endpoint"
    assert "/sync" in routes, "Missing /sync endpoint"
    assert "/cases/offline" in routes, "Missing /cases/offline endpoint"
    assert "/model/download" in routes, "Missing /model/download endpoint"

    print(f"✓ Mobile router has {len(routes)} endpoints")


def test_mobile_router_endpoint_count():
    """Test that mobile router has the correct number of endpoints."""
    import src.api.routers.mobile as mobile

    routes = list(mobile.router.routes)

    # We expect 4 endpoints:
    # 1. POST /register-device
    # 2. GET /sync
    # 3. GET /cases/offline
    # 4. GET /model/download

    assert len(routes) == 4, f"Expected 4 endpoints, found {len(routes)}"
    print(f"✓ Mobile router has {len(routes)} endpoints (expected 4)")


def test_pydantic_models_defined():
    """Test that required Pydantic models are defined in the router."""
    import src.api.routers.mobile as mobile

    # Check that models are defined
    assert hasattr(mobile, "DeviceRegistration")
    assert hasattr(mobile, "DeviceRegistrationResponse")
    assert hasattr(mobile, "SyncResponse")
    assert hasattr(mobile, "OfflineCasesResponse")
    assert hasattr(mobile, "ModelDownloadResponse")
    print("✓ All required Pydantic models are defined")


def test_mobile_router_methods():
    """Test that endpoints have correct HTTP methods."""
    import src.api.routers.mobile as mobile

    # Check methods for each route
    routes = {route.path: route.methods for route in mobile.router.routes}

    # POST for device registration
    assert "POST" in routes["/register-device"], "register-device should be POST"

    # GET for other endpoints
    assert "GET" in routes["/sync"], "sync should be GET"
    assert "GET" in routes["/cases/offline"], "cases/offline should be GET"
    assert "GET" in routes["/model/download"], "model/download should be GET"

    print("✓ All endpoints have correct HTTP methods")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
