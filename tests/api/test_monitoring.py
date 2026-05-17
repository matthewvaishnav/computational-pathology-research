"""
Tests for Monitoring Router

Tests for system monitoring endpoints including health checks, readiness probes,
metrics, IDS alerts, and SIEM incidents.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_monitoring_router_exists():
    """Test that monitoring router module exists and is importable."""
    try:
        from src.api.routers import monitoring

        assert hasattr(monitoring, "router")
        print("✓ Monitoring router module imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import monitoring router: {e}")


def test_monitoring_router_configuration():
    """Test that monitoring router has correct configuration."""
    from src.api.routers import monitoring

    # Check router configuration
    assert "monitoring" in monitoring.router.tags
    print("✓ Monitoring router configured correctly")


def test_monitoring_router_has_endpoints():
    """Test that monitoring router has the expected endpoints."""
    from src.api.routers import monitoring

    # Get all routes from the router
    routes = [route.path for route in monitoring.router.routes]

    # Check for monitoring endpoints
    assert "/health" in routes, "Missing /health endpoint"
    assert "/api/v1/system/readiness" in routes, "Missing /api/v1/system/readiness endpoint"
    assert "/metrics" in routes, "Missing /metrics endpoint"

    # Check for IDS endpoints
    assert "/api/v1/security/ids/alerts" in routes, "Missing /api/v1/security/ids/alerts endpoint"

    # Check for SIEM endpoints
    assert (
        "/api/v1/security/siem/incidents" in routes
    ), "Missing /api/v1/security/siem/incidents endpoint"

    print(f"✓ Monitoring router has {len(routes)} endpoints")


def test_monitoring_router_endpoint_count():
    """Test that monitoring router has the correct number of endpoints."""
    from src.api.routers import monitoring

    routes = list(monitoring.router.routes)

    # We expect at least 5 endpoints:
    # 1. GET /health
    # 2. GET /api/v1/system/readiness
    # 3. GET /metrics
    # 4. GET /api/v1/security/ids/alerts
    # 5. GET /api/v1/security/siem/incidents

    assert len(routes) >= 5, f"Expected at least 5 endpoints, found {len(routes)}"
    print(f"✓ Monitoring router has {len(routes)} endpoints (expected >= 5)")


def test_pydantic_models_defined():
    """Test that required Pydantic models are defined in the router."""
    from src.api.routers import monitoring

    # Check that models are defined
    assert hasattr(monitoring, "HealthResponse")
    assert hasattr(monitoring, "BuildInfo")
    print("✓ All required Pydantic models are defined")


def test_require_admin_dependency():
    """Test that require_admin dependency function exists."""
    from src.api.routers import monitoring

    # Check that require_admin function exists
    assert hasattr(monitoring, "require_admin")
    print("✓ require_admin dependency function is defined")


def test_health_endpoint_exists():
    """Test that health check endpoint is defined."""
    from src.api.routers import monitoring

    # Find the health endpoint
    health_routes = [route for route in monitoring.router.routes if route.path == "/health"]
    assert len(health_routes) > 0, "Health endpoint not found"

    # Check that it's a GET endpoint
    health_route = health_routes[0]
    assert "GET" in health_route.methods
    print("✓ Health endpoint is defined as GET /health")


def test_readiness_endpoint_exists():
    """Test that readiness check endpoint is defined."""
    from src.api.routers import monitoring

    # Find the readiness endpoint
    readiness_routes = [
        route for route in monitoring.router.routes if route.path == "/api/v1/system/readiness"
    ]
    assert len(readiness_routes) > 0, "Readiness endpoint not found"

    # Check that it's a GET endpoint
    readiness_route = readiness_routes[0]
    assert "GET" in readiness_route.methods
    print("✓ Readiness endpoint is defined as GET /api/v1/system/readiness")


def test_metrics_endpoint_exists():
    """Test that metrics endpoint is defined."""
    from src.api.routers import monitoring

    # Find the metrics endpoint
    metrics_routes = [route for route in monitoring.router.routes if route.path == "/metrics"]
    assert len(metrics_routes) > 0, "Metrics endpoint not found"

    # Check that it's a GET endpoint
    metrics_route = metrics_routes[0]
    assert "GET" in metrics_route.methods
    print("✓ Metrics endpoint is defined as GET /metrics")


def test_ids_alerts_endpoint_exists():
    """Test that IDS alerts endpoint is defined."""
    from src.api.routers import monitoring

    # Find the IDS alerts endpoint
    ids_routes = [
        route for route in monitoring.router.routes if route.path == "/api/v1/security/ids/alerts"
    ]
    assert len(ids_routes) > 0, "IDS alerts endpoint not found"

    # Check that it's a GET endpoint
    ids_route = ids_routes[0]
    assert "GET" in ids_route.methods
    print("✓ IDS alerts endpoint is defined as GET /api/v1/security/ids/alerts")


def test_siem_incidents_endpoint_exists():
    """Test that SIEM incidents endpoint is defined."""
    from src.api.routers import monitoring

    # Find the SIEM incidents endpoint
    siem_routes = [
        route
        for route in monitoring.router.routes
        if route.path == "/api/v1/security/siem/incidents"
    ]
    assert len(siem_routes) > 0, "SIEM incidents endpoint not found"

    # Check that it's a GET endpoint
    siem_route = siem_routes[0]
    assert "GET" in siem_route.methods
    print("✓ SIEM incidents endpoint is defined as GET /api/v1/security/siem/incidents")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
