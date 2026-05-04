"""
Tests for Admin Router

Tests for administrative endpoints including user management, system configuration,
audit logs, and reporting.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_admin_router_exists():
    """Test that admin router module exists and is importable."""
    try:
        from src.api.routers import admin
        assert hasattr(admin, "router")
        print("✓ Admin router module imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import admin router: {e}")


def test_admin_router_configuration():
    """Test that admin router has correct configuration."""
    from src.api.routers import admin
    
    # Check router configuration
    assert admin.router.prefix == "/api/v1/admin"
    assert "admin" in admin.router.tags
    print("✓ Admin router configured correctly")


def test_admin_router_has_endpoints():
    """Test that admin router has the expected endpoints."""
    from src.api.routers import admin
    
    # Get all routes from the router
    routes = [route.path for route in admin.router.routes]
    
    # Check for admin endpoints (with full prefix)
    assert "/api/v1/admin/users" in routes, "Missing /api/v1/admin/users endpoint"
    assert "/api/v1/admin/config" in routes, "Missing /api/v1/admin/config endpoint"
    assert "/api/v1/admin/audit-logs" in routes, "Missing /api/v1/admin/audit-logs endpoint"
    
    # Check for report endpoints
    assert "/api/v1/admin/reports/generate" in routes, "Missing /api/v1/admin/reports/generate endpoint"
    assert "/api/v1/admin/reports/{report_id}/status" in routes, "Missing /api/v1/admin/reports/{report_id}/status endpoint"
    
    print(f"✓ Admin router has {len(routes)} endpoints")


def test_admin_router_endpoint_count():
    """Test that admin router has the correct number of endpoints."""
    from src.api.routers import admin
    
    routes = list(admin.router.routes)
    
    # We expect at least 5 endpoints:
    # 1. GET /users
    # 2. GET /config
    # 3. GET /audit-logs
    # 4. POST /reports/generate
    # 5. GET /reports/{report_id}/status
    
    assert len(routes) >= 5, f"Expected at least 5 endpoints, found {len(routes)}"
    print(f"✓ Admin router has {len(routes)} endpoints (expected >= 5)")


def test_pydantic_models_defined():
    """Test that required Pydantic models are defined in the router."""
    from src.api.routers import admin
    
    # Check that models are defined
    assert hasattr(admin, "ReportRequest")
    print("✓ All required Pydantic models are defined")


def test_require_admin_dependency():
    """Test that require_admin dependency function exists."""
    from src.api.routers import admin
    
    # Check that require_admin function exists
    assert hasattr(admin, "require_admin")
    print("✓ require_admin dependency function is defined")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
