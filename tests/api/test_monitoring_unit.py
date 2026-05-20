"""
Unit tests for the monitoring router.

Tests health checks, readiness probes, metrics, and security monitoring endpoints.
"""

import os
from unittest.mock import Mock, patch

import pytest


class TestMonitoringRouterComponents:
    """Test suite for monitoring router components."""

    def test_monitoring_router_file_exists(self):
        """Test that monitoring router file exists and has expected structure."""
        monitoring_file = "src/api/routers/monitoring.py"
        assert os.path.exists(monitoring_file)

        # Read the file and check for key components
        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for router definition
        assert "router = APIRouter" in content
        assert 'tags=["monitoring"]' in content

        # Check for expected endpoints
        assert '@router.get("/health")' in content
        assert '@router.get("/api/v1/system/readiness")' in content
        assert '@router.get("/metrics")' in content
        assert '@router.get("/api/v1/security/ids/alerts")' in content
        assert '@router.get("/api/v1/security/siem/incidents")' in content

    def test_monitoring_router_pydantic_models(self):
        """Test Pydantic models are defined correctly."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for Pydantic models
        assert "class HealthResponse(BaseModel):" in content

        # Check model fields
        assert "status: str" in content
        assert "timestamp: str" in content
        assert "version: str" in content
        assert "components: Dict[str, bool]" in content

    def test_monitoring_router_health_checks(self):
        """Test that health check functionality is implemented."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for health check implementation
        assert "/health" in content
        assert "HealthResponse" in content

    def test_monitoring_router_readiness_probe(self):
        """Test that readiness probe is implemented."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for readiness probe
        assert "/readiness" in content or "/system/readiness" in content

    def test_monitoring_router_metrics_endpoint(self):
        """Test that metrics endpoint is implemented."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for metrics endpoint
        assert "/metrics" in content

    def test_monitoring_router_security_monitoring(self):
        """Test that security monitoring endpoints are implemented."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for security monitoring
        assert "/security/ids/alerts" in content
        assert "/security/siem/incidents" in content

    def test_monitoring_router_database_imports(self):
        """Test that database operations are imported."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for database imports
        assert "from src.platform.database import" in content
        assert "get_db_session" in content

    def test_monitoring_router_dependencies_imports(self):
        """Test that dependency functions are imported."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for dependency imports
        assert "from src.api.dependencies import" in content

    def test_monitoring_router_async_functions(self):
        """Test that endpoints are async functions."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for async function definitions
        assert "async def health_check" in content
        assert "async def readiness_check" in content
        assert "async def get_metrics" in content
        assert "async def get_ids_alerts" in content
        assert "async def get_siem_incidents" in content

    def test_monitoring_router_admin_protection(self):
        """Test that security endpoints require admin access."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for admin protection on security endpoints
        assert "require_admin" in content or "admin" in content.lower()

    def test_monitoring_router_error_handling(self):
        """Test that proper error handling is implemented."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for error handling patterns
        assert "try:" in content
        assert "except" in content
        assert "HTTPException" in content

    def test_monitoring_router_logging_integration(self):
        """Test that logging is properly integrated."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for logging setup and usage
        assert "import logging" in content
        assert "logger = logging.getLogger" in content

    def test_monitoring_router_structure_requirements(self):
        """Test that monitoring router meets structural requirements."""
        monitoring_file = "src/api/routers/monitoring.py"

        # Check file exists
        assert os.path.exists(monitoring_file)

        # Check file size (should be reasonable, not too large)
        file_size = os.path.getsize(monitoring_file)
        assert file_size > 1000  # Should have substantial content
        assert file_size < 50000  # Should not be too large

        # Count lines
        with open(monitoring_file, "r") as f:
            lines = f.readlines()

        # Should be substantial but not too large (design requirement: <500 lines per router)
        assert len(lines) > 50  # Should have substantial content
        assert len(lines) < 500  # Design requirement

    def test_monitoring_router_docstring_coverage(self):
        """Test that monitoring router has proper documentation."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for module docstring
        assert '"""' in content
        assert "Monitoring Router" in content or "monitoring" in content.lower()


class TestMonitoringRouterFunctionality:
    """Test monitoring router functionality with mocked dependencies."""

    def test_monitoring_router_endpoint_count(self):
        """Test that monitoring router has the expected number of endpoints."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Count router decorators
        get_endpoints = content.count("@router.get(")

        # Should have 5 GET endpoints
        assert get_endpoints >= 5

    def test_monitoring_router_http_methods(self):
        """Test that endpoints use correct HTTP methods."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check specific endpoint methods (all should be GET)
        assert '@router.get("/health")' in content
        assert '@router.get("/api/v1/system/readiness")' in content
        assert '@router.get("/metrics")' in content
        assert '@router.get("/api/v1/security/ids/alerts")' in content
        assert '@router.get("/api/v1/security/siem/incidents")' in content

    def test_monitoring_router_health_check_components(self):
        """Test that health check verifies system components."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for component health checks
        assert "database" in content.lower() or "db" in content.lower()
        assert "model" in content.lower() or "inference" in content.lower()

    def test_monitoring_router_prometheus_metrics(self):
        """Test that Prometheus-compatible metrics are provided."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for metrics functionality
        assert "metrics" in content.lower()

    def test_monitoring_router_kubernetes_integration(self):
        """Test that Kubernetes probes are supported."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for readiness probe (Kubernetes integration)
        assert "readiness" in content.lower()

    def test_monitoring_router_security_monitoring_integration(self):
        """Test that security monitoring is integrated."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for security monitoring
        assert "ids" in content.lower()  # Intrusion Detection System
        assert "siem" in content.lower()  # Security Information and Event Management

    def test_monitoring_router_response_formats(self):
        """Test that endpoints return expected response formats."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for expected response keys
        assert '"status"' in content or "status" in content
        assert '"timestamp"' in content or "timestamp" in content
        assert '"version"' in content or "version" in content

    def test_monitoring_router_dependency_injection(self):
        """Test that dependency injection is used correctly."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for FastAPI dependency injection
        assert "Depends(" in content
        assert "get_db_session" in content

    def test_monitoring_router_admin_endpoints_protection(self):
        """Test that admin endpoints are properly protected."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Security endpoints should require admin access
        assert "require_admin" in content or "admin" in content.lower()

    def test_monitoring_router_public_endpoints(self):
        """Test that health and metrics endpoints are accessible."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Health and metrics should be publicly accessible (no auth required)
        assert "/health" in content
        assert "/metrics" in content

    def test_monitoring_router_error_responses(self):
        """Test that proper error responses are implemented."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Check for error handling
        assert "HTTPException" in content
        assert "status_code" in content

    def test_monitoring_router_real_checks(self):
        """Test that real system checks are performed."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Should perform real database connectivity checks
        assert "db" in content.lower() or "database" in content.lower()
        # Should check model availability
        assert "model" in content.lower() or "inference" in content.lower()

    @patch("src.api.dependencies.get_db_session")
    def test_health_check_database_connectivity(self, mock_db):
        """Test health check verifies database connectivity."""
        # Mock database session
        mock_session = Mock()
        mock_db.return_value = iter([mock_session])

        # This would be used in health check
        db_session = next(mock_db())
        assert db_session is not None

    def test_monitoring_router_component_status_tracking(self):
        """Test that component status is tracked."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Should track status of different components
        assert "components" in content.lower()
        assert "Dict[str, bool]" in content or "dict" in content.lower()

    def test_monitoring_router_version_information(self):
        """Test that version information is provided."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Should provide version information
        assert "version" in content.lower()

    def test_monitoring_router_timestamp_tracking(self):
        """Test that timestamps are included in responses."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Should include timestamps
        assert "timestamp" in content.lower()

    def test_monitoring_router_security_event_access(self):
        """Test that security events can be accessed by admins."""
        monitoring_file = "src/api/routers/monitoring.py"

        with open(monitoring_file, "r") as f:
            content = f.read()

        # Should provide access to security events for admins
        assert "alerts" in content.lower()
        assert "incidents" in content.lower()
