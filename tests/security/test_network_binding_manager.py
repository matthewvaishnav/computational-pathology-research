"""
Unit tests for NetworkBindingManager.

Tests network binding security policies across different environments.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from src.security.network_binding_manager import NetworkBindingManager
from src.security.models import SecurityEnvironment
from src.security.exceptions import NetworkBindingSecurityError


class TestNetworkBindingManager:
    """Test NetworkBindingManager functionality."""

    def test_production_blocks_0_0_0_0_without_explicit_config(self):
        """Test production blocks 0.0.0.0 binding without explicit configuration."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = NetworkBindingManager()

            with pytest.raises(NetworkBindingSecurityError, match="0.0.0.0 binding not allowed"):
                manager.get_safe_host(requested_host="0.0.0.0")

    def test_production_allows_0_0_0_0_with_explicit_config(self):
        """Test production allows 0.0.0.0 with explicit configuration."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "ALLOW_PUBLIC_BINDING": "true"}):
            manager = NetworkBindingManager()
            host = manager.get_safe_host(requested_host="0.0.0.0")
            assert host == "0.0.0.0"

    def test_production_allows_localhost(self):
        """Test production allows localhost binding."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = NetworkBindingManager()

            assert manager.get_safe_host(requested_host="127.0.0.1") == "127.0.0.1"
            assert manager.get_safe_host(requested_host="localhost") == "localhost"

    def test_development_defaults_to_127_0_0_1(self):
        """Test development defaults to 127.0.0.1."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            manager = NetworkBindingManager()
            host = manager.get_safe_host()
            assert host == "127.0.0.1"

    def test_development_allows_0_0_0_0_with_warning(self, caplog):
        """Test development allows 0.0.0.0 with warning."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            manager = NetworkBindingManager()
            host = manager.get_safe_host(requested_host="0.0.0.0")

            assert host == "0.0.0.0"
            assert "0.0.0.0 binding in development" in caplog.text

    def test_research_allows_0_0_0_0_with_warning(self, caplog):
        """Test research allows 0.0.0.0 with warning."""
        with patch.dict(os.environ, {"ENVIRONMENT": "research"}):
            manager = NetworkBindingManager()
            host = manager.get_safe_host(requested_host="0.0.0.0")

            assert host == "0.0.0.0"
            assert "0.0.0.0 binding in research" in caplog.text

    def test_explicit_host_configuration(self):
        """Test explicit host configuration is respected."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = NetworkBindingManager()

            # Specific IP addresses should be allowed
            assert manager.get_safe_host(requested_host="192.168.1.100") == "192.168.1.100"
            assert manager.get_safe_host(requested_host="10.0.0.5") == "10.0.0.5"

    def test_invalid_hosts_rejected(self):
        """Test invalid host values are rejected."""
        manager = NetworkBindingManager()

        invalid_hosts = [
            "invalid_host",
            "256.256.256.256",
            "not.a.valid.ip",
            "",
            None,
        ]

        for invalid_host in invalid_hosts:
            with pytest.raises((NetworkBindingSecurityError, ValueError)):
                manager.get_safe_host(requested_host=invalid_host)

    def test_validate_binding_success(self):
        """Test validate_binding succeeds for valid bindings."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            manager = NetworkBindingManager()

            # Should not raise
            manager.validate_binding("127.0.0.1", 8000)
            manager.validate_binding("localhost", 8080)

    def test_validate_binding_failure(self):
        """Test validate_binding fails for invalid bindings."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = NetworkBindingManager()

            with pytest.raises(NetworkBindingSecurityError):
                manager.validate_binding("0.0.0.0", 8000)

    def test_audit_logging_for_binding_decisions(self, caplog):
        """Test audit logging for binding decisions."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = NetworkBindingManager()

            # Successful binding
            manager.get_safe_host(requested_host="127.0.0.1")
            assert "Network binding" in caplog.text

            # Failed binding
            try:
                manager.get_safe_host(requested_host="0.0.0.0")
            except NetworkBindingSecurityError:
                pass
            assert "blocked" in caplog.text.lower()

    def test_ipv6_localhost_support(self):
        """Test IPv6 localhost is supported."""
        manager = NetworkBindingManager()

        assert manager.get_safe_host(requested_host="::1") == "::1"
        assert manager.get_safe_host(requested_host="::") == "::"

    def test_port_validation(self):
        """Test port number validation."""
        manager = NetworkBindingManager()

        # Valid ports
        manager.validate_binding("127.0.0.1", 80)
        manager.validate_binding("127.0.0.1", 8000)
        manager.validate_binding("127.0.0.1", 65535)

        # Invalid ports
        with pytest.raises(ValueError):
            manager.validate_binding("127.0.0.1", 0)

        with pytest.raises(ValueError):
            manager.validate_binding("127.0.0.1", 70000)

        with pytest.raises(ValueError):
            manager.validate_binding("127.0.0.1", -1)

    def test_get_safe_host_with_port(self):
        """Test get_safe_host with port parameter."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            manager = NetworkBindingManager()

            host, port = manager.get_safe_host_and_port(
                requested_host="127.0.0.1", requested_port=8000
            )

            assert host == "127.0.0.1"
            assert port == 8000

    def test_environment_specific_defaults(self):
        """Test environment-specific default hosts."""
        # Production defaults to localhost
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = NetworkBindingManager()
            assert manager.get_safe_host() == "127.0.0.1"

        # Development defaults to localhost
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            manager = NetworkBindingManager()
            assert manager.get_safe_host() == "127.0.0.1"

        # Research defaults to localhost
        with patch.dict(os.environ, {"ENVIRONMENT": "research"}):
            manager = NetworkBindingManager()
            assert manager.get_safe_host() == "127.0.0.1"

    def test_manager_caches_environment(self):
        """Test manager caches environment detection."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = NetworkBindingManager()

            # Change environment
            os.environ["ENVIRONMENT"] = "development"

            # Should still use production rules
            with pytest.raises(NetworkBindingSecurityError):
                manager.get_safe_host(requested_host="0.0.0.0")
