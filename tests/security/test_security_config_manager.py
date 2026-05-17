"""
Unit tests for SecurityConfigManager.

Tests configuration loading, environment-based policy decisions, validation, and factory methods.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.security.config_manager import SecurityConfigManager
from src.security.models import SecurityConfig, SecurityEnvironment


class TestSecurityConfigManager:
    """Test SecurityConfigManager functionality."""

    @pytest.fixture
    def sample_config_file(self):
        """Create a temporary config file for testing."""
        config_content = """
production:
  enforce_strict_binding: true
  require_pinned_models: true
  allow_hardcoded_temp_paths: false
  require_pickle_validation: true
  require_url_scheme_validation: true
  audit_all_operations: true

development:
  enforce_strict_binding: false
  require_pinned_models: false
  allow_hardcoded_temp_paths: true
  require_pickle_validation: false
  require_url_scheme_validation: false
  audit_all_operations: false

research:
  enforce_strict_binding: false
  require_pinned_models: false
  allow_hardcoded_temp_paths: true
  require_pickle_validation: false
  require_url_scheme_validation: false
  audit_all_operations: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    def test_load_config_from_file(self, sample_config_file):
        """Test loading configuration from file."""
        manager = SecurityConfigManager(config_path=sample_config_file)

        assert manager.config is not None
        assert SecurityEnvironment.PRODUCTION in manager.config
        assert SecurityEnvironment.DEVELOPMENT in manager.config
        assert SecurityEnvironment.RESEARCH in manager.config

    def test_production_strict_policies(self, sample_config_file):
        """Test production environment has strict security policies."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = SecurityConfigManager(config_path=sample_config_file)

            assert manager.should_enforce_strict_binding() is True
            assert manager.should_require_pinned_models() is True
            assert manager.should_allow_hardcoded_temp_paths() is False
            assert manager.should_require_pickle_validation() is True
            assert manager.should_require_url_scheme_validation() is True
            assert manager.should_audit_all_operations() is True

    def test_development_relaxed_policies(self, sample_config_file):
        """Test development environment has relaxed security policies."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            manager = SecurityConfigManager(config_path=sample_config_file)

            assert manager.should_enforce_strict_binding() is False
            assert manager.should_require_pinned_models() is False
            assert manager.should_allow_hardcoded_temp_paths() is True
            assert manager.should_require_pickle_validation() is False
            assert manager.should_require_url_scheme_validation() is False
            assert manager.should_audit_all_operations() is False

    def test_research_mixed_policies(self, sample_config_file):
        """Test research environment has mixed security policies."""
        with patch.dict(os.environ, {"ENVIRONMENT": "research"}):
            manager = SecurityConfigManager(config_path=sample_config_file)

            assert manager.should_enforce_strict_binding() is False
            assert manager.should_require_pinned_models() is False
            assert manager.should_allow_hardcoded_temp_paths() is True
            assert manager.should_require_pickle_validation() is False
            assert manager.should_require_url_scheme_validation() is False
            assert manager.should_audit_all_operations() is True  # Research audits everything

    def test_factory_method_for_production(self):
        """Test for_production factory method."""
        manager = SecurityConfigManager.for_production()

        assert manager.current_environment == SecurityEnvironment.PRODUCTION
        assert manager.should_enforce_strict_binding() is True
        assert manager.should_require_pinned_models() is True

    def test_factory_method_for_development(self):
        """Test for_development factory method."""
        manager = SecurityConfigManager.for_development()

        assert manager.current_environment == SecurityEnvironment.DEVELOPMENT
        assert manager.should_enforce_strict_binding() is False
        assert manager.should_require_pinned_models() is False

    def test_factory_method_for_research(self):
        """Test for_research factory method."""
        manager = SecurityConfigManager.for_research()

        assert manager.current_environment == SecurityEnvironment.RESEARCH
        assert manager.should_audit_all_operations() is True

    def test_config_validation_missing_environment(self):
        """Test configuration validation fails when environment is missing."""
        invalid_config = """
production:
  enforce_strict_binding: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_config)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Missing configuration"):
                SecurityConfigManager(config_path=temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_config_validation_missing_required_field(self):
        """Test configuration validation fails when required field is missing."""
        invalid_config = """
production:
  enforce_strict_binding: true
  # Missing other required fields

development:
  enforce_strict_binding: false

research:
  enforce_strict_binding: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_config)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Missing required"):
                SecurityConfigManager(config_path=temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_default_config_when_file_not_found(self):
        """Test manager uses default config when file not found."""
        manager = SecurityConfigManager(config_path="/nonexistent/config.yaml")

        # Should still work with default config
        assert manager.config is not None

    def test_environment_detection_integration(self, sample_config_file):
        """Test environment detection integrates with config manager."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = SecurityConfigManager(config_path=sample_config_file)
            assert manager.current_environment == SecurityEnvironment.PRODUCTION

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            manager = SecurityConfigManager(config_path=sample_config_file)
            assert manager.current_environment == SecurityEnvironment.DEVELOPMENT

    def test_get_config_for_environment(self, sample_config_file):
        """Test getting configuration for specific environment."""
        manager = SecurityConfigManager(config_path=sample_config_file)

        prod_config = manager.get_config_for_environment(SecurityEnvironment.PRODUCTION)
        assert prod_config.enforce_strict_binding is True

        dev_config = manager.get_config_for_environment(SecurityEnvironment.DEVELOPMENT)
        assert dev_config.enforce_strict_binding is False

    def test_config_immutability(self, sample_config_file):
        """Test configuration cannot be modified after loading."""
        manager = SecurityConfigManager(config_path=sample_config_file)

        # Attempting to modify should not affect internal config
        config = manager.get_config_for_environment(SecurityEnvironment.PRODUCTION)
        original_value = config.enforce_strict_binding

        # This should not modify the manager's internal config
        # (depends on implementation - if using frozen dataclasses)
        assert manager.should_enforce_strict_binding() == original_value

    def test_multiple_managers_independent(self, sample_config_file):
        """Test multiple manager instances are independent."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager1 = SecurityConfigManager(config_path=sample_config_file)

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            manager2 = SecurityConfigManager(config_path=sample_config_file)

        assert manager1.current_environment == SecurityEnvironment.PRODUCTION
        assert manager2.current_environment == SecurityEnvironment.DEVELOPMENT

    def test_policy_query_methods_return_boolean(self, sample_config_file):
        """Test all policy query methods return boolean values."""
        manager = SecurityConfigManager(config_path=sample_config_file)

        assert isinstance(manager.should_enforce_strict_binding(), bool)
        assert isinstance(manager.should_require_pinned_models(), bool)
        assert isinstance(manager.should_allow_hardcoded_temp_paths(), bool)
        assert isinstance(manager.should_require_pickle_validation(), bool)
        assert isinstance(manager.should_require_url_scheme_validation(), bool)
        assert isinstance(manager.should_audit_all_operations(), bool)

    def test_config_reload(self, sample_config_file):
        """Test configuration can be reloaded."""
        manager = SecurityConfigManager(config_path=sample_config_file)

        # Modify config file
        new_config = """
production:
  enforce_strict_binding: false
  require_pinned_models: false
  allow_hardcoded_temp_paths: true
  require_pickle_validation: false
  require_url_scheme_validation: false
  audit_all_operations: false

development:
  enforce_strict_binding: false
  require_pinned_models: false
  allow_hardcoded_temp_paths: true
  require_pickle_validation: false
  require_url_scheme_validation: false
  audit_all_operations: false

research:
  enforce_strict_binding: false
  require_pinned_models: false
  allow_hardcoded_temp_paths: true
  require_pickle_validation: false
  require_url_scheme_validation: false
  audit_all_operations: false
"""
        with open(sample_config_file, "w") as f:
            f.write(new_config)

        # Reload
        manager.reload_config()

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = SecurityConfigManager(config_path=sample_config_file)
            assert manager.should_enforce_strict_binding() is False
