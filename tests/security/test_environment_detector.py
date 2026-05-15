"""
Unit tests for SecurityEnvironmentDetector.

Tests environment variable detection, default behavior, and validation.
"""

import os
from unittest.mock import patch

import pytest

from src.security.environment_detector import SecurityEnvironmentDetector
from src.security.models import SecurityEnvironment


class TestEnvironmentDetector:
    """Test environment detection logic."""

    def test_detect_production_from_environment_var(self):
        """Test production environment detection from ENVIRONMENT variable."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            detector = SecurityEnvironmentDetector()
            env = detector.detect()
            assert env == SecurityEnvironment.PRODUCTION

    def test_detect_production_from_deployment_env_var(self):
        """Test production environment detection from DEPLOYMENT_ENV variable."""
        with patch.dict(os.environ, {"DEPLOYMENT_ENV": "production"}):
            detector = SecurityEnvironmentDetector()
            env = detector.detect()
            assert env == SecurityEnvironment.PRODUCTION

    def test_detect_development_from_environment_var(self):
        """Test development environment detection from ENVIRONMENT variable."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            detector = SecurityEnvironmentDetector()
            env = detector.detect()
            assert env == SecurityEnvironment.DEVELOPMENT

    def test_detect_research_from_environment_var(self):
        """Test research environment detection from ENVIRONMENT variable."""
        with patch.dict(os.environ, {"ENVIRONMENT": "research"}):
            detector = SecurityEnvironmentDetector()
            env = detector.detect()
            assert env == SecurityEnvironment.RESEARCH

    def test_default_to_development_with_warning(self, caplog):
        """Test default to development mode when no environment variable set."""
        with patch.dict(os.environ, {}, clear=True):
            detector = SecurityEnvironmentDetector()
            env = detector.detect()

            assert env == SecurityEnvironment.DEVELOPMENT
            assert "No environment specified" in caplog.text
            assert "defaulting to DEVELOPMENT" in caplog.text

    def test_case_insensitive_environment_detection(self):
        """Test environment detection is case-insensitive."""
        test_cases = [
            ("PRODUCTION", SecurityEnvironment.PRODUCTION),
            ("Production", SecurityEnvironment.PRODUCTION),
            ("production", SecurityEnvironment.PRODUCTION),
            ("DEVELOPMENT", SecurityEnvironment.DEVELOPMENT),
            ("Development", SecurityEnvironment.DEVELOPMENT),
            ("development", SecurityEnvironment.DEVELOPMENT),
            ("RESEARCH", SecurityEnvironment.RESEARCH),
            ("Research", SecurityEnvironment.RESEARCH),
            ("research", SecurityEnvironment.RESEARCH),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"ENVIRONMENT": env_value}):
                detector = SecurityEnvironmentDetector()
                env = detector.detect()
                assert env == expected, f"Failed for {env_value}"

    def test_invalid_environment_raises_error(self):
        """Test invalid environment value raises ValueError."""
        with patch.dict(os.environ, {"ENVIRONMENT": "invalid_env"}):
            detector = SecurityEnvironmentDetector()
            with pytest.raises(ValueError, match="Invalid environment"):
                detector.detect()

    def test_environment_priority_environment_over_deployment_env(self):
        """Test ENVIRONMENT variable takes priority over DEPLOYMENT_ENV."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "DEPLOYMENT_ENV": "development"}):
            detector = SecurityEnvironmentDetector()
            env = detector.detect()
            assert env == SecurityEnvironment.PRODUCTION

    def test_deployment_env_used_when_environment_not_set(self):
        """Test DEPLOYMENT_ENV is used when ENVIRONMENT is not set."""
        with patch.dict(os.environ, {"DEPLOYMENT_ENV": "research"}, clear=True):
            detector = SecurityEnvironmentDetector()
            env = detector.detect()
            assert env == SecurityEnvironment.RESEARCH

    def test_whitespace_handling(self):
        """Test environment detection handles whitespace correctly."""
        test_cases = [
            " production ",
            "  development  ",
            "\tresearch\t",
            "\nproduction\n",
        ]

        expected = [
            SecurityEnvironment.PRODUCTION,
            SecurityEnvironment.DEVELOPMENT,
            SecurityEnvironment.RESEARCH,
            SecurityEnvironment.PRODUCTION,
        ]

        for env_value, expected_env in zip(test_cases, expected):
            with patch.dict(os.environ, {"ENVIRONMENT": env_value}):
                detector = SecurityEnvironmentDetector()
                env = detector.detect()
                assert env == expected_env

    def test_empty_string_defaults_to_development(self, caplog):
        """Test empty string environment variable defaults to development."""
        with patch.dict(os.environ, {"ENVIRONMENT": ""}):
            detector = SecurityEnvironmentDetector()
            env = detector.detect()

            assert env == SecurityEnvironment.DEVELOPMENT
            assert "No environment specified" in caplog.text

    def test_detector_caches_result(self):
        """Test detector caches environment detection result."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            detector = SecurityEnvironmentDetector()

            # First call
            env1 = detector.detect()

            # Change environment variable
            os.environ["ENVIRONMENT"] = "development"

            # Second call should return cached result
            env2 = detector.detect()

            assert env1 == env2 == SecurityEnvironment.PRODUCTION

    def test_multiple_detectors_independent(self):
        """Test multiple detector instances are independent."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            detector1 = SecurityEnvironmentDetector()
            env1 = detector1.detect()

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            detector2 = SecurityEnvironmentDetector()
            env2 = detector2.detect()

        assert env1 == SecurityEnvironment.PRODUCTION
        assert env2 == SecurityEnvironment.DEVELOPMENT
