"""
Tests for hypothesis generator API key validation.
"""

import os
import pytest
from src.hypothesis.generator import HypothesisGenerator


def test_hypothesis_generator_requires_api_key():
    """Test that HypothesisGenerator raises ValueError when API key is missing."""
    # Ensure no API key in environment
    old_key = os.environ.pop("ANTHROPIC_API_KEY", None)

    try:
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY must be provided"):
            HypothesisGenerator()
    finally:
        # Restore original key if it existed
        if old_key:
            os.environ["ANTHROPIC_API_KEY"] = old_key


def test_hypothesis_generator_accepts_api_key_parameter():
    """Test that HypothesisGenerator accepts API key via parameter."""
    # Should not raise
    generator = HypothesisGenerator(api_key="test-key-123")
    assert generator.api_key == "test-key-123"


def test_hypothesis_generator_reads_env_variable():
    """Test that HypothesisGenerator reads API key from environment."""
    old_key = os.environ.get("ANTHROPIC_API_KEY")

    try:
        os.environ["ANTHROPIC_API_KEY"] = "env-test-key"
        generator = HypothesisGenerator()
        assert generator.api_key == "env-test-key"
    finally:
        # Restore original state
        if old_key:
            os.environ["ANTHROPIC_API_KEY"] = old_key
        else:
            os.environ.pop("ANTHROPIC_API_KEY", None)


def test_hypothesis_generator_parameter_overrides_env():
    """Test that explicit API key parameter overrides environment variable."""
    old_key = os.environ.get("ANTHROPIC_API_KEY")

    try:
        os.environ["ANTHROPIC_API_KEY"] = "env-key"
        generator = HypothesisGenerator(api_key="param-key")
        assert generator.api_key == "param-key"
    finally:
        # Restore original state
        if old_key:
            os.environ["ANTHROPIC_API_KEY"] = old_key
        else:
            os.environ.pop("ANTHROPIC_API_KEY", None)
