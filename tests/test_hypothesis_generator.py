"""
Tests for hypothesis generator.
"""

import os
from unittest.mock import Mock, patch

import pytest
import requests

from src.features.research.testing.generator import HypothesisGenerator, create_ollama_llm


def test_ollama_timeout_parameter():
    """Test that create_ollama_llm passes timeout to requests.post()."""
    llm_fn = create_ollama_llm(timeout=60)

    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"message": {"content": "test response"}}

        llm_fn("system prompt", "user prompt")

        # Verify timeout passed correctly
        assert mock_post.call_args[1]["timeout"] == 60


def test_ollama_timeout_default():
    """Test that create_ollama_llm uses 30s default timeout."""
    llm_fn = create_ollama_llm()

    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"message": {"content": "test response"}}

        llm_fn("system prompt", "user prompt")

        # Verify default timeout
        assert mock_post.call_args[1]["timeout"] == 30


def test_ollama_timeout_exception():
    """Test that timeout exception is raised when request times out."""
    llm_fn = create_ollama_llm(timeout=1)

    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        with pytest.raises(requests.exceptions.Timeout):
            llm_fn("system prompt", "user prompt")


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
