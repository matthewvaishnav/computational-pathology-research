"""
Unit tests for URLFetcherControl.

Tests URL scheme validation for safe URL opening.
"""

import pytest
from unittest.mock import patch, MagicMock
from urllib.error import URLError

from src.security.url_fetcher_control import URLFetcherControl
from src.security.exceptions import URLSecurityError


class TestURLFetcherControl:
    """Test URLFetcherControl functionality."""

    def test_file_urls_blocked(self):
        """Test file:// URLs are blocked."""
        control = URLFetcherControl()

        with pytest.raises(URLSecurityError, match="file:// scheme not allowed"):
            control.safe_urlopen("file:///etc/passwd")

    def test_http_urls_allowed(self):
        """Test http:// URLs are allowed."""
        control = URLFetcherControl()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()

            control.safe_urlopen("http://example.com/data")
            mock_urlopen.assert_called_once()

    def test_https_urls_allowed(self):
        """Test https:// URLs are allowed."""
        control = URLFetcherControl()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()

            control.safe_urlopen("https://example.com/data")
            mock_urlopen.assert_called_once()

    def test_invalid_schemes_rejected(self):
        """Test invalid URL schemes are rejected."""
        control = URLFetcherControl()

        invalid_urls = [
            "ftp://example.com/file",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "gopher://example.com",
        ]

        for url in invalid_urls:
            with pytest.raises(URLSecurityError, match="scheme not allowed"):
                control.safe_urlopen(url)

    def test_validate_url_scheme_success(self):
        """Test validate_url_scheme succeeds for valid schemes."""
        control = URLFetcherControl()

        # Should not raise
        control.validate_url_scheme("http://example.com")
        control.validate_url_scheme("https://example.com")

    def test_validate_url_scheme_failure(self):
        """Test validate_url_scheme fails for invalid schemes."""
        control = URLFetcherControl()

        with pytest.raises(URLSecurityError):
            control.validate_url_scheme("file:///etc/passwd")

    def test_audit_logging_for_url_operations(self, caplog):
        """Test audit logging for URL operations."""
        control = URLFetcherControl()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()

            control.safe_urlopen("https://example.com/data")

            assert "URL fetch" in caplog.text or "url" in caplog.text.lower()

    def test_url_without_scheme_rejected(self):
        """Test URLs without scheme are rejected."""
        control = URLFetcherControl()

        with pytest.raises(URLSecurityError):
            control.safe_urlopen("example.com/data")

    def test_empty_url_rejected(self):
        """Test empty URLs are rejected."""
        control = URLFetcherControl()

        with pytest.raises(URLSecurityError):
            control.safe_urlopen("")

    def test_none_url_rejected(self):
        """Test None URLs are rejected."""
        control = URLFetcherControl()

        with pytest.raises((URLSecurityError, TypeError)):
            control.safe_urlopen(None)

    def test_url_with_credentials_allowed(self):
        """Test URLs with credentials are allowed (but logged)."""
        control = URLFetcherControl()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()

            # Should work but may log warning
            control.safe_urlopen("https://user:pass@example.com/data")
            mock_urlopen.assert_called_once()

    def test_url_with_port_allowed(self):
        """Test URLs with custom ports are allowed."""
        control = URLFetcherControl()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()

            control.safe_urlopen("https://example.com:8443/data")
            mock_urlopen.assert_called_once()

    def test_url_with_query_parameters_allowed(self):
        """Test URLs with query parameters are allowed."""
        control = URLFetcherControl()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()

            control.safe_urlopen("https://example.com/data?key=value&foo=bar")
            mock_urlopen.assert_called_once()

    def test_url_with_fragment_allowed(self):
        """Test URLs with fragments are allowed."""
        control = URLFetcherControl()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()

            control.safe_urlopen("https://example.com/data#section")
            mock_urlopen.assert_called_once()

    def test_case_insensitive_scheme_validation(self):
        """Test scheme validation is case-insensitive."""
        control = URLFetcherControl()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()

            # All should work
            control.safe_urlopen("HTTP://example.com")
            control.safe_urlopen("HTTPS://example.com")
            control.safe_urlopen("HtTpS://example.com")

    def test_file_scheme_variations_blocked(self):
        """Test various file:// scheme variations are blocked."""
        control = URLFetcherControl()

        file_urls = [
            "file:///etc/passwd",
            "FILE:///etc/passwd",
            "file://localhost/etc/passwd",
            "file:///C:/Windows/System32/config/sam",
        ]

        for url in file_urls:
            with pytest.raises(URLSecurityError):
                control.safe_urlopen(url)

    def test_safe_urlopen_passes_through_kwargs(self):
        """Test safe_urlopen passes through additional kwargs."""
        control = URLFetcherControl()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()

            control.safe_urlopen("https://example.com", timeout=30, data=b"test")

            # Check kwargs were passed
            call_args = mock_urlopen.call_args
            assert call_args.kwargs.get("timeout") == 30
            assert call_args.kwargs.get("data") == b"test"

    def test_safe_urlopen_returns_response(self):
        """Test safe_urlopen returns the response object."""
        control = URLFetcherControl()

        mock_response = MagicMock()
        with patch("urllib.request.urlopen", return_value=mock_response):
            response = control.safe_urlopen("https://example.com")
            assert response == mock_response

    def test_network_errors_propagated(self):
        """Test network errors are propagated correctly."""
        control = URLFetcherControl()

        with patch("urllib.request.urlopen", side_effect=URLError("Network error")):
            with pytest.raises(URLError):
                control.safe_urlopen("https://example.com")

    def test_allowed_schemes_configurable(self):
        """Test allowed schemes can be configured."""
        control = URLFetcherControl(allowed_schemes=["https"])

        # HTTPS should work
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            control.safe_urlopen("https://example.com")

        # HTTP should be blocked
        with pytest.raises(URLSecurityError):
            control.safe_urlopen("http://example.com")

    def test_default_allowed_schemes(self):
        """Test default allowed schemes are http and https."""
        control = URLFetcherControl()

        assert "http" in control.allowed_schemes
        assert "https" in control.allowed_schemes
        assert "file" not in control.allowed_schemes

    def test_url_validation_before_opening(self):
        """Test URL is validated before attempting to open."""
        control = URLFetcherControl()

        # Mock urlopen to track if it's called
        with patch("urllib.request.urlopen") as mock_urlopen:
            try:
                control.safe_urlopen("file:///etc/passwd")
            except URLSecurityError:
                pass

            # urlopen should never be called for invalid schemes
            mock_urlopen.assert_not_called()
