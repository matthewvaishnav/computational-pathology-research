"""URL scheme validation wrapper for network fetches."""

from __future__ import annotations

import logging
import urllib.request
from urllib.parse import urlparse

from src.platform.security.exceptions import URLSecurityError

logger = logging.getLogger(__name__)


class URLFetcherControl:
    """Validate URL schemes before opening external resources."""

    def __init__(self, allowed_schemes: list[str] | tuple[str, ...] | None = None) -> None:
        self.allowed_schemes = {scheme.lower() for scheme in (allowed_schemes or ("http", "https"))}

    def validate_url_scheme(self, url: str) -> None:
        if not isinstance(url, str) or not url:
            raise URLSecurityError("URL must be a non-empty string")

        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        if not scheme:
            raise URLSecurityError("URL scheme is required")
        if scheme not in self.allowed_schemes:
            raise URLSecurityError(f"{scheme}:// scheme not allowed")

    def safe_urlopen(self, url: str, *args, **kwargs):
        """Validate the URL and then delegate to urllib.request.urlopen."""
        self.validate_url_scheme(url)
        logger.warning("URL fetch allowed: %s", urlparse(url).scheme.lower())
        return urllib.request.urlopen(url, *args, **kwargs)
