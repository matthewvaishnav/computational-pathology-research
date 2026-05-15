"""
URL Validation and Sanitization

Prevents SSRF and open redirect vulnerabilities.
"""

from typing import Optional
from urllib.parse import urlparse


def validate_url_safe(
    url: str, allowed_schemes: Optional[set] = None, allow_private: bool = False
) -> bool:
    """Validate URL is safe (not SSRF vulnerable).

    Args:
        url: URL to validate
        allowed_schemes: Allowed URL schemes (default: http, https)
        allow_private: Allow private IP ranges

    Returns:
        True if URL is safe

    Raises:
        ValueError: If URL is unsafe
    """
    if allowed_schemes is None:
        allowed_schemes = {"http", "https"}

    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValueError(f"Invalid URL: {e}")

    # Check scheme
    if parsed.scheme not in allowed_schemes:
        raise ValueError(f"URL scheme not allowed: {parsed.scheme}")

    # Check for localhost/private IPs if not allowed
    if not allow_private:
        hostname = parsed.hostname

        if not hostname:
            raise ValueError("URL must have hostname")

        # Check for localhost
        if hostname.lower() in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Localhost URLs not allowed")

        # Check for private IP ranges
        if is_private_ip(hostname):
            raise ValueError(f"Private IP address not allowed: {hostname}")

        # Check for link-local
        if hostname.startswith("169.254."):
            raise ValueError("Link-local address not allowed")

    return True


def is_private_ip(hostname: str) -> bool:
    """Check if hostname is a private IP address.

    Args:
        hostname: Hostname or IP address

    Returns:
        True if private IP
    """
    import ipaddress

    try:
        ip = ipaddress.ip_address(hostname)
        return ip.is_private
    except ValueError:
        # Not an IP address, assume hostname is OK
        return False


def sanitize_redirect_url(url: str, allowed_domains: Optional[set] = None) -> str:
    """Sanitize redirect URL to prevent open redirect.

    Args:
        url: Redirect URL
        allowed_domains: Set of allowed domains

    Returns:
        Sanitized URL

    Raises:
        ValueError: If URL not allowed
    """
    parsed = urlparse(url)

    # Only allow relative URLs or specific domains
    if parsed.netloc:
        if allowed_domains and parsed.netloc not in allowed_domains:
            raise ValueError(f"Redirect to {parsed.netloc} not allowed")

    # Prevent javascript: and data: URLs
    if parsed.scheme and parsed.scheme not in ("http", "https", ""):
        raise ValueError(f"Redirect scheme not allowed: {parsed.scheme}")

    return url
