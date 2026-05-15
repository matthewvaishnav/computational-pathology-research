"""
Log Sanitization

Prevents log injection and sensitive data leakage in logs.
"""

import re
from typing import Any


def sanitize_for_log(value: Any) -> str:
    """Sanitize value for safe logging.

    Prevents:
    - Log injection (newlines, control characters)
    - Sensitive data leakage

    Args:
        value: Value to sanitize

    Returns:
        Sanitized string safe for logging
    """
    if value is None:
        return "None"

    # Convert to string
    text = str(value)

    # Remove newlines and carriage returns (log injection)
    text = text.replace("\n", " ").replace("\r", " ")

    # Remove other control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

    # Truncate if too long
    max_length = 1000
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return text


def mask_sensitive_data(text: str) -> str:
    """Mask sensitive data in logs.

    Args:
        text: Text that may contain sensitive data

    Returns:
        Text with sensitive data masked
    """
    # Mask email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "***@***.***", text)

    # Mask credit card numbers
    text = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "****-****-****-****", text)

    # Mask SSN
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "***-**-****", text)

    # Mask API keys (common patterns)
    text = re.sub(r"\b[A-Za-z0-9]{32,}\b", "***API_KEY***", text)

    # Mask JWT tokens
    text = re.sub(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "***JWT_TOKEN***", text)

    return text


def safe_log_format(message: str, **kwargs) -> str:
    """Format log message safely.

    Args:
        message: Log message template
        **kwargs: Values to include in log

    Returns:
        Safely formatted log message
    """
    # Sanitize all values
    safe_kwargs = {k: sanitize_for_log(v) for k, v in kwargs.items()}

    # Format message
    formatted = message.format(**safe_kwargs)

    # Mask sensitive data
    formatted = mask_sensitive_data(formatted)

    return formatted
