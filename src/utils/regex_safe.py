"""
Regex DoS Protection

Prevents ReDoS (Regular Expression Denial of Service) attacks.
"""

import re
import signal
from contextlib import contextmanager
from typing import Optional, Pattern


class RegexTimeout(Exception):
    """Raised when regex execution times out."""

    pass


@contextmanager
def timeout_context(seconds: int):
    """Context manager for timeout.

    Args:
        seconds: Timeout in seconds
    """

    def timeout_handler(signum, frame):
        raise RegexTimeout(f"Regex execution timed out after {seconds} seconds")

    # Set alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def safe_regex_match(pattern: str, text: str, timeout: int = 1) -> Optional[re.Match]:
    """Match regex with timeout protection.

    Args:
        pattern: Regex pattern
        text: Text to match
        timeout: Timeout in seconds

    Returns:
        Match object or None

    Raises:
        RegexTimeout: If regex takes too long
    """
    try:
        with timeout_context(timeout):
            return re.match(pattern, text)
    except RegexTimeout:
        raise


def safe_regex_search(pattern: str, text: str, timeout: int = 1) -> Optional[re.Match]:
    """Search regex with timeout protection.

    Args:
        pattern: Regex pattern
        text: Text to search
        timeout: Timeout in seconds

    Returns:
        Match object or None

    Raises:
        RegexTimeout: If regex takes too long
    """
    try:
        with timeout_context(timeout):
            return re.search(pattern, text)
    except RegexTimeout:
        raise


def validate_regex_safe(pattern: str) -> bool:
    """Check if regex pattern is safe (not vulnerable to ReDoS).

    Checks for common ReDoS patterns:
    - Nested quantifiers: (a+)+
    - Overlapping alternation: (a|a)+
    - Excessive backtracking

    Args:
        pattern: Regex pattern to validate

    Returns:
        True if pattern appears safe
    """
    # Check for nested quantifiers
    if re.search(r"\([^)]*[+*]\)[+*]", pattern):
        return False

    # Check for nested groups with quantifiers
    if re.search(r"\([^)]*\([^)]*[+*]\)[^)]*\)[+*]", pattern):
        return False

    # Check for alternation with quantifiers
    if re.search(r"\([^)]*\|[^)]*\)[+*]", pattern):
        return False

    return True
