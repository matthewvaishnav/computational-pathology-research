"""
Security-related exceptions for the HistoCore framework.

This module defines custom exceptions for security violations and errors.
"""


class SecurityError(Exception):
    """Base exception for security-related errors."""
    pass


class PickleSecurityError(SecurityError):
    """Exception raised when pickle deserialization security check fails."""
    pass


class ValidationError(SecurityError):
    """Exception raised when input validation fails."""
    pass


class AuthenticationError(SecurityError):
    """Exception raised when authentication fails."""
    pass


class AuthorizationError(SecurityError):
    """Exception raised when authorization check fails."""
    pass


class RateLimitError(SecurityError):
    """Exception raised when rate limit is exceeded."""
    pass


class InputSanitizationError(SecurityError):
    """Exception raised when input sanitization fails."""
    pass
