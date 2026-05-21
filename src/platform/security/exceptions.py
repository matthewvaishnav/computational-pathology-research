"""
Security-related exceptions for the HistoCore framework.

This module defines custom exceptions for security violations and errors.
"""


class SecurityError(Exception):
    """Base exception for security-related errors."""



class PickleSecurityError(SecurityError):
    """Exception raised when pickle deserialization security check fails."""



class ValidationError(SecurityError):
    """Exception raised when input validation fails."""



class AuthenticationError(SecurityError):
    """Exception raised when authentication fails."""



class AuthorizationError(SecurityError):
    """Exception raised when authorization check fails."""



class RateLimitError(SecurityError):
    """Exception raised when rate limit is exceeded."""



class InputSanitizationError(SecurityError):
    """Exception raised when input sanitization fails."""

