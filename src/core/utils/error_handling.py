"""
Secure Error Handling

Prevents information disclosure through error messages.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class SecureErrorResponse:
    """Generate secure error responses."""

    @staticmethod
    def generic_error(status_code: int = 500) -> Dict[str, Any]:
        """Return generic error message.

        Args:
            status_code: HTTP status code

        Returns:
            Generic error response
        """
        messages = {
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            500: "Internal Server Error",
        }

        return {
            "error": messages.get(status_code, "Error"),
            "status_code": status_code,
        }

    @staticmethod
    def safe_error(
        exception: Exception,
        status_code: int = 500,
        include_details: bool = False,
    ) -> Dict[str, Any]:
        """Create safe error response.

        Args:
            exception: Exception that occurred
            status_code: HTTP status code
            include_details: Whether to include details (dev only)

        Returns:
            Safe error response
        """
        # Log full error internally
        logger.error(f"Error occurred: {type(exception).__name__}", exc_info=True)

        # Return safe response to client
        response = SecureErrorResponse.generic_error(status_code)

        # Only include details in development
        if include_details:
            response["details"] = str(exception)
            response["type"] = type(exception).__name__

        return response


def handle_exception_safely(
    exception: Exception,
    log_traceback: bool = True,
) -> Dict[str, Any]:
    """Handle exception safely without exposing internals.

    Args:
        exception: Exception to handle
        log_traceback: Whether to log full traceback

    Returns:
        Safe error response
    """
    # Log internally
    if log_traceback:
        logger.error(f"Exception: {type(exception).__name__}: {str(exception)}", exc_info=True)
    else:
        logger.error(f"Exception: {type(exception).__name__}")

    # Determine status code
    status_code = 500
    if hasattr(exception, "status_code"):
        status_code = exception.status_code

    # Return safe response
    return SecureErrorResponse.generic_error(status_code)


def sanitize_error_message(message: str) -> str:
    """Sanitize error message to remove sensitive info.

    Args:
        message: Error message

    Returns:
        Sanitized message
    """
    # Remove file paths
    import re

    message = re.sub(r"/[^\s]+", "[PATH]", message)

    # Remove IP addresses
    message = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP]", message)

    # Remove potential credentials
    message = re.sub(r"password[=:]\S+", "password=[REDACTED]", message, flags=re.IGNORECASE)
    message = re.sub(r"token[=:]\S+", "token=[REDACTED]", message, flags=re.IGNORECASE)

    return message
