"""
HTTP Status Codes

Enum for HTTP status codes to avoid magic numbers.
"""

from enum import IntEnum


class HTTPStatus(IntEnum):
    """HTTP status codes."""

    # Success
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204

    # Redirection
    MOVED_PERMANENTLY = 301
    FOUND = 302
    NOT_MODIFIED = 304

    # Client Errors
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    GONE = 410
    PAYLOAD_TOO_LARGE = 413
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429

    # Server Errors
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504
