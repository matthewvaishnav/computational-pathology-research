"""
Error Handlers Module

Centralized error handling functions for FastAPI application.
Provides consistent error responses across all endpoints.
"""

import logging
from typing import Any, Dict

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


async def not_found_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle 404 Not Found errors.
    
    Args:
        request: The FastAPI request object
        exc: The HTTP exception
        
    Returns:
        JSON response with error details
    """
    logger.warning(f"404 Not Found: {request.method} {request.url}")
    
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": "The requested endpoint was not found",
            "path": str(request.url.path),
            "method": request.method
        }
    )


async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle 500 Internal Server Error.
    
    Args:
        request: The FastAPI request object
        exc: The exception that occurred
        
    Returns:
        JSON response with error details
    """
    logger.error(f"Internal Server Error: {request.method} {request.url} - {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "path": str(request.url.path),
            "method": request.method
        }
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle request validation errors (422 Unprocessable Entity).
    
    Args:
        request: The FastAPI request object
        exc: The validation error
        
    Returns:
        JSON response with validation error details
    """
    logger.warning(f"Validation Error: {request.method} {request.url} - {exc.errors()}")
    
    # Format validation errors for better readability
    formatted_errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        formatted_errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": "The request data is invalid",
            "path": str(request.url.path),
            "method": request.method,
            "errors": formatted_errors
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle general HTTP exceptions.
    
    Args:
        request: The FastAPI request object
        exc: The HTTP exception
        
    Returns:
        JSON response with error details
    """
    logger.warning(f"HTTP Exception: {request.method} {request.url} - {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": get_error_name(exc.status_code),
            "detail": exc.detail,
            "path": str(request.url.path),
            "method": request.method
        }
    )


def get_error_name(status_code: int) -> str:
    """
    Get human-readable error name from HTTP status code.
    
    Args:
        status_code: HTTP status code
        
    Returns:
        Human-readable error name
    """
    error_names = {
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        409: "Conflict",
        413: "Payload Too Large",
        422: "Unprocessable Entity",
        429: "Too Many Requests",
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout"
    }
    
    return error_names.get(status_code, f"HTTP {status_code}")


def create_error_response(
    status_code: int,
    message: str,
    detail: str = None,
    extra_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response dictionary.
    
    Args:
        status_code: HTTP status code
        message: Error message
        detail: Optional detailed error description
        extra_data: Optional additional data to include
        
    Returns:
        Standardized error response dictionary
    """
    response = {
        "error": get_error_name(status_code),
        "message": message
    }
    
    if detail:
        response["detail"] = detail
    
    if extra_data:
        response.update(extra_data)
    
    return response