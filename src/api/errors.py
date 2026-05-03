"""
API Error Handlers

Custom error handlers for FastAPI application.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )


async def internal_error_handler(request: Request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


async def validation_error_handler(request: Request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": exc.errors()}
    )


def register_error_handlers(app):
    """Register all error handlers with FastAPI app."""
    app.add_exception_handler(404, not_found_handler)
    app.add_exception_handler(500, internal_error_handler)
