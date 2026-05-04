"""
API Middleware Components

Security and request handling middleware for the FastAPI application.
"""

import asyncio
import logging
import os

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse

from src.api.security import get_security_headers, log_security_event

logger = logging.getLogger(__name__)


async def https_redirect_middleware(request: Request, call_next):
    """Redirect HTTP to HTTPS in production."""
    if (
        request.url.scheme != "https"
        and os.getenv("ENVIRONMENT") == "production"
        and not request.url.path.startswith("/health")
    ):
        # Validate hostname to prevent open redirect attacks
        allowed_hosts = os.getenv("ALLOWED_HOSTS", "").split(",")
        if not allowed_hosts or not allowed_hosts[0]:
            logger.error("ALLOWED_HOSTS not configured for production")
            return JSONResponse(status_code=500, content={"error": "Server misconfiguration"})

        # Parse and validate host
        host = request.headers.get("host", "").split(":")[0]  # Remove port

        # Reject if host contains @ (credential injection)
        if "@" in host:
            log_security_event(
                "open_redirect_attempt",
                ip_address=request.client.host,
                details=f"Host contains @: {host}",
                success=False,
            )
            return JSONResponse(status_code=400, content={"error": "Invalid host"})

        # Validate host is in allowed list
        if host not in allowed_hosts:
            log_security_event(
                "open_redirect_attempt",
                ip_address=request.client.host,
                details=f"Host not in allowed list: {host}",
                success=False,
            )
            return JSONResponse(status_code=400, content={"error": "Invalid host"})

        # Build safe HTTPS URL
        url = f"https://{host}{request.url.path}"
        if request.url.query:
            url += f"?{request.url.query}"

        return RedirectResponse(url=url, status_code=301)
    return await call_next(request)


async def request_size_and_timeout_middleware(request: Request, call_next):
    """Add size limit and timeout to all requests to prevent DoS attacks."""
    # Check content-length header for size limit (10MB for non-upload endpoints)
    content_length = request.headers.get("content-length")
    if content_length:
        size_mb = int(content_length) / (1024 * 1024)
        # Allow larger uploads only for specific endpoints
        if request.url.path.startswith("/api/v1/analyze/upload"):
            max_size_mb = 100
        else:
            max_size_mb = 10

        if size_mb > max_size_mb:
            log_security_event(
                "request_too_large",
                ip_address=request.client.host,
                details=f"Size: {size_mb:.1f}MB, Max: {max_size_mb}MB",
                success=False,
            )
            return JSONResponse(
                status_code=413,
                content={"error": f"Request too large. Maximum size is {max_size_mb}MB"},
            )

    # Add timeout to prevent slowloris attacks
    try:
        return await asyncio.wait_for(call_next(request), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning(f"Request timeout: {request.url.path}")
        log_security_event(
            "request_timeout",
            ip_address=request.client.host,
            details=f"Path: {request.url.path}",
            success=False,
        )
        return JSONResponse(status_code=504, content={"detail": "Request timeout"})


async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)

    # Add all security headers
    security_headers = get_security_headers()
    for header, value in security_headers.items():
        response.headers[header] = value

    return response
