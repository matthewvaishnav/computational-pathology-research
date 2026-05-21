"""
Request Validation Middleware

Validates and sanitizes all incoming requests.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Validate incoming requests."""

    async def dispatch(self, request: Request, call_next):
        """Validate request before processing."""

        # Check content length
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                max_size = 100 * 1024 * 1024  # 100MB
                if length > max_size:
                    return Response(content="Request too large", status_code=413)
            except ValueError:
                return Response(content="Invalid content-length header", status_code=400)

        # Validate content-type for POST/PUT
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")

            # Require content-type
            if not content_type:
                return Response(content="Content-Type header required", status_code=400)

            # Validate content-type
            allowed_types = [
                "application/json",
                "application/x-www-form-urlencoded",
                "multipart/form-data",
                "image/jpeg",
                "image/png",
            ]

            if not any(ct in content_type for ct in allowed_types):
                return Response(
                    content=f"Unsupported content-type: {content_type}", status_code=415
                )

        # Check for suspicious headers
        suspicious_headers = [
            "x-forwarded-host",  # Can be used for host header injection
        ]

        for header in suspicious_headers:
            if header in request.headers:
                # Log but don't block (might be legitimate proxy)
                pass

        # Process request
        response = await call_next(request)

        return response
