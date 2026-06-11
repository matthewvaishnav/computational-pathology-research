"""Minimal API surface for automated baseline security scanning.

This app intentionally avoids database, model, and tracing startup. It exercises
FastAPI routing, CORS policy, request identifiers, and common security headers so
OWASP ZAP can scan a deterministic HTTP target in CI.
"""

from __future__ import annotations

import os
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a stable request identifier to every response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Apply baseline browser-facing security headers."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'"
        )
        return response


app = FastAPI(
    title="Computational Pathology API Security Scan",
    description="Deterministic CI target for baseline dynamic security checks.",
    version="0.1.0",
)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)


@app.get("/")
async def root() -> dict[str, str]:
    """Return service metadata without initializing external resources."""
    return {"service": "computational-pathology-research", "mode": "security-scan"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Readiness endpoint used by the security workflow."""
    return {"status": "healthy", "mode": "security-scan"}
