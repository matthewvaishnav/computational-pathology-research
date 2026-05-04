#!/usr/bin/env python3
"""
Medical AI Platform - Production API Server

FastAPI-based REST API server for the Medical AI platform providing endpoints
for image analysis, DICOM integration, case management, and system monitoring.

This is the PRODUCTION version with real database and model inference.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import database components
from src.database import initialize_database

# Import tracing
from src.monitoring.tracing import get_tracer

# Import security utilities
from src.api.security import (
    get_security_headers,
    limiter,
    log_security_event,
    validate_security_configuration,
)

# Import routers
from src.api.routers import admin, analysis, auth, mobile, monitoring

# Import dependencies
from src.api.dependencies import get_inference_engine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI(
    title="Medical AI Platform API",
    description="Production REST API for Medical AI pathology analysis platform with real database and model inference",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Track application start time for uptime monitoring
app.state.start_time = time.time()

# Add CORS middleware with environment-specific origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=3600,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add WAF middleware
from src.api.waf import create_waf_middleware

waf_middleware = create_waf_middleware(app)
app.middleware("http")(waf_middleware)

# Include routers
app.include_router(auth.router)
app.include_router(analysis.router)
app.include_router(admin.router)
app.include_router(mobile.router)
app.include_router(monitoring.router)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database and models on startup."""
    try:
        # Validate security configuration
        validate_security_configuration()

        # Initialize database
        initialize_database()
        logger.info("Database initialized successfully")

        # Initialize inference engine
        get_inference_engine()
        logger.info("Inference engine initialized successfully")

        # Initialize distributed tracing
        tracer = get_tracer("histocore-api")
        tracer.initialize(
            jaeger_endpoint=os.getenv("JAEGER_ENDPOINT"),
            otlp_endpoint=os.getenv("OTLP_ENDPOINT"),
            service_version="2.0.0",
            environment=os.getenv("ENVIRONMENT", "development"),
        )
        tracer.instrument_fastapi(app)
        logger.info("Distributed tracing initialized successfully")

        log_security_event("system_startup", details="API server started", success=True)

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        log_security_event("system_startup", details=f"Startup failed: {e}", success=False)
        raise


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"detail": "Endpoint not found"})


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def main():
    """Run the API server."""
    # Add middleware
    from src.api.middleware import (
        https_redirect_middleware,
        request_size_and_timeout_middleware,
        security_headers_middleware,
    )

    app.middleware("http")(https_redirect_middleware)
    app.middleware("http")(request_size_and_timeout_middleware)
    app.middleware("http")(security_headers_middleware)

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
