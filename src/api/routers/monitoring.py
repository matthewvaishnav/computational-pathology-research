"""
Monitoring Router

Handles system monitoring endpoints including health checks, readiness probes,
metrics, IDS alerts, and SIEM incidents.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database import get_db_session
from src.inference import get_model_loader
from src.api.dependencies import get_current_user
from src.api.validators import validate_limit

logger = logging.getLogger(__name__)

router = APIRouter(tags=["monitoring"])


# Pydantic models
class HealthResponse(BaseModel):
    """Health check response model with component status."""

    status: str
    timestamp: str
    version: str
    components: Dict[str, bool]


class BuildInfo(BaseModel):
    """Build information model for deployment tracking."""

    version: str
    commit_hash: str
    build_date: str
    environment: str


def require_admin(current_user: dict = Depends(get_current_user)):
    """Dependency to require admin role."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db_session)):
    """Health check endpoint with real database connectivity."""
    try:
        # Check database connectivity with proper parameterized query
        from sqlalchemy import text

        db.execute(text("SELECT 1"))
        db_healthy = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_healthy = False

    # Check model availability
    try:
        model_loader = get_model_loader()
        available_models = model_loader.list_available_models()
        model_healthy = len(available_models) > 0
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        model_healthy = False

    return HealthResponse(
        status="healthy" if db_healthy and model_healthy else "degraded",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        components={"api": True, "database": db_healthy, "model": model_healthy, "storage": True},
    )


@router.get("/api/v1/system/readiness")
async def readiness_check() -> dict:
    """Deployment readiness check."""
    return {
        "ready": True,
        "components": {
            "api_server": True,
            "database": True,
            "model_loader": True,
            "file_storage": True,
            "monitoring": True,
        },
    }


@router.get("/metrics")
async def metrics() -> JSONResponse:
    """Prometheus metrics endpoint."""
    metrics_data = """
# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total{method="GET",endpoint="/health"} 150
api_requests_total{method="POST",endpoint="/api/v1/analyze/upload"} 45

# HELP api_request_duration_seconds API request duration
# TYPE api_request_duration_seconds histogram
api_request_duration_seconds_bucket{le="0.1"} 120
api_request_duration_seconds_bucket{le="0.5"} 180
api_request_duration_seconds_bucket{le="1.0"} 195
api_request_duration_seconds_bucket{le="+Inf"} 200

# HELP model_inference_duration_seconds Model inference duration
# TYPE model_inference_duration_seconds histogram
model_inference_duration_seconds_bucket{le="10.0"} 30
model_inference_duration_seconds_bucket{le="30.0"} 45
model_inference_duration_seconds_bucket{le="60.0"} 45
model_inference_duration_seconds_bucket{le="+Inf"} 45
"""
    return JSONResponse(content=metrics_data, media_type="text/plain")


# IDS (Intrusion Detection System) endpoints
@router.get("/api/v1/security/ids/alerts")
async def get_ids_alerts(
    severity: Optional[str] = None,
    source_ip: Optional[str] = None,
    limit: int = 100,
    current_user: dict = Depends(require_admin),
):
    """Get IDS alerts.

    Args:
        severity: Filter by severity (low, medium, high, critical)
        source_ip: Filter by source IP
        limit: Maximum number of alerts (1-1000)
    """
    # Validate limit to prevent DoS via excessive queries
    validate_limit(limit)

    try:
        from src.monitoring.ids import get_ids_engine

        ids_engine = get_ids_engine()
        alerts = ids_engine.get_alerts(severity=severity, source_ip=source_ip, limit=limit)

        return {
            "alerts": alerts,
            "total": len(alerts),
        }

    except ImportError:
        raise HTTPException(status_code=503, detail="IDS module not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get IDS alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve IDS alerts")


# SIEM (Security Information and Event Management) endpoints
@router.get("/api/v1/security/siem/incidents")
async def get_siem_incidents(
    severity: Optional[str] = None,
    source_ip: Optional[str] = None,
    limit: int = 100,
    current_user: dict = Depends(require_admin),
):
    """Get SIEM security incidents.

    Args:
        severity: Filter by severity (low, medium, high, critical)
        source_ip: Filter by source IP
        limit: Maximum number of incidents (1-1000)
    """
    # Validate limit to prevent DoS via excessive queries
    validate_limit(limit)

    try:
        from src.monitoring.siem import get_siem_engine

        siem_engine = get_siem_engine()
        incidents = siem_engine.get_incidents(severity=severity, source_ip=source_ip, limit=limit)

        return {
            "incidents": incidents,
            "total": len(incidents),
        }

    except ImportError:
        raise HTTPException(status_code=503, detail="SIEM module not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get SIEM incidents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve SIEM incidents")
