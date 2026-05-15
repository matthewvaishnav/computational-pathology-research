"""
Admin Router

Handles administrative endpoints including user management, system configuration,
audit logs, and reporting.
"""

import logging
import uuid
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.dependencies import get_current_user
from src.api.validators import validate_limit
from src.database import AuditOperations, UserOperations, get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

# Temporary in-memory user storage (will be replaced with database)
users: Dict[str, Dict] = {}


class ReportRequest(BaseModel):
    """Admin report generation request model."""

    report_type: str
    parameters: Optional[Dict] = None


def require_admin(current_user: dict = Depends(get_current_user)):
    """Dependency to require admin role."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@router.get("/users")
async def get_users(
    limit: int = 10,
    current_user: dict = Depends(require_admin),
):
    """Get user list (admin only).

    Args:
        limit: Maximum number of users (1-1000)
    """
    # Validate limit to prevent DoS via excessive queries
    validate_limit(limit)

    user_list = list(users.values())
    return {"users": user_list[:limit], "total": len(user_list)}


@router.get("/config")
async def get_system_config(
    current_user: dict = Depends(require_admin),
):
    """Get system configuration (admin only)."""
    return {
        "max_file_size_mb": 100,
        "supported_formats": ["PNG", "JPEG", "TIFF", "DICOM"],
        "model_version": "1.0.0",
        "inference_timeout": 60,
    }


@router.get("/audit-logs")
async def get_audit_logs(
    limit: int = 10,
    current_user: dict = Depends(require_admin),
    db: Session = Depends(get_db_session),
):
    """Get audit logs (admin only).

    Args:
        limit: Maximum number of logs (1-1000)
    """
    # Validate limit to prevent DoS via excessive queries
    validate_limit(limit)

    try:
        audit_ops = AuditOperations(db)
        logs = audit_ops.list_audit_logs(limit=limit)

        log_list = []
        for log in logs:
            log_dict = {
                "id": str(log.id),
                "event_type": log.event_type,
                "username": log.username,
                "ip_address": log.ip_address,
                "details": log.details,
                "success": log.success,
                "timestamp": log.timestamp.isoformat(),
            }
            log_list.append(log_dict)

        return {"logs": log_list, "total": len(log_list)}

    except Exception as e:
        logger.error(f"Failed to get audit logs: {e}")
        # Return empty logs if audit operations not available
        return {"logs": [], "total": 0}


@router.post("/reports/generate")
async def generate_report(
    report_data: ReportRequest,
    current_user: dict = Depends(require_admin),
):
    """Generate report (admin only)."""
    report_id = str(uuid.uuid4())
    return {"report_id": report_id, "status": "generating"}


@router.get("/reports/{report_id}/status")
async def get_report_status(
    report_id: str,
    current_user: dict = Depends(require_admin),
):
    """Get report generation status (admin only)."""
    return {
        "report_id": report_id,
        "status": "completed",
        "download_url": f"/reports/{report_id}.pdf",
    }
