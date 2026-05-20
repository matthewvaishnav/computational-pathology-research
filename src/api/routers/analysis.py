"""
Analysis Router

Handles image upload, analysis results, DICOM processing, and case management.
"""

import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from src.api.dependencies import get_current_user, get_inference_engine
from src.api.security import limiter, log_security_event, sanitize_for_log
from src.api.validators import validate_file_upload, validate_limit
from src.database import (
    AnalysisOperations,
    CaseOperations,
    get_db_session,
)
from src.platform.database.models import Case

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analysis"])
limiter = Limiter(key_func=get_remote_address)


# Pydantic models
class AnalysisRequest(BaseModel):
    """Analysis request model for pathology case processing."""

    case_id: Optional[str] = None
    priority: str = "normal"
    case_type: str = "breast_cancer_screening"


class CaseData(BaseModel):
    """Case data model for pathology analysis."""

    patient_id: str
    study_id: str
    priority: str = "normal"
    case_type: str = "breast_cancer_screening"


class CaseStatusUpdate(BaseModel):
    """Case status update model for tracking analysis progress."""

    status: str
    notes: Optional[str] = None


# Analysis endpoints
@router.post("/analyze/upload")
@limiter.limit("10/minute")
async def upload_for_analysis(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    request_data: AnalysisRequest = AnalysisRequest(),
    request: Request = None,
    db: Session = Depends(get_db_session),
    current_user: dict = Depends(get_current_user),
):
    """Upload image for AI analysis with comprehensive security validation."""

    try:
        # Enforce size limit before reading entire file into memory (DoS prevention)
        max_size = 100 * 1024 * 1024  # 100MB
        content_length = request.headers.get("content-length") if request else None
        if content_length and int(content_length) > max_size:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 100MB")

        # Read file content (bounded by max_size)
        file_content = await file.read(max_size + 1)
        if len(file_content) > max_size:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 100MB")

        # Comprehensive file validation using centralized validator
        detected_mime, safe_filename = validate_file_upload(file_content, file.filename)
        detected_type = detected_mime  # Alias for consistency

        file_size = len(file_content)

        # Create secure temporary file with restricted permissions
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp", prefix="medical_ai_", dir=tempfile.gettempdir()
        )

        try:
            # Set restrictive permissions (owner read/write only)
            os.chmod(temp_path, 0o600)

            # Write content atomically
            with os.fdopen(fd, "wb") as f:
                f.write(file_content)
                f.flush()
                os.fsync(f.fileno())

            # Create analysis record in database
            analysis_ops = AnalysisOperations(db)
            analysis = analysis_ops.create_analysis(
                filename=safe_filename,
                content_type=detected_type,
                file_size=file_size,
                file_path=temp_path,
                case_id=uuid.UUID(request_data.case_id) if request_data.case_id else None,
            )

            # Commit to database
            db.commit()

            # Start background processing with real inference
            # Pass only file path to avoid keeping file content in memory
            background_tasks.add_task(process_real_analysis, str(analysis.id), temp_path)

            logger.info(
                f"Analysis created: {analysis.id} for file {sanitize_for_log(safe_filename)}"
            )

            log_security_event(
                "file_upload",
                ip_address=request.client.host if request else None,
                details=f"File: {safe_filename}, Size: {file_size}, Type: {detected_type}",
                success=True,
            )

            return {"analysis_id": str(analysis.id), "status": "queued"}

        except Exception as e:
            # Clean up temporary file on error
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temp file: {cleanup_error}")
            raise e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        db.rollback()
        log_security_event(
            "file_upload_failed",
            ip_address=request.client.host if request else None,
            details=str(e),
            success=False,
        )
        raise HTTPException(status_code=500, detail="Upload failed. Please try again.")


async def process_real_analysis(analysis_id: str, file_path: str) -> None:
    """Background task to process analysis with real AI model."""

    # Get database session for background task
    from src.platform.database.connection import get_database_manager

    db_manager = get_database_manager()

    with db_manager.get_session() as db:
        try:
            analysis_ops = AnalysisOperations(db)

            # Update status to in_progress
            analysis_ops.update_analysis_status(uuid.UUID(analysis_id), "in_progress")
            db.commit()

            # Get inference engine
            engine = get_inference_engine()

            # Read file content for inference
            with open(file_path, "rb") as f:
                file_content = f.read()

            # Run real model inference
            result = engine.analyze_image_bytes(
                image_bytes=file_content,
                filename=Path(file_path).name,
                disease_type="breast_cancer",
            )

            # Update analysis with results
            analysis_ops.update_analysis_status(
                uuid.UUID(analysis_id),
                "completed",
                processing_time_ms=result.processing_time_ms,
                model_version=result.model_version,
            )

            # Add model result
            analysis_ops.add_model_result(
                analysis_id=uuid.UUID(analysis_id),
                prediction_class=result.prediction_class,
                confidence_score=result.confidence_score,
                model_name=result.model_name,
                model_version=result.model_version,
                probability_scores=result.probability_scores,
                uncertainty_score=result.uncertainty_score,
            )

            db.commit()

            logger.info(
                f"Analysis {analysis_id} completed: {result.prediction_class} ({result.confidence_score:.3f})"
            )

        except Exception as e:
            logger.error(f"Analysis processing failed for {analysis_id}: {e}")

            # Update status to failed
            try:
                analysis_ops.update_analysis_status(uuid.UUID(analysis_id), "failed")
                db.commit()
            except Exception as db_error:
                logger.error(f"Failed to update analysis status: {db_error}")

        finally:
            # Clean up temporary file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up file {file_path}: {cleanup_error}")


@router.get("/analyze/{analysis_id}")
async def get_analysis_result(
    analysis_id: str,
    db: Session = Depends(get_db_session),
    current_user: dict = Depends(get_current_user),
):
    """Get analysis result from database."""

    try:
        analysis_ops = AnalysisOperations(db)
        analysis = analysis_ops.get_analysis_by_id(uuid.UUID(analysis_id))

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        # IDOR protection: Verify user has access to this analysis
        if current_user.role != "admin":
            if analysis.case_id:
                case_ops = CaseOperations(db)
                case = case_ops.get_case_by_id(analysis.case_id)
                if case and case.assigned_user_id != current_user.id:
                    log_security_event(
                        "unauthorized_access_attempt",
                        username=current_user.username,
                        details=f"Attempted to access analysis {analysis_id}",
                        success=False,
                    )
                    raise HTTPException(status_code=403, detail="Access denied")

        # Build response
        response = {
            "analysis_id": str(analysis.id),
            "status": analysis.status,
            "filename": analysis.filename,
            "content_type": analysis.content_type,
            "file_size": analysis.file_size,
            "created_at": analysis.created_at.isoformat(),
            "updated_at": analysis.updated_at.isoformat(),
        }

        # Add processing info if available
        if analysis.processing_time_ms:
            response["processing_time_ms"] = analysis.processing_time_ms
        if analysis.model_version:
            response["model_version"] = analysis.model_version

        # Add model results if completed
        if analysis.results:
            result = analysis.results[0]  # Get first (and typically only) result
            response.update(
                {
                    "prediction_class": result.prediction_class,
                    "confidence_score": result.confidence_score,
                    "probability_scores": result.probability_scores,
                    "model_name": result.model_name,
                    "uncertainty_score": result.uncertainty_score,
                }
            )

        return response

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid analysis ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")


# DICOM endpoints
@router.post("/dicom/upload")
@limiter.limit("5/minute")
async def upload_dicom(
    file: UploadFile = File(...),
    request: Request = None,
    current_user: dict = Depends(get_current_user),
):
    """Upload DICOM file."""

    # Enforce size limit before reading entire file into memory (DoS prevention)
    max_size = 500 * 1024 * 1024  # 500MB for DICOM files
    content_length = request.headers.get("content-length") if request else None
    if content_length and int(content_length) > max_size:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 500MB")

    # Read file content with size limit
    file_content = await file.read(max_size + 1)
    if len(file_content) > max_size:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 500MB")

    # Use centralized file validator for DICOM files
    detected_mime, safe_filename = validate_file_upload(file_content, file.filename)

    # Ensure it's actually a DICOM file
    if detected_mime != "application/dicom":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only DICOM files are supported."
        )

    study_id = str(uuid.uuid4())

    # Mock DICOM processing
    dicom_data = {
        "study_id": study_id,
        "filename": safe_filename,
        "upload_time": datetime.now().isoformat(),
        "status": "processed",
    }

    return dicom_data


@router.get("/dicom/study/{study_id}")
async def get_dicom_study(
    study_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get DICOM study information."""

    # Mock study data
    return {
        "study_id": study_id,
        "study_instance_uid": f"1.2.840.10008.5.1.{study_id[:8]}",
        "patient_id": f"PATIENT_{study_id[:6]}",
        "study_date": "20260427",
        "study_time": "120000",
        "modality": "SM",
        "series_count": 1,
        "instance_count": 1,
    }


# Case management endpoints
@router.get("/cases")
async def get_cases(
    limit: int = 10,
    status: Optional[str] = None,
    db: Session = Depends(get_db_session),
    current_user: dict = Depends(get_current_user),
):
    """Get list of cases from database.

    Args:
        limit: Maximum number of cases (1-1000)
        status: Filter by status
    """
    # Validate limit to prevent DoS via excessive queries
    validate_limit(limit)

    try:
        case_ops = CaseOperations(db)
        # Use joined loading to avoid N+1 queries when accessing assigned_user
        from sqlalchemy.orm import joinedload

        cases = case_ops.list_cases(
            status=status, limit=limit, options=[joinedload(Case.assigned_user)]
        )

        case_list = []
        for case in cases:
            case_dict = {
                "case_id": str(case.id),
                "patient_id": case.patient_id,
                "study_id": case.study_id,
                "case_type": case.case_type,
                "priority": case.priority,
                "status": case.status,
                "notes": case.notes,
                "created_at": case.created_at.isoformat(),
                "updated_at": case.updated_at.isoformat(),
            }
            if case.assigned_user:
                case_dict["assigned_user"] = case.assigned_user.username
            case_list.append(case_dict)

        return {"cases": case_list, "total": len(case_list)}

    except Exception as e:
        logger.error(f"Failed to get cases: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cases")


@router.post("/cases")
@limiter.limit("10/minute")
async def create_case(
    case_data: CaseData,
    request: Request = None,
    db: Session = Depends(get_db_session),
    current_user: dict = Depends(get_current_user),
):
    """Create a new case in database."""

    try:
        case_ops = CaseOperations(db)
        case = case_ops.create_case(
            patient_id=case_data.patient_id,
            study_id=case_data.study_id,
            case_type=case_data.case_type,
            priority=case_data.priority,
        )

        db.commit()

        logger.info(f"Created case: {case.patient_id}/{case.study_id}")

        return {"case_id": str(case.id), "status": "created"}

    except Exception as e:
        logger.error(f"Failed to create case: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create case")


@router.get("/cases/{case_id}")
async def get_case(
    case_id: str,
    db: Session = Depends(get_db_session),
    current_user: dict = Depends(get_current_user),
):
    """Get case details from database."""

    try:
        case_ops = CaseOperations(db)
        case = case_ops.get_case_by_id(uuid.UUID(case_id))

        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        # IDOR protection: Verify user has access to this case
        if current_user.role != "admin" and case.assigned_user_id != current_user.id:
            log_security_event(
                "unauthorized_access_attempt",
                username=current_user.username,
                details=f"Attempted to access case {case_id}",
                success=False,
            )
            raise HTTPException(status_code=403, detail="Access denied")

        case_dict = {
            "case_id": str(case.id),
            "patient_id": case.patient_id,
            "study_id": case.study_id,
            "case_type": case.case_type,
            "priority": case.priority,
            "status": case.status,
            "notes": case.notes,
            "created_at": case.created_at.isoformat(),
            "updated_at": case.updated_at.isoformat(),
        }

        if case.assigned_user:
            case_dict["assigned_user"] = {
                "id": str(case.assigned_user.id),
                "username": case.assigned_user.username,
                "role": case.assigned_user.role,
            }

        # Add analysis count using efficient count query
        case_ops = CaseOperations(db)
        case_dict["analysis_count"] = case_ops.get_analysis_count_by_case(case.id)

        return case_dict

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get case {case_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve case")


@router.put("/cases/{case_id}/status")
async def update_case_status(
    case_id: str,
    status_data: CaseStatusUpdate,
    db: Session = Depends(get_db_session),
    current_user: dict = Depends(get_current_user),
):
    """Update case status in database."""

    try:
        case_ops = CaseOperations(db)

        # IDOR protection: Verify user has access to this case
        case = case_ops.get_case_by_id(uuid.UUID(case_id))
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        if current_user.role != "admin" and case.assigned_user_id != current_user.id:
            log_security_event(
                "unauthorized_access_attempt",
                username=current_user.username,
                details=f"Attempted to update case {case_id}",
                success=False,
            )
            raise HTTPException(status_code=403, detail="Access denied")

        success = case_ops.update_case_status(
            uuid.UUID(case_id), status_data.status, status_data.notes
        )

        if not success:
            raise HTTPException(status_code=404, detail="Case not found")

        db.commit()

        return {"message": "Status updated successfully"}

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update case {case_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update case")
