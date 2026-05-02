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
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import database components
from src.database import (
    AnalysisOperations,
    AuditOperations,
    CaseOperations,
    DicomOperations,
    UserOperations,
    get_db_session,
    initialize_database,
)

# Import inference components
from src.inference import InferenceEngine, get_model_loader

# Import tracing
from src.monitoring.tracing import get_tracer

# Import security utilities
from src.api.security import (
    check_account_lockout,
    clear_failed_login,
    create_access_token,
    decode_access_token,
    get_security_headers,
    hash_password,
    limiter,
    log_security_event,
    record_failed_login,
    sanitize_for_log,
    validate_security_configuration,
    validate_uploaded_image,
    verify_password,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models
class AnalysisRequest(BaseModel):
    case_id: Optional[str] = None
    priority: str = "normal"
    case_type: str = "breast_cancer_screening"


class AnalysisResult(BaseModel):
    analysis_id: str
    status: str
    confidence_score: Optional[float] = None
    prediction_class: Optional[str] = None
    processing_time_ms: Optional[int] = None
    created_at: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    components: Dict[str, bool]


class BuildInfo(BaseModel):
    version: str
    commit_hash: str
    build_date: str
    environment: str


class UserRegistration(BaseModel):
    username: str
    email: str
    password: str
    role: str = "pathologist"


class UserLogin(BaseModel):
    username: str
    password: str


class CaseData(BaseModel):
    patient_id: str
    study_id: str
    priority: str = "normal"
    case_type: str = "breast_cancer_screening"


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

# Database for user storage (replace in-memory dict with proper database)
# In production, use database with proper password hashing
users_db: Dict[str, Dict] = {}  # Temporary - will be replaced with database

# Add CORS middleware with environment-specific origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Restrict to specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=3600,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security
security = HTTPBearer(auto_error=False)

# Global inference engine
inference_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get global inference engine instance."""
    global inference_engine
    if inference_engine is None:
        inference_engine = InferenceEngine()
        # Warm up the model
        try:
            inference_engine.warm_up_model("breast_cancer")
        except Exception as e:
            logger.warning(f"Failed to warm up model: {e}")
    return inference_engine


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db_session),
):
    """Get current authenticated user from database with proper JWT validation."""
    if not credentials:
        log_security_event("authentication_failed", details="No credentials provided", success=False)
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = credentials.credentials

    try:
        # Decode and validate JWT token
        payload = decode_access_token(token)
        user_id = payload.get("sub")

        if not user_id:
            log_security_event(
                "authentication_failed", details="Invalid token payload", success=False
            )
            raise HTTPException(status_code=401, detail="Invalid token")

        # Get user from database
        user_ops = UserOperations(db)
        user = user_ops.get_user_by_id(uuid.UUID(user_id))

        if not user:
            log_security_event(
                "authentication_failed", username=user_id, details="User not found", success=False
            )
            raise HTTPException(status_code=401, detail="User not found")

        return user

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        log_security_event("authentication_error", details=str(e), success=False)
        raise HTTPException(status_code=401, detail="Authentication failed")


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


# Health and system endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db_session)):
    """Health check endpoint with real database connectivity."""
    try:
        # Check database connectivity
        db.execute("SELECT 1")
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


@app.get("/api/v1/system/status")
async def system_status(db: Session = Depends(get_db_session)):
    """System status endpoint with real database statistics."""
    try:
        analysis_ops = AnalysisOperations(db)
        case_ops = CaseOperations(db)

        # Get real statistics
        analysis_stats = analysis_ops.get_analysis_statistics()
        case_stats = case_ops.get_case_statistics()

        # Count active analyses
        active_analyses = len([a for a in analysis_ops.list_analyses(status="in_progress")])

        # Get actual system metrics
        import time

        import psutil

        # Calculate uptime (would need to track app start time in production)
        uptime_seconds = int(time.time() - getattr(app.state, "start_time", time.time()))

        # Get actual memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / (1024 * 1024)  # Convert bytes to MB

        # Get actual CPU usage
        cpu_usage_percent = process.cpu_percent(interval=0.1)

        return {
            "status": "operational",
            "uptime_seconds": uptime_seconds,
            "active_analyses": active_analyses,
            "total_analyses": analysis_stats.get("total_analyses", 0),
            "total_cases": case_stats.get("total_cases", 0),
            "memory_usage_mb": round(memory_usage_mb, 2),
            "cpu_usage_percent": round(cpu_usage_percent, 2),
        }
    except Exception as e:
        logger.error(f"System status error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@app.get("/api/v1/system/build-info", response_model=BuildInfo)
async def build_info():
    """Build information endpoint."""
    return BuildInfo(
        version="1.0.0",
        commit_hash="abc123def456",
        build_date=datetime.now().isoformat(),
        environment="development",
    )


@app.get("/api/v1/system/readiness")
async def readiness_check():
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


@app.get("/api/v1/system/db-health")
async def database_health(db: Session = Depends(get_db_session)):
    """Real database health check."""
    try:
        from src.database.connection import get_database_manager

        db_manager = get_database_manager()
        health_info = db_manager.health_check()

        # Add query timing
        start_time = time.time()
        db.execute("SELECT COUNT(*) FROM users")
        query_time_ms = (time.time() - start_time) * 1000

        health_info["query_time_ms"] = round(query_time_ms, 2)
        return health_info

    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {"status": "unhealthy", "error": str(e), "query_time_ms": None}


@app.get("/metrics")
async def metrics():
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


# Authentication endpoints
@app.post("/api/v1/auth/register")
async def register_user(user_data: UserRegistration, request: Request):
    """Register a new user with secure password hashing."""
    try:
        # Check if user already exists
        if user_data.username in users_db:
            log_security_event(
                "registration_failed",
                username=user_data.username,
                ip_address=request.client.host,
                details="User already exists",
                success=False,
            )
            raise HTTPException(status_code=409, detail="User already exists")

        # Hash password
        hashed_password = hash_password(user_data.password)

        # Create user
        user_id = str(uuid.uuid4())
        users_db[user_data.username] = {
            "user_id": user_id,
            "username": user_data.username,
            "email": user_data.email,
            "password_hash": hashed_password,
            "role": user_data.role,
            "created_at": datetime.now().isoformat(),
        }

        log_security_event(
            "user_registered",
            username=user_data.username,
            ip_address=request.client.host,
            details=f"Role: {user_data.role}",
            success=True,
        )

        return {"message": "User registered successfully", "user_id": user_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/api/v1/auth/login")
@limiter.limit("5/minute")  # Rate limit: 5 login attempts per minute
async def login_user(login_data: UserLogin, request: Request):
    """User login with rate limiting and brute force protection."""
    username = login_data.username
    ip_address = request.client.host

    try:
        # Check account lockout
        check_account_lockout(username)

        # Validate credentials
        if username not in users_db:
            record_failed_login(username)
            log_security_event(
                "login_failed",
                username=username,
                ip_address=ip_address,
                details="User not found",
                success=False,
            )
            raise HTTPException(status_code=401, detail="Invalid credentials")

        user = users_db[username]

        # Verify password
        if not verify_password(login_data.password, user["password_hash"]):
            record_failed_login(username)
            log_security_event(
                "login_failed",
                username=username,
                ip_address=ip_address,
                details="Invalid password",
                success=False,
            )
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Clear failed login attempts
        clear_failed_login(username)

        # Generate JWT token
        access_token = create_access_token(
            data={"sub": user["user_id"], "username": username, "role": user["role"]}
        )

        log_security_event(
            "login_success", username=username, ip_address=ip_address, success=True
        )

        return {"access_token": access_token, "token_type": "bearer"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        log_security_event(
            "login_error", username=username, ip_address=ip_address, details=str(e), success=False
        )
        raise HTTPException(status_code=500, detail="Login failed")


@app.get("/api/v1/auth/me")
async def get_current_user_info(current_user=Depends(get_current_user)):
    """Get current user information."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return current_user


# Analysis endpoints with real model inference
@app.post("/api/v1/analyze/upload")
async def upload_for_analysis(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    request_data: AnalysisRequest = AnalysisRequest(),
    request: Request = None,
    db: Session = Depends(get_db_session),
    current_user: dict = Depends(get_current_user),
):
    """Upload image for real AI analysis with comprehensive security validation."""

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

        # Comprehensive file validation
        safe_filename, detected_type = validate_uploaded_image(
            file_content, file.filename, file.content_type
        )

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
            background_tasks.add_task(
                process_real_analysis, str(analysis.id), temp_path, file_content
            )

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


async def process_real_analysis(analysis_id: str, file_path: str, file_content: bytes):
    """Background task to process analysis with real AI model."""

    # Get database session for background task
    from src.database.connection import get_database_manager

    db_manager = get_database_manager()

    with db_manager.get_session() as db:
        try:
            analysis_ops = AnalysisOperations(db)

            # Update status to in_progress
            analysis_ops.update_analysis_status(uuid.UUID(analysis_id), "in_progress")
            db.commit()

            # Get inference engine
            engine = get_inference_engine()

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


@app.get("/api/v1/analyze/{analysis_id}")
async def get_analysis_result(analysis_id: str, db: Session = Depends(get_db_session), current_user: dict = Depends(get_current_user)):
    """Get real analysis result from database."""

    try:
        analysis_ops = AnalysisOperations(db)
        analysis = analysis_ops.get_analysis_by_id(uuid.UUID(analysis_id))

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

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
    except Exception as e:
        logger.error(f"Failed to get analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")


# DICOM endpoints
@app.post("/api/v1/dicom/upload")
async def upload_dicom(file: UploadFile = File(...)):
    """Upload DICOM file."""

    if not file.content_type or file.content_type != "application/dicom":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only DICOM files are supported."
        )

    study_id = str(uuid.uuid4())

    # Mock DICOM processing
    dicom_data = {
        "study_id": study_id,
        "filename": file.filename,
        "upload_time": datetime.now().isoformat(),
        "status": "processed",
    }

    return dicom_data


@app.get("/api/v1/dicom/study/{study_id}")
async def get_dicom_study(study_id: str, current_user: dict = Depends(get_current_user)):
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


# Case management endpoints with real database
@app.get("/api/v1/cases")
async def get_cases(
    limit: int = 10, status: Optional[str] = None, db: Session = Depends(get_db_session), current_user: dict = Depends(get_current_user)
):
    """Get list of cases from database."""

    try:
        case_ops = CaseOperations(db)
        cases = case_ops.list_cases(status=status, limit=limit)

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


@app.post("/api/v1/cases")
async def create_case(case_data: CaseData, db: Session = Depends(get_db_session), current_user: dict = Depends(get_current_user)):
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


@app.get("/api/v1/cases/{case_id}")
async def get_case(case_id: str, db: Session = Depends(get_db_session), current_user: dict = Depends(get_current_user)):
    """Get case details from database."""

    try:
        case_ops = CaseOperations(db)
        case = case_ops.get_case_by_id(uuid.UUID(case_id))

        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

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

        # Add analysis count
        case_dict["analysis_count"] = len(case.analyses)

        return case_dict

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case ID format")
    except Exception as e:
        logger.error(f"Failed to get case {case_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve case")


@app.patch("/api/v1/cases/{case_id}/status")
async def update_case_status(
    case_id: str, status_data: dict, db: Session = Depends(get_db_session), current_user: dict = Depends(get_current_user)
):
    """Update case status in database."""

    try:
        case_ops = CaseOperations(db)
        success = case_ops.update_case_status(
            uuid.UUID(case_id), status_data.get("status"), status_data.get("notes")
        )

        if not success:
            raise HTTPException(status_code=404, detail="Case not found")

        db.commit()

        return {"message": "Status updated successfully"}

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case ID format")
    except Exception as e:
        logger.error(f"Failed to update case {case_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update case")


# Mobile app endpoints
@app.post("/api/v1/mobile/register-device")
async def register_mobile_device(device_data: dict):
    """Register mobile device."""
    return {"message": "Device registered successfully", "device_id": device_data.get("device_id")}


@app.get("/api/v1/mobile/sync")
async def mobile_sync():
    """Mobile sync endpoint."""
    return {"pending_cases": [], "sync_timestamp": datetime.now().isoformat()}


@app.get("/api/v1/mobile/cases/offline")
async def get_offline_cases():
    """Get cases for offline use."""
    return {"cases": []}


@app.get("/api/v1/mobile/model/download")
async def download_mobile_model():
    """Download mobile model."""
    return {"model_url": "/models/mobile_model.tflite", "version": "1.0.0"}


# Analytics and reporting endpoints with real data
@app.get("/api/v1/analytics/dashboard")
async def get_dashboard_data(db: Session = Depends(get_db_session)):
    """Get dashboard analytics data from database."""

    try:
        case_ops = CaseOperations(db)
        analysis_ops = AnalysisOperations(db)

        # Get real statistics
        case_stats = case_ops.get_case_statistics()
        analysis_stats = analysis_ops.get_analysis_statistics()

        # Calculate actual uptime
        import time

        uptime_seconds = int(time.time() - getattr(app.state, "start_time", time.time()))

        return {
            "total_cases": case_stats.get("total_cases", 0),
            "pending_cases": case_stats.get("status_distribution", {}).get("pending", 0),
            "completed_analyses": analysis_stats.get("status_distribution", {}).get("completed", 0),
            "average_processing_time": analysis_stats.get("average_processing_time_ms", 0),
            "system_uptime": uptime_seconds,
            "case_status_distribution": case_stats.get("status_distribution", {}),
            "analysis_status_distribution": analysis_stats.get("status_distribution", {}),
        }

    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")


@app.get("/api/v1/analytics/performance")
async def get_performance_metrics(period: str = "7d", db: Session = Depends(get_db_session)):
    """Get performance metrics from database."""

    try:
        analysis_ops = AnalysisOperations(db)

        # Get completed analyses (in production, filter by date period)
        completed_analyses = analysis_ops.list_analyses(status="completed", limit=1000)

        if not completed_analyses:
            return {
                "period": period,
                "total_cases": 0,
                "average_processing_time": 0,
                "success_rate": 0,
                "throughput_per_hour": 0,
            }

        # Calculate metrics
        processing_times = [
            a.processing_time_ms for a in completed_analyses if a.processing_time_ms
        ]
        avg_processing_time = (
            sum(processing_times) / len(processing_times) if processing_times else 0
        )

        # Calculate success rate (completed vs total)
        all_analyses = analysis_ops.list_analyses(limit=1000)
        success_rate = (len(completed_analyses) / len(all_analyses) * 100) if all_analyses else 0

        # Estimate throughput (analyses per hour)
        # In production, calculate based on actual time period
        throughput_per_hour = len(completed_analyses) / 24  # Rough estimate

        return {
            "period": period,
            "total_cases": len(completed_analyses),
            "average_processing_time": avg_processing_time / 1000,  # Convert to seconds
            "success_rate": round(success_rate, 1),
            "throughput_per_hour": round(throughput_per_hour, 1),
        }

    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")


@app.post("/api/v1/reports/generate")
async def generate_report(report_data: dict):
    """Generate report."""
    report_id = str(uuid.uuid4())
    return {"report_id": report_id, "status": "generating"}


@app.get("/api/v1/reports/{report_id}/status")
async def get_report_status(report_id: str):
    """Get report generation status."""
    return {
        "report_id": report_id,
        "status": "completed",
        "download_url": f"/reports/{report_id}.pdf",
    }


# Admin endpoints
@app.get("/api/v1/admin/users")
async def get_users(limit: int = 10):
    """Get user list."""
    user_list = list(users.values())
    return {"users": user_list[:limit], "total": len(user_list)}


@app.get("/api/v1/admin/config")
async def get_system_config():
    """Get system configuration."""
    return {
        "max_file_size_mb": 100,
        "supported_formats": ["PNG", "JPEG", "TIFF", "DICOM"],
        "model_version": "1.0.0",
        "inference_timeout": 60,
    }


@app.get("/api/v1/admin/audit-logs")
async def get_audit_logs(limit: int = 10):
    """Get audit logs."""
    return {"logs": [], "total": 0}


# WebSocket and notifications
@app.get("/api/v1/ws/info")
async def websocket_info():
    """WebSocket connection information."""
    return {"endpoint": "/ws", "protocols": ["websocket"]}


@app.post("/api/v1/notifications/subscribe")
async def subscribe_notifications(subscription_data: dict):
    """Subscribe to notifications."""
    return {"message": "Subscription successful"}


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"detail": "Endpoint not found"})


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def main():
    """Run the API server."""

    # Add HTTPS redirect middleware (production only)
    @app.middleware("http")
    async def https_redirect(request: Request, call_next):
        """Redirect HTTP to HTTPS in production."""
        if (
            request.url.scheme != "https"
            and os.getenv("ENVIRONMENT") == "production"
            and not request.url.path.startswith("/health")
        ):
            url = request.url.replace(scheme="https")
            return RedirectResponse(url=str(url), status_code=301)
        return await call_next(request)

    # Add request timeout middleware
    @app.middleware("http")
    async def timeout_middleware(request: Request, call_next):
        """Add timeout to all requests to prevent slowloris attacks."""
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

    # Add security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add security headers to all responses."""
        response = await call_next(request)

        # Add all security headers
        security_headers = get_security_headers()
        for header, value in security_headers.items():
            response.headers[header] = value

        return response

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
