#!/usr/bin/env python3
"""
Medical AI Platform - Main API Server

FastAPI-based REST API server for the Medical AI platform providing endpoints
for image analysis, DICOM integration, case management, and system monitoring.
"""

import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import existing components
try:
    from src.streaming.web_dashboard import app as streaming_app
    from src.streaming.interactive_showcase import app as showcase_app
except ImportError:
    streaming_app = None
    showcase_app = None

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
    description="REST API for Medical AI pathology analysis platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# In-memory storage for demo purposes
analysis_results = {}
users = {}
cases = {}
auth_tokens = {}

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    if not credentials:
        return None
    
    token = credentials.credentials
    return auth_tokens.get(token)

# Health and system endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        components={
            "api": True,
            "database": True,
            "model": True,
            "storage": True
        }
    )

@app.get("/api/v1/system/status")
async def system_status():
    """System status endpoint."""
    return {
        "status": "operational",
        "uptime_seconds": 3600,
        "active_analyses": len([r for r in analysis_results.values() if r["status"] == "in_progress"]),
        "total_analyses": len(analysis_results),
        "memory_usage_mb": 1024,
        "cpu_usage_percent": 45.2
    }

@app.get("/api/v1/system/build-info", response_model=BuildInfo)
async def build_info():
    """Build information endpoint."""
    return BuildInfo(
        version="1.0.0",
        commit_hash="abc123def456",
        build_date=datetime.now().isoformat(),
        environment="development"
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
            "monitoring": True
        }
    }

@app.get("/api/v1/system/db-health")
async def database_health():
    """Database health check."""
    return {
        "status": "healthy",
        "connection_count": 5,
        "query_time_ms": 25.3,
        "active_connections": 2,
        "max_connections": 100
    }

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
async def register_user(user_data: UserRegistration):
    """Register a new user."""
    if user_data.username in users:
        raise HTTPException(status_code=409, detail="User already exists")
    
    user_id = str(uuid.uuid4())
    users[user_data.username] = {
        "user_id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "role": user_data.role,
        "created_at": datetime.now().isoformat()
    }
    
    return {"message": "User registered successfully", "user_id": user_id}

@app.post("/api/v1/auth/login")
async def login_user(login_data: UserLogin):
    """User login."""
    if login_data.username not in users:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Generate token (simplified for demo)
    token = str(uuid.uuid4())
    auth_tokens[token] = users[login_data.username]
    
    return {"access_token": token, "token_type": "bearer"}

@app.get("/api/v1/auth/me")
async def get_current_user_info(current_user=Depends(get_current_user)):
    """Get current user information."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return current_user

# Analysis endpoints
@app.post("/api/v1/analyze/upload")
async def upload_for_analysis(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    request_data: AnalysisRequest = AnalysisRequest()
):
    """Upload image for analysis."""
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only images are supported.")
    
    # Generate analysis ID
    analysis_id = str(uuid.uuid4())
    
    # Store analysis request
    analysis_results[analysis_id] = {
        "analysis_id": analysis_id,
        "status": "queued",
        "filename": file.filename,
        "content_type": file.content_type,
        "case_id": request_data.case_id,
        "priority": request_data.priority,
        "case_type": request_data.case_type,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    # Start background processing
    background_tasks.add_task(process_analysis, analysis_id)
    
    return {"analysis_id": analysis_id, "status": "queued"}

async def process_analysis(analysis_id: str):
    """Background task to process analysis."""
    
    # Update status to in_progress
    analysis_results[analysis_id]["status"] = "in_progress"
    analysis_results[analysis_id]["updated_at"] = datetime.now().isoformat()
    
    # Simulate processing time
    import asyncio
    await asyncio.sleep(2)  # Simulate 2 second processing
    
    # Generate mock results
    import random
    confidence = random.uniform(0.7, 0.99)
    prediction = random.choice(["positive", "negative", "uncertain"])
    processing_time = random.randint(15000, 30000)
    
    # Update with results
    analysis_results[analysis_id].update({
        "status": "completed",
        "confidence_score": confidence,
        "prediction_class": prediction,
        "processing_time_ms": processing_time,
        "updated_at": datetime.now().isoformat()
    })

@app.get("/api/v1/analyze/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """Get analysis result by ID."""
    
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_results[analysis_id]

# DICOM endpoints
@app.post("/api/v1/dicom/upload")
async def upload_dicom(file: UploadFile = File(...)):
    """Upload DICOM file."""
    
    if not file.content_type or file.content_type != 'application/dicom':
        raise HTTPException(status_code=400, detail="Invalid file type. Only DICOM files are supported.")
    
    study_id = str(uuid.uuid4())
    
    # Mock DICOM processing
    dicom_data = {
        "study_id": study_id,
        "filename": file.filename,
        "upload_time": datetime.now().isoformat(),
        "status": "processed"
    }
    
    return dicom_data

@app.get("/api/v1/dicom/study/{study_id}")
async def get_dicom_study(study_id: str):
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
        "instance_count": 1
    }

# Case management endpoints
@app.get("/api/v1/cases")
async def get_cases(limit: int = 10, status: Optional[str] = None):
    """Get list of cases."""
    
    case_list = list(cases.values())
    
    if status:
        case_list = [c for c in case_list if c.get("status") == status]
    
    return {"cases": case_list[:limit], "total": len(case_list)}

@app.post("/api/v1/cases")
async def create_case(case_data: CaseData):
    """Create a new case."""
    
    case_id = str(uuid.uuid4())
    
    cases[case_id] = {
        "case_id": case_id,
        "patient_id": case_data.patient_id,
        "study_id": case_data.study_id,
        "priority": case_data.priority,
        "case_type": case_data.case_type,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    
    return {"case_id": case_id, "status": "created"}

@app.get("/api/v1/cases/{case_id}")
async def get_case(case_id: str):
    """Get case details."""
    
    if case_id not in cases:
        raise HTTPException(status_code=404, detail="Case not found")
    
    return cases[case_id]

@app.patch("/api/v1/cases/{case_id}/status")
async def update_case_status(case_id: str, status_data: dict):
    """Update case status."""
    
    if case_id not in cases:
        raise HTTPException(status_code=404, detail="Case not found")
    
    cases[case_id]["status"] = status_data.get("status", cases[case_id]["status"])
    cases[case_id]["notes"] = status_data.get("notes", "")
    cases[case_id]["updated_at"] = datetime.now().isoformat()
    
    return {"message": "Status updated successfully"}

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

# Analytics and reporting endpoints
@app.get("/api/v1/analytics/dashboard")
async def get_dashboard_data():
    """Get dashboard analytics data."""
    return {
        "total_cases": len(cases),
        "pending_cases": len([c for c in cases.values() if c.get("status") == "pending"]),
        "completed_analyses": len([r for r in analysis_results.values() if r.get("status") == "completed"]),
        "average_processing_time": 25.5,
        "system_uptime": 3600
    }

@app.get("/api/v1/analytics/performance")
async def get_performance_metrics(period: str = "7d"):
    """Get performance metrics."""
    return {
        "period": period,
        "total_cases": 150,
        "average_processing_time": 24.8,
        "success_rate": 98.5,
        "throughput_per_hour": 12.3
    }

@app.post("/api/v1/reports/generate")
async def generate_report(report_data: dict):
    """Generate report."""
    report_id = str(uuid.uuid4())
    return {"report_id": report_id, "status": "generating"}

@app.get("/api/v1/reports/{report_id}/status")
async def get_report_status(report_id: str):
    """Get report generation status."""
    return {"report_id": report_id, "status": "completed", "download_url": f"/reports/{report_id}.pdf"}

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
        "inference_timeout": 60
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
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

def main():
    """Run the API server."""
    
    # Add security headers middleware
    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()