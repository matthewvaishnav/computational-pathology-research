"""Production FL Coordinator Server."""

import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import sentry_sdk
import structlog
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from ..aggregator.factory import AggregatorFactory
from ..common.data_models import ClientUpdate
from ..coordinator.orchestrator import TrainingOrchestrator
from .config import get_config, validate_production_config
from .database import get_db_manager, init_database
from .monitoring import get_metrics_manager, setup_logging
from .security import get_audit_logger, get_security_manager, validate_security_config

# Import distributed tracing
from src.monitoring.tracing import get_tracer

# Configuration
config = get_config()
db_manager = get_db_manager()
security_manager = get_security_manager()
audit_logger = get_audit_logger()

# Logging setup
logger = structlog.get_logger(__name__)

# Metrics
REQUEST_COUNT = Counter(
    "fl_coordinator_requests_total", "Total requests", ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram("fl_coordinator_request_duration_seconds", "Request duration")
ACTIVE_CLIENTS = Gauge("fl_coordinator_active_clients", "Number of active clients")
TRAINING_ROUNDS = Counter(
    "fl_coordinator_training_rounds_total", "Total training rounds", ["status"]
)

# Security
security = HTTPBearer()

# Server start time for uptime calculation
SERVER_START_TIME = datetime.now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting FL Coordinator Server")

    try:
        # Validate configuration
        validate_production_config()
        validate_security_config()

        # Initialize database
        init_database()

        # Initialize monitoring
        if config.monitoring.sentry_dsn:
            sentry_sdk.init(
                dsn=config.monitoring.sentry_dsn,
                environment=config.monitoring.sentry_environment,
                integrations=[
                    FastApiIntegration(auto_enable=True),
                    SqlalchemyIntegration(),
                ],
            )

        # Initialize distributed tracing
        tracer = get_tracer("histocore-fl-coordinator")
        tracer.initialize(
            jaeger_endpoint=os.getenv("JAEGER_ENDPOINT"),
            otlp_endpoint=os.getenv("OTLP_ENDPOINT"),
            service_version="1.0.0",
            environment=config.monitoring.sentry_environment if hasattr(config, "monitoring") else "development",
        )
        tracer.instrument_fastapi(app)
        logger.info("Distributed tracing initialized successfully")

        # Initialize orchestrator
        app.state.orchestrator = None

        logger.info("FL Coordinator Server started successfully")

    except Exception as e:
        logger.error(f"Failed to start FL Coordinator Server: {e}")
        sys.exit(1)

    yield

    # Shutdown
    logger.info("Shutting down FL Coordinator Server")


# FastAPI app
app = FastAPI(
    title="Federated Learning Coordinator",
    description="Production FL Coordinator API",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost:3000"],  # Restrict origins in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.internal"],  # Configure for your environment
)


# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Authenticate user from JWT token."""
    token = credentials.credentials
    payload = security_manager.verify_token(token)

    if not payload:
        audit_logger.log_authentication_attempt(
            user_id="unknown", success=False, error_message="Invalid token"
        )
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    audit_logger.log_authentication_attempt(user_id=user_id, success=True)
    return {"user_id": user_id, "payload": payload}


async def get_admin_user(current_user: dict = Depends(get_current_user)):
    """Verify user has admin role."""
    payload = current_user.get("payload", {})
    role = payload.get("role", "user")
    
    if role != "admin":
        audit_logger.log_event(
            event_type="unauthorized_access",
            event_category="security",
            description=f"Non-admin user {current_user['user_id']} attempted to access admin endpoint",
            user_id=current_user["user_id"],
        )
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return current_user


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for audit purposes."""
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Log metrics
    duration = time.time() - start_time
    REQUEST_DURATION.observe(duration)
    REQUEST_COUNT.labels(
        method=request.method, endpoint=request.url.path, status=response.status_code
    ).inc()

    # Audit log for sensitive endpoints
    if request.url.path.startswith("/api/v1/"):
        audit_logger.log_event(
            event_type="api_request",
            event_category="access",
            description=f"{request.method} {request.url.path}",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            success=response.status_code < 400,
            additional_data={"status_code": response.status_code, "duration": duration},
        )

    return response


# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        with db_manager.get_session() as session:
            session.execute("SELECT 1")

        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "version": "1.0.0"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Client management endpoints
@app.post("/api/v1/clients/register")
async def register_client(client_data: dict, current_user: dict = Depends(get_current_user)):
    """Register a new FL client."""
    try:
        client_id = client_data["client_id"]
        name = client_data["name"]
        organization = client_data.get("organization")

        # Register client in database
        client = db_manager.register_client(client_id, name, organization)

        # Audit log
        audit_logger.log_client_registration(client_id=client_id, user_id=current_user["user_id"])

        return {
            "status": "success",
            "client_id": client_id,
            "message": "Client registered successfully",
        }

    except Exception as e:
        logger.error(f"Client registration failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/clients")
async def list_clients(current_user: dict = Depends(get_current_user)):
    """List all registered clients."""
    try:
        clients = db_manager.get_active_clients()

        # Update metrics
        ACTIVE_CLIENTS.set(len(clients))

        return {
            "clients": [
                {
                    "client_id": client.client_id,
                    "name": client.name,
                    "organization": client.organization,
                    "status": client.status,
                    "last_seen": client.last_seen.isoformat() if client.last_seen else None,
                }
                for client in clients
            ]
        }

    except Exception as e:
        logger.error(f"Failed to list clients: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Training management endpoints
@app.post("/api/v1/training/start")
async def start_training_round(
    training_config: dict,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    """Start a new training round."""
    try:
        algorithm = training_config.get("algorithm", "fedavg")
        min_clients = training_config.get(
            "min_clients", config.federated_learning.min_clients_per_round
        )
        max_clients = training_config.get(
            "max_clients", config.federated_learning.max_clients_per_round
        )

        # Get active clients
        active_clients = db_manager.get_active_clients()
        if len(active_clients) < min_clients:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough active clients: {len(active_clients)} < {min_clients}",
            )

        # Create orchestrator if not exists
        if not app.state.orchestrator:
            # Initialize with a dummy model for now
            from torch import nn

            model = nn.Linear(10, 1)  # Replace with actual model
            aggregator = AggregatorFactory.create_aggregator(algorithm)
            app.state.orchestrator = TrainingOrchestrator(model, aggregator)

        # Start training round
        client_ids = [client.client_id for client in active_clients[:max_clients]]
        round_metadata = app.state.orchestrator.start_round(client_ids)

        # Create database record
        db_round = db_manager.create_training_round(
            round_id=round_metadata.round_id,
            algorithm=algorithm,
            min_clients=min_clients,
            max_clients=max_clients,
        )

        # Audit log
        audit_logger.log_training_round_start(
            round_id=round_metadata.round_id,
            participants=client_ids,
            user_id=current_user["user_id"],
        )

        # Update metrics
        TRAINING_ROUNDS.labels(status="started").inc()

        return {
            "status": "success",
            "round_id": round_metadata.round_id,
            "participants": client_ids,
            "algorithm": algorithm,
        }

    except Exception as e:
        logger.error(f"Failed to start training round: {e}")
        TRAINING_ROUNDS.labels(status="failed").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/training/status/{round_id}")
async def get_training_status(round_id: int, current_user: dict = Depends(get_current_user)):
    """Get training round status."""
    try:
        # Get from database
        with db_manager.get_session() as session:
            round_obj = (
                session.query(db_manager.TrainingRound)
                .filter(db_manager.TrainingRound.round_id == round_id)
                .first()
            )

            if not round_obj:
                raise HTTPException(status_code=404, detail="Training round not found")

            return {
                "round_id": round_obj.round_id,
                "status": round_obj.status,
                "algorithm": round_obj.algorithm,
                "participants": round_obj.participants,
                "started_at": round_obj.started_at.isoformat() if round_obj.started_at else None,
                "completed_at": (
                    round_obj.completed_at.isoformat() if round_obj.completed_at else None
                ),
                "metrics": round_obj.aggregated_metrics,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Model management endpoints
@app.get("/api/v1/models/latest")
async def get_latest_model(current_user: dict = Depends(get_current_user)):
    """Get the latest global model."""
    try:
        if not app.state.orchestrator:
            raise HTTPException(status_code=404, detail="No training in progress")

        model_state = app.state.orchestrator.get_global_model()

        # Audit log
        audit_logger.log_data_access(
            user_id=current_user["user_id"],
            resource_type="model",
            resource_id="latest",
            action="download",
        )

        return {
            "version": app.state.orchestrator.current_version,
            "round_id": app.state.orchestrator.current_round,
            "model_state": model_state,  # In production, this would be a download URL
        }

    except Exception as e:
        logger.error(f"Failed to get latest model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Admin endpoints
@app.get("/api/v1/admin/stats")
async def get_system_stats(current_user: dict = Depends(get_admin_user)):
    """Get system statistics (admin only)."""
    try:
        with db_manager.get_session() as session:
            # Get client stats
            total_clients = session.query(db_manager.Client).count()
            active_clients = (
                session.query(db_manager.Client)
                .filter(db_manager.Client.status == "active")
                .count()
            )

            # Get training stats
            total_rounds = session.query(db_manager.TrainingRound).count()
            completed_rounds = (
                session.query(db_manager.TrainingRound)
                .filter(db_manager.TrainingRound.status == "completed")
                .count()
            )
            
            # Calculate uptime
            uptime_delta = datetime.now() - SERVER_START_TIME
            uptime_seconds = int(uptime_delta.total_seconds())
            uptime_hours = uptime_seconds // 3600
            uptime_minutes = (uptime_seconds % 3600) // 60
            uptime_str = f"{uptime_hours}h {uptime_minutes}m"

            return {
                "clients": {"total": total_clients, "active": active_clients},
                "training": {"total_rounds": total_rounds, "completed_rounds": completed_rounds},
                "system": {
                    "uptime": uptime_str,
                    "uptime_seconds": uptime_seconds,
                    "version": "1.0.0",
                    "start_time": SERVER_START_TIME.isoformat(),
                },
            }

    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def main():
    """Main entry point."""
    # Setup logging
    setup_logging()

    # Handle shutdown gracefully
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start server
    uvicorn.run(
        "src.federated.production.coordinator_server:app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_config=None,  # We handle logging ourselves
        access_log=False,
        ssl_keyfile=config.security.tls_key_path,
        ssl_certfile=config.security.tls_cert_path,
        ssl_ca_certs=config.security.ca_cert_path,
        ssl_cert_reqs=2,  # Require client certificates
    )


if __name__ == "__main__":
    main()
