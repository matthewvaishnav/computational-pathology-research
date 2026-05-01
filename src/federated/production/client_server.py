"""Production FL Client Server."""

import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
import torch
import torch.nn as nn
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from ..client.trainer import FederatedTrainer
from ..common.data_models import ClientUpdate, TrainingMetadata
from ..communication.grpc_client import GRPCClient
from ..privacy.dp_sgd import DPSGDEngine
from .config import get_config
from .monitoring import get_metrics_manager, setup_logging

# Import distributed tracing
from src.monitoring.tracing import get_tracer

logger = structlog.get_logger(__name__)
config = get_config()

# Metrics
CLIENT_REQUESTS = Counter(
    "fl_client_requests_total", "Total client requests", ["endpoint", "status"]
)
TRAINING_DURATION = Histogram("fl_client_training_duration_seconds", "Training duration")
MODEL_UPDATES_SENT = Counter("fl_client_model_updates_sent_total", "Model updates sent")
PRIVACY_BUDGET_REMAINING = Gauge("fl_client_privacy_budget_remaining", "Remaining privacy budget")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting FL Client Server")

    try:
        # Initialize client components
        app.state.client_id = config.client_id
        app.state.coordinator_url = config.coordinator_url
        app.state.trainer = None
        app.state.grpc_client = None
        app.state.privacy_engine = None

        # Initialize gRPC client
        app.state.grpc_client = GRPCClient(config.coordinator_url)

        # Initialize privacy engine
        app.state.privacy_engine = DPSGDEngine(
            epsilon=config.federated_learning.default_epsilon,
            delta=config.federated_learning.default_delta,
        )

        # Initialize distributed tracing
        tracer = get_tracer(f"histocore-fl-client-{config.client_id}")
        tracer.initialize(
            jaeger_endpoint=os.getenv("JAEGER_ENDPOINT"),
            otlp_endpoint=os.getenv("OTLP_ENDPOINT"),
            service_version="1.0.0",
            environment=os.getenv("ENVIRONMENT", "development"),
        )
        tracer.instrument_fastapi(app)
        logger.info("Distributed tracing initialized successfully")

        logger.info("FL Client Server started successfully")

    except Exception as e:
        logger.error(f"Failed to start FL Client Server: {e}")
        sys.exit(1)

    yield

    # Shutdown
    logger.info("Shutting down FL Client Server")
    if app.state.grpc_client:
        await app.state.grpc_client.close()


# FastAPI app
app = FastAPI(
    title="Federated Learning Client",
    description="Production FL Client API",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    CLIENT_REQUESTS.labels(endpoint=request.url.path, status=response.status_code).inc()

    return response


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "client_id": app.state.client_id,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Client registration
@app.post("/api/v1/register")
async def register_client():
    """Register client with coordinator."""
    try:
        registration_data = {
            "client_id": app.state.client_id,
            "name": f"Hospital {app.state.client_id}",
            "organization": "HistoCore Medical Center",
            "capabilities": {
                "max_batch_size": 32,
                "supported_algorithms": ["fedavg", "fedprox", "krum"],
                "privacy_enabled": True,
            },
        }

        # Register with coordinator via gRPC
        success = await app.state.grpc_client.register_client(registration_data)

        if success:
            logger.info(f"Client {app.state.client_id} registered successfully")
            return {"status": "success", "message": "Client registered"}
        else:
            raise HTTPException(status_code=500, detail="Registration failed")

    except Exception as e:
        logger.error(f"Client registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Training endpoints
@app.post("/api/v1/training/start")
async def start_training(training_config: dict, background_tasks: BackgroundTasks):
    """Start local training."""
    try:
        round_id = training_config["round_id"]
        global_model = training_config["global_model"]
        algorithm = training_config.get("algorithm", "fedavg")

        logger.info(f"Starting training for round {round_id}")

        # Initialize trainer if not exists
        if not app.state.trainer:
            # Create dummy model for demo
            model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

            app.state.trainer = FederatedTrainer(
                model=model, privacy_engine=app.state.privacy_engine
            )

        # Load global model
        app.state.trainer.load_global_model(global_model)

        # Start training in background
        background_tasks.add_task(run_training_round, round_id, algorithm)

        return {"status": "started", "round_id": round_id, "client_id": app.state.client_id}

    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_training_round(round_id: int, algorithm: str):
    """Run training round in background."""
    try:
        start_time = time.time()

        # Simulate training data
        train_data = torch.randn(1000, 784)
        train_labels = torch.randint(0, 10, (1000,))

        # Train model
        metrics = app.state.trainer.train_epoch(
            train_data, train_labels, batch_size=32, learning_rate=0.01
        )

        # Get model update
        model_update = app.state.trainer.get_model_update()

        # Create client update
        client_update = ClientUpdate(
            client_id=app.state.client_id,
            round_id=round_id,
            model_update=model_update,
            dataset_size=len(train_data),
            training_metrics=metrics,
            privacy_epsilon=app.state.privacy_engine.get_epsilon_used(),
            privacy_delta=app.state.privacy_engine.get_delta_used(),
        )

        # Send update to coordinator
        success = await app.state.grpc_client.submit_update(client_update)

        if success:
            duration = time.time() - start_time
            TRAINING_DURATION.observe(duration)
            MODEL_UPDATES_SENT.inc()

            # Update privacy budget
            remaining_budget = app.state.privacy_engine.get_remaining_budget()
            PRIVACY_BUDGET_REMAINING.set(remaining_budget)

            logger.info(f"Training round {round_id} completed in {duration:.2f}s")
        else:
            logger.error(f"Failed to submit update for round {round_id}")

    except Exception as e:
        logger.error(f"Training round {round_id} failed: {e}")


@app.get("/api/v1/status")
async def get_client_status():
    """Get client status."""
    try:
        status = {
            "client_id": app.state.client_id,
            "status": "active",
            "privacy_budget": {
                "epsilon_used": (
                    app.state.privacy_engine.get_epsilon_used() if app.state.privacy_engine else 0.0
                ),
                "delta_used": (
                    app.state.privacy_engine.get_delta_used() if app.state.privacy_engine else 0.0
                ),
                "remaining_budget": (
                    app.state.privacy_engine.get_remaining_budget()
                    if app.state.privacy_engine
                    else 1.0
                ),
            },
            "training": {
                "current_round": (
                    getattr(app.state.trainer, "current_round", None) if app.state.trainer else None
                ),
                "is_training": (
                    getattr(app.state.trainer, "is_training", False) if app.state.trainer else False
                ),
            },
            "system": {
                "uptime": (
                    time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0
                ),
                "version": "1.0.0",
            },
        }

        return status

    except Exception as e:
        logger.error(f"Failed to get client status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/v1/privacy/reset")
async def reset_privacy_budget():
    """Reset privacy budget (admin only)."""
    try:
        if app.state.privacy_engine:
            app.state.privacy_engine.reset_budget()
            PRIVACY_BUDGET_REMAINING.set(1.0)

            logger.info(f"Privacy budget reset for client {app.state.client_id}")
            return {"status": "success", "message": "Privacy budget reset"}
        else:
            raise HTTPException(status_code=500, detail="Privacy engine not initialized")

    except Exception as e:
        logger.error(f"Failed to reset privacy budget: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/data/stats")
async def get_data_stats():
    """Get local data statistics."""
    try:
        # In production, this would analyze actual local data
        stats = {
            "total_samples": 10000,
            "classes": {"benign": 7000, "malignant": 3000},
            "data_quality": {"missing_values": 0.02, "outliers": 0.01, "duplicates": 0.005},
            "privacy_level": "high",
        }

        return stats

    except Exception as e:
        logger.error(f"Failed to get data stats: {e}")
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

    # Get client configuration
    client_id = os.getenv("CLIENT_ID", "hospital_a")
    coordinator_url = os.getenv("COORDINATOR_URL", "localhost:50051")

    # Update app state
    app.state.client_id = client_id
    app.state.coordinator_url = coordinator_url
    app.state.start_time = time.time()

    # Start server
    uvicorn.run(
        "src.federated.production.client_server:app",
        host="0.0.0.0",
        port=8081,
        workers=1,
        log_config=None,
        access_log=False,
        ssl_keyfile=config.security.tls_key_path if hasattr(config, "security") else None,
        ssl_certfile=config.security.tls_cert_path if hasattr(config, "security") else None,
    )


if __name__ == "__main__":
    import os

    main()
