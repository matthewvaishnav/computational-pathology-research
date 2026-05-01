"""
FastAPI backend for annotation interface
Provides REST endpoints and WebSocket for real-time collaboration
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .annotation_models import (
    AIPredictionOverlay,
    Annotation,
    AnnotationCreate,
    AnnotationQueueItem,
    AnnotationResponse,
    AnnotationUpdate,
    SlideInfo,
)
from .quality_control import (
    AnnotationValidator,
    InterRaterAgreement,
    QualityMetricsTracker,
    ValidationResult,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Expert Annotation Interface",
    description="Web-based annotation tool for pathologists",
    version="1.0.0",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# In-Memory Storage (Replace with database in production)
# ============================================================================

annotations_db: Dict[str, Annotation] = {}
slides_db: Dict[str, SlideInfo] = {}
queue_db: Dict[str, AnnotationQueueItem] = {}
validation_results_db: Dict[str, ValidationResult] = {}

# Initialize quality control components
annotation_validator = AnnotationValidator()
quality_metrics_tracker = QualityMetricsTracker()


# ============================================================================
# WebSocket Connection Manager for Real-Time Collaboration
# ============================================================================


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.logger = logging.getLogger(__name__)

    async def connect(self, websocket: WebSocket, slide_id: str):
        """Connect client to slide room"""
        await websocket.accept()
        if slide_id not in self.active_connections:
            self.active_connections[slide_id] = set()
        self.active_connections[slide_id].add(websocket)
        self.logger.info(f"Client connected to slide {slide_id}")

    def disconnect(self, websocket: WebSocket, slide_id: str):
        """Disconnect client from slide room"""
        if slide_id in self.active_connections:
            self.active_connections[slide_id].discard(websocket)
            if not self.active_connections[slide_id]:
                del self.active_connections[slide_id]
        self.logger.info(f"Client disconnected from slide {slide_id}")

    async def broadcast(self, slide_id: str, message: dict):
        """Broadcast message to all clients viewing a slide"""
        if slide_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[slide_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    self.logger.error(f"Error broadcasting to client: {e}")
                    disconnected.add(connection)

            # Remove disconnected clients
            for conn in disconnected:
                self.active_connections[slide_id].discard(conn)


manager = ConnectionManager()


# ============================================================================
# REST API Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint"""
    return {"service": "Expert Annotation Interface", "version": "1.0.0", "status": "running"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "annotations_count": len(annotations_db),
        "slides_count": len(slides_db),
        "queue_count": len(queue_db),
    }


# ============================================================================
# Annotation Queue Endpoints
# ============================================================================


@app.get("/api/queue", response_model=List[AnnotationQueueItem])
async def get_annotation_queue(
    expert_id: Optional[str] = Query(None), limit: int = Query(10, ge=1, le=100)
):
    """Get annotation queue for expert"""
    queue_items = list(queue_db.values())

    # Filter by expert if specified
    if expert_id:
        queue_items = [
            item
            for item in queue_items
            if item.assigned_expert is None or item.assigned_expert == expert_id
        ]

    # Filter pending/in-progress items
    queue_items = [item for item in queue_items if item.status in ["pending", "in_progress"]]

    # Sort by priority (highest first)
    queue_items.sort(key=lambda x: x.priority, reverse=True)

    return queue_items[:limit]


@app.post("/api/queue/{task_id}/assign")
async def assign_task(task_id: str, expert_id: str):
    """Assign task to expert"""
    if task_id not in queue_db:
        raise HTTPException(status_code=404, detail="Task not found")

    task = queue_db[task_id]
    task.assigned_expert = expert_id
    task.status = "in_progress"
    task.started_at = datetime.now()

    return {"success": True, "task": task}


@app.post("/api/queue/{task_id}/complete")
async def complete_task(task_id: str):
    """Mark task as completed"""
    if task_id not in queue_db:
        raise HTTPException(status_code=404, detail="Task not found")

    task = queue_db[task_id]
    task.status = "completed"
    task.completed_at = datetime.now()

    # Calculate annotation time if started_at is available
    if task.started_at:
        task.annotation_time_seconds = (task.completed_at - task.started_at).total_seconds()

    return {"success": True, "task": task}


# ============================================================================
# Slide Endpoints
# ============================================================================


@app.get("/api/slides/{slide_id}", response_model=SlideInfo)
async def get_slide_info(slide_id: str):
    """Get slide information"""
    if slide_id not in slides_db:
        raise HTTPException(status_code=404, detail="Slide not found")

    return slides_db[slide_id]


@app.get("/api/slides/{slide_id}/tile/{z}/{x}/{y}")
async def get_slide_tile(slide_id: str, z: int, x: int, y: int):
    """Get slide tile for OpenSeadragon viewer"""
    try:
        # Get slide info from database
        if slide_id not in slides_db:
            raise HTTPException(status_code=404, detail="Slide not found")
        
        slide_info = slides_db[slide_id]
        image_path = slide_info.image_path
        
        # Check if file exists
        import os
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Slide file not found")
        
        # Determine file type and extract tile
        from pathlib import Path
        file_ext = Path(image_path).suffix.lower()
        
        # For WSI formats, use OpenSlide
        if file_ext in ['.svs', '.tif', '.tiff', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs', '.bif']:
            try:
                import openslide
                from PIL import Image
                import io
                
                slide = openslide.OpenSlide(image_path)
                
                # Calculate tile position at the requested zoom level
                # OpenSeadragon uses zoom level 0 as highest resolution
                # OpenSlide uses level 0 as highest resolution
                level = min(z, slide.level_count - 1)
                
                # Get level dimensions and downsample factor
                level_dimensions = slide.level_dimensions[level]
                downsample = slide.level_downsamples[level]
                
                # Calculate tile position in level 0 coordinates
                tile_size = slide_info.tile_size
                x_pos = int(x * tile_size * downsample)
                y_pos = int(y * tile_size * downsample)
                
                # Read region from slide
                tile = slide.read_region(
                    (x_pos, y_pos),
                    level,
                    (tile_size, tile_size)
                )
                
                # Convert RGBA to RGB
                tile = tile.convert('RGB')
                
                # Save to bytes
                img_byte_arr = io.BytesIO()
                tile.save(img_byte_arr, format='JPEG', quality=85)
                img_byte_arr.seek(0)
                
                slide.close()
                
                return FileResponse(
                    img_byte_arr,
                    media_type="image/jpeg",
                    headers={
                        "Cache-Control": "public, max-age=31536000",
                        "Content-Type": "image/jpeg"
                    }
                )
                
            except Exception as e:
                logger.error(f"Error reading WSI tile: {e}")
                raise HTTPException(status_code=500, detail=f"Error reading tile: {str(e)}")
        
        # For regular images, use PIL
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            try:
                from PIL import Image
                import io
                
                img = Image.open(image_path)
                
                # Calculate tile position
                tile_size = slide_info.tile_size
                x_pos = x * tile_size
                y_pos = y * tile_size
                
                # Crop tile
                tile = img.crop((
                    x_pos,
                    y_pos,
                    x_pos + tile_size,
                    y_pos + tile_size
                ))
                
                # Save to bytes
                img_byte_arr = io.BytesIO()
                tile.save(img_byte_arr, format='JPEG', quality=85)
                img_byte_arr.seek(0)
                
                img.close()
                
                return FileResponse(
                    img_byte_arr,
                    media_type="image/jpeg",
                    headers={
                        "Cache-Control": "public, max-age=31536000",
                        "Content-Type": "image/jpeg"
                    }
                )
                
            except Exception as e:
                logger.error(f"Error reading image tile: {e}")
                raise HTTPException(status_code=500, detail=f"Error reading tile: {str(e)}")
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported image format")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_slide_tile: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/slides/{slide_id}/ai-prediction", response_model=AIPredictionOverlay)
async def get_ai_prediction(slide_id: str):
    """Get AI prediction overlay for slide"""
    try:
        # Get slide info from database
        if slide_id not in slides_db:
            raise HTTPException(status_code=404, detail="Slide not found")
        
        slide_info = slides_db[slide_id]
        
        # Check if we have a cached prediction
        prediction_cache_path = Path(f"./prediction_cache/{slide_id}.json")
        if prediction_cache_path.exists():
            with open(prediction_cache_path, 'r') as f:
                cached_prediction = json.load(f)
                return AIPredictionOverlay(**cached_prediction)
        
        # Generate prediction using inference engine
        try:
            from src.inference.inference_engine import InferenceEngine
            
            # Initialize inference engine (use cached instance if available)
            if not hasattr(get_ai_prediction, '_inference_engine'):
                get_ai_prediction._inference_engine = InferenceEngine(
                    model_path="checkpoints/best_model.pth",
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            
            engine = get_ai_prediction._inference_engine
            
            # Run inference on the slide
            result = engine.predict_slide(slide_info.image_path)
            
            # Convert result to overlay format
            regions = []
            if "regions" in result:
                for region in result["regions"]:
                    regions.append({
                        "x": region.get("x", 0),
                        "y": region.get("y", 0),
                        "width": region.get("width", 0),
                        "height": region.get("height", 0),
                        "confidence": region.get("confidence", 0.0),
                        "label": region.get("label", "unknown"),
                    })
            
            prediction = AIPredictionOverlay(
                slide_id=slide_id,
                prediction_type=result.get("prediction_type", "tumor_detection"),
                confidence=result.get("confidence", 0.0),
                regions=regions,
                metadata={
                    "model": result.get("model_name", "foundation_model_v1"),
                    "timestamp": datetime.now().isoformat(),
                    "inference_time_ms": result.get("inference_time_ms", 0),
                },
            )
            
            # Cache the prediction
            prediction_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(prediction_cache_path, 'w') as f:
                json.dump(json.loads(prediction.json()), f, indent=2)
            
            return prediction
            
        except ImportError:
            logger.warning("InferenceEngine not available, returning placeholder prediction")
            # Return placeholder if inference engine not available
            return AIPredictionOverlay(
                slide_id=slide_id,
                prediction_type="tumor_detection",
                confidence=0.0,
                regions=[],
                metadata={
                    "model": "placeholder",
                    "note": "InferenceEngine not available"
                },
            )
        except Exception as e:
            logger.error(f"Error generating AI prediction: {e}")
            # Return empty prediction on error
            return AIPredictionOverlay(
                slide_id=slide_id,
                prediction_type="tumor_detection",
                confidence=0.0,
                regions=[],
                metadata={
                    "error": str(e),
                    "note": "Prediction generation failed"
                },
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_ai_prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Annotation CRUD Endpoints
# ============================================================================


@app.post("/api/annotations", response_model=AnnotationResponse)
async def create_annotation(annotation_data: AnnotationCreate):
    """Create new annotation"""
    try:
        annotation_id = str(uuid.uuid4())
        now = datetime.now()

        annotation = Annotation(
            id=annotation_id,
            slide_id=annotation_data.slide_id,
            task_id=annotation_data.task_id,
            label=annotation_data.label,
            geometry=annotation_data.geometry,
            confidence=annotation_data.confidence,
            comments=annotation_data.comments,
            expert_id=annotation_data.expert_id,
            created_at=now,
            updated_at=now,
        )

        # Validate annotation quality
        validation_result = annotation_validator.validate(annotation)
        validation_results_db[annotation_id] = validation_result

        annotations_db[annotation_id] = annotation

        # Broadcast to other users viewing this slide
        await manager.broadcast(
            annotation_data.slide_id,
            {
                "type": "annotation_created",
                "annotation": json.loads(annotation.json()),
                "validation": validation_result.to_dict(),
            },
        )

        logger.info(f"Created annotation {annotation_id} for slide {annotation_data.slide_id}")

        return AnnotationResponse(
            success=True, annotation=annotation, message="Annotation created successfully"
        )

    except Exception as e:
        logger.error(f"Error creating annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/annotations", response_model=List[Annotation])
async def get_annotations(
    slide_id: Optional[str] = Query(None),
    expert_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get annotations with optional filters"""
    annotations = list(annotations_db.values())

    if slide_id:
        annotations = [a for a in annotations if a.slide_id == slide_id]

    if expert_id:
        annotations = [a for a in annotations if a.expert_id == expert_id]

    # Sort by creation time (newest first)
    annotations.sort(key=lambda x: x.created_at, reverse=True)

    return annotations[:limit]


@app.get("/api/annotations/{annotation_id}", response_model=Annotation)
async def get_annotation(annotation_id: str):
    """Get specific annotation"""
    if annotation_id not in annotations_db:
        raise HTTPException(status_code=404, detail="Annotation not found")

    return annotations_db[annotation_id]


@app.put("/api/annotations/{annotation_id}", response_model=AnnotationResponse)
async def update_annotation(annotation_id: str, update_data: AnnotationUpdate):
    """Update existing annotation"""
    if annotation_id not in annotations_db:
        raise HTTPException(status_code=404, detail="Annotation not found")

    annotation = annotations_db[annotation_id]

    # Update fields
    if update_data.label is not None:
        annotation.label = update_data.label
    if update_data.geometry is not None:
        annotation.geometry = update_data.geometry
    if update_data.confidence is not None:
        annotation.confidence = update_data.confidence
    if update_data.comments is not None:
        annotation.comments = update_data.comments

    annotation.updated_at = datetime.now()

    # Broadcast update
    await manager.broadcast(
        annotation.slide_id,
        {"type": "annotation_updated", "annotation": json.loads(annotation.json())},
    )

    logger.info(f"Updated annotation {annotation_id}")

    return AnnotationResponse(
        success=True, annotation=annotation, message="Annotation updated successfully"
    )


@app.delete("/api/annotations/{annotation_id}")
async def delete_annotation(annotation_id: str):
    """Delete annotation"""
    if annotation_id not in annotations_db:
        raise HTTPException(status_code=404, detail="Annotation not found")

    annotation = annotations_db.pop(annotation_id)

    # Broadcast deletion
    await manager.broadcast(
        annotation.slide_id, {"type": "annotation_deleted", "annotation_id": annotation_id}
    )

    logger.info(f"Deleted annotation {annotation_id}")

    return {"success": True, "message": "Annotation deleted successfully"}


# ============================================================================
# WebSocket Endpoint for Real-Time Collaboration
# ============================================================================


@app.websocket("/ws/{slide_id}")
async def websocket_endpoint(websocket: WebSocket, slide_id: str):
    """WebSocket endpoint for real-time collaboration"""
    await manager.connect(websocket, slide_id)

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()

            # Echo back to all clients (for cursor position, etc.)
            await manager.broadcast(slide_id, {"type": "user_action", "data": data})

    except WebSocketDisconnect:
        manager.disconnect(websocket, slide_id)
        logger.info(f"WebSocket disconnected for slide {slide_id}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, slide_id)


# ============================================================================
# Quality Control Endpoints
# ============================================================================


@app.get("/api/quality/validate/{annotation_id}")
async def validate_annotation(annotation_id: str):
    """Validate specific annotation"""
    if annotation_id not in annotations_db:
        raise HTTPException(status_code=404, detail="Annotation not found")

    annotation = annotations_db[annotation_id]
    validation_result = annotation_validator.validate(annotation)
    validation_results_db[annotation_id] = validation_result

    return validation_result.to_dict()


@app.get("/api/quality/metrics")
async def get_quality_metrics(time_window_days: int = Query(7, ge=1, le=90)):
    """Get quality metrics for annotations"""
    annotations = list(annotations_db.values())

    metrics = quality_metrics_tracker.compute_metrics(
        annotations, validation_results_db, time_window_days
    )

    # Detect quality alerts
    alerts = quality_metrics_tracker.detect_quality_alerts(metrics)

    return {"metrics": metrics, "alerts": alerts}


@app.get("/api/quality/metrics/history")
async def get_metrics_history(limit: int = Query(30, ge=1, le=100)):
    """Get historical quality metrics"""
    history = quality_metrics_tracker.get_metrics_history(limit)
    return {"history": history}


@app.get("/api/quality/inter-rater-agreement")
async def calculate_inter_rater_agreement(
    slide_id: str = Query(..., description="Slide ID to analyze"),
    expert_id_1: str = Query(..., description="First expert ID"),
    expert_id_2: str = Query(..., description="Second expert ID"),
):
    """Calculate inter-rater agreement (Cohen's kappa) between two experts"""
    # Get annotations for each expert
    annotations_expert1 = [a for a in annotations_db.values() if a.expert_id == expert_id_1]

    annotations_expert2 = [a for a in annotations_db.values() if a.expert_id == expert_id_2]

    if not annotations_expert1 or not annotations_expert2:
        raise HTTPException(status_code=400, detail="Insufficient annotations for both experts")

    # Calculate Cohen's kappa
    agreement_metrics = InterRaterAgreement.calculate_cohens_kappa(
        annotations_expert1, annotations_expert2, slide_id
    )

    return {
        "slide_id": slide_id,
        "expert_1": expert_id_1,
        "expert_2": expert_id_2,
        "agreement_metrics": agreement_metrics,
    }


@app.get("/api/quality/alerts")
async def get_quality_alerts():
    """Get current quality alerts"""
    annotations = list(annotations_db.values())

    # Compute current metrics
    metrics = quality_metrics_tracker.compute_metrics(
        annotations, validation_results_db, time_window_days=7
    )

    # Detect alerts
    alerts = quality_metrics_tracker.detect_quality_alerts(metrics)

    return {"timestamp": datetime.now().isoformat(), "alert_count": len(alerts), "alerts": alerts}


@app.get("/api/quality/expert/{expert_id}/performance")
async def get_expert_performance(expert_id: str, time_window_days: int = Query(30, ge=1, le=365)):
    """Get performance metrics for specific expert"""
    # Get expert's annotations
    expert_annotations = [a for a in annotations_db.values() if a.expert_id == expert_id]

    if not expert_annotations:
        raise HTTPException(status_code=404, detail=f"No annotations found for expert {expert_id}")

    # Filter by time window
    cutoff_time = datetime.now() - timedelta(days=time_window_days)
    recent_annotations = [a for a in expert_annotations if a.created_at >= cutoff_time]

    # Calculate metrics
    total = len(recent_annotations)
    confidences = [a.confidence for a in recent_annotations]

    # Validation metrics
    validation_results = [
        validation_results_db[a.id] for a in recent_annotations if a.id in validation_results_db
    ]

    quality_scores = [v.quality_score for v in validation_results]

    # Time metrics
    annotation_times = [
        a.annotation_time_seconds
        for a in recent_annotations
        if a.annotation_time_seconds is not None
    ]

    avg_annotation_time = np.mean(annotation_times) if annotation_times else None
    median_annotation_time = np.median(annotation_times) if annotation_times else None

    return {
        "expert_id": expert_id,
        "time_window_days": time_window_days,
        "total_annotations": total,
        "annotations_per_day": round(total / time_window_days, 2),
        "avg_confidence": round(np.mean(confidences), 3) if confidences else 0.0,
        "avg_quality_score": round(np.mean(quality_scores), 3) if quality_scores else 0.0,
        "avg_annotation_time_seconds": (
            round(avg_annotation_time, 2) if avg_annotation_time else None
        ),
        "median_annotation_time_seconds": (
            round(median_annotation_time, 2) if median_annotation_time else None
        ),
        "quality_distribution": {
            "excellent": sum(
                1 for v in validation_results if v.quality_status.value == "excellent"
            ),
            "good": sum(1 for v in validation_results if v.quality_status.value == "good"),
            "needs_review": sum(
                1 for v in validation_results if v.quality_status.value == "needs_review"
            ),
            "poor": sum(1 for v in validation_results if v.quality_status.value == "poor"),
        },
    }


@app.get("/api/quality/annotation-time-analytics")
async def get_annotation_time_analytics(
    time_window_days: int = Query(30, ge=1, le=365), expert_id: Optional[str] = Query(None)
):
    """Get annotation time analytics and trends"""
    # Get annotations
    annotations = list(annotations_db.values())

    # Filter by expert if specified
    if expert_id:
        annotations = [a for a in annotations if a.expert_id == expert_id]

    # Filter by time window
    cutoff_time = datetime.now() - timedelta(days=time_window_days)
    recent_annotations = [a for a in annotations if a.created_at >= cutoff_time]

    # Get annotations with time data
    timed_annotations = [a for a in recent_annotations if a.annotation_time_seconds is not None]

    if not timed_annotations:
        return {
            "time_window_days": time_window_days,
            "expert_id": expert_id,
            "total_annotations": len(recent_annotations),
            "annotations_with_time_data": 0,
            "message": "No annotation time data available",
        }

    # Calculate time metrics
    times = [a.annotation_time_seconds for a in timed_annotations]

    # Group by expert for comparison
    expert_times = {}
    for a in timed_annotations:
        if a.expert_id not in expert_times:
            expert_times[a.expert_id] = []
        expert_times[a.expert_id].append(a.annotation_time_seconds)

    expert_stats = {}
    for exp_id, exp_times in expert_times.items():
        expert_stats[exp_id] = {
            "count": len(exp_times),
            "avg_time": round(np.mean(exp_times), 2),
            "median_time": round(np.median(exp_times), 2),
            "min_time": round(np.min(exp_times), 2),
            "max_time": round(np.max(exp_times), 2),
            "std_time": round(np.std(exp_times), 2),
        }

    # Calculate quality vs time correlation
    quality_time_data = []
    for a in timed_annotations:
        if a.id in validation_results_db:
            quality_time_data.append(
                {
                    "time": a.annotation_time_seconds,
                    "quality_score": validation_results_db[a.id].quality_score,
                }
            )

    return {
        "time_window_days": time_window_days,
        "expert_id": expert_id,
        "total_annotations": len(recent_annotations),
        "annotations_with_time_data": len(timed_annotations),
        "overall_stats": {
            "avg_time_seconds": round(np.mean(times), 2),
            "median_time_seconds": round(np.median(times), 2),
            "min_time_seconds": round(np.min(times), 2),
            "max_time_seconds": round(np.max(times), 2),
            "std_time_seconds": round(np.std(times), 2),
        },
        "expert_stats": expert_stats,
        "quality_time_correlation": quality_time_data,
    }


@app.get("/api/quality/quality-trends")
async def get_quality_trends(
    time_window_days: int = Query(30, ge=1, le=365),
    expert_id: Optional[str] = Query(None),
    granularity: str = Query("daily", pattern="^(daily|weekly)$"),
):
    """Get quality trends over time"""
    # Get annotations
    annotations = list(annotations_db.values())

    # Filter by expert if specified
    if expert_id:
        annotations = [a for a in annotations if a.expert_id == expert_id]

    # Filter by time window
    cutoff_time = datetime.now() - timedelta(days=time_window_days)
    recent_annotations = [a for a in annotations if a.created_at >= cutoff_time]

    if not recent_annotations:
        return {
            "time_window_days": time_window_days,
            "expert_id": expert_id,
            "granularity": granularity,
            "trends": [],
            "message": "No annotations in time window",
        }

    # Group by time period
    from collections import defaultdict

    time_buckets = defaultdict(list)

    for a in recent_annotations:
        if granularity == "daily":
            bucket_key = a.created_at.date().isoformat()
        else:  # weekly
            week_start = a.created_at.date() - timedelta(days=a.created_at.weekday())
            bucket_key = week_start.isoformat()

        time_buckets[bucket_key].append(a)

    # Calculate metrics for each bucket
    trends = []
    for bucket_key in sorted(time_buckets.keys()):
        bucket_annotations = time_buckets[bucket_key]

        # Quality metrics
        validation_results = [
            validation_results_db[a.id] for a in bucket_annotations if a.id in validation_results_db
        ]

        quality_scores = [v.quality_score for v in validation_results]
        confidences = [a.confidence for a in bucket_annotations]

        # Time metrics
        annotation_times = [
            a.annotation_time_seconds
            for a in bucket_annotations
            if a.annotation_time_seconds is not None
        ]

        trends.append(
            {
                "period": bucket_key,
                "annotation_count": len(bucket_annotations),
                "avg_quality_score": round(np.mean(quality_scores), 3) if quality_scores else None,
                "avg_confidence": round(np.mean(confidences), 3) if confidences else None,
                "avg_annotation_time": (
                    round(np.mean(annotation_times), 2) if annotation_times else None
                ),
                "quality_distribution": {
                    "excellent": sum(
                        1 for v in validation_results if v.quality_status.value == "excellent"
                    ),
                    "good": sum(1 for v in validation_results if v.quality_status.value == "good"),
                    "needs_review": sum(
                        1 for v in validation_results if v.quality_status.value == "needs_review"
                    ),
                    "poor": sum(1 for v in validation_results if v.quality_status.value == "poor"),
                },
            }
        )

    return {
        "time_window_days": time_window_days,
        "expert_id": expert_id,
        "granularity": granularity,
        "trends": trends,
    }


# ============================================================================
# Utility Functions for Integration
# ============================================================================


def add_slide_to_db(slide_info: SlideInfo):
    """Add slide to database (for integration with active learning)"""
    slides_db[slide_info.slide_id] = slide_info
    logger.info(f"Added slide {slide_info.slide_id} to database")


def add_task_to_queue(task: AnnotationQueueItem):
    """Add task to annotation queue (for integration with active learning)"""
    queue_db[task.task_id] = task
    logger.info(f"Added task {task.task_id} to queue")


def get_annotations_for_slide(slide_id: str) -> List[Annotation]:
    """Get all annotations for a slide"""
    return [a for a in annotations_db.values() if a.slide_id == slide_id]


# ============================================================================
# Startup Event
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Expert Annotation Interface started")
    logger.info(f"API documentation available at /docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
