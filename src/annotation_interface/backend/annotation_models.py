"""
Data models for annotation interface
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class AnnotationType(str, Enum):
    """Types of annotations"""
    POLYGON = "polygon"
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    FREEHAND = "freehand"
    POINT = "point"


class AnnotationLabel(str, Enum):
    """Annotation labels for pathology"""
    TUMOR = "tumor"
    NORMAL = "normal"
    NECROSIS = "necrosis"
    INFLAMMATION = "inflammation"
    STROMA = "stroma"
    OTHER = "other"


class Point(BaseModel):
    """2D point coordinates"""
    x: float
    y: float


class AnnotationGeometry(BaseModel):
    """Geometry data for annotations"""
    type: AnnotationType
    points: List[Point] = Field(default_factory=list)
    center: Optional[Point] = None
    radius: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


class AnnotationCreate(BaseModel):
    """Request model for creating annotation"""
    slide_id: str = Field(..., description="Slide identifier")
    task_id: Optional[str] = Field(None, description="Associated annotation task ID")
    label: AnnotationLabel = Field(..., description="Annotation label")
    geometry: AnnotationGeometry = Field(..., description="Annotation geometry")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Annotator confidence")
    comments: str = Field("", description="Additional comments")
    expert_id: str = Field(..., description="Expert/pathologist identifier")


class AnnotationUpdate(BaseModel):
    """Request model for updating annotation"""
    label: Optional[AnnotationLabel] = None
    geometry: Optional[AnnotationGeometry] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    comments: Optional[str] = None


class Annotation(BaseModel):
    """Complete annotation model"""
    id: str
    slide_id: str
    task_id: Optional[str]
    label: AnnotationLabel
    geometry: AnnotationGeometry
    confidence: float
    comments: str
    expert_id: str
    created_at: datetime
    updated_at: datetime
    annotation_time_seconds: Optional[float] = None


class AnnotationResponse(BaseModel):
    """Response model for annotation operations"""
    success: bool
    annotation: Optional[Annotation] = None
    message: str = ""


class SlideInfo(BaseModel):
    """Slide information for annotation interface"""
    slide_id: str
    image_path: str
    width: int
    height: int
    tile_size: int = 256
    max_zoom: int = 10
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AIPredictionOverlay(BaseModel):
    """AI prediction overlay data"""
    slide_id: str
    prediction_type: str
    confidence: float
    heatmap_url: Optional[str] = None
    regions: List[AnnotationGeometry] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnnotationQueueItem(BaseModel):
    """Item in annotation queue"""
    task_id: str
    slide_id: str
    priority: float
    uncertainty_score: float
    ai_prediction: Dict[str, Any]
    status: str
    created_at: datetime
    assigned_expert: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    annotation_time_seconds: Optional[float] = None
