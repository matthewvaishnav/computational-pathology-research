"""
Expert Annotation Interface for Continuous Learning
Web-based tool for pathologists to annotate high-uncertainty cases
"""

from .backend.annotation_api import app as annotation_app
from .backend.annotation_models import (
    Annotation,
    AnnotationCreate,
    AnnotationResponse,
    AnnotationUpdate,
)

__all__ = [
    "annotation_app",
    "Annotation",
    "AnnotationCreate",
    "AnnotationUpdate",
    "AnnotationResponse",
]
