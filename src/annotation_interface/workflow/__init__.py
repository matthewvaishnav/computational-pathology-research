"""
Clinical Workflow Integration for Annotation Interface

This module integrates the annotation interface with:
- Active learning system for automatic case queuing
- PACS for slide retrieval
- WSI streaming for real-time viewing
- Notification system for pathologists
"""

from .active_learning_connector import ActiveLearningConnector
from .clinical_integration import ClinicalWorkflowIntegrator
from .notification_service import NotificationService
from .pacs_connector import PACSConnector

__all__ = [
    "ClinicalWorkflowIntegrator",
    "ActiveLearningConnector",
    "PACSConnector",
    "NotificationService",
]
