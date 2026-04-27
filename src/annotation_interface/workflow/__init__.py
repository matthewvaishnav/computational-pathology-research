"""
Clinical Workflow Integration for Annotation Interface

This module integrates the annotation interface with:
- Active learning system for automatic case queuing
- PACS for slide retrieval
- WSI streaming for real-time viewing
- Notification system for pathologists
"""

from .clinical_integration import ClinicalWorkflowIntegrator
from .active_learning_connector import ActiveLearningConnector
from .pacs_connector import PACSConnector
from .notification_service import NotificationService

__all__ = [
    'ClinicalWorkflowIntegrator',
    'ActiveLearningConnector',
    'PACSConnector',
    'NotificationService'
]
