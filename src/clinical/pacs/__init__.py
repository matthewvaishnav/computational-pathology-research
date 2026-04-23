"""
PACS Integration System for HistoCore

This module provides comprehensive PACS integration capabilities for clinical deployment,
including DICOM query/retrieve/store operations, multi-vendor support, security, and
workflow automation.

Components:
- Query Engine: DICOM C-FIND operations for study/series queries
- Retrieval Engine: DICOM C-MOVE operations for WSI file retrieval
- Storage Engine: DICOM C-STORE operations for AI results storage
- Security Manager: TLS encryption and certificate management
- Configuration Manager: Multi-environment PACS configuration
- Workflow Orchestrator: Automated processing workflows
- Audit Logger: HIPAA-compliant audit logging
- Notification System: Multi-channel clinical alerts
- PACS Adapter: Main orchestration interface
"""

from .query_engine import QueryEngine
from .retrieval_engine import RetrievalEngine
from .storage_engine import StorageEngine, StructuredReportBuilder
from .security_manager import SecurityManager
from .configuration_manager import ConfigurationManager
from .workflow_orchestrator import WorkflowOrchestrator
from .pacs_adapter import PACSAdapter
from .data_models import (
    StudyInfo,
    SeriesInfo,
    PACSEndpoint,
    PACSConfiguration,
    AnalysisResults,
    PACSMetadata,
    SecurityConfig,
    PerformanceConfig,
    PACSVendor,
    DicomPriority,
    ValidationResult,
    OperationResult,
    DetectedRegion,
    DiagnosticRecommendation,
)

__all__ = [
    "QueryEngine",
    "RetrievalEngine", 
    "StorageEngine",
    "StructuredReportBuilder",
    "SecurityManager",
    "ConfigurationManager",
    "WorkflowOrchestrator",
    "PACSAdapter",
    "StudyInfo",
    "SeriesInfo",
    "PACSEndpoint",
    "PACSConfiguration",
    "AnalysisResults",
    "PACSMetadata",
    "SecurityConfig",
    "PerformanceConfig",
    "PACSVendor",
    "DicomPriority",
    "ValidationResult",
    "OperationResult",
    "DetectedRegion",
    "DiagnosticRecommendation",
]