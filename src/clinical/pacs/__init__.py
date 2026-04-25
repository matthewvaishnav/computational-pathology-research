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
- PACS Service: Complete service integration
"""

from .audit_logger import AuditLogger
from .configuration_manager import ConfigurationManager
from .data_models import (
    AnalysisResults,
    DetectedRegion,
    DiagnosticRecommendation,
    DicomPriority,
    OperationResult,
    PACSConfiguration,
    PACSEndpoint,
    PACSMetadata,
    PACSVendor,
    PerformanceConfig,
    SecurityConfig,
    SeriesInfo,
    StudyInfo,
    ValidationResult,
)
from .error_handling import DeadLetterQueue, DicomErrorHandler, NetworkErrorHandler
from .failover import FailoverManager
from .notification_system import NotificationSystem
from .pacs_adapter import PACSAdapter
from .pacs_service import PACSService
from .query_engine import QueryEngine
from .retrieval_engine import RetrievalEngine
from .security_manager import SecurityManager
from .storage_engine import StorageEngine, StructuredReportBuilder
from .vendor_adapters import (
    AgfaAdapter,
    GEAdapter,
    PhilipsAdapter,
    SiemensAdapter,
    VendorAdapter,
)
from .workflow_orchestrator import WorkflowOrchestrator

__all__ = [
    # Core engines
    "QueryEngine",
    "RetrievalEngine",
    "StorageEngine",
    "StructuredReportBuilder",
    # Security and configuration
    "SecurityManager",
    "ConfigurationManager",
    # Workflow and orchestration
    "WorkflowOrchestrator",
    "PACSAdapter",
    "PACSService",
    # Error handling
    "NetworkErrorHandler",
    "DicomErrorHandler",
    "DeadLetterQueue",
    "FailoverManager",
    # Monitoring and compliance
    "AuditLogger",
    "NotificationSystem",
    # Vendor adapters
    "VendorAdapter",
    "GEAdapter",
    "PhilipsAdapter",
    "SiemensAdapter",
    "AgfaAdapter",
    # Data models
    "StudyInfo",
    "SeriesInfo",
    "PACSEndpoint",
    "PACSConfiguration",
    "PACSMetadata",
    "PACSVendor",
    "SecurityConfig",
    "PerformanceConfig",
    "AnalysisResults",
    "DetectedRegion",
    "DiagnosticRecommendation",
    "DicomPriority",
    "OperationResult",
    "ValidationResult",
]
