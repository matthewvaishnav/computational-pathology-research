"""
Audit logging infrastructure for regulatory compliance.

This module provides comprehensive audit logging functionality including:
- Recording all prediction operations with input identifiers, model versions, timestamps, and outputs
- Recording all user access events (authentication, data queries, report generation)
- Recording all data modifications (patient data updates, report amendments)
- Recording system errors with stack traces and input data states
- Tamper-evident records with cryptographic signatures
- Log retention for regulatory duration (minimum 7 years for FDA)
- Audit log export for regulatory submissions
- Model training and validation event recording
"""

# Re-export all components for backward compatibility
from .audit_analysis import (
    AuditLogAnalyzer,
    audit_operation,
    create_default_audit_logger,
)
from .audit_compliance import ComplianceAuditLogger
from .audit_crypto import CryptographicSigner
from .audit_logger import AuditContextManager, AuditLogger
from .audit_models import AuditEvent, AuditEventType, AuditSeverity, SignedAuditRecord
from .audit_storage import AuditStorage, FileAuditStorage

__all__ = [
    # Models
    "AuditEventType",
    "AuditSeverity",
    "AuditEvent",
    "SignedAuditRecord",
    # Crypto
    "CryptographicSigner",
    # Storage
    "AuditStorage",
    "FileAuditStorage",
    # Logger
    "AuditLogger",
    "AuditContextManager",
    # Compliance
    "ComplianceAuditLogger",
    # Analysis
    "AuditLogAnalyzer",
    # Utilities
    "audit_operation",
    "create_default_audit_logger",
]
