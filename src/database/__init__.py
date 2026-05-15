"""
Database module for Medical AI platform.

Provides PostgreSQL database connectivity, models, and operations
for production deployment.
"""

from .connection import DatabaseManager, get_db_session, initialize_database
from .models import Analysis, AuditLog, Case, DicomStudy, ModelResult, User
from .operations import (
    AnalysisOperations,
    AuditOperations,
    CaseOperations,
    DicomOperations,
    UserOperations,
)

__all__ = [
    "DatabaseManager",
    "get_db_session",
    "initialize_database",
    "Analysis",
    "Case",
    "User",
    "DicomStudy",
    "ModelResult",
    "AuditLog",
    "AnalysisOperations",
    "AuditOperations",
    "CaseOperations",
    "UserOperations",
    "DicomOperations",
]
