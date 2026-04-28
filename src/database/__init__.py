"""
Database module for Medical AI platform.

Provides PostgreSQL database connectivity, models, and operations
for production deployment.
"""

from .connection import DatabaseManager, get_db_session
from .models import (
    Analysis,
    Case,
    User,
    DicomStudy,
    ModelResult,
    AuditLog
)
from .operations import (
    AnalysisOperations,
    CaseOperations,
    UserOperations,
    DicomOperations
)

__all__ = [
    'DatabaseManager',
    'get_db_session',
    'Analysis',
    'Case', 
    'User',
    'DicomStudy',
    'ModelResult',
    'AuditLog',
    'AnalysisOperations',
    'CaseOperations',
    'UserOperations',
    'DicomOperations'
]