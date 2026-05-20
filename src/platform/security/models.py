"""
Security models and enumerations for the HistoCore framework.

This module defines security-related data models and enumerations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class SecurityEnvironment(Enum):
    """Security environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ThreatLevel(Enum):
    """Threat level classifications."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityContext:
    """Security context for operations."""

    environment: SecurityEnvironment
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    trusted_sources: List[str] = None

    def __post_init__(self):
        if self.trusted_sources is None:
            self.trusted_sources = []


@dataclass
class AuditLogEntry:
    """Audit log entry for security events."""

    timestamp: str
    event_type: str
    user_id: Optional[str]
    resource: str
    action: str
    result: str
    threat_level: ThreatLevel
    details: Optional[dict] = None
