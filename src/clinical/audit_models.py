"""Audit event data models and enumerations."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class AuditEventType(Enum):
    """Types of audit events."""

    PREDICTION_OPERATION = "prediction_operation"
    USER_ACCESS = "user_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_ERROR = "system_error"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    AUTHENTICATION = "authentication"
    DATA_EXPORT = "data_export"
    REPORT_GENERATION = "report_generation"
    CONFIGURATION_CHANGE = "configuration_change"


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Individual audit event record."""

    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_token: Optional[str]
    severity: AuditSeverity
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    input_data_hash: Optional[str] = None
    output_data_hash: Optional[str] = None
    model_version: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def __post_init__(self):
        """Validate and process audit event after initialization."""
        if isinstance(self.event_type, str):
            self.event_type = AuditEventType(self.event_type)
        if isinstance(self.severity, str):
            self.severity = AuditSeverity(self.severity)

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary for serialization."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """Create audit event from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def get_content_hash(self) -> str:
        """Get SHA-256 hash of event content for integrity verification."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class SignedAuditRecord:
    """Tamper-evident audit record with cryptographic signature."""

    event: AuditEvent
    signature: str
    public_key_fingerprint: str
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert signed record to dictionary."""
        return {
            "event": self.event.to_dict(),
            "signature": self.signature,
            "public_key_fingerprint": self.public_key_fingerprint,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignedAuditRecord":
        """Create signed record from dictionary."""
        return cls(
            event=AuditEvent.from_dict(data["event"]),
            signature=data["signature"],
            public_key_fingerprint=data["public_key_fingerprint"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )
