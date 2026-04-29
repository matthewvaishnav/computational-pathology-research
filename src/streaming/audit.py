"""
Audit logging for HistoCore Real-Time WSI Streaming.

HIPAA-compliant audit trails, user activity tracking, compliance reporting.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Audit Event Types
# ============================================================================


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    PASSWORD_CHANGE = "password_change"

    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGE = "permission_change"

    # Data access events
    SLIDE_VIEWED = "slide_viewed"
    SLIDE_PROCESSED = "slide_processed"
    SLIDE_EXPORTED = "slide_exported"
    SLIDE_DELETED = "slide_deleted"

    # Result events
    RESULT_VIEWED = "result_viewed"
    RESULT_MODIFIED = "result_modified"
    RESULT_APPROVED = "result_approved"
    RESULT_EXPORTED = "result_exported"

    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    USER_CREATED = "user_created"
    USER_MODIFIED = "user_modified"
    USER_DELETED = "user_deleted"

    # Data events
    DATA_EXPORT = "data_export"
    DATA_DELETE = "data_delete"
    DATA_BACKUP = "data_backup"
    DATA_RESTORE = "data_restore"

    # Security events
    SECURITY_ALERT = "security_alert"
    ENCRYPTION_KEY_ROTATION = "encryption_key_rotation"
    CERTIFICATE_RENEWAL = "certificate_renewal"


class AuditSeverity(str, Enum):
    """Audit event severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================================
# Audit Event Model
# ============================================================================


@dataclass
class AuditEvent:
    """Audit event record."""

    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str]
    username: Optional[str]
    organization: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    action: str
    outcome: str  # success, failure, partial
    details: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    def get_hash(self) -> str:
        """Get event hash for integrity verification."""
        event_str = f"{self.event_id}{self.timestamp}{self.event_type}{self.user_id}{self.action}"
        return hashlib.sha256(event_str.encode()).hexdigest()


# ============================================================================
# Audit Logger
# ============================================================================


class AuditLogger:
    """Audit event logger."""

    def __init__(
        self, log_file: str = "./audit.log", enable_console: bool = False, enable_json: bool = True
    ):
        """Initialize audit logger."""
        self.log_file = Path(log_file)
        self.enable_console = enable_console
        self.enable_json = enable_json

        # Create log directory
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Event counter
        self.event_counter = 0

        logger.info("Audit logger initialized: log_file=%s", log_file)

    def log_event(self, event: AuditEvent):
        """Log audit event."""
        self.event_counter += 1

        # Write to file
        with open(self.log_file, "a") as f:
            if self.enable_json:
                f.write(event.to_json() + "\n")
            else:
                f.write(self._format_event(event) + "\n")

        # Console output
        if self.enable_console:
            print(f"[AUDIT] {self._format_event(event)}")

        # System logger
        logger.info(
            "Audit event: %s - %s by %s",
            event.event_type.value,
            event.action,
            event.username or "system",
        )

    def _format_event(self, event: AuditEvent) -> str:
        """Format event for text logging."""
        return (
            f"{event.timestamp.isoformat()} | "
            f"{event.severity.value.upper()} | "
            f"{event.event_type.value} | "
            f"user={event.username or 'N/A'} | "
            f"action={event.action} | "
            f"outcome={event.outcome} | "
            f"resource={event.resource_type}:{event.resource_id or 'N/A'}"
        )

    def get_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
    ) -> List[AuditEvent]:
        """Query audit events."""
        events = []

        if not self.log_file.exists():
            return events

        with open(self.log_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    event = self._dict_to_event(data)

                    # Apply filters
                    if start_time and event.timestamp < start_time:
                        continue
                    if end_time and event.timestamp > end_time:
                        continue
                    if event_type and event.event_type != event_type:
                        continue
                    if user_id and event.user_id != user_id:
                        continue

                    events.append(event)

                except Exception as e:
                    logger.error("Failed to parse audit event: %s", e)

        return events

    def _dict_to_event(self, data: Dict[str, Any]) -> AuditEvent:
        """Convert dictionary to AuditEvent."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["event_type"] = AuditEventType(data["event_type"])
        data["severity"] = AuditSeverity(data["severity"])
        return AuditEvent(**data)


# ============================================================================
# Audit Manager
# ============================================================================


class AuditManager:
    """Main audit manager."""

    def __init__(self, log_file: str = "./audit.log", enable_console: bool = False):
        """Initialize audit manager."""
        self.logger = AuditLogger(log_file, enable_console)
        self.event_id_counter = 0

        logger.info("Audit manager initialized")

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        self.event_id_counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"AE-{timestamp}-{self.event_id_counter:06d}"

    def log_authentication(
        self,
        event_type: AuditEventType,
        username: str,
        outcome: str,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log authentication event."""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=AuditSeverity.INFO if outcome == "success" else AuditSeverity.WARNING,
            user_id=None,
            username=username,
            organization=None,
            ip_address=ip_address,
            user_agent=None,
            resource_type="authentication",
            resource_id=None,
            action=event_type.value,
            outcome=outcome,
            details=details or {},
        )

        self.logger.log_event(event)

    def log_data_access(
        self,
        event_type: AuditEventType,
        user_id: str,
        username: str,
        organization: str,
        resource_type: str,
        resource_id: str,
        action: str,
        outcome: str = "success",
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log data access event."""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=AuditSeverity.INFO,
            user_id=user_id,
            username=username,
            organization=organization,
            ip_address=ip_address,
            user_agent=None,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            details=details or {},
        )

        self.logger.log_event(event)

    def log_authorization(
        self,
        user_id: str,
        username: str,
        permission: str,
        resource_type: str,
        resource_id: str,
        granted: bool,
        ip_address: Optional[str] = None,
    ):
        """Log authorization event."""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.ACCESS_GRANTED if granted else AuditEventType.ACCESS_DENIED,
            severity=AuditSeverity.INFO if granted else AuditSeverity.WARNING,
            user_id=user_id,
            username=username,
            organization=None,
            ip_address=ip_address,
            user_agent=None,
            resource_type=resource_type,
            resource_id=resource_id,
            action=f"check_permission:{permission}",
            outcome="granted" if granted else "denied",
            details={"permission": permission},
        )

        self.logger.log_event(event)

    def log_system_event(
        self,
        event_type: AuditEventType,
        action: str,
        outcome: str = "success",
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log system event."""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=AuditSeverity.INFO,
            user_id=None,
            username="system",
            organization=None,
            ip_address=None,
            user_agent=None,
            resource_type="system",
            resource_id=None,
            action=action,
            outcome=outcome,
            details=details or {},
        )

        self.logger.log_event(event)

    def log_security_event(
        self, action: str, severity: AuditSeverity, details: Optional[Dict[str, Any]] = None
    ):
        """Log security event."""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.SECURITY_ALERT,
            severity=severity,
            user_id=None,
            username="system",
            organization=None,
            ip_address=None,
            user_agent=None,
            resource_type="security",
            resource_id=None,
            action=action,
            outcome="alert",
            details=details or {},
        )

        self.logger.log_event(event)

    def generate_compliance_report(
        self, start_time: datetime, end_time: datetime, output_file: str
    ):
        """Generate compliance report."""
        events = self.logger.get_events(start_time=start_time, end_time=end_time)

        # Aggregate statistics
        stats = {
            "total_events": len(events),
            "by_type": {},
            "by_user": {},
            "by_severity": {},
            "failed_access_attempts": 0,
            "data_exports": 0,
            "data_deletions": 0,
        }

        for event in events:
            # By type
            event_type = event.event_type.value
            stats["by_type"][event_type] = stats["by_type"].get(event_type, 0) + 1

            # By user
            if event.username:
                stats["by_user"][event.username] = stats["by_user"].get(event.username, 0) + 1

            # By severity
            severity = event.severity.value
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1

            # Special counts
            if event.event_type == AuditEventType.ACCESS_DENIED:
                stats["failed_access_attempts"] += 1
            if event.event_type == AuditEventType.DATA_EXPORT:
                stats["data_exports"] += 1
            if event.event_type == AuditEventType.DATA_DELETE:
                stats["data_deletions"] += 1

        # Generate report
        report = {
            "report_generated": datetime.utcnow().isoformat(),
            "period_start": start_time.isoformat(),
            "period_end": end_time.isoformat(),
            "statistics": stats,
            "events": [event.to_dict() for event in events],
        }

        # Write report
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Generated compliance report: %s (%d events)", output_file, len(events))

        return report


# ============================================================================
# Audit Decorators
# ============================================================================


def audit_data_access(audit_manager: AuditManager, resource_type: str):
    """Decorator to audit data access."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            user = kwargs.get("user")
            resource_id = kwargs.get("resource_id", "unknown")

            try:
                result = func(*args, **kwargs)

                if user:
                    audit_manager.log_data_access(
                        event_type=AuditEventType.SLIDE_VIEWED,
                        user_id=user.user_id,
                        username=user.username,
                        organization=user.organization,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        action=func.__name__,
                        outcome="success",
                    )

                return result

            except Exception as e:
                if user:
                    audit_manager.log_data_access(
                        event_type=AuditEventType.SLIDE_VIEWED,
                        user_id=user.user_id,
                        username=user.username,
                        organization=user.organization,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        action=func.__name__,
                        outcome="failure",
                        details={"error": str(e)},
                    )
                raise

        return wrapper

    return decorator


# ============================================================================
# Convenience Functions
# ============================================================================


def create_audit_manager(log_file: str = "./audit.log") -> AuditManager:
    """Create audit manager with default configuration."""
    return AuditManager(log_file=log_file)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create audit manager
    audit = create_audit_manager()

    # Log authentication
    audit.log_authentication(
        event_type=AuditEventType.LOGIN_SUCCESS,
        username="dr_smith",
        outcome="success",
        ip_address="192.168.1.100",
    )

    # Log data access
    audit.log_data_access(
        event_type=AuditEventType.SLIDE_VIEWED,
        user_id="user123",
        username="dr_smith",
        organization="General Hospital",
        resource_type="slide",
        resource_id="slide_001",
        action="view_slide",
        ip_address="192.168.1.100",
    )

    # Log authorization
    audit.log_authorization(
        user_id="user123",
        username="dr_smith",
        permission="view_slides",
        resource_type="slide",
        resource_id="slide_001",
        granted=True,
        ip_address="192.168.1.100",
    )

    # Generate compliance report
    report = audit.generate_compliance_report(
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        output_file="./compliance_report.json",
    )

    print(f"Generated report with {report['statistics']['total_events']} events")
