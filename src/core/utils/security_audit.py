"""
Security Audit Logging

Centralized logging for security-relevant events.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class SecurityEventType(Enum):
    """Types of security events to log."""

    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"

    # Authorization events
    ACCESS_DENIED = "access_denied"
    PERMISSION_ESCALATION = "permission_escalation"

    # Data access events
    PHI_ACCESS = "phi_access"
    PHI_EXPORT = "phi_export"
    PHI_DELETE = "phi_delete"

    # Attack indicators
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_TOKEN = "invalid_token"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    PATH_TRAVERSAL_ATTEMPT = "path_traversal_attempt"
    XSS_ATTEMPT = "xss_attempt"

    # System events
    CONFIG_CHANGE = "config_change"
    ENCRYPTION_KEY_ROTATION = "encryption_key_rotation"


class SecurityAuditLogger:
    """Log security events for compliance and forensics."""

    def __init__(self, log_dir: str = "logs/security"):
        """Initialize security audit logger.

        Args:
            log_dir: Directory for security logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up dedicated security logger
        self.logger = logging.getLogger("security_audit")
        self.logger.setLevel(logging.INFO)

        # File handler for security events
        log_file = self.log_dir / f"security_audit_{datetime.now().strftime('%Y%m%d')}.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)

        # JSON formatter for structured logging
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}'
        )
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def log_event(
        self,
        event_type: SecurityEventType,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
    ) -> None:
        """Log a security event.

        Args:
            event_type: Type of security event
            user_id: User identifier (if applicable)
            ip_address: Source IP address
            details: Additional event details
            severity: Log severity (INFO, WARNING, ERROR, CRITICAL)
        """
        event = {
            "event_type": event_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details or {},
            "severity": severity,
        }

        # Log as JSON
        log_message = json.dumps(event)

        if severity == "CRITICAL":
            self.logger.critical(log_message)
        elif severity == "ERROR":
            self.logger.error(log_message)
        elif severity == "WARNING":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def log_login_attempt(
        self, username: str, success: bool, ip_address: str, reason: Optional[str] = None
    ) -> None:
        """Log login attempt.

        Args:
            username: Username attempting login
            success: Whether login succeeded
            ip_address: Source IP
            reason: Failure reason (if applicable)
        """
        event_type = SecurityEventType.LOGIN_SUCCESS if success else SecurityEventType.LOGIN_FAILURE
        severity = "INFO" if success else "WARNING"

        self.log_event(
            event_type=event_type,
            user_id=username,
            ip_address=ip_address,
            details={"reason": reason} if reason else None,
            severity=severity,
        )

    def log_phi_access(self, user_id: str, patient_id: str, action: str, ip_address: str) -> None:
        """Log PHI (Protected Health Information) access.

        Args:
            user_id: User accessing PHI
            patient_id: Patient whose data was accessed
            action: Action performed (read, write, delete, export)
            ip_address: Source IP
        """
        self.log_event(
            event_type=SecurityEventType.PHI_ACCESS,
            user_id=user_id,
            ip_address=ip_address,
            details={"patient_id": patient_id, "action": action},
            severity="INFO",
        )

    def log_attack_indicator(
        self, attack_type: SecurityEventType, ip_address: str, details: Dict[str, Any]
    ) -> None:
        """Log potential attack indicator.

        Args:
            attack_type: Type of attack detected
            ip_address: Source IP
            details: Attack details
        """
        self.log_event(
            event_type=attack_type, ip_address=ip_address, details=details, severity="ERROR"
        )


# Global security audit logger instance
_security_logger: Optional[SecurityAuditLogger] = None


def get_security_logger() -> SecurityAuditLogger:
    """Get global security audit logger instance."""
    global _security_logger
    if _security_logger is None:
        _security_logger = SecurityAuditLogger()
    return _security_logger
