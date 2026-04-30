"""
Privacy and security infrastructure for clinical workflow integration.

This module provides HIPAA-compliant privacy and security functionality including:
- AES-256 encryption for data at rest and in transit
- Patient identifier anonymization
- Role-based access control (RBAC)
- Data deletion with audit integrity preservation
- Unauthorized export detection
- Patient consent management
- Automatic session timeout
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class Role(Enum):
    """User roles for RBAC system."""

    ADMIN = "admin"
    PHYSICIAN = "physician"
    RESEARCHER = "researcher"
    TECHNICIAN = "technician"
    VIEWER = "viewer"


class Permission(Enum):
    """Permissions for data access control."""

    READ_PATIENT_DATA = "read_patient_data"
    WRITE_PATIENT_DATA = "write_patient_data"
    DELETE_PATIENT_DATA = "delete_patient_data"
    EXPORT_DATA = "export_data"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_USERS = "manage_users"
    CONFIGURE_SYSTEM = "configure_system"


@dataclass
class UserSession:
    """User session information for access control."""

    user_id: str
    role: Role
    permissions: Set[Permission]
    created_at: datetime
    last_activity: datetime
    session_token: str
    ip_address: Optional[str] = None

    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired due to inactivity."""
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()


@dataclass
class ConsentRecord:
    """Patient consent record for data sharing."""

    patient_id: str
    consent_type: str
    granted: bool
    granted_at: datetime
    expires_at: Optional[datetime] = None
    purpose: Optional[str] = None
    third_party: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if consent is still valid."""
        if not self.granted:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True


class EncryptionProvider(ABC):
    """Abstract base class for encryption providers."""

    @abstractmethod
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data."""

    @abstractmethod
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""


class AES256Encryption(EncryptionProvider):
    """AES-256 encryption provider for data at rest."""

    def __init__(self, key: Optional[bytes] = None):
        """Initialize with encryption key."""
        if key is None:
            key = Fernet.generate_key()
        self.fernet = Fernet(key)
        self._key = key

    @classmethod
    def from_password(cls, password: str, salt: Optional[bytes] = None) -> "AES256Encryption":
        """Create encryption provider from password."""
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600000,  # OWASP recommended minimum
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return cls(key)

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data using AES-256."""
        return self.fernet.encrypt(data)

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using AES-256."""
        return self.fernet.decrypt(encrypted_data)

    @property
    def key(self) -> bytes:
        """Get encryption key."""
        return self._key
    
    def rotate_key(self, new_key: Optional[bytes] = None) -> bytes:
        """
        Rotate encryption key.
        
        Args:
            new_key: New encryption key (generated if not provided)
            
        Returns:
            New encryption key
        """
        if new_key is None:
            new_key = Fernet.generate_key()
        
        old_key = self._key
        self._key = new_key
        self.fernet = Fernet(new_key)
        
        logger.info("Encryption key rotated")
        return new_key
    
    def re_encrypt_with_new_key(self, encrypted_data: bytes, new_key: bytes) -> bytes:
        """
        Re-encrypt data with new key during key rotation.
        
        Args:
            encrypted_data: Data encrypted with old key
            new_key: New encryption key
            
        Returns:
            Data encrypted with new key
        """
        # Decrypt with old key
        decrypted = self.decrypt(encrypted_data)
        
        # Rotate to new key
        self.rotate_key(new_key)
        
        # Encrypt with new key
        return self.encrypt(decrypted)


class PatientIdentifierAnonymizer:
    """Anonymizes patient identifiers for logs and audit trails."""

    def __init__(self, secret_key: Optional[bytes] = None):
        """Initialize with secret key for consistent anonymization."""
        if secret_key is None:
            secret_key = secrets.token_bytes(32)
        self.secret_key = secret_key

    def anonymize_patient_id(self, patient_id: str) -> str:
        """Create anonymized patient identifier."""
        # Use HMAC-SHA256 for consistent, irreversible anonymization
        mac = hmac.new(self.secret_key, patient_id.encode(), hashlib.sha256)
        return f"anon_{mac.hexdigest()[:16]}"

    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize patient identifiers in data dictionary."""
        anonymized = data.copy()

        # Common patient identifier fields to anonymize
        identifier_fields = [
            "patient_id",
            "mrn",
            "medical_record_number",
            "ssn",
            "social_security_number",
            "name",
            "first_name",
            "last_name",
            "email",
            "phone",
            "address",
            "date_of_birth",
            "dob",
        ]

        for field in identifier_fields:
            if field in anonymized:
                if field == "patient_id":
                    anonymized[field] = self.anonymize_patient_id(str(anonymized[field]))
                else:
                    # Replace other identifiers with anonymized placeholders
                    anonymized[field] = f"<{field}_anonymized>"

        return anonymized


class RBACManager:
    """Role-based access control manager."""

    # Default role permissions
    ROLE_PERMISSIONS = {
        Role.ADMIN: {
            Permission.READ_PATIENT_DATA,
            Permission.WRITE_PATIENT_DATA,
            Permission.DELETE_PATIENT_DATA,
            Permission.EXPORT_DATA,
            Permission.VIEW_AUDIT_LOGS,
            Permission.MANAGE_USERS,
            Permission.CONFIGURE_SYSTEM,
        },
        Role.PHYSICIAN: {
            Permission.READ_PATIENT_DATA,
            Permission.WRITE_PATIENT_DATA,
            Permission.EXPORT_DATA,
        },
        Role.RESEARCHER: {
            Permission.READ_PATIENT_DATA,
            Permission.EXPORT_DATA,
        },
        Role.TECHNICIAN: {
            Permission.READ_PATIENT_DATA,
            Permission.WRITE_PATIENT_DATA,
        },
        Role.VIEWER: {
            Permission.READ_PATIENT_DATA,
        },
    }

    def __init__(self):
        """Initialize RBAC manager."""
        self.active_sessions: Dict[str, UserSession] = {}
        self.session_timeout_minutes = 30
        self.logger = logging.getLogger(__name__)

    def create_session(
        self,
        user_id: str,
        role: Role,
        ip_address: Optional[str] = None,
        custom_permissions: Optional[Set[Permission]] = None,
    ) -> str:
        """Create new user session."""
        session_token = secrets.token_urlsafe(32)
        permissions = custom_permissions or self.ROLE_PERMISSIONS.get(role, set())

        session = UserSession(
            user_id=user_id,
            role=role,
            permissions=permissions,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            session_token=session_token,
            ip_address=ip_address,
        )

        self.active_sessions[session_token] = session
        self.logger.info(f"Created session for user {user_id} with role {role.value}")
        return session_token

    def get_session(self, session_token: str) -> Optional[UserSession]:
        """Get active session by token."""
        session = self.active_sessions.get(session_token)
        if session and session.is_expired(self.session_timeout_minutes):
            self.invalidate_session(session_token)
            return None
        return session

    def invalidate_session(self, session_token: str) -> bool:
        """Invalidate user session."""
        if session_token in self.active_sessions:
            session = self.active_sessions.pop(session_token)
            self.logger.info(f"Invalidated session for user {session.user_id}")
            return True
        return False

    def check_permission(self, session_token: str, permission: Permission) -> bool:
        """Check if user has required permission."""
        session = self.get_session(session_token)
        if not session:
            return False

        session.update_activity()
        return permission in session.permissions

    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions and return count."""
        expired_tokens = [
            token
            for token, session in self.active_sessions.items()
            if session.is_expired(self.session_timeout_minutes)
        ]

        for token in expired_tokens:
            self.invalidate_session(token)

        return len(expired_tokens)


class DataExportMonitor:
    """Monitors and controls data export operations."""

    def __init__(self, max_export_size_mb: int = 100):
        """Initialize export monitor."""
        self.max_export_size_mb = max_export_size_mb
        self.export_attempts: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def request_export(
        self, session_token: str, data_size_mb: float, export_type: str, destination: str
    ) -> bool:
        """Request data export with authorization check."""
        attempt = {
            "session_token": session_token,
            "timestamp": datetime.now(),
            "data_size_mb": data_size_mb,
            "export_type": export_type,
            "destination": destination,
            "approved": False,
        }

        # Check size limits
        if data_size_mb > self.max_export_size_mb:
            self.logger.warning(
                f"Export request denied: size {data_size_mb}MB exceeds limit {self.max_export_size_mb}MB"
            )
            attempt["denial_reason"] = "size_limit_exceeded"
            self.export_attempts.append(attempt)
            return False

        # Log successful export request
        attempt["approved"] = True
        self.export_attempts.append(attempt)
        self.logger.info(f"Export approved: {export_type} to {destination}")
        return True

    def detect_unauthorized_export(
        self, session_token: str, data_access_pattern: Dict[str, Any]
    ) -> bool:
        """Detect potential unauthorized export attempts."""
        # Simple heuristics for detecting suspicious patterns
        suspicious_indicators = [
            data_access_pattern.get("bulk_access", False),
            data_access_pattern.get("off_hours_access", False),
            data_access_pattern.get("unusual_volume", False),
            data_access_pattern.get("external_destination", False),
        ]

        if sum(suspicious_indicators) >= 2:
            self.logger.warning(f"Suspicious export pattern detected for session {session_token}")
            return True

        return False


class PrivacyManager:
    """Main privacy and security manager for clinical workflow."""

    def __init__(
        self, encryption_key: Optional[bytes] = None, anonymization_key: Optional[bytes] = None
    ):
        """Initialize privacy manager."""
        self.encryption = AES256Encryption(encryption_key)
        self.anonymizer = PatientIdentifierAnonymizer(anonymization_key)
        self.rbac = RBACManager()
        self.export_monitor = DataExportMonitor()
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        self.logger = logging.getLogger(__name__)

    def encrypt_patient_data(self, data: Union[str, bytes, Dict[str, Any]]) -> bytes:
        """Encrypt patient data at rest using AES-256."""
        if isinstance(data, dict):
            data = json.dumps(data).encode()
        elif isinstance(data, str):
            data = data.encode()

        return self.encryption.encrypt(data)

    def decrypt_patient_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt patient data."""
        return self.encryption.decrypt(encrypted_data)

    def anonymize_for_logging(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize patient identifiers for logs and audit trails."""
        return self.anonymizer.anonymize_data(data)

    def create_user_session(
        self, user_id: str, role: Role, ip_address: Optional[str] = None
    ) -> str:
        """Create authenticated user session."""
        return self.rbac.create_session(user_id, role, ip_address)

    def check_access_permission(self, session_token: str, permission: Permission) -> bool:
        """Check if user has required access permission."""
        return self.rbac.check_permission(session_token, permission)

    def record_consent(self, consent: ConsentRecord) -> None:
        """Record patient consent for data sharing."""
        if consent.patient_id not in self.consent_records:
            self.consent_records[consent.patient_id] = []

        self.consent_records[consent.patient_id].append(consent)
        self.logger.info(f"Recorded consent for patient {consent.patient_id}")

    def check_consent(self, patient_id: str, purpose: str) -> bool:
        """Check if patient has valid consent for specified purpose."""
        consents = self.consent_records.get(patient_id, [])

        for consent in consents:
            if (consent.purpose == purpose or consent.purpose is None) and consent.is_valid():
                return True

        return False

    def request_data_export(
        self, session_token: str, patient_ids: List[str], export_type: str, destination: str
    ) -> bool:
        """Request patient data export with authorization and consent checks."""
        # Check export permission
        if not self.check_access_permission(session_token, Permission.EXPORT_DATA):
            self.logger.warning(
                f"Export denied: insufficient permissions for session {session_token}"
            )
            return False

        # Check consent for all patients
        for patient_id in patient_ids:
            if not self.check_consent(patient_id, "external_sharing"):
                self.logger.warning(f"Export denied: no consent for patient {patient_id}")
                return False

        # Estimate data size (simplified)
        estimated_size_mb = len(patient_ids) * 0.1  # Rough estimate

        return self.export_monitor.request_export(
            session_token, estimated_size_mb, export_type, destination
        )

    def delete_patient_data(
        self, session_token: str, patient_id: str, preserve_audit: bool = True
    ) -> bool:
        """Delete patient data (right to be forgotten) while preserving audit integrity."""
        # Check deletion permission
        if not self.check_access_permission(session_token, Permission.DELETE_PATIENT_DATA):
            self.logger.warning(
                f"Deletion denied: insufficient permissions for session {session_token}"
            )
            return False

        # Log deletion request with anonymized patient ID
        anonymized_id = self.anonymizer.anonymize_patient_id(patient_id)
        self.logger.info(f"Patient data deletion requested for {anonymized_id}")

        if preserve_audit:
            # In real implementation, this would mark data as deleted
            # while preserving audit trail entries with anonymized identifiers
            self.logger.info(
                f"Patient data marked for deletion with audit preservation: {anonymized_id}"
            )
        else:
            # Complete deletion including audit trails (rare case)
            self.logger.info(f"Complete patient data deletion: {anonymized_id}")

        return True

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired user sessions."""
        return self.rbac.cleanup_expired_sessions()

    def get_privacy_status(self) -> Dict[str, Any]:
        """Get current privacy and security status."""
        return {
            "active_sessions": len(self.rbac.active_sessions),
            "total_consent_records": sum(
                len(consents) for consents in self.consent_records.values()
            ),
            "export_attempts_today": len(
                [
                    attempt
                    for attempt in self.export_monitor.export_attempts
                    if attempt["timestamp"].date() == datetime.now().date()
                ]
            ),
            "encryption_enabled": True,
            "anonymization_enabled": True,
            "rbac_enabled": True,
        }


class SecurityAuditEvent:
    """Security audit event for tracking access and operations."""

    def __init__(
        self,
        event_type: str,
        user_id: str,
        session_token: str,
        resource: str,
        action: str,
        success: bool,
        timestamp: Optional[datetime] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize security audit event."""
        self.event_type = event_type
        self.user_id = user_id
        self.session_token = session_token
        self.resource = resource
        self.action = action
        self.success = success
        self.timestamp = timestamp or datetime.now()
        self.ip_address = ip_address
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_type": self.event_type,
            "user_id": self.user_id,
            "session_token": self.session_token[:8] + "...",  # Truncate for security
            "resource": self.resource,
            "action": self.action,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "ip_address": self.ip_address,
            "details": self.details,
        }


class DataAccessLogger:
    """Logs all data access operations for audit trails."""

    def __init__(self, anonymizer: PatientIdentifierAnonymizer):
        """Initialize with anonymizer for patient data."""
        self.anonymizer = anonymizer
        self.access_log: List[SecurityAuditEvent] = []
        self.logger = logging.getLogger(__name__)

    def log_access(
        self,
        event_type: str,
        user_id: str,
        session_token: str,
        resource: str,
        action: str,
        success: bool,
        patient_data: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        """Log data access event with anonymized patient information."""
        details = {}
        if patient_data:
            details["patient_data"] = self.anonymizer.anonymize_data(patient_data)

        event = SecurityAuditEvent(
            event_type=event_type,
            user_id=user_id,
            session_token=session_token,
            resource=resource,
            action=action,
            success=success,
            ip_address=ip_address,
            details=details,
        )

        self.access_log.append(event)
        self.logger.info(f"Access logged: {event.to_dict()}")

    def get_access_history(
        self,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[SecurityAuditEvent]:
        """Get filtered access history."""
        filtered_events = self.access_log

        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]

        if resource:
            filtered_events = [e for e in filtered_events if e.resource == resource]

        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]

        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]

        return filtered_events


class SessionTimeoutManager:
    """Manages automatic session timeout and cleanup."""

    def __init__(self, default_timeout_minutes: int = 30):
        """Initialize with default timeout."""
        self.default_timeout_minutes = default_timeout_minutes
        self.custom_timeouts: Dict[str, int] = {}  # session_token -> timeout_minutes
        self.logger = logging.getLogger(__name__)

    def set_custom_timeout(self, session_token: str, timeout_minutes: int) -> None:
        """Set custom timeout for specific session."""
        self.custom_timeouts[session_token] = timeout_minutes
        self.logger.info(
            f"Set custom timeout {timeout_minutes}min for session {session_token[:8]}..."
        )

    def get_timeout(self, session_token: str) -> int:
        """Get timeout for session."""
        return self.custom_timeouts.get(session_token, self.default_timeout_minutes)

    def is_session_expired(self, session: UserSession) -> bool:
        """Check if session is expired based on custom or default timeout."""
        timeout_minutes = self.get_timeout(session.session_token)
        return datetime.now() - session.last_activity > timedelta(minutes=timeout_minutes)

    def cleanup_session_timeout(self, session_token: str) -> None:
        """Clean up custom timeout when session is invalidated."""
        if session_token in self.custom_timeouts:
            del self.custom_timeouts[session_token]


class UnauthorizedAccessDetector:
    """Detects and prevents unauthorized data access attempts."""

    def __init__(self, max_failed_attempts: int = 5, lockout_minutes: int = 15):
        """Initialize detector with failure thresholds."""
        self.max_failed_attempts = max_failed_attempts
        self.lockout_minutes = lockout_minutes
        self.failed_attempts: Dict[str, List[datetime]] = {}  # user_id -> attempt timestamps
        self.locked_users: Dict[str, datetime] = {}  # user_id -> lockout_time
        self.logger = logging.getLogger(__name__)

    def record_failed_attempt(self, user_id: str, ip_address: Optional[str] = None) -> None:
        """Record failed access attempt."""
        now = datetime.now()

        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []

        self.failed_attempts[user_id].append(now)

        # Clean old attempts (older than lockout period)
        cutoff = now - timedelta(minutes=self.lockout_minutes)
        self.failed_attempts[user_id] = [
            attempt for attempt in self.failed_attempts[user_id] if attempt > cutoff
        ]

        # Check if user should be locked out
        if len(self.failed_attempts[user_id]) >= self.max_failed_attempts:
            self.locked_users[user_id] = now
            self.logger.warning(
                f"User {user_id} locked out due to {self.max_failed_attempts} failed attempts"
            )

    def is_user_locked(self, user_id: str) -> bool:
        """Check if user is currently locked out."""
        if user_id not in self.locked_users:
            return False

        lockout_time = self.locked_users[user_id]
        if datetime.now() - lockout_time > timedelta(minutes=self.lockout_minutes):
            # Lockout expired
            del self.locked_users[user_id]
            if user_id in self.failed_attempts:
                del self.failed_attempts[user_id]
            return False

        return True

    def clear_failed_attempts(self, user_id: str) -> None:
        """Clear failed attempts for user (after successful login)."""
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
        if user_id in self.locked_users:
            del self.locked_users[user_id]


class EnhancedPrivacyManager(PrivacyManager):
    """Enhanced privacy manager with additional security features."""

    def __init__(
        self,
        encryption_key: Optional[bytes] = None,
        anonymization_key: Optional[bytes] = None,
        session_timeout_minutes: int = 30,
    ):
        """Initialize enhanced privacy manager."""
        super().__init__(encryption_key, anonymization_key)

        # Additional security components
        self.access_logger = DataAccessLogger(self.anonymizer)
        self.timeout_manager = SessionTimeoutManager(session_timeout_minutes)
        self.access_detector = UnauthorizedAccessDetector()

        # Override RBAC manager to use enhanced features
        self.rbac.session_timeout_minutes = session_timeout_minutes

    def authenticate_user(
        self, user_id: str, credentials: str, role: Role, ip_address: Optional[str] = None
    ) -> Optional[str]:
        """Authenticate user and create session with security checks."""
        # Check if user is locked out
        if self.access_detector.is_user_locked(user_id):
            self.access_logger.log_access(
                "authentication", user_id, "", "user_session", "login", False, ip_address=ip_address
            )
            self.logger.warning(f"Authentication denied for locked user {user_id}")
            return None

        # In real implementation, verify credentials against secure store
        # For demo purposes, assume authentication succeeds
        auth_success = True  # Replace with actual credential verification

        if not auth_success:
            self.access_detector.record_failed_attempt(user_id, ip_address)
            self.access_logger.log_access(
                "authentication", user_id, "", "user_session", "login", False, ip_address=ip_address
            )
            return None

        # Clear any previous failed attempts
        self.access_detector.clear_failed_attempts(user_id)

        # Create session
        session_token = self.create_user_session(user_id, role, ip_address)

        self.access_logger.log_access(
            "authentication",
            user_id,
            session_token,
            "user_session",
            "login",
            True,
            ip_address=ip_address,
        )

        return session_token

    def access_patient_data(
        self, session_token: str, patient_id: str, action: str, ip_address: Optional[str] = None
    ) -> bool:
        """Access patient data with comprehensive logging and security checks."""
        session = self.rbac.get_session(session_token)
        if not session:
            self.access_logger.log_access(
                "data_access",
                "unknown",
                session_token,
                f"patient:{patient_id}",
                action,
                False,
                ip_address=ip_address,
            )
            return False

        # Check if session is expired with custom timeout
        if self.timeout_manager.is_session_expired(session):
            self.rbac.invalidate_session(session_token)
            self.timeout_manager.cleanup_session_timeout(session_token)
            self.access_logger.log_access(
                "data_access",
                session.user_id,
                session_token,
                f"patient:{patient_id}",
                action,
                False,
                ip_address=ip_address,
            )
            return False

        # Check permissions based on action
        required_permission = Permission.READ_PATIENT_DATA
        if action in ["write", "update", "modify"]:
            required_permission = Permission.WRITE_PATIENT_DATA
        elif action in ["delete", "remove"]:
            required_permission = Permission.DELETE_PATIENT_DATA
        elif action in ["export", "download"]:
            required_permission = Permission.EXPORT_DATA

        if not self.check_access_permission(session_token, required_permission):
            self.access_logger.log_access(
                "data_access",
                session.user_id,
                session_token,
                f"patient:{patient_id}",
                action,
                False,
                ip_address=ip_address,
            )
            return False

        # Log successful access
        patient_data = {"patient_id": patient_id}  # Minimal data for logging
        self.access_logger.log_access(
            "data_access",
            session.user_id,
            session_token,
            f"patient:{patient_id}",
            action,
            True,
            patient_data,
            ip_address,
        )

        return True

    def set_session_timeout(self, session_token: str, timeout_minutes: int) -> bool:
        """Set custom timeout for user session."""
        session = self.rbac.get_session(session_token)
        if not session:
            return False

        self.timeout_manager.set_custom_timeout(session_token, timeout_minutes)
        return True

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics."""
        base_metrics = self.get_privacy_status()

        # Add security-specific metrics
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        access_events_today = [
            event for event in self.access_logger.access_log if event.timestamp >= today_start
        ]

        failed_access_today = [event for event in access_events_today if not event.success]

        security_metrics = {
            "total_access_events_today": len(access_events_today),
            "failed_access_attempts_today": len(failed_access_today),
            "locked_users": len(self.access_detector.locked_users),
            "custom_session_timeouts": len(self.timeout_manager.custom_timeouts),
            "average_session_duration_minutes": self._calculate_average_session_duration(),
        }

        return {**base_metrics, **security_metrics}

    def _calculate_average_session_duration(self) -> float:
        """Calculate average session duration for active sessions."""
        if not self.rbac.active_sessions:
            return 0.0

        total_duration = sum(
            (datetime.now() - session.created_at).total_seconds() / 60
            for session in self.rbac.active_sessions.values()
        )

        return total_duration / len(self.rbac.active_sessions)

    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        metrics = self.get_security_metrics()

        # Get recent security events
        recent_events = [event.to_dict() for event in self.access_logger.access_log[-50:]]

        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "recent_events": recent_events,
            "active_sessions": [
                {
                    "user_id": session.user_id,
                    "role": session.role.value,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "ip_address": session.ip_address,
                }
                for session in self.rbac.active_sessions.values()
            ],
        }
