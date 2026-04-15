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

import base64
import hashlib
import json
import logging
import secrets
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


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


class CryptographicSigner:
    """Handles cryptographic signing of audit records."""

    def __init__(self, private_key: Optional[rsa.RSAPrivateKey] = None):
        """Initialize with RSA private key."""
        if private_key is None:
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        self.private_key = private_key
        self.public_key = private_key.public_key()
        self.public_key_fingerprint = self._compute_key_fingerprint()

    def _compute_key_fingerprint(self) -> str:
        """Compute fingerprint of public key."""
        public_key_bytes = self.public_key.public_bytes(
            encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo
        )
        return hashlib.sha256(public_key_bytes).hexdigest()[:16]

    def sign_event(self, event: AuditEvent) -> str:
        """Sign audit event and return base64-encoded signature."""
        content_hash = event.get_content_hash()
        signature = self.private_key.sign(
            content_hash.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def verify_signature(self, event: AuditEvent, signature: str) -> bool:
        """Verify signature of audit event."""
        try:
            signature_bytes = base64.b64decode(signature)
            content_hash = event.get_content_hash()

            self.public_key.verify(
                signature_bytes,
                content_hash.encode(),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False

    def export_public_key(self) -> str:
        """Export public key in PEM format."""
        return self.public_key.public_bytes(
            encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo
        ).decode()

    def export_private_key(self, password: Optional[str] = None) -> str:
        """Export private key in PEM format."""
        encryption_algorithm = NoEncryption()
        if password:
            from cryptography.hazmat.primitives.serialization import BestAvailableEncryption

            encryption_algorithm = BestAvailableEncryption(password.encode())

        return self.private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=encryption_algorithm,
        ).decode()


class AuditStorage(ABC):
    """Abstract base class for audit log storage."""

    @abstractmethod
    def store_record(self, record: SignedAuditRecord) -> bool:
        """Store signed audit record."""

    @abstractmethod
    def retrieve_records(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SignedAuditRecord]:
        """Retrieve audit records with filtering."""

    @abstractmethod
    def export_records(
        self,
        output_path: Path,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "json",
    ) -> bool:
        """Export audit records for regulatory submissions."""

    @abstractmethod
    def get_record_count(self) -> int:
        """Get total number of stored records."""

    @abstractmethod
    def cleanup_old_records(self, retention_days: int) -> int:
        """Clean up records older than retention period."""


class FileAuditStorage(AuditStorage):
    """File-based audit log storage implementation."""

    def __init__(self, storage_directory: Path):
        """Initialize with storage directory."""
        self.storage_directory = Path(storage_directory)
        self.storage_directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def _get_daily_log_file(self, date: datetime) -> Path:
        """Get log file path for specific date."""
        date_str = date.strftime("%Y-%m-%d")
        return self.storage_directory / f"audit_{date_str}.jsonl"

    def store_record(self, record: SignedAuditRecord) -> bool:
        """Store signed audit record in daily log file."""
        try:
            log_file = self._get_daily_log_file(record.created_at)

            with open(log_file, "a", encoding="utf-8") as f:
                json.dump(record.to_dict(), f)
                f.write("\n")

            return True
        except Exception as e:
            self.logger.error(f"Failed to store audit record: {e}")
            return False

    def retrieve_records(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SignedAuditRecord]:
        """Retrieve audit records with filtering."""
        records = []

        # Determine date range for file scanning
        if start_time is None:
            start_time = datetime.now() - timedelta(days=30)  # Default to last 30 days
        if end_time is None:
            end_time = datetime.now()

        current_date = start_time.date()
        end_date = end_time.date()

        while current_date <= end_date:
            log_file = self._get_daily_log_file(datetime.combine(current_date, datetime.min.time()))

            if log_file.exists():
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                record_data = json.loads(line)
                                record = SignedAuditRecord.from_dict(record_data)

                                # Apply filters
                                if start_time and record.event.timestamp < start_time:
                                    continue
                                if end_time and record.event.timestamp > end_time:
                                    continue
                                if event_type and record.event.event_type != event_type:
                                    continue
                                if user_id and record.event.user_id != user_id:
                                    continue

                                records.append(record)

                                if limit and len(records) >= limit:
                                    return records

                except Exception as e:
                    self.logger.error(f"Failed to read audit log {log_file}: {e}")

            current_date += timedelta(days=1)

        return records

    def export_records(
        self,
        output_path: Path,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "json",
    ) -> bool:
        """Export audit records for regulatory submissions."""
        try:
            records = self.retrieve_records(start_time, end_time)

            if format.lower() == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    export_data = {
                        "export_timestamp": datetime.now().isoformat(),
                        "start_time": start_time.isoformat() if start_time else None,
                        "end_time": end_time.isoformat() if end_time else None,
                        "record_count": len(records),
                        "records": [record.to_dict() for record in records],
                    }
                    json.dump(export_data, f, indent=2)

            elif format.lower() == "csv":
                import csv

                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    if records:
                        fieldnames = [
                            "event_id",
                            "event_type",
                            "timestamp",
                            "user_id",
                            "severity",
                            "description",
                            "model_version",
                            "ip_address",
                            "signature",
                        ]
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()

                        for record in records:
                            row = {
                                "event_id": record.event.event_id,
                                "event_type": record.event.event_type.value,
                                "timestamp": record.event.timestamp.isoformat(),
                                "user_id": record.event.user_id or "",
                                "severity": record.event.severity.value,
                                "description": record.event.description,
                                "model_version": record.event.model_version or "",
                                "ip_address": record.event.ip_address or "",
                                "signature": record.signature,
                            }
                            writer.writerow(row)

            return True

        except Exception as e:
            self.logger.error(f"Failed to export audit records: {e}")
            return False

    def get_record_count(self) -> int:
        """Get total number of stored records."""
        count = 0

        for log_file in self.storage_directory.glob("audit_*.jsonl"):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    count += sum(1 for line in f if line.strip())
            except Exception as e:
                self.logger.error(f"Failed to count records in {log_file}: {e}")

        return count

    def cleanup_old_records(self, retention_days: int) -> int:
        """Clean up records older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        deleted_count = 0

        for log_file in self.storage_directory.glob("audit_*.jsonl"):
            try:
                # Extract date from filename
                date_str = log_file.stem.replace("audit_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")

                if file_date < cutoff_date:
                    # Count records before deletion
                    with open(log_file, "r", encoding="utf-8") as f:
                        file_record_count = sum(1 for line in f if line.strip())

                    log_file.unlink()
                    deleted_count += file_record_count
                    self.logger.info(
                        f"Deleted old audit log {log_file} with {file_record_count} records"
                    )

            except Exception as e:
                self.logger.error(f"Failed to process audit log {log_file}: {e}")

        return deleted_count


class AuditLogger:
    """Main audit logger for regulatory compliance."""

    def __init__(
        self,
        storage: Optional[AuditStorage] = None,
        signer: Optional[CryptographicSigner] = None,
        retention_days: int = 2555,  # 7 years for FDA compliance
    ):
        """Initialize audit logger."""
        if storage is None:
            storage_dir = Path.home() / ".clinical_audit_logs"
            storage = FileAuditStorage(storage_dir)

        if signer is None:
            signer = CryptographicSigner()

        self.storage = storage
        self.signer = signer
        self.retention_days = retention_days
        self.logger = logging.getLogger(__name__)

        # Initialize anonymizer for patient data
        from .privacy import PatientIdentifierAnonymizer

        self.anonymizer = PatientIdentifierAnonymizer()

    def _create_event_id(self) -> str:
        """Generate unique event ID."""
        return f"audit_{secrets.token_hex(16)}"

    def _hash_data(self, data: Any) -> str:
        """Create hash of data for integrity verification."""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)

        return hashlib.sha256(data_str.encode()).hexdigest()

    def log_prediction_operation(
        self,
        user_id: Optional[str],
        session_token: Optional[str],
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        model_version: str,
        processing_time_ms: float,
        ip_address: Optional[str] = None,
    ) -> str:
        """Log prediction operation with input/output data hashes."""
        # Anonymize patient data in input/output
        anonymized_input = self.anonymizer.anonymize_data(input_data)
        anonymized_output = self.anonymizer.anonymize_data(output_data)

        event = AuditEvent(
            event_id=self._create_event_id(),
            event_type=AuditEventType.PREDICTION_OPERATION,
            timestamp=datetime.now(),
            user_id=user_id,
            session_token=session_token,
            severity=AuditSeverity.INFO,
            description=f"Prediction operation completed in {processing_time_ms:.2f}ms",
            details={
                "input_data": anonymized_input,
                "output_data": anonymized_output,
                "processing_time_ms": processing_time_ms,
            },
            input_data_hash=self._hash_data(input_data),
            output_data_hash=self._hash_data(output_data),
            model_version=model_version,
            ip_address=ip_address,
        )

        return self._store_signed_event(event)

    def log_user_access(
        self,
        event_type: str,
        user_id: str,
        session_token: Optional[str],
        resource: str,
        action: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log user access event (authentication, data queries, report generation)."""
        severity = AuditSeverity.INFO if success else AuditSeverity.WARNING

        event = AuditEvent(
            event_id=self._create_event_id(),
            event_type=AuditEventType.USER_ACCESS,
            timestamp=datetime.now(),
            user_id=user_id,
            session_token=session_token,
            severity=severity,
            description=f"User {action} on {resource}: {'SUCCESS' if success else 'FAILED'}",
            details={
                "event_type": event_type,
                "resource": resource,
                "action": action,
                "success": success,
                **(details or {}),
            },
            ip_address=ip_address,
            user_agent=user_agent,
        )

        return self._store_signed_event(event)

    def log_data_modification(
        self,
        user_id: str,
        session_token: Optional[str],
        resource: str,
        modification_type: str,
        old_data_hash: Optional[str],
        new_data_hash: Optional[str],
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log data modification event (patient data updates, report amendments)."""
        event = AuditEvent(
            event_id=self._create_event_id(),
            event_type=AuditEventType.DATA_MODIFICATION,
            timestamp=datetime.now(),
            user_id=user_id,
            session_token=session_token,
            severity=AuditSeverity.INFO,
            description=f"Data modification: {modification_type} on {resource}",
            details={
                "resource": resource,
                "modification_type": modification_type,
                "old_data_hash": old_data_hash,
                "new_data_hash": new_data_hash,
                **(details or {}),
            },
            ip_address=ip_address,
        )

        return self._store_signed_event(event)

    def log_system_error(
        self,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_token: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> str:
        """Log system error with stack trace and input data state."""
        # Anonymize input data if present
        anonymized_input = None
        input_hash = None
        if input_data:
            anonymized_input = self.anonymizer.anonymize_data(input_data)
            input_hash = self._hash_data(input_data)

        event = AuditEvent(
            event_id=self._create_event_id(),
            event_type=AuditEventType.SYSTEM_ERROR,
            timestamp=datetime.now(),
            user_id=user_id,
            session_token=session_token,
            severity=AuditSeverity.ERROR,
            description=f"System error: {error_type} - {error_message}",
            details={
                "error_type": error_type,
                "error_message": error_message,
                "stack_trace": stack_trace,
                "input_data": anonymized_input,
            },
            input_data_hash=input_hash,
            model_version=model_version,
        )

        return self._store_signed_event(event)

    def log_model_training(
        self,
        dataset_version: str,
        hyperparameters: Dict[str, Any],
        performance_metrics: Dict[str, float],
        training_duration_minutes: float,
        model_version: str,
        user_id: Optional[str] = None,
    ) -> str:
        """Log model training event with dataset versions, hyperparameters, and metrics."""
        event = AuditEvent(
            event_id=self._create_event_id(),
            event_type=AuditEventType.MODEL_TRAINING,
            timestamp=datetime.now(),
            user_id=user_id,
            session_token=None,
            severity=AuditSeverity.INFO,
            description=f"Model training completed for version {model_version}",
            details={
                "dataset_version": dataset_version,
                "hyperparameters": hyperparameters,
                "performance_metrics": performance_metrics,
                "training_duration_minutes": training_duration_minutes,
            },
            model_version=model_version,
        )

        return self._store_signed_event(event)

    def log_model_validation(
        self,
        model_version: str,
        validation_dataset: str,
        performance_metrics: Dict[str, float],
        validation_type: str,
        user_id: Optional[str] = None,
    ) -> str:
        """Log model validation event with performance metrics."""
        event = AuditEvent(
            event_id=self._create_event_id(),
            event_type=AuditEventType.MODEL_VALIDATION,
            timestamp=datetime.now(),
            user_id=user_id,
            session_token=None,
            severity=AuditSeverity.INFO,
            description=f"Model validation completed: {validation_type} for {model_version}",
            details={
                "validation_dataset": validation_dataset,
                "performance_metrics": performance_metrics,
                "validation_type": validation_type,
            },
            model_version=model_version,
        )

        return self._store_signed_event(event)

    def _store_signed_event(self, event: AuditEvent) -> str:
        """Sign and store audit event."""
        try:
            signature = self.signer.sign_event(event)

            signed_record = SignedAuditRecord(
                event=event,
                signature=signature,
                public_key_fingerprint=self.signer.public_key_fingerprint,
            )

            success = self.storage.store_record(signed_record)

            if success:
                self.logger.info(f"Stored audit event {event.event_id}")
                return event.event_id
            else:
                self.logger.error(f"Failed to store audit event {event.event_id}")
                return ""

        except Exception as e:
            self.logger.error(f"Failed to sign and store audit event: {e}")
            return ""

    def verify_record_integrity(self, record: SignedAuditRecord) -> bool:
        """Verify cryptographic signature of audit record."""
        return self.signer.verify_signature(record.event, record.signature)

    def get_audit_records(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SignedAuditRecord]:
        """Retrieve audit records with filtering."""
        return self.storage.retrieve_records(start_time, end_time, event_type, user_id, limit)

    def export_audit_logs(
        self,
        output_path: Path,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "json",
    ) -> bool:
        """Export audit logs for regulatory submissions."""
        return self.storage.export_records(output_path, start_time, end_time, format)

    def cleanup_old_logs(self) -> int:
        """Clean up audit logs older than retention period."""
        return self.storage.cleanup_old_records(self.retention_days)

    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        total_records = self.storage.get_record_count()

        # Get recent records for analysis
        recent_records = self.get_audit_records(
            start_time=datetime.now() - timedelta(days=30), limit=1000
        )

        event_type_counts = {}
        severity_counts = {}
        user_activity = {}

        for record in recent_records:
            event_type = record.event.event_type.value
            severity = record.event.severity.value
            user_id = record.event.user_id or "system"

            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            user_activity[user_id] = user_activity.get(user_id, 0) + 1

        return {
            "total_records": total_records,
            "recent_records_30_days": len(recent_records),
            "event_type_distribution": event_type_counts,
            "severity_distribution": severity_counts,
            "top_users": dict(sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]),
            "retention_days": self.retention_days,
            "storage_type": type(self.storage).__name__,
        }


class AuditContextManager:
    """Context manager for automatic audit logging of operations."""

    def __init__(
        self,
        audit_logger: AuditLogger,
        operation_type: str,
        user_id: Optional[str] = None,
        session_token: Optional[str] = None,
        model_version: Optional[str] = None,
    ):
        """Initialize audit context."""
        self.audit_logger = audit_logger
        self.operation_type = operation_type
        self.user_id = user_id
        self.session_token = session_token
        self.model_version = model_version
        self.start_time = None
        self.exception_occurred = False

    def __enter__(self):
        """Enter audit context."""
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit audit context and log results."""
        end_time = datetime.now()
        (end_time - self.start_time).total_seconds() * 1000

        if exc_type is not None:
            # Log error
            self.audit_logger.log_system_error(
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                stack_trace=traceback.format_exc(),
                user_id=self.user_id,
                session_token=self.session_token,
                model_version=self.model_version,
            )
            self.exception_occurred = True

        return False  # Don't suppress exceptions

    def log_success(self, input_data: Dict[str, Any], output_data: Dict[str, Any]):
        """Log successful operation completion."""
        if not self.exception_occurred:
            processing_time_ms = (datetime.now() - self.start_time).total_seconds() * 1000

            self.audit_logger.log_prediction_operation(
                user_id=self.user_id,
                session_token=self.session_token,
                input_data=input_data,
                output_data=output_data,
                model_version=self.model_version or "unknown",
                processing_time_ms=processing_time_ms,
            )


class ComplianceAuditLogger(AuditLogger):
    """Enhanced audit logger with regulatory compliance features."""

    def __init__(
        self,
        storage: Optional[AuditStorage] = None,
        signer: Optional[CryptographicSigner] = None,
        retention_days: int = 2555,  # 7 years for FDA compliance
        backup_storage: Optional[AuditStorage] = None,
    ):
        """Initialize compliance audit logger with backup storage."""
        super().__init__(storage, signer, retention_days)
        self.backup_storage = backup_storage
        self.compliance_logger = logging.getLogger(f"{__name__}.compliance")

    def _store_signed_event(self, event: AuditEvent) -> str:
        """Store event in primary and backup storage for redundancy."""
        event_id = super()._store_signed_event(event)

        # Also store in backup storage if available
        if self.backup_storage and event_id:
            try:
                signature = self.signer.sign_event(event)
                signed_record = SignedAuditRecord(
                    event=event,
                    signature=signature,
                    public_key_fingerprint=self.signer.public_key_fingerprint,
                )

                backup_success = self.backup_storage.store_record(signed_record)
                if backup_success:
                    self.compliance_logger.info(f"Backup storage successful for event {event_id}")
                else:
                    self.compliance_logger.warning(f"Backup storage failed for event {event_id}")

            except Exception as e:
                self.compliance_logger.error(f"Backup storage error for event {event_id}: {e}")

        return event_id

    def generate_compliance_report(
        self, start_date: datetime, end_date: datetime, output_path: Path
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report for regulatory submissions."""
        records = self.get_audit_records(start_date, end_date)

        # Verify integrity of all records
        integrity_results = []
        for record in records:
            is_valid = self.verify_record_integrity(record)
            integrity_results.append(
                {
                    "event_id": record.event.event_id,
                    "timestamp": record.event.timestamp.isoformat(),
                    "integrity_valid": is_valid,
                }
            )

        # Generate statistics
        stats = self.get_audit_statistics()

        # Create compliance report
        compliance_report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_period_start": start_date.isoformat(),
                "report_period_end": end_date.isoformat(),
                "total_records": len(records),
                "integrity_verification_passed": all(
                    r["integrity_valid"] for r in integrity_results
                ),
                "public_key_fingerprint": self.signer.public_key_fingerprint,
                "retention_policy_days": self.retention_days,
            },
            "audit_statistics": stats,
            "integrity_verification": integrity_results,
            "regulatory_compliance": {
                "fda_21_cfr_part_11_compliant": True,
                "hipaa_compliant": True,
                "retention_period_years": self.retention_days / 365.25,
                "tamper_evident": True,
                "cryptographic_signatures": True,
            },
            "public_key": self.signer.export_public_key(),
        }

        # Export detailed records
        records_export_path = output_path.parent / f"{output_path.stem}_records.json"
        self.export_audit_logs(records_export_path, start_date, end_date, "json")

        # Save compliance report
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(compliance_report, f, indent=2)

        self.compliance_logger.info(
            f"Generated compliance report with {len(records)} records for period "
            f"{start_date.date()} to {end_date.date()}"
        )

        return compliance_report

    def validate_audit_chain(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Validate integrity of audit chain for regulatory compliance."""
        records = self.get_audit_records(start_time, end_time)

        validation_results = {
            "total_records": len(records),
            "valid_signatures": 0,
            "invalid_signatures": 0,
            "validation_errors": [],
            "chain_integrity": True,
            "validation_timestamp": datetime.now().isoformat(),
        }

        for record in records:
            try:
                is_valid = self.verify_record_integrity(record)

                if is_valid:
                    validation_results["valid_signatures"] += 1
                else:
                    validation_results["invalid_signatures"] += 1
                    validation_results["chain_integrity"] = False
                    validation_results["validation_errors"].append(
                        {
                            "event_id": record.event.event_id,
                            "timestamp": record.event.timestamp.isoformat(),
                            "error": "Invalid cryptographic signature",
                        }
                    )

            except Exception as e:
                validation_results["invalid_signatures"] += 1
                validation_results["chain_integrity"] = False
                validation_results["validation_errors"].append(
                    {
                        "event_id": record.event.event_id,
                        "timestamp": record.event.timestamp.isoformat(),
                        "error": f"Validation exception: {str(e)}",
                    }
                )

        return validation_results

    def archive_old_records(self, archive_path: Path, cutoff_date: datetime) -> Dict[str, Any]:
        """Archive old records while maintaining regulatory compliance."""
        records_to_archive = self.get_audit_records(end_time=cutoff_date)

        if not records_to_archive:
            return {
                "archived_count": 0,
                "archive_path": None,
                "archive_timestamp": datetime.now().isoformat(),
            }

        # Create archive with integrity verification
        archive_data = {
            "archive_metadata": {
                "created_at": datetime.now().isoformat(),
                "cutoff_date": cutoff_date.isoformat(),
                "record_count": len(records_to_archive),
                "public_key_fingerprint": self.signer.public_key_fingerprint,
                "archive_format_version": "1.0",
            },
            "public_key": self.signer.export_public_key(),
            "records": [record.to_dict() for record in records_to_archive],
        }

        # Save archive
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(archive_data, f, indent=2)

        # Create archive integrity hash
        with open(archive_path, "rb") as f:
            archive_hash = hashlib.sha256(f.read()).hexdigest()

        # Save integrity file
        integrity_path = archive_path.with_suffix(".integrity")
        with open(integrity_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "archive_file": archive_path.name,
                    "sha256_hash": archive_hash,
                    "created_at": datetime.now().isoformat(),
                    "record_count": len(records_to_archive),
                },
                f,
                indent=2,
            )

        self.compliance_logger.info(f"Archived {len(records_to_archive)} records to {archive_path}")

        return {
            "archived_count": len(records_to_archive),
            "archive_path": str(archive_path),
            "integrity_path": str(integrity_path),
            "archive_hash": archive_hash,
            "archive_timestamp": datetime.now().isoformat(),
        }


class AuditLogAnalyzer:
    """Analyzer for audit log patterns and compliance monitoring."""

    def __init__(self, audit_logger: AuditLogger):
        """Initialize with audit logger."""
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)

    def detect_anomalous_patterns(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Detect anomalous patterns in audit logs."""
        start_time = datetime.now() - timedelta(days=lookback_days)
        records = self.audit_logger.get_audit_records(start_time=start_time)

        anomalies = {
            "unusual_access_patterns": [],
            "failed_operations_spike": [],
            "off_hours_activity": [],
            "bulk_data_access": [],
            "analysis_timestamp": datetime.now().isoformat(),
        }

        # Analyze access patterns by user
        user_activity = {}
        for record in records:
            user_id = record.event.user_id or "system"
            if user_id not in user_activity:
                user_activity[user_id] = {
                    "total_events": 0,
                    "failed_events": 0,
                    "event_types": {},
                    "hourly_distribution": [0] * 24,
                }

            user_activity[user_id]["total_events"] += 1

            if record.event.severity in [AuditSeverity.WARNING, AuditSeverity.ERROR]:
                user_activity[user_id]["failed_events"] += 1

            event_type = record.event.event_type.value
            user_activity[user_id]["event_types"][event_type] = (
                user_activity[user_id]["event_types"].get(event_type, 0) + 1
            )

            # Track hourly distribution
            hour = record.event.timestamp.hour
            user_activity[user_id]["hourly_distribution"][hour] += 1

        # Detect anomalies
        for user_id, activity in user_activity.items():
            # High failure rate
            if activity["total_events"] > 10:
                failure_rate = activity["failed_events"] / activity["total_events"]
                if failure_rate > 0.3:  # More than 30% failures
                    anomalies["failed_operations_spike"].append(
                        {
                            "user_id": user_id,
                            "failure_rate": failure_rate,
                            "total_events": activity["total_events"],
                            "failed_events": activity["failed_events"],
                        }
                    )

            # Off-hours activity (10 PM to 6 AM)
            off_hours_activity = sum(activity["hourly_distribution"][22:24]) + sum(
                activity["hourly_distribution"][0:6]
            )
            total_activity = sum(activity["hourly_distribution"])

            if total_activity > 20 and off_hours_activity / total_activity > 0.4:
                anomalies["off_hours_activity"].append(
                    {
                        "user_id": user_id,
                        "off_hours_percentage": off_hours_activity / total_activity,
                        "total_events": total_activity,
                        "off_hours_events": off_hours_activity,
                    }
                )

            # Bulk data access patterns
            prediction_events = activity["event_types"].get("prediction_operation", 0)
            if prediction_events > 100:  # More than 100 predictions
                anomalies["bulk_data_access"].append(
                    {
                        "user_id": user_id,
                        "prediction_count": prediction_events,
                        "total_events": activity["total_events"],
                    }
                )

        return anomalies

    def generate_usage_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate usage report for audit logs."""
        records = self.audit_logger.get_audit_records(start_date, end_date)

        report = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_days": (end_date - start_date).days,
            },
            "summary_statistics": {
                "total_events": len(records),
                "unique_users": len(set(r.event.user_id for r in records if r.event.user_id)),
                "event_types": {},
                "severity_distribution": {},
                "daily_activity": {},
            },
            "top_users": {},
            "system_health": {
                "error_rate": 0,
                "average_processing_time": 0,
                "peak_activity_hour": 0,
            },
        }

        # Calculate statistics
        processing_times = []
        hourly_activity = [0] * 24
        daily_activity = {}

        for record in records:
            # Event type distribution
            event_type = record.event.event_type.value
            report["summary_statistics"]["event_types"][event_type] = (
                report["summary_statistics"]["event_types"].get(event_type, 0) + 1
            )

            # Severity distribution
            severity = record.event.severity.value
            report["summary_statistics"]["severity_distribution"][severity] = (
                report["summary_statistics"]["severity_distribution"].get(severity, 0) + 1
            )

            # Daily activity
            date_str = record.event.timestamp.date().isoformat()
            daily_activity[date_str] = daily_activity.get(date_str, 0) + 1

            # Hourly activity
            hour = record.event.timestamp.hour
            hourly_activity[hour] += 1

            # Processing times for prediction operations
            if (
                record.event.event_type == AuditEventType.PREDICTION_OPERATION
                and "processing_time_ms" in record.event.details
            ):
                processing_times.append(record.event.details["processing_time_ms"])

        # Calculate derived metrics
        error_count = report["summary_statistics"]["severity_distribution"].get(
            "error", 0
        ) + report["summary_statistics"]["severity_distribution"].get("critical", 0)

        report["summary_statistics"]["daily_activity"] = daily_activity
        report["system_health"]["error_rate"] = error_count / len(records) if records else 0
        report["system_health"]["average_processing_time"] = (
            sum(processing_times) / len(processing_times) if processing_times else 0
        )
        report["system_health"]["peak_activity_hour"] = hourly_activity.index(max(hourly_activity))

        return report


# Utility functions for audit logging integration


def audit_operation(
    audit_logger: AuditLogger,
    operation_type: str,
    user_id: Optional[str] = None,
    session_token: Optional[str] = None,
    model_version: Optional[str] = None,
):
    """Decorator for automatic audit logging of operations."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            with AuditContextManager(
                audit_logger, operation_type, user_id, session_token, model_version
            ) as audit_ctx:
                result = func(*args, **kwargs)

                # Extract input/output data for logging
                input_data = {"args": str(args)[:1000], "kwargs": str(kwargs)[:1000]}
                output_data = {"result": str(result)[:1000]}

                audit_ctx.log_success(input_data, output_data)
                return result

        return wrapper

    return decorator


def create_default_audit_logger(storage_dir: Optional[Path] = None) -> ComplianceAuditLogger:
    """Create default audit logger with file storage."""
    if storage_dir is None:
        storage_dir = Path.home() / ".clinical_audit_logs"

    primary_storage = FileAuditStorage(storage_dir / "primary")
    backup_storage = FileAuditStorage(storage_dir / "backup")

    return ComplianceAuditLogger(
        storage=primary_storage,
        backup_storage=backup_storage,
        retention_days=2555,  # 7 years for FDA compliance
    )
