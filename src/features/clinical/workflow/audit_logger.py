"""Main audit logger for regulatory compliance."""

import hashlib
import json
import logging
import secrets
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .audit_crypto import CryptographicSigner
from .audit_models import AuditEvent, AuditEventType, AuditSeverity, SignedAuditRecord
from .audit_storage import AuditStorage, FileAuditStorage


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
