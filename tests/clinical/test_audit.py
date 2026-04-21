"""
Unit tests for audit logging infrastructure.

Tests audit record creation and retrieval, cryptographic signature verification,
and log retention and export functionality.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from .audit import (
    AuditContextManager,
    AuditEvent,
    AuditEventType,
    AuditLogAnalyzer,
    AuditLogger,
    AuditSeverity,
    ComplianceAuditLogger,
    CryptographicSigner,
    FileAuditStorage,
    SignedAuditRecord,
    create_default_audit_logger,
)


class TestAuditEvent:
    """Test audit event functionality."""

    def test_audit_event_creation(self):
        """Test audit event creation and validation."""
        event = AuditEvent(
            event_id="test_001",
            event_type=AuditEventType.PREDICTION_OPERATION,
            timestamp=datetime.now(),
            user_id="user123",
            session_token="session_abc",
            severity=AuditSeverity.INFO,
            description="Test prediction operation",
            model_version="v1.0.0",
        )

        assert event.event_id == "test_001"
        assert event.event_type == AuditEventType.PREDICTION_OPERATION
        assert event.severity == AuditSeverity.INFO
        assert event.user_id == "user123"
        assert event.model_version == "v1.0.0"

    def test_audit_event_string_enum_conversion(self):
        """Test automatic conversion of string enums."""
        event = AuditEvent(
            event_id="test_002",
            event_type="prediction_operation",  # String instead of enum
            timestamp=datetime.now(),
            user_id="user123",
            session_token=None,
            severity="info",  # String instead of enum
            description="Test event",
        )

        assert event.event_type == AuditEventType.PREDICTION_OPERATION
        assert event.severity == AuditSeverity.INFO

    def test_audit_event_serialization(self):
        """Test audit event to/from dictionary conversion."""
        original_event = AuditEvent(
            event_id="test_003",
            event_type=AuditEventType.USER_ACCESS,
            timestamp=datetime.now(),
            user_id="user456",
            session_token="session_xyz",
            severity=AuditSeverity.WARNING,
            description="Failed login attempt",
            details={"ip_address": "192.168.1.1", "attempts": 3},
        )

        # Convert to dict and back
        event_dict = original_event.to_dict()
        restored_event = AuditEvent.from_dict(event_dict)

        assert restored_event.event_id == original_event.event_id
        assert restored_event.event_type == original_event.event_type
        assert restored_event.user_id == original_event.user_id
        assert restored_event.details == original_event.details

    def test_audit_event_content_hash(self):
        """Test content hash generation for integrity verification."""
        event = AuditEvent(
            event_id="test_004",
            event_type=AuditEventType.DATA_MODIFICATION,
            timestamp=datetime.now(),
            user_id="user789",
            session_token=None,
            severity=AuditSeverity.INFO,
            description="Patient data updated",
        )

        hash1 = event.get_content_hash()
        hash2 = event.get_content_hash()

        # Same event should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length

        # Different event should produce different hash
        event.description = "Different description"
        hash3 = event.get_content_hash()
        assert hash3 != hash1


class TestCryptographicSigner:
    """Test cryptographic signing functionality."""

    def test_signer_initialization(self):
        """Test signer initialization with key generation."""
        signer = CryptographicSigner()

        assert signer.private_key is not None
        assert signer.public_key is not None
        assert len(signer.public_key_fingerprint) == 16

    def test_sign_and_verify_event(self):
        """Test event signing and verification."""
        signer = CryptographicSigner()

        event = AuditEvent(
            event_id="test_005",
            event_type=AuditEventType.PREDICTION_OPERATION,
            timestamp=datetime.now(),
            user_id="user123",
            session_token=None,
            severity=AuditSeverity.INFO,
            description="Test prediction",
        )

        # Sign event
        signature = signer.sign_event(event)
        assert signature is not None
        assert len(signature) > 0

        # Verify signature
        is_valid = signer.verify_signature(event, signature)
        assert is_valid

        # Tamper with event and verify signature fails
        event.description = "Tampered description"
        is_valid_after_tampering = signer.verify_signature(event, signature)
        assert not is_valid_after_tampering

    def test_key_export(self):
        """Test public and private key export."""
        signer = CryptographicSigner()

        public_key_pem = signer.export_public_key()
        private_key_pem = signer.export_private_key()

        assert "-----BEGIN PUBLIC KEY-----" in public_key_pem
        assert "-----END PUBLIC KEY-----" in public_key_pem
        assert "-----BEGIN PRIVATE KEY-----" in private_key_pem
        assert "-----END PRIVATE KEY-----" in private_key_pem

    def test_key_export_with_password(self):
        """Test private key export with password protection."""
        signer = CryptographicSigner()

        private_key_pem = signer.export_private_key("test_password")

        assert "-----BEGIN ENCRYPTED PRIVATE KEY-----" in private_key_pem
        assert "-----END ENCRYPTED PRIVATE KEY-----" in private_key_pem


class TestFileAuditStorage:
    """Test file-based audit storage."""

    def test_store_and_retrieve_record(self):
        """Test storing and retrieving audit records."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            signer = CryptographicSigner()

            # Create test event
            event = AuditEvent(
                event_id="test_006",
                event_type=AuditEventType.USER_ACCESS,
                timestamp=datetime.now(),
                user_id="user123",
                session_token="session_abc",
                severity=AuditSeverity.INFO,
                description="User login",
            )

            # Create signed record
            signature = signer.sign_event(event)
            record = SignedAuditRecord(
                event=event,
                signature=signature,
                public_key_fingerprint=signer.public_key_fingerprint,
            )

            # Store record
            success = storage.store_record(record)
            assert success

            # Retrieve records
            retrieved_records = storage.retrieve_records()
            assert len(retrieved_records) == 1
            assert retrieved_records[0].event.event_id == "test_006"

    def test_retrieve_records_with_filters(self):
        """Test retrieving records with various filters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            signer = CryptographicSigner()

            # Create multiple test events
            events = [
                AuditEvent(
                    event_id=f"test_{i:03d}",
                    event_type=(
                        AuditEventType.PREDICTION_OPERATION
                        if i % 2 == 0
                        else AuditEventType.USER_ACCESS
                    ),
                    timestamp=datetime.now() - timedelta(days=i),
                    user_id=f"user{i % 3}",
                    session_token=None,
                    severity=AuditSeverity.INFO,
                    description=f"Test event {i}",
                )
                for i in range(5)
            ]

            # Store all events
            for event in events:
                signature = signer.sign_event(event)
                record = SignedAuditRecord(
                    event=event,
                    signature=signature,
                    public_key_fingerprint=signer.public_key_fingerprint,
                )
                storage.store_record(record)

            # Test filtering by event type
            prediction_records = storage.retrieve_records(
                event_type=AuditEventType.PREDICTION_OPERATION
            )
            assert len(prediction_records) == 3  # Events 0, 2, 4

            # Test filtering by user
            user0_records = storage.retrieve_records(user_id="user0")
            assert len(user0_records) == 2  # Events 0, 3

            # Test limit
            limited_records = storage.retrieve_records(limit=2)
            assert len(limited_records) == 2

    def test_export_records_json(self):
        """Test exporting records to JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            signer = CryptographicSigner()

            # Create and store test event
            event = AuditEvent(
                event_id="test_007",
                event_type=AuditEventType.SYSTEM_ERROR,
                timestamp=datetime.now(),
                user_id="system",
                session_token=None,
                severity=AuditSeverity.ERROR,
                description="Test system error",
            )

            signature = signer.sign_event(event)
            record = SignedAuditRecord(
                event=event,
                signature=signature,
                public_key_fingerprint=signer.public_key_fingerprint,
            )
            storage.store_record(record)

            # Export to JSON
            export_path = Path(temp_dir) / "export.json"
            success = storage.export_records(export_path, format="json")
            assert success
            assert export_path.exists()

            # Verify export content
            with open(export_path, "r") as f:
                export_data = json.load(f)

            assert export_data["record_count"] == 1
            assert len(export_data["records"]) == 1
            assert export_data["records"][0]["event"]["event_id"] == "test_007"

    def test_export_records_csv(self):
        """Test exporting records to CSV format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            signer = CryptographicSigner()

            # Create and store test event
            event = AuditEvent(
                event_id="test_008",
                event_type=AuditEventType.DATA_MODIFICATION,
                timestamp=datetime.now(),
                user_id="user123",
                session_token="session_abc",
                severity=AuditSeverity.INFO,
                description="Test data modification",
            )

            signature = signer.sign_event(event)
            record = SignedAuditRecord(
                event=event,
                signature=signature,
                public_key_fingerprint=signer.public_key_fingerprint,
            )
            storage.store_record(record)

            # Export to CSV
            export_path = Path(temp_dir) / "export.csv"
            success = storage.export_records(export_path, format="csv")
            assert success
            assert export_path.exists()

            # Verify CSV has header and data
            with open(export_path, "r") as f:
                content = f.read()
                assert "event_id,event_type,timestamp" in content
                assert "test_008" in content

    def test_get_record_count(self):
        """Test getting total record count."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            signer = CryptographicSigner()

            # Initially no records
            assert storage.get_record_count() == 0

            # Add some records
            for i in range(3):
                event = AuditEvent(
                    event_id=f"test_{i:03d}",
                    event_type=AuditEventType.PREDICTION_OPERATION,
                    timestamp=datetime.now(),
                    user_id="user123",
                    session_token=None,
                    severity=AuditSeverity.INFO,
                    description=f"Test event {i}",
                )

                signature = signer.sign_event(event)
                record = SignedAuditRecord(
                    event=event,
                    signature=signature,
                    public_key_fingerprint=signer.public_key_fingerprint,
                )
                storage.store_record(record)

            assert storage.get_record_count() == 3


class TestAuditLogger:
    """Test main audit logger functionality."""

    def test_audit_logger_initialization(self):
        """Test audit logger initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            logger = AuditLogger(storage=storage)

            assert logger.storage == storage
            assert logger.signer is not None
            assert logger.retention_days == 2555  # 7 years

    def test_log_prediction_operation(self):
        """Test logging prediction operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            logger = AuditLogger(storage=storage)

            input_data = {
                "patient_id": "PATIENT_123",
                "image_path": "/path/to/image.svs",
                "model_config": {"version": "v1.0.0"},
            }

            output_data = {
                "prediction": "benign",
                "confidence": 0.85,
                "attention_weights": [0.1, 0.2, 0.7],
            }

            event_id = logger.log_prediction_operation(
                user_id="user123",
                session_token="session_abc",
                input_data=input_data,
                output_data=output_data,
                model_version="v1.0.0",
                processing_time_ms=1500.0,
                ip_address="192.168.1.100",
            )

            assert event_id != ""

            # Verify record was stored
            records = logger.get_audit_records()
            assert len(records) == 1

            record = records[0]
            assert record.event.event_type == AuditEventType.PREDICTION_OPERATION
            assert record.event.user_id == "user123"
            assert record.event.model_version == "v1.0.0"
            assert "processing_time_ms" in record.event.details

            # Verify patient data was anonymized
            assert record.event.details["input_data"]["patient_id"].startswith("anon_")

    def test_log_user_access(self):
        """Test logging user access events."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            logger = AuditLogger(storage=storage)

            event_id = logger.log_user_access(
                event_type="login",
                user_id="user456",
                session_token="session_xyz",
                resource="patient_data",
                action="read",
                success=True,
                ip_address="192.168.1.101",
                user_agent="Mozilla/5.0...",
                details={"login_method": "password"},
            )

            assert event_id != ""

            # Verify record
            records = logger.get_audit_records()
            assert len(records) == 1

            record = records[0]
            assert record.event.event_type == AuditEventType.USER_ACCESS
            assert record.event.user_id == "user456"
            assert record.event.severity == AuditSeverity.INFO
            assert record.event.ip_address == "192.168.1.101"

    def test_log_data_modification(self):
        """Test logging data modification events."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            logger = AuditLogger(storage=storage)

            event_id = logger.log_data_modification(
                user_id="user789",
                session_token="session_def",
                resource="patient_record",
                modification_type="update",
                old_data_hash="abc123",
                new_data_hash="def456",
                ip_address="192.168.1.102",
                details={"fields_modified": ["diagnosis", "treatment_plan"]},
            )

            assert event_id != ""

            # Verify record
            records = logger.get_audit_records()
            assert len(records) == 1

            record = records[0]
            assert record.event.event_type == AuditEventType.DATA_MODIFICATION
            assert record.event.details["old_data_hash"] == "abc123"
            assert record.event.details["new_data_hash"] == "def456"

    def test_log_system_error(self):
        """Test logging system errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            logger = AuditLogger(storage=storage)

            event_id = logger.log_system_error(
                error_type="ValidationError",
                error_message="Invalid input data format",
                stack_trace="Traceback (most recent call last):\n  File...",
                input_data={"patient_id": "PATIENT_456", "invalid_field": "bad_value"},
                user_id="user123",
                session_token="session_ghi",
                model_version="v1.0.0",
            )

            assert event_id != ""

            # Verify record
            records = logger.get_audit_records()
            assert len(records) == 1

            record = records[0]
            assert record.event.event_type == AuditEventType.SYSTEM_ERROR
            assert record.event.severity == AuditSeverity.ERROR
            assert "stack_trace" in record.event.details

            # Verify patient data was anonymized
            assert record.event.details["input_data"]["patient_id"].startswith("anon_")

    def test_log_model_training(self):
        """Test logging model training events."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            logger = AuditLogger(storage=storage)

            event_id = logger.log_model_training(
                dataset_version="dataset_v2.1",
                hyperparameters={"learning_rate": 0.001, "batch_size": 32},
                performance_metrics={"accuracy": 0.92, "auc": 0.95},
                training_duration_minutes=120.5,
                model_version="v1.1.0",
                user_id="researcher123",
            )

            assert event_id != ""

            # Verify record
            records = logger.get_audit_records()
            assert len(records) == 1

            record = records[0]
            assert record.event.event_type == AuditEventType.MODEL_TRAINING
            assert record.event.model_version == "v1.1.0"
            assert record.event.details["performance_metrics"]["accuracy"] == 0.92

    def test_verify_record_integrity(self):
        """Test record integrity verification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            logger = AuditLogger(storage=storage)

            # Log an event
            logger.log_user_access(
                event_type="login",
                user_id="user123",
                session_token="session_abc",
                resource="system",
                action="authenticate",
                success=True,
            )

            # Get the record
            records = logger.get_audit_records()
            assert len(records) == 1

            record = records[0]

            # Verify integrity
            is_valid = logger.verify_record_integrity(record)
            assert is_valid

            # Tamper with record and verify it fails
            record.event.description = "Tampered description"
            is_valid_after_tampering = logger.verify_record_integrity(record)
            assert not is_valid_after_tampering

    def test_get_audit_statistics(self):
        """Test audit statistics generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            logger = AuditLogger(storage=storage)

            # Log various events
            logger.log_prediction_operation(
                user_id="user1",
                session_token=None,
                input_data={},
                output_data={},
                model_version="v1.0",
                processing_time_ms=1000,
            )
            logger.log_user_access(
                event_type="login",
                user_id="user2",
                session_token=None,
                resource="system",
                action="authenticate",
                success=True,
            )
            logger.log_system_error(error_type="TestError", error_message="Test error")

            stats = logger.get_audit_statistics()

            assert stats["total_records"] == 3
            assert "event_type_distribution" in stats
            assert "severity_distribution" in stats
            assert "top_users" in stats
            assert stats["retention_days"] == 2555


class TestComplianceAuditLogger:
    """Test compliance audit logger with regulatory features."""

    def test_compliance_logger_with_backup(self):
        """Test compliance logger with backup storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            primary_storage = FileAuditStorage(Path(temp_dir) / "primary")
            backup_storage = FileAuditStorage(Path(temp_dir) / "backup")

            logger = ComplianceAuditLogger(storage=primary_storage, backup_storage=backup_storage)

            # Log an event
            event_id = logger.log_prediction_operation(
                user_id="user123",
                session_token=None,
                input_data={},
                output_data={},
                model_version="v1.0",
                processing_time_ms=1500,
            )

            assert event_id != ""

            # Verify event is in both storages
            primary_records = primary_storage.retrieve_records()
            backup_records = backup_storage.retrieve_records()

            assert len(primary_records) == 1
            assert len(backup_records) == 1
            assert primary_records[0].event.event_id == backup_records[0].event.event_id

    def test_generate_compliance_report(self):
        """Test compliance report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir) / "audit")
            logger = ComplianceAuditLogger(storage=storage)

            # Log some events
            logger.log_prediction_operation(
                user_id="user123",
                session_token=None,
                input_data={},
                output_data={},
                model_version="v1.0",
                processing_time_ms=1200,
            )
            logger.log_user_access(
                event_type="login",
                user_id="user123",
                session_token=None,
                resource="system",
                action="authenticate",
                success=True,
            )

            # Use broader date range to capture all events
            start_date = datetime.now() - timedelta(days=2)
            end_date = datetime.now() + timedelta(days=1)

            # Generate compliance report
            report_path = Path(temp_dir) / "compliance_report.json"
            report = logger.generate_compliance_report(start_date, end_date, report_path)

            assert report_path.exists()
            assert report["report_metadata"]["total_records"] == 2
            assert report["report_metadata"]["integrity_verification_passed"] is True
            assert report["regulatory_compliance"]["fda_21_cfr_part_11_compliant"] is True
            assert report["regulatory_compliance"]["tamper_evident"] is True

    def test_validate_audit_chain(self):
        """Test audit chain validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            logger = ComplianceAuditLogger(storage=storage)

            # Log events
            logger.log_prediction_operation(
                user_id="user123",
                session_token=None,
                input_data={},
                output_data={},
                model_version="v1.0",
                processing_time_ms=1000,
            )
            logger.log_user_access(
                event_type="login",
                user_id="user456",
                session_token=None,
                resource="system",
                action="authenticate",
                success=True,
            )

            # Validate chain
            validation_results = logger.validate_audit_chain()

            assert validation_results["total_records"] == 2
            assert validation_results["valid_signatures"] == 2
            assert validation_results["invalid_signatures"] == 0
            assert validation_results["chain_integrity"] is True
            assert len(validation_results["validation_errors"]) == 0


class TestAuditContextManager:
    """Test audit context manager for automatic logging."""

    def test_context_manager_success(self):
        """Test context manager with successful operation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            logger = AuditLogger(storage=storage)

            with AuditContextManager(
                logger, "test_operation", "user123", "session_abc", "v1.0"
            ) as audit_ctx:
                # Simulate successful operation
                result = {"status": "success", "value": 42}
                audit_ctx.log_success({"input": "test"}, result)

            # Verify audit record was created
            records = logger.get_audit_records()
            assert len(records) == 1
            assert records[0].event.event_type == AuditEventType.PREDICTION_OPERATION

    def test_context_manager_with_exception(self):
        """Test context manager with exception handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            logger = AuditLogger(storage=storage)

            try:
                with AuditContextManager(
                    logger, "test_operation", "user123", "session_abc", "v1.0"
                ):
                    raise ValueError("Test exception")
            except ValueError:
                pass  # Expected exception

            # Verify error was logged
            records = logger.get_audit_records()
            assert len(records) == 1
            assert records[0].event.event_type == AuditEventType.SYSTEM_ERROR
            assert records[0].event.severity == AuditSeverity.ERROR


class TestAuditLogAnalyzer:
    """Test audit log analysis functionality."""

    def test_detect_anomalous_patterns(self):
        """Test anomaly detection in audit logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            logger = AuditLogger(storage=storage)
            analyzer = AuditLogAnalyzer(logger)

            # Create normal activity
            for i in range(10):
                logger.log_user_access(
                    event_type="data_access",
                    user_id="normal_user",
                    session_token=None,
                    resource="patient_data",
                    action="read",
                    success=True,
                )

            # Create suspicious activity (high failure rate)
            for i in range(15):
                logger.log_user_access(
                    event_type="login",
                    user_id="suspicious_user",
                    session_token=None,
                    resource="system",
                    action="authenticate",
                    success=(i < 5),  # Only first 5 succeed
                )

            # Analyze patterns
            anomalies = analyzer.detect_anomalous_patterns(lookback_days=1)

            assert len(anomalies["failed_operations_spike"]) > 0
            suspicious_user_anomaly = next(
                (
                    a
                    for a in anomalies["failed_operations_spike"]
                    if a["user_id"] == "suspicious_user"
                ),
                None,
            )
            assert suspicious_user_anomaly is not None
            assert suspicious_user_anomaly["failure_rate"] > 0.3

    def test_generate_usage_report(self):
        """Test usage report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileAuditStorage(Path(temp_dir))
            logger = AuditLogger(storage=storage)
            analyzer = AuditLogAnalyzer(logger)

            # Log various events
            logger.log_prediction_operation(
                user_id="user1",
                session_token=None,
                input_data={},
                output_data={},
                model_version="v1.0",
                processing_time_ms=1200,
            )
            logger.log_user_access(
                event_type="login",
                user_id="user2",
                session_token=None,
                resource="system",
                action="authenticate",
                success=True,
            )

            # Use broader date range to capture all events
            start_date = datetime.now() - timedelta(days=2)
            end_date = datetime.now() + timedelta(days=1)

            # Generate report
            report = analyzer.generate_usage_report(start_date, end_date)

            assert report["summary_statistics"]["total_events"] == 2
            assert report["summary_statistics"]["unique_users"] == 2
            assert "prediction_operation" in report["summary_statistics"]["event_types"]
            assert "user_access" in report["summary_statistics"]["event_types"]


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_default_audit_logger(self):
        """Test default audit logger creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = create_default_audit_logger(Path(temp_dir))

            assert isinstance(logger, ComplianceAuditLogger)
            assert logger.retention_days == 2555
            assert logger.backup_storage is not None


if __name__ == "__main__":
    pytest.main([__file__])
