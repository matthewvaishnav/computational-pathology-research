"""
Unit tests for privacy and security infrastructure.

Tests encryption/decryption operations, anonymization functions,
and access control enforcement.
"""

import json
from datetime import datetime, timedelta

import pytest

from src.clinical.privacy import (
    AES256Encryption,
    ConsentRecord,
    DataExportMonitor,
    EnhancedPrivacyManager,
    PatientIdentifierAnonymizer,
    Permission,
    PrivacyManager,
    RBACManager,
    Role,
    SessionTimeoutManager,
    UnauthorizedAccessDetector,
    UserSession,
)


class TestAES256Encryption:
    """Test AES-256 encryption functionality."""

    def test_encrypt_decrypt_bytes(self):
        """Test encryption and decryption of bytes."""
        encryption = AES256Encryption()
        original_data = b"sensitive patient data"

        encrypted = encryption.encrypt(original_data)
        decrypted = encryption.decrypt(encrypted)

        assert decrypted == original_data
        assert encrypted != original_data

    def test_encrypt_decrypt_consistency(self):
        """Test that same data produces different ciphertext but same plaintext."""
        encryption = AES256Encryption()
        data = b"test data"

        encrypted1 = encryption.encrypt(data)
        encrypted2 = encryption.encrypt(data)

        # Different ciphertext due to random IV
        assert encrypted1 != encrypted2

        # Same plaintext when decrypted
        assert encryption.decrypt(encrypted1) == data
        assert encryption.decrypt(encrypted2) == data

    def test_from_password(self):
        """Test creating encryption from password."""
        password = "secure_password_123"
        encryption = AES256Encryption.from_password(password)

        data = b"test data"
        encrypted = encryption.encrypt(data)
        decrypted = encryption.decrypt(encrypted)

        assert decrypted == data

    def test_key_property(self):
        """Test key property access."""
        encryption = AES256Encryption()
        key = encryption.key

        assert isinstance(key, bytes)
        assert len(key) == 44  # Base64 encoded 32-byte key


class TestPatientIdentifierAnonymizer:
    """Test patient identifier anonymization."""

    def test_anonymize_patient_id_consistency(self):
        """Test that same patient ID produces same anonymized ID."""
        anonymizer = PatientIdentifierAnonymizer()
        patient_id = "PATIENT_12345"

        anon1 = anonymizer.anonymize_patient_id(patient_id)
        anon2 = anonymizer.anonymize_patient_id(patient_id)

        assert anon1 == anon2
        assert anon1.startswith("anon_")
        assert len(anon1) == 21  # "anon_" + 16 hex chars

    def test_anonymize_patient_id_different_inputs(self):
        """Test that different patient IDs produce different anonymized IDs."""
        anonymizer = PatientIdentifierAnonymizer()

        anon1 = anonymizer.anonymize_patient_id("PATIENT_001")
        anon2 = anonymizer.anonymize_patient_id("PATIENT_002")

        assert anon1 != anon2
        assert anon1.startswith("anon_")
        assert anon2.startswith("anon_")

    def test_anonymize_data_dictionary(self):
        """Test anonymization of data dictionary."""
        anonymizer = PatientIdentifierAnonymizer()

        data = {
            "patient_id": "PATIENT_123",
            "name": "John Doe",
            "age": 45,
            "diagnosis": "Benign",
            "email": "john@example.com",
        }

        anonymized = anonymizer.anonymize_data(data)

        assert anonymized["patient_id"].startswith("anon_")
        assert anonymized["name"] == "<name_anonymized>"
        assert anonymized["age"] == 45  # Non-identifier field unchanged
        assert anonymized["diagnosis"] == "Benign"  # Non-identifier field unchanged
        assert anonymized["email"] == "<email_anonymized>"

    def test_anonymize_data_preserves_non_identifiers(self):
        """Test that non-identifier fields are preserved."""
        anonymizer = PatientIdentifierAnonymizer()

        data = {
            "scan_date": "2024-01-15",
            "tissue_type": "breast",
            "prediction_score": 0.85,
            "model_version": "v1.2.3",
        }

        anonymized = anonymizer.anonymize_data(data)

        assert anonymized == data  # No changes for non-identifier fields


class TestUserSession:
    """Test user session functionality."""

    def test_session_creation(self):
        """Test user session creation."""
        session = UserSession(
            user_id="user123",
            role=Role.PHYSICIAN,
            permissions={Permission.READ_PATIENT_DATA},
            created_at=datetime.now(),
            last_activity=datetime.now(),
            session_token="token123",
        )

        assert session.user_id == "user123"
        assert session.role == Role.PHYSICIAN
        assert Permission.READ_PATIENT_DATA in session.permissions

    def test_session_expiry(self):
        """Test session expiry logic."""
        old_time = datetime.now() - timedelta(minutes=45)
        session = UserSession(
            user_id="user123",
            role=Role.PHYSICIAN,
            permissions=set(),
            created_at=old_time,
            last_activity=old_time,
            session_token="token123",
        )

        assert session.is_expired(timeout_minutes=30)
        assert not session.is_expired(timeout_minutes=60)

    def test_update_activity(self):
        """Test activity timestamp update."""
        old_time = datetime.now() - timedelta(minutes=10)
        session = UserSession(
            user_id="user123",
            role=Role.PHYSICIAN,
            permissions=set(),
            created_at=old_time,
            last_activity=old_time,
            session_token="token123",
        )

        session.update_activity()

        assert session.last_activity > old_time


class TestConsentRecord:
    """Test patient consent record functionality."""

    def test_valid_consent(self):
        """Test valid consent record."""
        consent = ConsentRecord(
            patient_id="PATIENT_123",
            consent_type="data_sharing",
            granted=True,
            granted_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=365),
        )

        assert consent.is_valid()

    def test_expired_consent(self):
        """Test expired consent record."""
        consent = ConsentRecord(
            patient_id="PATIENT_123",
            consent_type="data_sharing",
            granted=True,
            granted_at=datetime.now() - timedelta(days=400),
            expires_at=datetime.now() - timedelta(days=1),
        )

        assert not consent.is_valid()

    def test_revoked_consent(self):
        """Test revoked consent record."""
        consent = ConsentRecord(
            patient_id="PATIENT_123",
            consent_type="data_sharing",
            granted=False,
            granted_at=datetime.now(),
        )

        assert not consent.is_valid()


class TestRBACManager:
    """Test role-based access control manager."""

    def test_create_session(self):
        """Test session creation."""
        rbac = RBACManager()

        token = rbac.create_session("user123", Role.PHYSICIAN)

        assert token in rbac.active_sessions
        session = rbac.active_sessions[token]
        assert session.user_id == "user123"
        assert session.role == Role.PHYSICIAN

    def test_get_session(self):
        """Test session retrieval."""
        rbac = RBACManager()
        token = rbac.create_session("user123", Role.PHYSICIAN)

        session = rbac.get_session(token)

        assert session is not None
        assert session.user_id == "user123"

    def test_get_expired_session(self):
        """Test retrieval of expired session."""
        rbac = RBACManager()
        rbac.session_timeout_minutes = 1  # 1 minute timeout

        token = rbac.create_session("user123", Role.PHYSICIAN)
        session = rbac.active_sessions[token]
        session.last_activity = datetime.now() - timedelta(minutes=2)

        retrieved_session = rbac.get_session(token)

        assert retrieved_session is None
        assert token not in rbac.active_sessions

    def test_check_permission(self):
        """Test permission checking."""
        rbac = RBACManager()
        token = rbac.create_session("user123", Role.PHYSICIAN)

        # Physician should have read permission
        assert rbac.check_permission(token, Permission.READ_PATIENT_DATA)

        # Physician should not have admin permissions
        assert not rbac.check_permission(token, Permission.MANAGE_USERS)

    def test_invalidate_session(self):
        """Test session invalidation."""
        rbac = RBACManager()
        token = rbac.create_session("user123", Role.PHYSICIAN)

        success = rbac.invalidate_session(token)

        assert success
        assert token not in rbac.active_sessions

    def test_cleanup_expired_sessions(self):
        """Test cleanup of expired sessions."""
        rbac = RBACManager()
        rbac.session_timeout_minutes = 1

        # Create sessions
        token1 = rbac.create_session("user1", Role.PHYSICIAN)
        token2 = rbac.create_session("user2", Role.PHYSICIAN)

        # Expire one session
        rbac.active_sessions[token1].last_activity = datetime.now() - timedelta(minutes=2)

        cleaned_count = rbac.cleanup_expired_sessions()

        assert cleaned_count == 1
        assert token1 not in rbac.active_sessions
        assert token2 in rbac.active_sessions


class TestDataExportMonitor:
    """Test data export monitoring."""

    def test_request_export_within_limits(self):
        """Test export request within size limits."""
        monitor = DataExportMonitor(max_export_size_mb=100)

        approved = monitor.request_export("token123", 50.0, "csv", "external_system")

        assert approved
        assert len(monitor.export_attempts) == 1
        assert monitor.export_attempts[0]["approved"]

    def test_request_export_exceeds_limits(self):
        """Test export request exceeding size limits."""
        monitor = DataExportMonitor(max_export_size_mb=100)

        approved = monitor.request_export("token123", 150.0, "csv", "external_system")

        assert not approved
        assert len(monitor.export_attempts) == 1
        assert not monitor.export_attempts[0]["approved"]
        assert monitor.export_attempts[0]["denial_reason"] == "size_limit_exceeded"

    def test_detect_unauthorized_export(self):
        """Test detection of unauthorized export patterns."""
        monitor = DataExportMonitor()

        # Suspicious pattern
        suspicious_pattern = {
            "bulk_access": True,
            "off_hours_access": True,
            "unusual_volume": False,
            "external_destination": False,
        }

        is_suspicious = monitor.detect_unauthorized_export("token123", suspicious_pattern)
        assert is_suspicious

        # Normal pattern
        normal_pattern = {
            "bulk_access": False,
            "off_hours_access": False,
            "unusual_volume": False,
            "external_destination": False,
        }

        is_suspicious = monitor.detect_unauthorized_export("token123", normal_pattern)
        assert not is_suspicious


class TestPrivacyManager:
    """Test main privacy manager functionality."""

    def test_encrypt_decrypt_patient_data(self):
        """Test patient data encryption and decryption."""
        manager = PrivacyManager()

        # Test string data
        data = "sensitive patient information"
        encrypted = manager.encrypt_patient_data(data)
        decrypted = manager.decrypt_patient_data(encrypted)

        assert decrypted.decode() == data

        # Test dictionary data
        data_dict = {"patient_id": "123", "diagnosis": "benign"}
        encrypted = manager.encrypt_patient_data(data_dict)
        decrypted = manager.decrypt_patient_data(encrypted)

        assert json.loads(decrypted.decode()) == data_dict

    def test_anonymize_for_logging(self):
        """Test anonymization for logging."""
        manager = PrivacyManager()

        data = {"patient_id": "PATIENT_123", "name": "John Doe", "diagnosis": "benign"}

        anonymized = manager.anonymize_for_logging(data)

        assert anonymized["patient_id"].startswith("anon_")
        assert anonymized["name"] == "<name_anonymized>"
        assert anonymized["diagnosis"] == "benign"

    def test_create_user_session(self):
        """Test user session creation."""
        manager = PrivacyManager()

        token = manager.create_user_session("user123", Role.PHYSICIAN)

        assert token in manager.rbac.active_sessions
        session = manager.rbac.active_sessions[token]
        assert session.user_id == "user123"
        assert session.role == Role.PHYSICIAN

    def test_check_access_permission(self):
        """Test access permission checking."""
        manager = PrivacyManager()
        token = manager.create_user_session("user123", Role.PHYSICIAN)

        assert manager.check_access_permission(token, Permission.READ_PATIENT_DATA)
        assert not manager.check_access_permission(token, Permission.MANAGE_USERS)

    def test_record_and_check_consent(self):
        """Test consent recording and checking."""
        manager = PrivacyManager()

        consent = ConsentRecord(
            patient_id="PATIENT_123",
            consent_type="external_sharing",
            granted=True,
            granted_at=datetime.now(),
            purpose="external_sharing",
        )

        manager.record_consent(consent)

        assert manager.check_consent("PATIENT_123", "external_sharing")
        assert not manager.check_consent("PATIENT_123", "research")
        assert not manager.check_consent("PATIENT_456", "external_sharing")

    def test_request_data_export_with_consent(self):
        """Test data export request with proper consent."""
        manager = PrivacyManager()
        token = manager.create_user_session("user123", Role.PHYSICIAN)

        # Record consent
        consent = ConsentRecord(
            patient_id="PATIENT_123",
            consent_type="external_sharing",
            granted=True,
            granted_at=datetime.now(),
            purpose="external_sharing",
        )
        manager.record_consent(consent)

        approved = manager.request_data_export(token, ["PATIENT_123"], "csv", "external_system")

        assert approved

    def test_request_data_export_without_consent(self):
        """Test data export request without consent."""
        manager = PrivacyManager()
        token = manager.create_user_session("user123", Role.PHYSICIAN)

        approved = manager.request_data_export(token, ["PATIENT_123"], "csv", "external_system")

        assert not approved

    def test_delete_patient_data(self):
        """Test patient data deletion."""
        manager = PrivacyManager()
        token = manager.create_user_session("admin123", Role.ADMIN)

        success = manager.delete_patient_data(token, "PATIENT_123")

        assert success

    def test_delete_patient_data_insufficient_permission(self):
        """Test patient data deletion with insufficient permissions."""
        manager = PrivacyManager()
        token = manager.create_user_session("viewer123", Role.VIEWER)

        success = manager.delete_patient_data(token, "PATIENT_123")

        assert not success

    def test_get_privacy_status(self):
        """Test privacy status reporting."""
        manager = PrivacyManager()
        manager.create_user_session("user123", Role.PHYSICIAN)

        status = manager.get_privacy_status()

        assert status["active_sessions"] == 1
        assert status["encryption_enabled"] is True
        assert status["anonymization_enabled"] is True
        assert status["rbac_enabled"] is True


class TestSessionTimeoutManager:
    """Test session timeout management."""

    def test_default_timeout(self):
        """Test default timeout behavior."""
        manager = SessionTimeoutManager(default_timeout_minutes=30)

        timeout = manager.get_timeout("any_token")
        assert timeout == 30

    def test_custom_timeout(self):
        """Test custom timeout setting."""
        manager = SessionTimeoutManager(default_timeout_minutes=30)

        manager.set_custom_timeout("token123", 60)
        timeout = manager.get_timeout("token123")

        assert timeout == 60

    def test_session_expiry_check(self):
        """Test session expiry with custom timeout."""
        manager = SessionTimeoutManager()

        # Create session with old activity
        session = UserSession(
            user_id="user123",
            role=Role.PHYSICIAN,
            permissions=set(),
            created_at=datetime.now(),
            last_activity=datetime.now() - timedelta(minutes=45),
            session_token="token123",
        )

        # Set custom timeout
        manager.set_custom_timeout("token123", 60)

        assert not manager.is_session_expired(session)

        # Default timeout would expire this session
        assert session.is_expired(30)


class TestUnauthorizedAccessDetector:
    """Test unauthorized access detection."""

    def test_record_failed_attempts(self):
        """Test recording of failed attempts."""
        detector = UnauthorizedAccessDetector(max_failed_attempts=3)

        detector.record_failed_attempt("user123")
        detector.record_failed_attempt("user123")

        assert not detector.is_user_locked("user123")

        detector.record_failed_attempt("user123")

        assert detector.is_user_locked("user123")

    def test_lockout_expiry(self):
        """Test that lockout expires after timeout."""
        detector = UnauthorizedAccessDetector(max_failed_attempts=2, lockout_minutes=1)

        # Lock user
        detector.record_failed_attempt("user123")
        detector.record_failed_attempt("user123")
        assert detector.is_user_locked("user123")

        # Simulate time passing
        detector.locked_users["user123"] = datetime.now() - timedelta(minutes=2)

        assert not detector.is_user_locked("user123")

    def test_clear_failed_attempts(self):
        """Test clearing failed attempts."""
        detector = UnauthorizedAccessDetector()

        detector.record_failed_attempt("user123")
        detector.clear_failed_attempts("user123")

        assert "user123" not in detector.failed_attempts


class TestEnhancedPrivacyManager:
    """Test enhanced privacy manager with additional security features."""

    def test_authenticate_user_success(self):
        """Test successful user authentication."""
        manager = EnhancedPrivacyManager()

        token = manager.authenticate_user("user123", "password", Role.PHYSICIAN)

        assert token is not None
        assert token in manager.rbac.active_sessions

    def test_authenticate_locked_user(self):
        """Test authentication of locked user."""
        manager = EnhancedPrivacyManager()

        # Lock user
        manager.access_detector.locked_users["user123"] = datetime.now()

        token = manager.authenticate_user("user123", "password", Role.PHYSICIAN)

        assert token is None

    def test_access_patient_data_success(self):
        """Test successful patient data access."""
        manager = EnhancedPrivacyManager()
        token = manager.create_user_session("user123", Role.PHYSICIAN)

        success = manager.access_patient_data(token, "PATIENT_123", "read")

        assert success
        assert len(manager.access_logger.access_log) > 0

    def test_access_patient_data_insufficient_permission(self):
        """Test patient data access with insufficient permissions."""
        manager = EnhancedPrivacyManager()
        token = manager.create_user_session("user123", Role.VIEWER)

        success = manager.access_patient_data(token, "PATIENT_123", "write")

        assert not success

    def test_set_session_timeout(self):
        """Test setting custom session timeout."""
        manager = EnhancedPrivacyManager()
        token = manager.create_user_session("user123", Role.PHYSICIAN)

        success = manager.set_session_timeout(token, 60)

        assert success
        assert manager.timeout_manager.get_timeout(token) == 60

    def test_get_security_metrics(self):
        """Test security metrics generation."""
        manager = EnhancedPrivacyManager()
        token = manager.create_user_session("user123", Role.PHYSICIAN)
        manager.access_patient_data(token, "PATIENT_123", "read")

        metrics = manager.get_security_metrics()

        assert "total_access_events_today" in metrics
        assert "failed_access_attempts_today" in metrics
        assert "locked_users" in metrics
        assert metrics["active_sessions"] == 1

    def test_generate_security_report(self):
        """Test security report generation."""
        manager = EnhancedPrivacyManager()
        manager.create_user_session("user123", Role.PHYSICIAN)

        report = manager.generate_security_report()

        assert "timestamp" in report
        assert "metrics" in report
        assert "recent_events" in report
        assert "active_sessions" in report
        assert len(report["active_sessions"]) == 1


if __name__ == "__main__":
    pytest.main([__file__])
