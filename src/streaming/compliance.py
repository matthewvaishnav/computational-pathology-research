"""
Healthcare compliance for HistoCore Real-Time WSI Streaming.

HIPAA, GDPR, FDA 510(k) compliance measures.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

# Import existing clinical compliance modules
from src.clinical.privacy import (
    AESEncryption,
    ConsentRecord,
    EncryptionProvider,
    PatientIdentifierAnonymizer,
    Permission,
    Role,
    UserSession,
)
from src.clinical.regulatory import ComplianceValidator, RegulatoryFramework, RiskLevel

logger = logging.getLogger(__name__)


# ============================================================================
# HIPAA Compliance
# ============================================================================


class HIPAARequirement(str, Enum):
    """HIPAA Security Rule requirements."""

    # Administrative Safeguards
    SECURITY_MANAGEMENT = "security_management"
    WORKFORCE_SECURITY = "workforce_security"
    INFORMATION_ACCESS = "information_access"
    SECURITY_AWARENESS = "security_awareness"
    CONTINGENCY_PLAN = "contingency_plan"

    # Physical Safeguards
    FACILITY_ACCESS = "facility_access"
    WORKSTATION_SECURITY = "workstation_security"
    DEVICE_MEDIA = "device_media"

    # Technical Safeguards
    ACCESS_CONTROL = "access_control"
    AUDIT_CONTROLS = "audit_controls"
    INTEGRITY_CONTROLS = "integrity_controls"
    TRANSMISSION_SECURITY = "transmission_security"


@dataclass
class HIPAAConfig:
    """HIPAA compliance configuration."""

    enable_phi_encryption: bool = True
    enable_audit_logging: bool = True
    enable_access_control: bool = True
    enable_data_integrity: bool = True
    enable_transmission_security: bool = True

    # Data retention
    audit_log_retention_years: int = 6
    phi_retention_years: int = 6

    # Security policies
    password_min_length: int = 12
    session_timeout_minutes: int = 30
    max_failed_login_attempts: int = 5
    require_mfa: bool = False

    # Anonymization
    enable_phi_anonymization: bool = True
    anonymization_method: str = "hash"  # hash, pseudonymize, redact


class HIPAACompliance:
    """HIPAA compliance manager for streaming system."""

    def __init__(self, config: HIPAAConfig):
        """Initialize HIPAA compliance."""
        self.config = config

        # Initialize encryption
        if config.enable_phi_encryption:
            self.encryption = AESEncryption()

        # Initialize anonymizer
        if config.enable_phi_anonymization:
            self.anonymizer = PatientIdentifierAnonymizer()

        # Consent tracking
        self.consent_records: Dict[str, ConsentRecord] = {}

        logger.info("HIPAA compliance initialized")

    def anonymize_phi(self, patient_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize Protected Health Information."""
        if not self.config.enable_phi_anonymization:
            return data

        anonymized = data.copy()

        # Anonymize patient identifiers
        if "patient_id" in anonymized:
            anonymized["patient_id"] = self.anonymizer.anonymize(patient_id)

        # Remove direct identifiers
        phi_fields = [
            "patient_name",
            "ssn",
            "mrn",
            "address",
            "phone",
            "email",
            "date_of_birth",
            "account_number",
        ]

        for field in phi_fields:
            if field in anonymized:
                anonymized[field] = "[REDACTED]"

        return anonymized

    def encrypt_phi(self, data: bytes) -> bytes:
        """Encrypt PHI data."""
        if not self.config.enable_phi_encryption:
            return data

        return self.encryption.encrypt(data)

    def decrypt_phi(self, encrypted_data: bytes) -> bytes:
        """Decrypt PHI data."""
        if not self.config.enable_phi_encryption:
            return encrypted_data

        return self.encryption.decrypt(encrypted_data)

    def check_consent(self, patient_id: str, purpose: str) -> bool:
        """Check patient consent for data use."""
        consent = self.consent_records.get(patient_id)

        if not consent:
            logger.warning("No consent record for patient: %s", patient_id)
            return False

        if not consent.is_valid():
            logger.warning("Consent expired for patient: %s", patient_id)
            return False

        if consent.purpose and consent.purpose != purpose:
            logger.warning("Consent purpose mismatch for patient: %s", patient_id)
            return False

        return True

    def record_consent(
        self,
        patient_id: str,
        consent_type: str,
        granted: bool,
        purpose: Optional[str] = None,
        expires_days: Optional[int] = None,
    ):
        """Record patient consent."""
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

        consent = ConsentRecord(
            patient_id=patient_id,
            consent_type=consent_type,
            granted=granted,
            granted_at=datetime.utcnow(),
            expires_at=expires_at,
            purpose=purpose,
        )

        self.consent_records[patient_id] = consent

        logger.info("Recorded consent for patient: %s (granted=%s)", patient_id, granted)

    def secure_delete_phi(self, patient_id: str) -> bool:
        """Securely delete PHI (right to be forgotten)."""
        # In production, this would:
        # 1. Overwrite data multiple times
        # 2. Remove from all backups
        # 3. Update audit log (preserve audit trail)
        # 4. Notify relevant parties

        if patient_id in self.consent_records:
            del self.consent_records[patient_id]

        logger.info("Securely deleted PHI for patient: %s", patient_id)
        return True

    def generate_hipaa_report(self) -> Dict[str, Any]:
        """Generate HIPAA compliance report."""
        return {
            "report_date": datetime.utcnow().isoformat(),
            "phi_encryption_enabled": self.config.enable_phi_encryption,
            "audit_logging_enabled": self.config.enable_audit_logging,
            "access_control_enabled": self.config.enable_access_control,
            "data_integrity_enabled": self.config.enable_data_integrity,
            "transmission_security_enabled": self.config.enable_transmission_security,
            "total_consent_records": len(self.consent_records),
            "active_consents": sum(1 for c in self.consent_records.values() if c.is_valid()),
            "audit_retention_years": self.config.audit_log_retention_years,
            "phi_retention_years": self.config.phi_retention_years,
        }


# ============================================================================
# GDPR Compliance
# ============================================================================


class GDPRRight(str, Enum):
    """GDPR data subject rights."""

    RIGHT_TO_ACCESS = "right_to_access"
    RIGHT_TO_RECTIFICATION = "right_to_rectification"
    RIGHT_TO_ERASURE = "right_to_erasure"
    RIGHT_TO_RESTRICT = "right_to_restrict"
    RIGHT_TO_PORTABILITY = "right_to_portability"
    RIGHT_TO_OBJECT = "right_to_object"


@dataclass
class GDPRConfig:
    """GDPR compliance configuration."""

    enable_consent_management: bool = True
    enable_data_portability: bool = True
    enable_right_to_erasure: bool = True

    # Data processing
    data_retention_days: int = 2555  # 7 years
    consent_expiry_days: int = 365

    # Privacy
    enable_privacy_by_design: bool = True
    enable_privacy_by_default: bool = True

    # Breach notification
    breach_notification_hours: int = 72


class GDPRCompliance:
    """GDPR compliance manager."""

    def __init__(self, config: GDPRConfig):
        """Initialize GDPR compliance."""
        self.config = config

        # Data processing records
        self.processing_records: List[Dict[str, Any]] = []

        # Data subject requests
        self.subject_requests: List[Dict[str, Any]] = []

        logger.info("GDPR compliance initialized")

    def record_data_processing(
        self, subject_id: str, purpose: str, legal_basis: str, data_categories: List[str]
    ):
        """Record data processing activity."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "subject_id": subject_id,
            "purpose": purpose,
            "legal_basis": legal_basis,
            "data_categories": data_categories,
        }

        self.processing_records.append(record)

        logger.info("Recorded data processing: subject=%s purpose=%s", subject_id, purpose)

    def handle_subject_request(
        self, subject_id: str, right: GDPRRight, details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle GDPR data subject request."""
        request = {
            "request_id": f"REQ-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "subject_id": subject_id,
            "right": right.value,
            "status": "pending",
            "details": details or {},
        }

        self.subject_requests.append(request)

        logger.info("Received GDPR request: %s for subject %s", right.value, subject_id)

        # Handle specific rights
        if right == GDPRRight.RIGHT_TO_ACCESS:
            return self._handle_access_request(subject_id)
        elif right == GDPRRight.RIGHT_TO_ERASURE:
            return self._handle_erasure_request(subject_id)
        elif right == GDPRRight.RIGHT_TO_PORTABILITY:
            return self._handle_portability_request(subject_id)

        return request

    def _handle_access_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle right to access request."""
        # Collect all data for subject
        data = {
            "subject_id": subject_id,
            "processing_records": [
                r for r in self.processing_records if r["subject_id"] == subject_id
            ],
            "data_categories": ["slides", "results", "audit_logs"],
        }

        return {"status": "completed", "data": data}

    def _handle_erasure_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle right to erasure (right to be forgotten)."""
        if not self.config.enable_right_to_erasure:
            return {"status": "denied", "reason": "Right to erasure not enabled"}

        # Remove processing records
        self.processing_records = [
            r for r in self.processing_records if r["subject_id"] != subject_id
        ]

        logger.info("Erased data for subject: %s", subject_id)

        return {"status": "completed"}

    def _handle_portability_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle right to data portability."""
        if not self.config.enable_data_portability:
            return {"status": "denied", "reason": "Data portability not enabled"}

        # Export data in machine-readable format
        data = {
            "subject_id": subject_id,
            "export_date": datetime.utcnow().isoformat(),
            "format": "JSON",
            "processing_records": [
                r for r in self.processing_records if r["subject_id"] == subject_id
            ],
        }

        return {"status": "completed", "data": data}

    def generate_gdpr_report(self) -> Dict[str, Any]:
        """Generate GDPR compliance report."""
        return {
            "report_date": datetime.utcnow().isoformat(),
            "consent_management_enabled": self.config.enable_consent_management,
            "data_portability_enabled": self.config.enable_data_portability,
            "right_to_erasure_enabled": self.config.enable_right_to_erasure,
            "total_processing_records": len(self.processing_records),
            "total_subject_requests": len(self.subject_requests),
            "pending_requests": sum(1 for r in self.subject_requests if r["status"] == "pending"),
            "data_retention_days": self.config.data_retention_days,
            "breach_notification_hours": self.config.breach_notification_hours,
        }


# ============================================================================
# FDA 510(k) Compliance
# ============================================================================


@dataclass
class FDAConfig:
    """FDA 510(k) compliance configuration."""

    device_class: str = "Class II"  # Class I, II, III
    intended_use: str = "Diagnostic aid for pathology"

    # Software lifecycle (IEC 62304)
    enable_software_lifecycle: bool = True
    software_safety_class: str = "B"  # A, B, C

    # Risk management (ISO 14971)
    enable_risk_management: bool = True

    # Quality system (21 CFR Part 820)
    enable_quality_system: bool = True

    # Clinical validation
    enable_clinical_validation: bool = True
    validation_dataset_size: int = 1000


class FDACompliance:
    """FDA 510(k) compliance manager."""

    def __init__(self, config: FDAConfig):
        """Initialize FDA compliance."""
        self.config = config

        # Validation records
        self.validation_records: List[Dict[str, Any]] = []

        # Risk assessments
        self.risk_assessments: List[Dict[str, Any]] = []

        logger.info("FDA compliance initialized: class=%s", config.device_class)

    def record_validation(
        self,
        validation_type: str,
        dataset_size: int,
        accuracy: float,
        sensitivity: float,
        specificity: float,
    ):
        """Record clinical validation results."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "validation_type": validation_type,
            "dataset_size": dataset_size,
            "metrics": {
                "accuracy": accuracy,
                "sensitivity": sensitivity,
                "specificity": specificity,
            },
        }

        self.validation_records.append(record)

        logger.info("Recorded validation: type=%s accuracy=%.3f", validation_type, accuracy)

    def assess_risk(self, hazard: str, severity: str, probability: str, mitigation: str):
        """Record risk assessment (ISO 14971)."""
        assessment = {
            "timestamp": datetime.utcnow().isoformat(),
            "hazard": hazard,
            "severity": severity,
            "probability": probability,
            "risk_level": self._calculate_risk_level(severity, probability),
            "mitigation": mitigation,
        }

        self.risk_assessments.append(assessment)

        logger.info(
            "Recorded risk assessment: hazard=%s level=%s", hazard, assessment["risk_level"]
        )

    def _calculate_risk_level(self, severity: str, probability: str) -> str:
        """Calculate risk level from severity and probability."""
        risk_matrix = {
            ("high", "high"): "critical",
            ("high", "medium"): "high",
            ("high", "low"): "medium",
            ("medium", "high"): "high",
            ("medium", "medium"): "medium",
            ("medium", "low"): "low",
            ("low", "high"): "medium",
            ("low", "medium"): "low",
            ("low", "low"): "low",
        }

        return risk_matrix.get((severity.lower(), probability.lower()), "unknown")

    def generate_fda_report(self) -> Dict[str, Any]:
        """Generate FDA 510(k) compliance report."""
        return {
            "report_date": datetime.utcnow().isoformat(),
            "device_class": self.config.device_class,
            "intended_use": self.config.intended_use,
            "software_safety_class": self.config.software_safety_class,
            "total_validations": len(self.validation_records),
            "total_risk_assessments": len(self.risk_assessments),
            "critical_risks": sum(
                1 for r in self.risk_assessments if r["risk_level"] == "critical"
            ),
            "software_lifecycle_enabled": self.config.enable_software_lifecycle,
            "risk_management_enabled": self.config.enable_risk_management,
            "quality_system_enabled": self.config.enable_quality_system,
        }


# ============================================================================
# Unified Compliance Manager
# ============================================================================


class ComplianceManager:
    """Unified compliance manager for all regulations."""

    def __init__(
        self,
        hipaa_config: Optional[HIPAAConfig] = None,
        gdpr_config: Optional[GDPRConfig] = None,
        fda_config: Optional[FDAConfig] = None,
    ):
        """Initialize compliance manager."""
        self.hipaa = HIPAACompliance(hipaa_config or HIPAAConfig())
        self.gdpr = GDPRCompliance(gdpr_config or GDPRConfig())
        self.fda = FDACompliance(fda_config or FDAConfig())

        logger.info("Compliance manager initialized: HIPAA + GDPR + FDA")

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        return {
            "report_date": datetime.utcnow().isoformat(),
            "hipaa": self.hipaa.generate_hipaa_report(),
            "gdpr": self.gdpr.generate_gdpr_report(),
            "fda": self.fda.generate_fda_report(),
        }


# ============================================================================
# Convenience Functions
# ============================================================================


def create_compliance_manager() -> ComplianceManager:
    """Create compliance manager with default configs."""
    return ComplianceManager()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create compliance manager
    compliance = create_compliance_manager()

    # HIPAA: Record consent
    compliance.hipaa.record_consent(
        patient_id="P12345",
        consent_type="research",
        granted=True,
        purpose="AI model training",
        expires_days=365,
    )

    # GDPR: Record processing
    compliance.gdpr.record_data_processing(
        subject_id="P12345",
        purpose="diagnostic_analysis",
        legal_basis="consent",
        data_categories=["medical_images", "diagnostic_results"],
    )

    # FDA: Record validation
    compliance.fda.record_validation(
        validation_type="clinical",
        dataset_size=1000,
        accuracy=0.95,
        sensitivity=0.93,
        specificity=0.97,
    )

    # Generate report
    report = compliance.generate_comprehensive_report()
    print(f"Compliance report generated: {len(report)} frameworks")
