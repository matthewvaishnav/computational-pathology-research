"""
Clinical workflow integration module for computational pathology.

This module provides clinical-grade functionality including:
- Multi-class disease state classification
- Patient context integration
- Risk factor analysis
- Uncertainty quantification
- Longitudinal patient tracking
- Clinical document parsing
- Clinical standards integration (DICOM, HL7 FHIR)
- Regulatory compliance infrastructure
"""

from .classifier import MultiClassDiseaseClassifier
from .dicom_adapter import (
    DICOMAdapter,
    DICOMMetadata,
    PredictionResult,
    TransferSyntax,
)
from .document_parser import (
    ClinicalDocumentParser,
    DocumentFormat,
    ExtractedEntity,
    ExtractionConfidence,
    ParsedDocument,
)
from .fhir_adapter import (
    AuthenticationMethod,
    DiagnosticReportData,
    FHIRAdapter,
    FHIRResourceType,
    FHIRServerConfig,
    PatientClinicalMetadata,
)
from .longitudinal import (
    LongitudinalTracker,
    PatientTimeline,
    ScanRecord,
    TreatmentEvent,
    TreatmentResponseCategory,
)
from .ood_detection import OODDetector
from .patient_context import (
    AlcoholConsumption,
    ClinicalMetadata,
    ClinicalMetadataEncoder,
    ExerciseFrequency,
    PatientContextIntegrator,
    Sex,
    SmokingStatus,
)
from .privacy import (
    AES256Encryption,
    ConsentRecord,
    DataAccessLogger,
    DataExportMonitor,
    EnhancedPrivacyManager,
    PatientIdentifierAnonymizer,
    Permission,
    PrivacyManager,
    RBACManager,
    Role,
    SecurityAuditEvent,
    SessionTimeoutManager,
    UnauthorizedAccessDetector,
    UserSession,
)
from .reporting import (
    ClinicalReportGenerator,
    DiagnosisResult,
    ExportFormat,
    ReportData,
    ReportSpecialty,
)
from .risk_analysis import ClinicalRiskFactorEncoder, RiskAnalyzer
from .taxonomy import DiseaseTaxonomy
from .temporal_progression import (
    ClinicalMetadataEncoder,
    TemporalProgressionModel,
    TreatmentEffectEncoder,
)
from .thresholds import ClinicalThresholdSystem, ThresholdConfig
from .uncertainty import UncertaintyQuantifier

__all__ = [
    "DiseaseTaxonomy",
    "MultiClassDiseaseClassifier",
    "ClinicalMetadata",
    "PatientContextIntegrator",
    "ClinicalMetadataEncoder",
    "SmokingStatus",
    "AlcoholConsumption",
    "ExerciseFrequency",
    "Sex",
    "RiskAnalyzer",
    "ClinicalRiskFactorEncoder",
    "ClinicalThresholdSystem",
    "ThresholdConfig",
    "UncertaintyQuantifier",
    "OODDetector",
    "PatientTimeline",
    "ScanRecord",
    "TreatmentEvent",
    "TreatmentResponseCategory",
    "LongitudinalTracker",
    "TemporalProgressionModel",
    "TreatmentEffectEncoder",
    "ClinicalMetadataEncoder",
    "ClinicalDocumentParser",
    "DocumentFormat",
    "ExtractedEntity",
    "ExtractionConfidence",
    "ParsedDocument",
    "DICOMAdapter",
    "DICOMMetadata",
    "PredictionResult",
    "TransferSyntax",
    "FHIRAdapter",
    "FHIRServerConfig",
    "FHIRResourceType",
    "AuthenticationMethod",
    "PatientClinicalMetadata",
    "DiagnosticReportData",
    "ClinicalReportGenerator",
    "ReportData",
    "DiagnosisResult",
    "ReportSpecialty",
    "ExportFormat",
    # Privacy and Security
    "PrivacyManager",
    "EnhancedPrivacyManager",
    "AES256Encryption",
    "PatientIdentifierAnonymizer",
    "RBACManager",
    "Role",
    "Permission",
    "UserSession",
    "ConsentRecord",
    "DataAccessLogger",
    "DataExportMonitor",
    "SecurityAuditEvent",
    "SessionTimeoutManager",
    "UnauthorizedAccessDetector",
]
