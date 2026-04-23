"""
Data models for PACS Integration System.

This module defines the core data structures used throughout the PACS integration
system, including study information, endpoint configuration, analysis results,
and enhanced DICOM metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydicom.dataset import Dataset


class PACSVendor(Enum):
    """Supported PACS vendors."""
    GE = "GE"
    PHILIPS = "Philips"
    SIEMENS = "Siemens"
    AGFA = "Agfa"
    GENERIC = "Generic"


class DicomPriority(Enum):
    """DICOM priority levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"


class OperationStatus(Enum):
    """Status of PACS operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class StudyInfo:
    """Information about a DICOM study."""
    study_instance_uid: str
    patient_id: str
    patient_name: str
    study_date: datetime
    study_description: str
    modality: str
    series_count: int
    priority: DicomPriority = DicomPriority.MEDIUM
    accession_number: Optional[str] = None
    referring_physician: Optional[str] = None
    study_time: Optional[str] = None
    
    def to_dicom_query(self) -> Dict[str, str]:
        """Convert to DICOM query parameters."""
        query = {
            "StudyInstanceUID": self.study_instance_uid,
            "PatientID": self.patient_id,
            "PatientName": self.patient_name,
            "StudyDate": self.study_date.strftime("%Y%m%d"),
            "StudyDescription": self.study_description,
            "Modality": self.modality,
        }
        
        if self.accession_number:
            query["AccessionNumber"] = self.accession_number
        if self.referring_physician:
            query["ReferringPhysicianName"] = self.referring_physician
        if self.study_time:
            query["StudyTime"] = self.study_time
            
        return query
    
    def validate_for_processing(self) -> bool:
        """Validate study is suitable for AI processing."""
        # Check required fields
        if not all([self.study_instance_uid, self.patient_id, self.modality]):
            return False
        
        # Check if modality is suitable for pathology AI
        pathology_modalities = ["SM", "XC", "GM"]  # Slide Microscopy, External Camera, General Microscopy
        return self.modality in pathology_modalities


@dataclass
class SeriesInfo:
    """Information about a DICOM series."""
    series_instance_uid: str
    study_instance_uid: str
    series_number: str
    series_description: str
    modality: str
    instance_count: int
    series_date: Optional[datetime] = None
    series_time: Optional[str] = None
    body_part_examined: Optional[str] = None
    
    def to_dicom_query(self) -> Dict[str, str]:
        """Convert to DICOM query parameters."""
        query = {
            "SeriesInstanceUID": self.series_instance_uid,
            "StudyInstanceUID": self.study_instance_uid,
            "SeriesNumber": self.series_number,
            "SeriesDescription": self.series_description,
            "Modality": self.modality,
        }
        
        if self.series_date:
            query["SeriesDate"] = self.series_date.strftime("%Y%m%d")
        if self.series_time:
            query["SeriesTime"] = self.series_time
        if self.body_part_examined:
            query["BodyPartExamined"] = self.body_part_examined
            
        return query


@dataclass
class SecurityConfig:
    """Security configuration for PACS endpoints."""
    tls_enabled: bool = True
    tls_version: str = "1.3"
    certificate_path: Optional[Path] = None
    client_cert_path: Optional[Path] = None
    client_key_path: Optional[Path] = None
    ca_bundle_path: Optional[Path] = None
    verify_certificates: bool = True
    mutual_authentication: bool = False
    
    def validate(self) -> List[str]:
        """Validate security configuration."""
        errors = []
        
        if self.tls_enabled:
            if self.certificate_path and not self.certificate_path.exists():
                errors.append(f"Certificate file not found: {self.certificate_path}")
            
            if self.mutual_authentication:
                if not self.client_cert_path or not self.client_cert_path.exists():
                    errors.append(f"Client certificate required for mutual auth: {self.client_cert_path}")
                if not self.client_key_path or not self.client_key_path.exists():
                    errors.append(f"Client key required for mutual auth: {self.client_key_path}")
        
        return errors


@dataclass
class PerformanceConfig:
    """Performance configuration for PACS operations."""
    max_concurrent_studies: int = 50
    connection_pool_size: int = 10
    query_timeout: int = 30
    retrieval_timeout: int = 300
    storage_timeout: int = 120
    retry_attempts: int = 3
    retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    
    def validate(self) -> List[str]:
        """Validate performance configuration."""
        errors = []
        
        if self.max_concurrent_studies <= 0:
            errors.append("max_concurrent_studies must be positive")
        if self.connection_pool_size <= 0:
            errors.append("connection_pool_size must be positive")
        if self.query_timeout <= 0:
            errors.append("query_timeout must be positive")
        if self.retrieval_timeout <= 0:
            errors.append("retrieval_timeout must be positive")
        if self.retry_attempts < 0:
            errors.append("retry_attempts must be non-negative")
        
        return errors


@dataclass
class PACSEndpoint:
    """Configuration for a PACS endpoint."""
    endpoint_id: str
    ae_title: str
    host: str
    port: int
    vendor: PACSVendor
    security_config: SecurityConfig
    performance_config: PerformanceConfig
    description: Optional[str] = None
    is_primary: bool = False
    
    def __post_init__(self):
        """Validate endpoint configuration after initialization."""
        errors = []
        
        if not self.endpoint_id:
            errors.append("endpoint_id is required")
        if not self.ae_title:
            errors.append("ae_title is required")
        if not self.host:
            errors.append("host is required")
        if not (1 <= self.port <= 65535):
            errors.append("port must be between 1 and 65535")
        
        errors.extend(self.security_config.validate())
        errors.extend(self.performance_config.validate())
        
        if errors:
            raise ValueError(f"Invalid PACS endpoint configuration: {'; '.join(errors)}")
    
    def create_association_parameters(self) -> Dict[str, Any]:
        """Create pynetdicom association parameters."""
        return {
            "ae_title": self.ae_title,
            "peer_ae_title": self.ae_title,
            "address": self.host,
            "port": self.port,
            "timeout": self.performance_config.query_timeout,
        }
    
    def supports_transfer_syntax(self, syntax_uid: str) -> bool:
        """Check if endpoint supports specific transfer syntax."""
        # Vendor-specific transfer syntax support
        vendor_syntaxes = {
            PACSVendor.GE: [
                "1.2.840.10008.1.2",  # Implicit VR Little Endian
                "1.2.840.10008.1.2.1",  # Explicit VR Little Endian
                "1.2.840.10008.1.2.4.90",  # JPEG 2000 Lossless
            ],
            PACSVendor.PHILIPS: [
                "1.2.840.10008.1.2",
                "1.2.840.10008.1.2.1",
                "1.2.840.10008.1.2.4.70",  # JPEG Lossless
            ],
            PACSVendor.SIEMENS: [
                "1.2.840.10008.1.2",
                "1.2.840.10008.1.2.1",
                "1.2.840.10008.1.2.4.90",
                "1.2.840.10008.1.2.4.80",  # JPEG-LS Lossless
            ],
            PACSVendor.AGFA: [
                "1.2.840.10008.1.2",
                "1.2.840.10008.1.2.1",
            ],
        }
        
        supported = vendor_syntaxes.get(self.vendor, [
            "1.2.840.10008.1.2",
            "1.2.840.10008.1.2.1",
        ])
        
        return syntax_uid in supported


@dataclass
class PACSConfiguration:
    """Complete PACS configuration for an environment."""
    profile_name: str
    pacs_endpoints: Dict[str, PACSEndpoint]
    storage_config: Dict[str, Any]
    notification_config: Dict[str, Any] = field(default_factory=dict)
    audit_config: Dict[str, Any] = field(default_factory=dict)
    
    def get_primary_endpoint(self) -> Optional[PACSEndpoint]:
        """Get the primary PACS endpoint."""
        for endpoint in self.pacs_endpoints.values():
            if endpoint.is_primary:
                return endpoint
        return None
    
    def get_backup_endpoints(self) -> List[PACSEndpoint]:
        """Get backup PACS endpoints."""
        return [ep for ep in self.pacs_endpoints.values() if not ep.is_primary]
    
    def validate(self) -> List[str]:
        """Validate complete configuration."""
        errors = []
        
        if not self.pacs_endpoints:
            errors.append("At least one PACS endpoint is required")
        
        primary_count = sum(1 for ep in self.pacs_endpoints.values() if ep.is_primary)
        if primary_count == 0:
            errors.append("At least one primary endpoint is required")
        elif primary_count > 1:
            errors.append("Only one primary endpoint is allowed")
        
        return errors


@dataclass
class DetectedRegion:
    """Detected region in WSI analysis."""
    region_id: str
    coordinates: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    region_type: str
    description: Optional[str] = None


@dataclass
class DiagnosticRecommendation:
    """Diagnostic recommendation from AI analysis."""
    recommendation_id: str
    recommendation_text: str
    confidence: float
    urgency_level: str
    supporting_evidence: Optional[List[str]] = None


@dataclass
class AnalysisResults:
    """AI analysis results for DICOM Structured Report generation."""
    study_instance_uid: str
    series_instance_uid: str
    algorithm_name: str
    algorithm_version: str
    confidence_score: float
    detected_regions: List[DetectedRegion]
    diagnostic_recommendations: List[DiagnosticRecommendation]
    processing_timestamp: datetime
    primary_diagnosis: Optional[str] = None
    probability_distribution: Optional[Dict[str, float]] = None
    
    def to_structured_report(self) -> Dataset:
        """Convert to DICOM Structured Report."""
        # This will be implemented in the StorageEngine
        raise NotImplementedError("Implemented in StorageEngine")
    
    def validate_clinical_thresholds(self) -> bool:
        """Validate results meet clinical acceptance criteria."""
        # Check minimum confidence threshold
        if self.confidence_score < 0.7:
            return False
        
        # Check that detected regions have reasonable confidence
        for region in self.detected_regions:
            if region.confidence < 0.5:
                return False
        
        # Check diagnostic recommendations
        for rec in self.diagnostic_recommendations:
            if rec.confidence < 0.6:
                return False
        
        return True


@dataclass
class PACSMetadata:
    """Enhanced DICOM metadata with PACS-specific fields."""
    # Base DICOM metadata
    patient_id: str
    patient_name: str
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    modality: str
    
    # PACS-specific fields
    source_pacs_ae_title: str
    retrieval_timestamp: datetime
    original_transfer_syntax: str
    compression_ratio: Optional[float] = None
    network_transfer_time: Optional[timedelta] = None
    
    # Optional DICOM fields
    study_date: Optional[str] = None
    series_date: Optional[str] = None
    acquisition_date: Optional[str] = None
    institution_name: Optional[str] = None
    manufacturer: Optional[str] = None
    manufacturer_model: Optional[str] = None
    image_type: Optional[List[str]] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    number_of_frames: Optional[int] = None
    additional_metadata: Optional[Dict[str, Any]] = None
    
    def calculate_quality_metrics(self) -> Dict[str, float]:
        """Calculate image quality metrics for clinical validation."""
        metrics = {}
        
        if self.compression_ratio is not None:
            metrics["compression_ratio"] = self.compression_ratio
            metrics["compression_quality"] = min(1.0, 1.0 / self.compression_ratio)
        
        if self.network_transfer_time is not None:
            # Calculate transfer rate (assuming some file size)
            transfer_seconds = self.network_transfer_time.total_seconds()
            if transfer_seconds > 0:
                metrics["transfer_time_seconds"] = transfer_seconds
        
        return metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "study_instance_uid": self.study_instance_uid,
            "series_instance_uid": self.series_instance_uid,
            "sop_instance_uid": self.sop_instance_uid,
            "modality": self.modality,
            "source_pacs_ae_title": self.source_pacs_ae_title,
            "retrieval_timestamp": self.retrieval_timestamp.isoformat(),
            "original_transfer_syntax": self.original_transfer_syntax,
            "compression_ratio": self.compression_ratio,
            "network_transfer_time": self.network_transfer_time.total_seconds() if self.network_transfer_time else None,
            "study_date": self.study_date,
            "series_date": self.series_date,
            "acquisition_date": self.acquisition_date,
            "institution_name": self.institution_name,
            "manufacturer": self.manufacturer,
            "manufacturer_model": self.manufacturer_model,
            "image_type": self.image_type,
            "rows": self.rows,
            "columns": self.columns,
            "number_of_frames": self.number_of_frames,
            "additional_metadata": self.additional_metadata or {},
        }


@dataclass
class ValidationResult:
    """Result of validation operations."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)


@dataclass
class OperationResult:
    """Result of PACS operations."""
    success: bool
    operation_id: str
    timestamp: datetime
    message: str
    data: Optional[Any] = None
    errors: List[str] = field(default_factory=list)
    
    @classmethod
    def success_result(cls, operation_id: str, message: str, data: Any = None) -> "OperationResult":
        """Create a success result."""
        return cls(
            success=True,
            operation_id=operation_id,
            timestamp=datetime.now(),
            message=message,
            data=data
        )
    
    @classmethod
    def error_result(cls, operation_id: str, message: str, errors: List[str] = None) -> "OperationResult":
        """Create an error result."""
        return cls(
            success=False,
            operation_id=operation_id,
            timestamp=datetime.now(),
            message=message,
            errors=errors or []
        )


# Type aliases for common operations
QueryResult = List[StudyInfo]
RetrievalResult = OperationResult
StorageResult = OperationResult