"""FHIR streaming client for clinical system integration.

Integrates HL7 FHIR w/ PACS WSI streaming for healthcare interoperability.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from src.clinical.fhir_adapter import (
    FHIRAdapter,
    FHIRServerConfig,
    PatientClinicalMetadata,
    DiagnosticReportData,
    AuthenticationMethod
)
from .pacs_wsi_client import AnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class StreamingDiagnosticReport:
    """Streaming diagnostic report for real-time analysis."""
    patient_id: str
    study_uid: str
    series_uid: str
    prediction: str
    confidence: float
    processing_time: float
    timestamp: datetime
    attention_weights: Optional[Dict[str, float]] = None
    model_version: str = "1.0.0"
    
    def to_fhir_diagnostic_report(self) -> DiagnosticReportData:
        """Convert to FHIR DiagnosticReport format."""
        return DiagnosticReportData(
            patient_id=self.patient_id,
            imaging_study_id=self.study_uid,
            issued_datetime=self.timestamp,
            status="final",
            code={
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "60567-5",
                    "display": "Comprehensive pathology report"
                }]
            },
            conclusion=f"AI Analysis: {self.prediction}",
            primary_diagnosis=self.prediction,
            confidence_score=self.confidence,
            uncertainty_explanation=f"Model confidence: {self.confidence:.2%}",
            model_version=self.model_version
        )


class FHIRStreamingClient:
    """Client for FHIR integration w/ streaming WSI analysis.
    
    Features:
    - Patient metadata retrieval from FHIR servers
    - Study metadata exchange w/ EHR systems
    - Diagnostic report generation in FHIR format
    - Real-time result delivery to clinical systems
    """
    
    def __init__(self,
                 fhir_base_url: str,
                 auth_method: str = "none",
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None,
                 token_url: Optional[str] = None):
        """Init FHIR streaming client.
        
        Args:
            fhir_base_url: FHIR server base URL
            auth_method: Authentication method (oauth2/smart_on_fhir/basic/none)
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            token_url: OAuth2 token endpoint URL
        """
        # Map string to enum
        auth_enum = {
            "oauth2": AuthenticationMethod.OAUTH2,
            "smart_on_fhir": AuthenticationMethod.SMART_ON_FHIR,
            "basic": AuthenticationMethod.BASIC,
            "none": AuthenticationMethod.NONE
        }.get(auth_method.lower(), AuthenticationMethod.NONE)
        
        config = FHIRServerConfig(
            base_url=fhir_base_url,
            auth_method=auth_enum,
            client_id=client_id,
            client_secret=client_secret,
            token_url=token_url
        )
        
        self.fhir_adapter = FHIRAdapter(config)
        
        logger.info(f"Init FHIRStreamingClient: url={fhir_base_url}, auth={auth_method}")
    
    def get_patient_metadata(self, patient_id: str) -> Optional[PatientClinicalMetadata]:
        """Retrieve patient clinical metadata from FHIR.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            PatientClinicalMetadata or None on failure
        """
        logger.info(f"Retrieve patient metadata: {patient_id}")
        
        try:
            metadata = self.fhir_adapter.get_patient_clinical_metadata(patient_id)
            
            if metadata:
                logger.info(f"Retrieved metadata for patient {patient_id}")
                return metadata
            else:
                logger.warning(f"No metadata found for patient {patient_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve patient metadata: {e}")
            return None
    
    def get_study_metadata(self,
                          patient_id: str,
                          study_uid: str) -> Optional[Dict[str, Any]]:
        """Retrieve study metadata from FHIR ImagingStudy.
        
        Args:
            patient_id: Patient identifier
            study_uid: Study Instance UID
            
        Returns:
            Study metadata dict or None on failure
        """
        logger.info(f"Retrieve study metadata: patient={patient_id}, study={study_uid}")
        
        try:
            # Query FHIR ImagingStudy resource
            imaging_study = self.fhir_adapter.get_imaging_study(
                patient_id=patient_id,
                study_uid=study_uid
            )
            
            if imaging_study:
                logger.info(f"Retrieved study metadata for {study_uid}")
                return imaging_study
            else:
                logger.warning(f"No study metadata found for {study_uid}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve study metadata: {e}")
            return None
    
    def create_diagnostic_report(self,
                                report: StreamingDiagnosticReport) -> Optional[str]:
        """Create FHIR DiagnosticReport from streaming analysis.
        
        Args:
            report: StreamingDiagnosticReport object
            
        Returns:
            FHIR resource ID or None on failure
        """
        logger.info(f"Create diagnostic report: patient={report.patient_id}, "
                   f"study={report.study_uid}")
        
        try:
            # Convert to FHIR format
            fhir_report = report.to_fhir_diagnostic_report()
            
            # Create resource on FHIR server
            resource_id = self.fhir_adapter.create_diagnostic_report(fhir_report)
            
            if resource_id:
                logger.info(f"Created diagnostic report: {resource_id}")
                return resource_id
            else:
                logger.error("Failed to create diagnostic report")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create diagnostic report: {e}")
            return None
    
    def update_diagnostic_report(self,
                                resource_id: str,
                                report: StreamingDiagnosticReport) -> bool:
        """Update existing FHIR DiagnosticReport.
        
        Args:
            resource_id: FHIR resource ID
            report: Updated StreamingDiagnosticReport
            
        Returns:
            True if update successful
        """
        logger.info(f"Update diagnostic report: {resource_id}")
        
        try:
            fhir_report = report.to_fhir_diagnostic_report()
            
            success = self.fhir_adapter.update_diagnostic_report(
                resource_id=resource_id,
                report_data=fhir_report
            )
            
            if success:
                logger.info(f"Updated diagnostic report: {resource_id}")
            else:
                logger.error(f"Failed to update diagnostic report: {resource_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update diagnostic report: {e}")
            return False
    
    def convert_analysis_result_to_report(self,
                                         result: AnalysisResult,
                                         patient_id: str) -> StreamingDiagnosticReport:
        """Convert AnalysisResult to StreamingDiagnosticReport.
        
        Args:
            result: AnalysisResult from PACS workflow
            patient_id: Patient identifier
            
        Returns:
            StreamingDiagnosticReport object
        """
        return StreamingDiagnosticReport(
            patient_id=patient_id,
            study_uid=result.study_uid,
            series_uid=result.series_uid,
            prediction=result.prediction,
            confidence=result.confidence,
            processing_time=result.processing_time,
            timestamp=datetime.fromtimestamp(result.timestamp),
            attention_weights=result.attention_weights
        )
    
    def validate_patient_match(self,
                              pacs_patient_id: str,
                              fhir_patient_id: str) -> bool:
        """Validate patient ID match between PACS and FHIR.
        
        Args:
            pacs_patient_id: Patient ID from PACS
            fhir_patient_id: Patient ID from FHIR
            
        Returns:
            True if IDs match
        """
        # Simple exact match for now
        # Production would use MPI (Master Patient Index) matching
        match = pacs_patient_id == fhir_patient_id
        
        if not match:
            logger.warning(f"Patient ID mismatch: PACS={pacs_patient_id}, "
                          f"FHIR={fhir_patient_id}")
        
        return match
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "fhir_base_url": self.fhir_adapter.config.base_url,
            "auth_method": self.fhir_adapter.config.auth_method.value,
            "timeout": self.fhir_adapter.config.timeout
        }
