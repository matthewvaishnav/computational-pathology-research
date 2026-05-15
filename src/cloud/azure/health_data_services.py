"""
Azure Health Data Services Integration

Provides integration with Azure Health Data Services (AHDS) for FHIR-compliant
healthcare data exchange, patient data management, and clinical workflow integration.

This module implements:
- FHIR R4 resource management
- Patient data synchronization
- Diagnostic report submission
- Clinical workflow integration
- Audit logging and compliance
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

try:
    from azure.identity import DefaultAzureCredential
    from azure.core.exceptions import AzureError
    import requests

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class FHIRResourceType(Enum):
    """FHIR resource types supported by HistoCore."""

    PATIENT = "Patient"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    OBSERVATION = "Observation"
    IMAGING_STUDY = "ImagingStudy"
    SPECIMEN = "Specimen"
    ORGANIZATION = "Organization"
    PRACTITIONER = "Practitioner"


@dataclass
class HealthDataConfig:
    """Configuration for Azure Health Data Services."""

    workspace_url: str
    fhir_service_name: str
    tenant_id: str
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    use_managed_identity: bool = True
    api_version: str = "2022-06-01"
    timeout: int = 30


@dataclass
class PatientData:
    """Patient data structure for FHIR Patient resource."""

    patient_id: str
    given_name: str
    family_name: str
    birth_date: Optional[str] = None
    gender: Optional[str] = None
    mrn: Optional[str] = None  # Medical Record Number
    organization_id: Optional[str] = None


@dataclass
class DiagnosticReportData:
    """Diagnostic report data for FHIR DiagnosticReport resource."""

    report_id: str
    patient_id: str
    specimen_id: str
    practitioner_id: str
    status: str  # registered, partial, preliminary, final, amended, corrected, cancelled
    category: str  # pathology, radiology, etc.
    code: str  # LOINC code for the diagnostic procedure
    conclusion: str
    confidence_score: float
    ai_generated: bool = True
    effective_datetime: Optional[str] = None
    issued_datetime: Optional[str] = None


@dataclass
class ObservationData:
    """Observation data for FHIR Observation resource."""

    observation_id: str
    patient_id: str
    diagnostic_report_id: str
    code: str  # LOINC or SNOMED code
    value: Union[str, float, bool]
    unit: Optional[str] = None
    status: str = "final"
    effective_datetime: Optional[str] = None


class AzureHealthDataServices:
    """Azure Health Data Services integration for FHIR-compliant healthcare data exchange."""

    def __init__(self, config: HealthDataConfig):
        """Initialize Azure Health Data Services client."""
        if not AZURE_AVAILABLE:
            raise ImportError(
                "Azure SDK not available. Install with: "
                "pip install azure-identity azure-core requests"
            )

        self.config = config
        self.credential = None
        self.access_token = None
        self.token_expires_at = None
        self.base_url = f"{config.workspace_url}/fhir"

        self._initialize_authentication()
        logger.info(
            "Azure Health Data Services initialized: workspace=%s, service=%s",
            config.workspace_url,
            config.fhir_service_name,
        )

    def _initialize_authentication(self) -> None:
        """Initialize Azure authentication."""
        try:
            if self.config.use_managed_identity:
                self.credential = DefaultAzureCredential()
            else:
                from azure.identity import ClientSecretCredential

                self.credential = ClientSecretCredential(
                    tenant_id=self.config.tenant_id,
                    client_id=self.config.client_id,
                    client_secret=self.config.client_secret,
                )
            logger.info("Azure authentication initialized")
        except Exception as e:
            logger.error("Failed to initialize Azure authentication: %s", e)
            raise

    def _get_access_token(self) -> str:
        """Get valid access token for Azure Health Data Services."""
        if (
            self.access_token
            and self.token_expires_at
            and datetime.now(timezone.utc) < self.token_expires_at
        ):
            return self.access_token

        try:
            # Request token for Healthcare APIs scope
            token = self.credential.get_token("https://azurehealthcareapis.com/.default")
            self.access_token = token.token
            self.token_expires_at = datetime.fromtimestamp(token.expires_on, timezone.utc)
            logger.debug("Access token refreshed, expires at: %s", self.token_expires_at)
            return self.access_token
        except Exception as e:
            logger.error("Failed to get access token: %s", e)
            raise

    def _make_fhir_request(
        self,
        method: str,
        resource_path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        """Make authenticated FHIR API request."""
        url = f"{self.base_url}/{resource_path}"
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/fhir+json",
            "Accept": "application/fhir+json",
        }

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params,
                timeout=self.config.timeout,
            )
            response.raise_for_status()

            if response.content:
                return response.json()
            return {}

        except requests.exceptions.RequestException as e:
            logger.error("FHIR API request failed: %s", e)
            raise
        except json.JSONDecodeError as e:
            logger.error("Failed to parse FHIR response: %s", e)
            raise

    def create_patient(self, patient_data: PatientData) -> Dict:
        """Create FHIR Patient resource."""
        fhir_patient = {
            "resourceType": "Patient",
            "id": patient_data.patient_id,
            "identifier": [
                {
                    "use": "usual",
                    "type": {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                                "code": "MR",
                                "display": "Medical Record Number",
                            }
                        ]
                    },
                    "value": patient_data.mrn or patient_data.patient_id,
                }
            ],
            "name": [
                {
                    "use": "official",
                    "family": patient_data.family_name,
                    "given": [patient_data.given_name],
                }
            ],
        }

        if patient_data.birth_date:
            fhir_patient["birthDate"] = patient_data.birth_date

        if patient_data.gender:
            fhir_patient["gender"] = patient_data.gender.lower()

        if patient_data.organization_id:
            fhir_patient["managingOrganization"] = {
                "reference": f"Organization/{patient_data.organization_id}"
            }

        try:
            result = self._make_fhir_request("POST", "Patient", fhir_patient)
            logger.info("Created FHIR Patient: %s", patient_data.patient_id)
            return result
        except Exception as e:
            logger.error("Failed to create patient %s: %s", patient_data.patient_id, e)
            raise

    def create_diagnostic_report(self, report_data: DiagnosticReportData) -> Dict:
        """Create FHIR DiagnosticReport resource for AI pathology analysis."""
        current_time = datetime.now(timezone.utc).isoformat()

        fhir_report = {
            "resourceType": "DiagnosticReport",
            "id": report_data.report_id,
            "status": report_data.status,
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "PAT",
                            "display": "Pathology",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": report_data.code,
                        "display": "Histopathology report",
                    }
                ]
            },
            "subject": {"reference": f"Patient/{report_data.patient_id}"},
            "performer": [{"reference": f"Practitioner/{report_data.practitioner_id}"}],
            "specimen": [{"reference": f"Specimen/{report_data.specimen_id}"}],
            "conclusion": report_data.conclusion,
            "effectiveDateTime": report_data.effective_datetime or current_time,
            "issued": report_data.issued_datetime or current_time,
        }

        # Add AI-specific extensions
        fhir_report["extension"] = [
            {
                "url": "http://histocore.ai/fhir/StructureDefinition/ai-generated",
                "valueBoolean": report_data.ai_generated,
            },
            {
                "url": "http://histocore.ai/fhir/StructureDefinition/confidence-score",
                "valueDecimal": report_data.confidence_score,
            },
        ]

        try:
            result = self._make_fhir_request("POST", "DiagnosticReport", fhir_report)
            logger.info("Created FHIR DiagnosticReport: %s", report_data.report_id)
            return result
        except Exception as e:
            logger.error("Failed to create diagnostic report %s: %s", report_data.report_id, e)
            raise

    def create_observation(self, observation_data: ObservationData) -> Dict:
        """Create FHIR Observation resource for specific findings."""
        current_time = datetime.now(timezone.utc).isoformat()

        fhir_observation = {
            "resourceType": "Observation",
            "id": observation_data.observation_id,
            "status": observation_data.status,
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "laboratory",
                            "display": "Laboratory",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": observation_data.code,
                        "display": "Pathology finding",
                    }
                ]
            },
            "subject": {"reference": f"Patient/{observation_data.patient_id}"},
            "effectiveDateTime": observation_data.effective_datetime or current_time,
        }

        # Add value based on type
        if isinstance(observation_data.value, str):
            fhir_observation["valueString"] = observation_data.value
        elif isinstance(observation_data.value, bool):
            fhir_observation["valueBoolean"] = observation_data.value
        elif isinstance(observation_data.value, (int, float)):
            value_quantity = {"value": observation_data.value}
            if observation_data.unit:
                value_quantity["unit"] = observation_data.unit
            fhir_observation["valueQuantity"] = value_quantity

        # Link to diagnostic report
        if observation_data.diagnostic_report_id:
            fhir_observation["derivedFrom"] = [
                {"reference": f"DiagnosticReport/{observation_data.diagnostic_report_id}"}
            ]

        try:
            result = self._make_fhir_request("POST", "Observation", fhir_observation)
            logger.info("Created FHIR Observation: %s", observation_data.observation_id)
            return result
        except Exception as e:
            logger.error("Failed to create observation %s: %s", observation_data.observation_id, e)
            raise

    def get_patient(self, patient_id: str) -> Optional[Dict]:
        """Retrieve FHIR Patient resource."""
        try:
            result = self._make_fhir_request("GET", f"Patient/{patient_id}")
            logger.debug("Retrieved FHIR Patient: %s", patient_id)
            return result
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning("Patient not found: %s", patient_id)
                return None
            logger.error("Failed to get patient %s: %s", patient_id, e)
            raise
        except Exception as e:
            logger.error("Failed to get patient %s: %s", patient_id, e)
            raise

    def search_patients(self, **search_params) -> List[Dict]:
        """Search for FHIR Patient resources."""
        try:
            result = self._make_fhir_request("GET", "Patient", params=search_params)

            if "entry" in result:
                patients = [entry["resource"] for entry in result["entry"]]
                logger.debug("Found %d patients matching search criteria", len(patients))
                return patients

            logger.debug("No patients found matching search criteria")
            return []

        except Exception as e:
            logger.error("Failed to search patients: %s", e)
            raise

    def get_diagnostic_reports(self, patient_id: str) -> List[Dict]:
        """Get all diagnostic reports for a patient."""
        try:
            search_params = {"subject": f"Patient/{patient_id}"}
            result = self._make_fhir_request("GET", "DiagnosticReport", params=search_params)

            if "entry" in result:
                reports = [entry["resource"] for entry in result["entry"]]
                logger.debug("Found %d diagnostic reports for patient %s", len(reports), patient_id)
                return reports

            logger.debug("No diagnostic reports found for patient %s", patient_id)
            return []

        except Exception as e:
            logger.error("Failed to get diagnostic reports for patient %s: %s", patient_id, e)
            raise

    def update_diagnostic_report_status(self, report_id: str, status: str) -> Dict:
        """Update diagnostic report status."""
        try:
            # First get the existing report
            existing_report = self._make_fhir_request("GET", f"DiagnosticReport/{report_id}")

            # Update status
            existing_report["status"] = status

            # Update the resource
            result = self._make_fhir_request(
                "PUT", f"DiagnosticReport/{report_id}", existing_report
            )
            logger.info("Updated diagnostic report %s status to %s", report_id, status)
            return result

        except Exception as e:
            logger.error("Failed to update diagnostic report %s status: %s", report_id, e)
            raise

    def create_imaging_study(self, study_data: Dict) -> Dict:
        """Create FHIR ImagingStudy resource for WSI data."""
        fhir_study = {
            "resourceType": "ImagingStudy",
            "id": study_data["study_id"],
            "status": "available",
            "subject": {"reference": f"Patient/{study_data['patient_id']}"},
            "started": study_data.get("started", datetime.now(timezone.utc).isoformat()),
            "numberOfSeries": study_data.get("number_of_series", 1),
            "numberOfInstances": study_data.get("number_of_instances", 1),
            "modality": [
                {
                    "system": "http://dicom.nema.org/resources/ontology/DCM",
                    "code": "SM",
                    "display": "Slide Microscopy",
                }
            ],
        }

        if "series" in study_data:
            fhir_study["series"] = study_data["series"]

        try:
            result = self._make_fhir_request("POST", "ImagingStudy", fhir_study)
            logger.info("Created FHIR ImagingStudy: %s", study_data["study_id"])
            return result
        except Exception as e:
            logger.error("Failed to create imaging study %s: %s", study_data["study_id"], e)
            raise

    def validate_fhir_resource(self, resource: Dict) -> bool:
        """Validate FHIR resource against profile."""
        try:
            resource_type = resource.get("resourceType")
            if not resource_type:
                logger.error("Resource missing resourceType")
                return False

            # Use FHIR validation endpoint
            result = self._make_fhir_request("POST", f"{resource_type}/$validate", resource)

            # Check for validation issues
            if "issue" in result:
                issues = result["issue"]
                errors = [issue for issue in issues if issue.get("severity") == "error"]
                if errors:
                    logger.error("FHIR validation errors: %s", errors)
                    return False

            logger.debug("FHIR resource validation passed")
            return True

        except Exception as e:
            logger.error("FHIR validation failed: %s", e)
            return False

    def get_capability_statement(self) -> Dict:
        """Get FHIR server capability statement."""
        try:
            result = self._make_fhir_request("GET", "metadata")
            logger.debug("Retrieved FHIR capability statement")
            return result
        except Exception as e:
            logger.error("Failed to get capability statement: %s", e)
            raise

    def health_check(self) -> bool:
        """Check Azure Health Data Services connectivity and authentication."""
        try:
            # Try to get capability statement as health check
            self.get_capability_statement()
            logger.info("Azure Health Data Services health check passed")
            return True
        except Exception as e:
            logger.error("Azure Health Data Services health check failed: %s", e)
            return False


# Factory function for easy initialization
def create_health_data_services(
    workspace_url: str, fhir_service_name: str, tenant_id: str, **kwargs
) -> AzureHealthDataServices:
    """Create Azure Health Data Services instance with configuration."""
    config = HealthDataConfig(
        workspace_url=workspace_url,
        fhir_service_name=fhir_service_name,
        tenant_id=tenant_id,
        **kwargs,
    )
    return AzureHealthDataServices(config)
