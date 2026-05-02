"""
HL7 FHIR integration adapter for clinical workflow.

This module provides FHIR R4 standard integration for reading patient clinical
metadata and writing prediction results as FHIR DiagnosticReport resources.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import requests


class FHIRResourceType(Enum):
    """FHIR resource types supported by the adapter."""

    PATIENT = "Patient"
    OBSERVATION = "Observation"
    CONDITION = "Condition"
    MEDICATION_STATEMENT = "MedicationStatement"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    IMAGING_STUDY = "ImagingStudy"


class AuthenticationMethod(Enum):
    """FHIR authentication methods."""

    OAUTH2 = "oauth2"
    SMART_ON_FHIR = "smart_on_fhir"
    BASIC = "basic"
    NONE = "none"


@dataclass
class FHIRServerConfig:
    """Configuration for FHIR server connection."""

    base_url: str
    auth_method: AuthenticationMethod = AuthenticationMethod.NONE
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    token_url: Optional[str] = None
    access_token: Optional[str] = None
    timeout: int = 30
    verify_ssl: bool = True


@dataclass
class PatientClinicalMetadata:
    """Patient clinical metadata extracted from FHIR resources."""

    patient_id: str
    age: Optional[int] = None
    sex: Optional[str] = None
    smoking_status: Optional[str] = None
    alcohol_consumption: Optional[str] = None
    medications: Optional[List[str]] = None
    conditions: Optional[List[str]] = None
    observations: Optional[Dict[str, Any]] = None
    family_history: Optional[List[str]] = None


@dataclass
class DiagnosticReportData:
    """Data for creating FHIR DiagnosticReport resources."""

    patient_id: str
    imaging_study_id: Optional[str] = None
    issued_datetime: Optional[datetime] = None
    status: str = "final"
    code: Optional[Dict[str, Any]] = None
    conclusion: Optional[str] = None
    probability_distribution: Optional[Dict[str, float]] = None
    primary_diagnosis: Optional[str] = None
    confidence_score: Optional[float] = None
    uncertainty_explanation: Optional[str] = None
    risk_scores: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None


class FHIRAdapter:
    """
    HL7 FHIR R4 adapter for clinical workflow integration.

    This adapter provides:
    - Reading patient clinical metadata from FHIR resources
    - Writing prediction results as FHIR DiagnosticReport resources
    - Querying FHIR servers for patient historical data
    - FHIR authentication and validation support

    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
    """

    def __init__(self, config: FHIRServerConfig):
        """
        Initialize FHIR adapter with server configuration.

        Args:
            config: FHIR server configuration including URL and authentication
        """
        self.config = config
        self.session = requests.Session()
        self.session.verify = config.verify_ssl

        # Set up authentication
        if config.auth_method != AuthenticationMethod.NONE:
            self._setup_authentication()

    def _setup_authentication(self) -> None:
        """
        Set up FHIR authentication based on configured method.

        Supports OAuth 2.0, SMART on FHIR, and basic authentication.
        Requirement: 7.4
        """
        if self.config.auth_method == AuthenticationMethod.OAUTH2:
            if self.config.access_token:
                self.session.headers.update({"Authorization": f"Bearer {self.config.access_token}"})
            elif self.config.client_id and self.config.client_secret and self.config.token_url:
                # Request access token
                token_response = requests.post(
                    self.config.token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self.config.client_id,
                        "client_secret": self.config.client_secret,
                    },
                    timeout=self.config.timeout,
                , timeout=30)
                token_response.raise_for_status()
                access_token = token_response.json().get("access_token")
                self.session.headers.update({"Authorization": f"Bearer {access_token}"})

        elif self.config.auth_method == AuthenticationMethod.SMART_ON_FHIR:
            # SMART on FHIR uses OAuth 2.0 with specific scopes
            if self.config.access_token:
                self.session.headers.update({"Authorization": f"Bearer {self.config.access_token}"})

        elif self.config.auth_method == AuthenticationMethod.BASIC:
            if self.config.client_id and self.config.client_secret:
                from requests.auth import HTTPBasicAuth

                self.session.auth = HTTPBasicAuth(self.config.client_id, self.config.client_secret)

    def read_patient_metadata(self, patient_id: str) -> PatientClinicalMetadata:
        """
        Read patient clinical metadata from FHIR resources.

        Queries FHIR server for Patient, Observation, Condition, and
        MedicationStatement resources to build comprehensive patient context.

        Args:
            patient_id: FHIR patient identifier

        Returns:
            PatientClinicalMetadata with extracted information

        Requirements: 7.1, 7.3
        """
        metadata = PatientClinicalMetadata(patient_id=patient_id)

        # Read Patient resource
        patient = self._read_resource(FHIRResourceType.PATIENT, patient_id)
        if patient:
            metadata.age = self._extract_age(patient)
            metadata.sex = self._extract_sex(patient)

        # Read Observations
        observations = self._search_resources(FHIRResourceType.OBSERVATION, {"patient": patient_id})
        if observations:
            metadata.observations = self._extract_observations(observations)
            metadata.smoking_status = self._extract_smoking_status(observations)

        # Read Conditions
        conditions = self._search_resources(FHIRResourceType.CONDITION, {"patient": patient_id})
        if conditions:
            metadata.conditions = self._extract_conditions(conditions)

        # Read MedicationStatements
        medications = self._search_resources(
            FHIRResourceType.MEDICATION_STATEMENT, {"patient": patient_id}
        )
        if medications:
            metadata.medications = self._extract_medications(medications)

        return metadata

    def write_diagnostic_report(self, report_data: DiagnosticReportData) -> Dict[str, Any]:
        """
        Write prediction results as FHIR DiagnosticReport resource.

        Creates a DiagnosticReport resource with prediction results,
        probability distributions, and uncertainty quantification.

        Args:
            report_data: Diagnostic report data including predictions

        Returns:
            Created FHIR DiagnosticReport resource

        Requirements: 7.2, 7.6
        """
        # Build DiagnosticReport resource
        report = {
            "resourceType": "DiagnosticReport",
            "status": report_data.status,
            "subject": {"reference": f"Patient/{report_data.patient_id}"},
            "issued": (report_data.issued_datetime or datetime.now()).isoformat(),
        }

        # Link to ImagingStudy if available
        if report_data.imaging_study_id:
            report["imagingStudy"] = [{"reference": f"ImagingStudy/{report_data.imaging_study_id}"}]

        # Add diagnostic code
        if report_data.code:
            report["code"] = report_data.code
        else:
            report["code"] = {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "60570-9",
                        "display": "Pathology report",
                    }
                ]
            }

        # Add conclusion text
        if report_data.conclusion:
            report["conclusion"] = report_data.conclusion

        # Add structured results as extensions
        if report_data.probability_distribution:
            report.setdefault("extension", []).append(
                {
                    "url": "http://example.org/fhir/StructureDefinition/probability-distribution",
                    "valueString": str(report_data.probability_distribution),
                }
            )

        if report_data.primary_diagnosis:
            report.setdefault("extension", []).append(
                {
                    "url": "http://example.org/fhir/StructureDefinition/primary-diagnosis",
                    "valueString": report_data.primary_diagnosis,
                }
            )

        if report_data.confidence_score is not None:
            report.setdefault("extension", []).append(
                {
                    "url": "http://example.org/fhir/StructureDefinition/confidence-score",
                    "valueDecimal": report_data.confidence_score,
                }
            )

        if report_data.uncertainty_explanation:
            report.setdefault("extension", []).append(
                {
                    "url": "http://example.org/fhir/StructureDefinition/uncertainty-explanation",
                    "valueString": report_data.uncertainty_explanation,
                }
            )

        if report_data.risk_scores:
            report.setdefault("extension", []).append(
                {
                    "url": "http://example.org/fhir/StructureDefinition/risk-scores",
                    "valueString": str(report_data.risk_scores),
                }
            )

        if report_data.model_version:
            report.setdefault("extension", []).append(
                {
                    "url": "http://example.org/fhir/StructureDefinition/model-version",
                    "valueString": report_data.model_version,
                }
            )

        # Create resource on FHIR server
        response = self._create_resource(FHIRResourceType.DIAGNOSTIC_REPORT, report)
        return response

    def query_patient_history(
        self, patient_id: str, resource_type: FHIRResourceType
    ) -> List[Dict[str, Any]]:
        """
        Query FHIR server for patient historical data.

        Args:
            patient_id: FHIR patient identifier
            resource_type: Type of FHIR resource to query

        Returns:
            List of FHIR resources matching the query

        Requirement: 7.3
        """
        return self._search_resources(resource_type, {"patient": patient_id})

    def validate_resource_conformance(
        self, resource: Dict[str, Any], profile_url: Optional[str] = None
    ) -> bool:
        """
        Validate FHIR resource conformance to specified profiles.

        Args:
            resource: FHIR resource to validate
            profile_url: Optional profile URL for validation

        Returns:
            True if resource is valid, False otherwise

        Requirement: 7.5
        """
        # Basic validation: check required fields
        if "resourceType" not in resource:
            return False

        resource_type = resource["resourceType"]

        # Resource-specific validation
        if resource_type == "Patient":
            return self._validate_patient(resource)
        elif resource_type == "DiagnosticReport":
            return self._validate_diagnostic_report(resource)
        elif resource_type == "Observation":
            return self._validate_observation(resource)

        # If profile URL is provided, validate against profile
        if profile_url:
            # In production, this would use a FHIR validator library
            # For now, we perform basic validation
            return True

        return True

    def subscribe_to_imaging_studies(
        self, callback_url: str, patient_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create FHIR subscription for real-time notifications of new imaging studies.

        Args:
            callback_url: URL to receive subscription notifications
            patient_id: Optional patient ID to filter subscriptions

        Returns:
            Created FHIR Subscription resource

        Requirement: 7.7
        """
        subscription = {
            "resourceType": "Subscription",
            "status": "requested",
            "reason": "Monitor new imaging studies for computational pathology",
            "criteria": "ImagingStudy?modality=SM",  # SM = Slide Microscopy
            "channel": {
                "type": "rest-hook",
                "endpoint": callback_url,
                "payload": "application/fhir+json",
            },
        }

        # Add patient filter if specified
        if patient_id:
            subscription["criteria"] += f"&patient={patient_id}"

        # Create subscription on FHIR server
        response = self.session.post(
            f"{self.config.base_url}/Subscription",
            json=subscription,
            headers={"Content-Type": "application/fhir+json"},
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()

    # Private helper methods

    def _read_resource(
        self, resource_type: FHIRResourceType, resource_id: str
    ) -> Optional[Dict[str, Any]]:
        """Read a single FHIR resource by ID."""
        try:
            response = self.session.get(
                f"{self.config.base_url}/{resource_type.value}/{resource_id}",
                headers={"Accept": "application/fhir+json"},
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return None

    def _search_resources(
        self, resource_type: FHIRResourceType, params: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Search for FHIR resources matching parameters."""
        try:
            response = self.session.get(
                f"{self.config.base_url}/{resource_type.value}",
                params=params,
                headers={"Accept": "application/fhir+json"},
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            bundle = response.json()

            # Extract resources from bundle
            if bundle.get("resourceType") == "Bundle":
                entries = bundle.get("entry", [])
                return [entry.get("resource") for entry in entries if "resource" in entry]

            return []
        except requests.exceptions.RequestException:
            return []

    def _create_resource(
        self, resource_type: FHIRResourceType, resource: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new FHIR resource on the server."""
        response = self.session.post(
            f"{self.config.base_url}/{resource_type.value}",
            json=resource,
            headers={"Content-Type": "application/fhir+json"},
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _extract_age(self, patient: Dict[str, Any]) -> Optional[int]:
        """Extract patient age from Patient resource."""
        birth_date = patient.get("birthDate")
        if birth_date:
            from datetime import date

            birth = datetime.strptime(birth_date, "%Y-%m-%d").date()
            today = date.today()
            age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
            return age
        return None

    def _extract_sex(self, patient: Dict[str, Any]) -> Optional[str]:
        """Extract patient sex from Patient resource."""
        return patient.get("gender")

    def _extract_observations(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract observation values from Observation resources."""
        obs_dict = {}
        for obs in observations:
            code = obs.get("code", {})
            coding = code.get("coding", [])
            if coding:
                code_value = coding[0].get("code")
                display = coding[0].get("display", code_value)

                # Extract value
                value = None
                if "valueQuantity" in obs:
                    value = obs["valueQuantity"].get("value")
                elif "valueString" in obs:
                    value = obs["valueString"]
                elif "valueCodeableConcept" in obs:
                    value_coding = obs["valueCodeableConcept"].get("coding", [])
                    if value_coding:
                        value = value_coding[0].get("display")

                if value is not None:
                    obs_dict[display] = value

        return obs_dict

    def _extract_smoking_status(self, observations: List[Dict[str, Any]]) -> Optional[str]:
        """Extract smoking status from Observation resources."""
        for obs in observations:
            code = obs.get("code", {})
            coding = code.get("coding", [])
            for c in coding:
                # LOINC code for smoking status
                if c.get("code") == "72166-2":
                    if "valueCodeableConcept" in obs:
                        value_coding = obs["valueCodeableConcept"].get("coding", [])
                        if value_coding:
                            return value_coding[0].get("display")
        return None

    def _extract_conditions(self, conditions: List[Dict[str, Any]]) -> List[str]:
        """Extract condition descriptions from Condition resources."""
        condition_list = []
        for condition in conditions:
            code = condition.get("code", {})
            coding = code.get("coding", [])
            if coding:
                display = coding[0].get("display")
                if display:
                    condition_list.append(display)
        return condition_list

    def _extract_medications(self, medications: List[Dict[str, Any]]) -> List[str]:
        """Extract medication names from MedicationStatement resources."""
        med_list = []
        for med in medications:
            # Check medicationCodeableConcept
            if "medicationCodeableConcept" in med:
                coding = med["medicationCodeableConcept"].get("coding", [])
                if coding:
                    display = coding[0].get("display")
                    if display:
                        med_list.append(display)
            # Check medicationReference
            elif "medicationReference" in med:
                ref = med["medicationReference"].get("display")
                if ref:
                    med_list.append(ref)
        return med_list

    def _validate_patient(self, resource: Dict[str, Any]) -> bool:
        """Validate Patient resource structure."""
        # Patient must have identifier or name
        return "identifier" in resource or "name" in resource

    def _validate_diagnostic_report(self, resource: Dict[str, Any]) -> bool:
        """Validate DiagnosticReport resource structure."""
        required_fields = ["status", "code", "subject"]
        return all(field in resource for field in required_fields)

    def _validate_observation(self, resource: Dict[str, Any]) -> bool:
        """Validate Observation resource structure."""
        required_fields = ["status", "code"]
        return all(field in resource for field in required_fields)
