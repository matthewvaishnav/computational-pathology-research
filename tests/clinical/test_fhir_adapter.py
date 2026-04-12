"""
Integration tests for FHIR adapter.

Tests FHIR resource reading, DiagnosticReport generation, resource linking,
and validation.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
"""

import json
import unittest
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

from src.clinical.fhir_adapter import (
    AuthenticationMethod,
    DiagnosticReportData,
    FHIRAdapter,
    FHIRResourceType,
    FHIRServerConfig,
    PatientClinicalMetadata,
)


class TestFHIRAdapter(unittest.TestCase):
    """Test FHIR adapter functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = FHIRServerConfig(
            base_url="https://fhir.example.com",
            auth_method=AuthenticationMethod.NONE,
        )
        self.adapter = FHIRAdapter(self.config)

    @patch("src.clinical.fhir_adapter.requests.Session")
    def test_read_patient_metadata(self, mock_session_class):
        """
        Test reading patient clinical metadata from FHIR resources.

        Requirement: 7.1
        """
        # Mock session
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Mock Patient resource
        patient_response = Mock()
        patient_response.json.return_value = {
            "resourceType": "Patient",
            "id": "patient123",
            "birthDate": "1980-01-15",
            "gender": "male",
        }
        patient_response.raise_for_status = Mock()

        # Mock Observation bundle
        obs_response = Mock()
        obs_response.json.return_value = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {
                            "coding": [
                                {
                                    "system": "http://loinc.org",
                                    "code": "72166-2",
                                    "display": "Tobacco smoking status",
                                }
                            ]
                        },
                        "valueCodeableConcept": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "8517006",
                                    "display": "Former smoker",
                                }
                            ]
                        },
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {
                            "coding": [
                                {
                                    "system": "http://loinc.org",
                                    "code": "8867-4",
                                    "display": "Heart rate",
                                }
                            ]
                        },
                        "valueQuantity": {"value": 72, "unit": "beats/minute"},
                    }
                },
            ],
        }
        obs_response.raise_for_status = Mock()

        # Mock Condition bundle
        condition_response = Mock()
        condition_response.json.return_value = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "code": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "38341003",
                                    "display": "Hypertension",
                                }
                            ]
                        },
                    }
                }
            ],
        }
        condition_response.raise_for_status = Mock()

        # Mock MedicationStatement bundle
        med_response = Mock()
        med_response.json.return_value = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "MedicationStatement",
                        "medicationCodeableConcept": {
                            "coding": [
                                {
                                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                    "code": "197361",
                                    "display": "Lisinopril",
                                }
                            ]
                        },
                    }
                }
            ],
        }
        med_response.raise_for_status = Mock()

        # Configure mock session to return different responses
        mock_session.get.side_effect = [
            patient_response,
            obs_response,
            condition_response,
            med_response,
        ]

        # Create adapter with mocked session
        adapter = FHIRAdapter(self.config)
        adapter.session = mock_session

        # Read patient metadata
        metadata = adapter.read_patient_metadata("patient123")

        # Verify results
        self.assertEqual(metadata.patient_id, "patient123")
        self.assertIsNotNone(metadata.age)
        self.assertEqual(metadata.sex, "male")
        self.assertEqual(metadata.smoking_status, "Former smoker")
        self.assertIn("Heart rate", metadata.observations)
        self.assertEqual(metadata.observations["Heart rate"], 72)
        self.assertIn("Hypertension", metadata.conditions)
        self.assertIn("Lisinopril", metadata.medications)

    @patch("src.clinical.fhir_adapter.requests.Session")
    def test_write_diagnostic_report(self, mock_session_class):
        """
        Test writing prediction results as FHIR DiagnosticReport.

        Requirement: 7.2
        """
        # Mock session
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Mock POST response
        post_response = Mock()
        post_response.json.return_value = {
            "resourceType": "DiagnosticReport",
            "id": "report123",
            "status": "final",
        }
        post_response.raise_for_status = Mock()
        mock_session.post.return_value = post_response

        # Create adapter with mocked session
        adapter = FHIRAdapter(self.config)
        adapter.session = mock_session

        # Create diagnostic report data
        report_data = DiagnosticReportData(
            patient_id="patient123",
            imaging_study_id="study456",
            status="final",
            conclusion="Malignant tumor detected with high confidence",
            probability_distribution={
                "benign": 0.15,
                "malignant": 0.75,
                "uncertain": 0.10,
            },
            primary_diagnosis="malignant",
            confidence_score=0.75,
            uncertainty_explanation="High confidence prediction based on clear tumor markers",
            risk_scores={"1-year": 0.8, "5-year": 0.6},
            model_version="v1.2.3",
        )

        # Write diagnostic report
        result = adapter.write_diagnostic_report(report_data)

        # Verify POST was called
        self.assertTrue(mock_session.post.called)
        call_args = mock_session.post.call_args

        # Verify URL
        self.assertIn("DiagnosticReport", call_args[0][0])

        # Verify request body
        request_body = call_args[1]["json"]
        self.assertEqual(request_body["resourceType"], "DiagnosticReport")
        self.assertEqual(request_body["status"], "final")
        self.assertEqual(request_body["subject"]["reference"], "Patient/patient123")
        self.assertIn("ImagingStudy/study456", str(request_body["imagingStudy"]))

        # Verify extensions contain prediction data
        extensions = request_body.get("extension", [])
        self.assertTrue(len(extensions) > 0)

        # Check for specific extensions
        extension_urls = [ext["url"] for ext in extensions]
        self.assertIn(
            "http://example.org/fhir/StructureDefinition/primary-diagnosis",
            extension_urls,
        )
        self.assertIn(
            "http://example.org/fhir/StructureDefinition/confidence-score",
            extension_urls,
        )
        self.assertIn("http://example.org/fhir/StructureDefinition/model-version", extension_urls)

        # Verify result
        self.assertEqual(result["id"], "report123")

    @patch("src.clinical.fhir_adapter.requests.Session")
    def test_query_patient_history(self, mock_session_class):
        """
        Test querying FHIR server for patient historical data.

        Requirement: 7.3
        """
        # Mock session
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Mock search response
        search_response = Mock()
        search_response.json.return_value = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "DiagnosticReport",
                        "id": "report1",
                        "status": "final",
                        "issued": "2023-01-15T10:30:00Z",
                    }
                },
                {
                    "resource": {
                        "resourceType": "DiagnosticReport",
                        "id": "report2",
                        "status": "final",
                        "issued": "2023-06-20T14:15:00Z",
                    }
                },
            ],
        }
        search_response.raise_for_status = Mock()
        mock_session.get.return_value = search_response

        # Create adapter with mocked session
        adapter = FHIRAdapter(self.config)
        adapter.session = mock_session

        # Query patient history
        history = adapter.query_patient_history("patient123", FHIRResourceType.DIAGNOSTIC_REPORT)

        # Verify results
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["id"], "report1")
        self.assertEqual(history[1]["id"], "report2")

        # Verify search parameters
        call_args = mock_session.get.call_args
        self.assertIn("patient", call_args[1]["params"])
        self.assertEqual(call_args[1]["params"]["patient"], "patient123")

    def test_validate_resource_conformance_patient(self):
        """
        Test FHIR resource validation for Patient resources.

        Requirement: 7.5
        """
        # Valid patient resource
        valid_patient = {
            "resourceType": "Patient",
            "id": "patient123",
            "identifier": [{"system": "http://hospital.org", "value": "12345"}],
            "name": [{"family": "Smith", "given": ["John"]}],
        }

        self.assertTrue(self.adapter.validate_resource_conformance(valid_patient))

        # Invalid patient (missing identifier and name)
        invalid_patient = {"resourceType": "Patient", "id": "patient123"}

        self.assertFalse(self.adapter.validate_resource_conformance(invalid_patient))

    def test_validate_resource_conformance_diagnostic_report(self):
        """
        Test FHIR resource validation for DiagnosticReport resources.

        Requirement: 7.5
        """
        # Valid diagnostic report
        valid_report = {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "60570-9",
                        "display": "Pathology report",
                    }
                ]
            },
            "subject": {"reference": "Patient/patient123"},
        }

        self.assertTrue(self.adapter.validate_resource_conformance(valid_report))

        # Invalid report (missing required fields)
        invalid_report = {"resourceType": "DiagnosticReport", "status": "final"}

        self.assertFalse(self.adapter.validate_resource_conformance(invalid_report))

    def test_validate_resource_conformance_observation(self):
        """
        Test FHIR resource validation for Observation resources.

        Requirement: 7.5
        """
        # Valid observation
        valid_obs = {
            "resourceType": "Observation",
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "8867-4",
                        "display": "Heart rate",
                    }
                ]
            },
        }

        self.assertTrue(self.adapter.validate_resource_conformance(valid_obs))

        # Invalid observation (missing code)
        invalid_obs = {"resourceType": "Observation", "status": "final"}

        self.assertFalse(self.adapter.validate_resource_conformance(invalid_obs))

    @patch("src.clinical.fhir_adapter.requests.Session")
    def test_subscribe_to_imaging_studies(self, mock_session_class):
        """
        Test FHIR subscription for real-time notifications.

        Requirement: 7.7
        """
        # Mock session
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Mock POST response
        post_response = Mock()
        post_response.json.return_value = {
            "resourceType": "Subscription",
            "id": "sub123",
            "status": "active",
        }
        post_response.raise_for_status = Mock()
        mock_session.post.return_value = post_response

        # Create adapter with mocked session
        adapter = FHIRAdapter(self.config)
        adapter.session = mock_session

        # Create subscription
        result = adapter.subscribe_to_imaging_studies(
            callback_url="https://example.com/callback", patient_id="patient123"
        )

        # Verify POST was called
        self.assertTrue(mock_session.post.called)
        call_args = mock_session.post.call_args

        # Verify URL
        self.assertIn("Subscription", call_args[0][0])

        # Verify request body
        request_body = call_args[1]["json"]
        self.assertEqual(request_body["resourceType"], "Subscription")
        self.assertEqual(request_body["status"], "requested")
        self.assertIn("ImagingStudy", request_body["criteria"])
        self.assertIn("patient123", request_body["criteria"])
        self.assertEqual(request_body["channel"]["endpoint"], "https://example.com/callback")

        # Verify result
        self.assertEqual(result["id"], "sub123")

    @patch("src.clinical.fhir_adapter.requests.Session")
    @patch("src.clinical.fhir_adapter.requests.post")
    def test_oauth2_authentication(self, mock_post, mock_session_class):
        """
        Test OAuth 2.0 authentication setup.

        Requirement: 7.4
        """
        # Mock token response
        token_response = Mock()
        token_response.json.return_value = {"access_token": "test_token_12345"}
        token_response.raise_for_status = Mock()
        mock_post.return_value = token_response

        # Mock session with headers attribute
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session_class.return_value = mock_session

        # Create config with OAuth2
        config = FHIRServerConfig(
            base_url="https://fhir.example.com",
            auth_method=AuthenticationMethod.OAUTH2,
            client_id="test_client",
            client_secret="test_secret",
            token_url="https://auth.example.com/token",
        )

        # Create adapter (should trigger authentication)
        adapter = FHIRAdapter(config)

        # Verify token request was made
        self.assertTrue(mock_post.called)
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], "https://auth.example.com/token")
        self.assertEqual(call_args[1]["data"]["grant_type"], "client_credentials")
        self.assertEqual(call_args[1]["data"]["client_id"], "test_client")

        # Verify authorization header was set
        self.assertIn("Authorization", adapter.session.headers)
        self.assertEqual(adapter.session.headers["Authorization"], "Bearer test_token_12345")

    @patch("src.clinical.fhir_adapter.requests.Session")
    def test_smart_on_fhir_authentication(self, mock_session_class):
        """
        Test SMART on FHIR authentication setup.

        Requirement: 7.4
        """
        # Mock session with headers attribute
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session_class.return_value = mock_session

        # Create config with SMART on FHIR
        config = FHIRServerConfig(
            base_url="https://fhir.example.com",
            auth_method=AuthenticationMethod.SMART_ON_FHIR,
            access_token="smart_token_67890",
        )

        # Create adapter
        adapter = FHIRAdapter(config)

        # Verify authorization header was set
        self.assertIn("Authorization", adapter.session.headers)
        self.assertEqual(adapter.session.headers["Authorization"], "Bearer smart_token_67890")

    @patch("src.clinical.fhir_adapter.requests.Session")
    def test_resource_linking(self, mock_session_class):
        """
        Test linking DiagnosticReport to Patient and ImagingStudy resources.

        Requirement: 7.6
        """
        # Mock session
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Mock POST response
        post_response = Mock()
        post_response.json.return_value = {
            "resourceType": "DiagnosticReport",
            "id": "report123",
        }
        post_response.raise_for_status = Mock()
        mock_session.post.return_value = post_response

        # Create adapter with mocked session
        adapter = FHIRAdapter(self.config)
        adapter.session = mock_session

        # Create diagnostic report with links
        report_data = DiagnosticReportData(
            patient_id="patient123",
            imaging_study_id="study456",
            primary_diagnosis="malignant",
        )

        # Write diagnostic report
        adapter.write_diagnostic_report(report_data)

        # Verify request body contains proper references
        call_args = mock_session.post.call_args
        request_body = call_args[1]["json"]

        # Check Patient reference
        self.assertEqual(request_body["subject"]["reference"], "Patient/patient123")

        # Check ImagingStudy reference
        self.assertIn("imagingStudy", request_body)
        self.assertEqual(request_body["imagingStudy"][0]["reference"], "ImagingStudy/study456")

    def test_patient_clinical_metadata_dataclass(self):
        """Test PatientClinicalMetadata dataclass structure."""
        metadata = PatientClinicalMetadata(
            patient_id="patient123",
            age=45,
            sex="female",
            smoking_status="never",
            alcohol_consumption="moderate",
            medications=["Aspirin", "Metformin"],
            conditions=["Type 2 Diabetes", "Hypertension"],
            observations={"BMI": 28.5, "Blood Pressure": "130/85"},
            family_history=["Breast Cancer", "Heart Disease"],
        )

        self.assertEqual(metadata.patient_id, "patient123")
        self.assertEqual(metadata.age, 45)
        self.assertEqual(metadata.sex, "female")
        self.assertEqual(len(metadata.medications), 2)
        self.assertEqual(len(metadata.conditions), 2)
        self.assertIn("BMI", metadata.observations)

    def test_diagnostic_report_data_dataclass(self):
        """Test DiagnosticReportData dataclass structure."""
        report_data = DiagnosticReportData(
            patient_id="patient123",
            imaging_study_id="study456",
            status="final",
            primary_diagnosis="benign",
            confidence_score=0.92,
            probability_distribution={"benign": 0.92, "malignant": 0.08},
            model_version="v2.0.0",
        )

        self.assertEqual(report_data.patient_id, "patient123")
        self.assertEqual(report_data.imaging_study_id, "study456")
        self.assertEqual(report_data.status, "final")
        self.assertEqual(report_data.confidence_score, 0.92)
        self.assertEqual(report_data.probability_distribution["benign"], 0.92)


class TestFHIRResourceExtraction(unittest.TestCase):
    """Test FHIR resource data extraction methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = FHIRServerConfig(
            base_url="https://fhir.example.com",
            auth_method=AuthenticationMethod.NONE,
        )
        self.adapter = FHIRAdapter(self.config)

    def test_extract_age(self):
        """Test age extraction from Patient resource."""
        patient = {"birthDate": "1980-05-15"}
        age = self.adapter._extract_age(patient)
        self.assertIsNotNone(age)
        self.assertGreater(age, 40)  # Should be over 40 years old

    def test_extract_sex(self):
        """Test sex extraction from Patient resource."""
        patient = {"gender": "female"}
        sex = self.adapter._extract_sex(patient)
        self.assertEqual(sex, "female")

    def test_extract_observations(self):
        """Test observation extraction from Observation resources."""
        observations = [
            {
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "8867-4",
                            "display": "Heart rate",
                        }
                    ]
                },
                "valueQuantity": {"value": 75, "unit": "beats/minute"},
            },
            {
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "39156-5",
                            "display": "BMI",
                        }
                    ]
                },
                "valueQuantity": {"value": 24.5, "unit": "kg/m2"},
            },
        ]

        obs_dict = self.adapter._extract_observations(observations)
        self.assertIn("Heart rate", obs_dict)
        self.assertEqual(obs_dict["Heart rate"], 75)
        self.assertIn("BMI", obs_dict)
        self.assertEqual(obs_dict["BMI"], 24.5)

    def test_extract_smoking_status(self):
        """Test smoking status extraction from Observation resources."""
        observations = [
            {
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "72166-2",
                            "display": "Tobacco smoking status",
                        }
                    ]
                },
                "valueCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "266919005",
                            "display": "Never smoker",
                        }
                    ]
                },
            }
        ]

        smoking_status = self.adapter._extract_smoking_status(observations)
        self.assertEqual(smoking_status, "Never smoker")

    def test_extract_conditions(self):
        """Test condition extraction from Condition resources."""
        conditions = [
            {
                "code": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "44054006",
                            "display": "Type 2 Diabetes",
                        }
                    ]
                }
            },
            {
                "code": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "38341003",
                            "display": "Hypertension",
                        }
                    ]
                }
            },
        ]

        condition_list = self.adapter._extract_conditions(conditions)
        self.assertEqual(len(condition_list), 2)
        self.assertIn("Type 2 Diabetes", condition_list)
        self.assertIn("Hypertension", condition_list)

    def test_extract_medications(self):
        """Test medication extraction from MedicationStatement resources."""
        medications = [
            {
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "860975",
                            "display": "Metformin",
                        }
                    ]
                }
            },
            {"medicationReference": {"display": "Aspirin 81mg"}},
        ]

        med_list = self.adapter._extract_medications(medications)
        self.assertEqual(len(med_list), 2)
        self.assertIn("Metformin", med_list)
        self.assertIn("Aspirin 81mg", med_list)


if __name__ == "__main__":
    unittest.main()
