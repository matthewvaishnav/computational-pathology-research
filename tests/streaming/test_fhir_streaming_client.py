"""Tests for FHIR streaming client."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.clinical.fhir_adapter import DiagnosticReportData, PatientClinicalMetadata
from src.streaming.fhir_streaming_client import FHIRStreamingClient, StreamingDiagnosticReport
from src.streaming.pacs_wsi_client import AnalysisResult


@pytest.fixture
def mock_fhir_adapter():
    """Mock FHIR adapter."""
    with patch("src.streaming.fhir_streaming_client.FHIRAdapter") as mock:
        adapter = Mock()
        adapter.config = Mock()
        adapter.config.base_url = "https://fhir.example.com"
        adapter.config.auth_method = Mock(value="none")
        adapter.config.timeout = 30
        mock.return_value = adapter
        yield adapter


class TestFHIRStreamingClient:
    """Tests for FHIRStreamingClient."""

    def test_init(self, mock_fhir_adapter):
        """Test client initialization."""
        client = FHIRStreamingClient(
            fhir_base_url="https://fhir.example.com",
            auth_method="oauth2",
            client_id="test_client",
            client_secret="test_secret",
        )

        assert client.fhir_adapter is not None

    def test_get_patient_metadata_success(self, mock_fhir_adapter):
        """Test successful patient metadata retrieval."""
        metadata = PatientClinicalMetadata(
            patient_id="PAT001",
            age=45,
            sex="M",
            smoking_status="never",
            medications=["aspirin", "metformin"],
        )

        mock_fhir_adapter.get_patient_clinical_metadata.return_value = metadata

        client = FHIRStreamingClient(fhir_base_url="https://fhir.example.com")

        result = client.get_patient_metadata("PAT001")

        assert result is not None
        assert result.patient_id == "PAT001"
        assert result.age == 45
        assert len(result.medications) == 2

    def test_get_patient_metadata_not_found(self, mock_fhir_adapter):
        """Test patient metadata not found."""
        mock_fhir_adapter.get_patient_clinical_metadata.return_value = None

        client = FHIRStreamingClient(fhir_base_url="https://fhir.example.com")

        result = client.get_patient_metadata("PAT999")

        assert result is None

    def test_get_patient_metadata_error(self, mock_fhir_adapter):
        """Test patient metadata retrieval error."""
        mock_fhir_adapter.get_patient_clinical_metadata.side_effect = Exception("Connection error")

        client = FHIRStreamingClient(fhir_base_url="https://fhir.example.com")

        result = client.get_patient_metadata("PAT001")

        assert result is None

    def test_get_study_metadata_success(self, mock_fhir_adapter):
        """Test successful study metadata retrieval."""
        study_data = {
            "resourceType": "ImagingStudy",
            "id": "study123",
            "status": "available",
            "subject": {"reference": "Patient/PAT001"},
        }

        mock_fhir_adapter.get_imaging_study.return_value = study_data

        client = FHIRStreamingClient(fhir_base_url="https://fhir.example.com")

        result = client.get_study_metadata("PAT001", "1.2.3.4")

        assert result is not None
        assert result["resourceType"] == "ImagingStudy"
        assert result["id"] == "study123"

    def test_create_diagnostic_report_success(self, mock_fhir_adapter):
        """Test successful diagnostic report creation."""
        mock_fhir_adapter.create_diagnostic_report.return_value = "report123"

        client = FHIRStreamingClient(fhir_base_url="https://fhir.example.com")

        report = StreamingDiagnosticReport(
            patient_id="PAT001",
            study_uid="1.2.3.4",
            series_uid="1.2.3.4.1",
            prediction="Positive",
            confidence=0.95,
            processing_time=25.3,
            timestamp=datetime(2026, 4, 15, 10, 30),
        )

        resource_id = client.create_diagnostic_report(report)

        assert resource_id == "report123"
        mock_fhir_adapter.create_diagnostic_report.assert_called_once()

    def test_create_diagnostic_report_failure(self, mock_fhir_adapter):
        """Test diagnostic report creation failure."""
        mock_fhir_adapter.create_diagnostic_report.return_value = None

        client = FHIRStreamingClient(fhir_base_url="https://fhir.example.com")

        report = StreamingDiagnosticReport(
            patient_id="PAT001",
            study_uid="1.2.3.4",
            series_uid="1.2.3.4.1",
            prediction="Negative",
            confidence=0.88,
            processing_time=22.1,
            timestamp=datetime(2026, 4, 15, 10, 30),
        )

        resource_id = client.create_diagnostic_report(report)

        assert resource_id is None

    def test_update_diagnostic_report_success(self, mock_fhir_adapter):
        """Test successful diagnostic report update."""
        mock_fhir_adapter.update_diagnostic_report.return_value = True

        client = FHIRStreamingClient(fhir_base_url="https://fhir.example.com")

        report = StreamingDiagnosticReport(
            patient_id="PAT001",
            study_uid="1.2.3.4",
            series_uid="1.2.3.4.1",
            prediction="Positive",
            confidence=0.97,
            processing_time=24.5,
            timestamp=datetime(2026, 4, 15, 10, 35),
        )

        success = client.update_diagnostic_report("report123", report)

        assert success is True

    def test_convert_analysis_result_to_report(self, mock_fhir_adapter):
        """Test convert AnalysisResult to StreamingDiagnosticReport."""
        client = FHIRStreamingClient(fhir_base_url="https://fhir.example.com")

        analysis_result = AnalysisResult(
            study_uid="1.2.3.4",
            series_uid="1.2.3.4.1",
            patient_id="PAT001",
            confidence=0.92,
            prediction="Negative",
            processing_time=23.7,
            timestamp=1745577000.0,  # Unix timestamp
            attention_weights={"region1": 0.8, "region2": 0.2},
        )

        report = client.convert_analysis_result_to_report(analysis_result, patient_id="PAT001")

        assert report.patient_id == "PAT001"
        assert report.study_uid == "1.2.3.4"
        assert report.prediction == "Negative"
        assert report.confidence == 0.92
        assert report.attention_weights is not None

    def test_validate_patient_match_success(self, mock_fhir_adapter):
        """Test patient ID validation success."""
        client = FHIRStreamingClient(fhir_base_url="https://fhir.example.com")

        match = client.validate_patient_match("PAT001", "PAT001")

        assert match is True

    def test_validate_patient_match_failure(self, mock_fhir_adapter):
        """Test patient ID validation failure."""
        client = FHIRStreamingClient(fhir_base_url="https://fhir.example.com")

        match = client.validate_patient_match("PAT001", "PAT002")

        assert match is False

    def test_get_statistics(self, mock_fhir_adapter):
        """Test get statistics."""
        client = FHIRStreamingClient(fhir_base_url="https://fhir.example.com")

        stats = client.get_statistics()

        assert "fhir_base_url" in stats
        assert "auth_method" in stats
        assert "timeout" in stats


class TestStreamingDiagnosticReport:
    """Tests for StreamingDiagnosticReport."""

    def test_report_creation(self):
        """Test report creation."""
        report = StreamingDiagnosticReport(
            patient_id="PAT001",
            study_uid="1.2.3.4",
            series_uid="1.2.3.4.1",
            prediction="Positive",
            confidence=0.95,
            processing_time=25.3,
            timestamp=datetime(2026, 4, 15, 10, 30),
        )

        assert report.patient_id == "PAT001"
        assert report.prediction == "Positive"
        assert report.confidence == 0.95

    def test_to_fhir_diagnostic_report(self):
        """Test convert to FHIR DiagnosticReport."""
        report = StreamingDiagnosticReport(
            patient_id="PAT001",
            study_uid="1.2.3.4",
            series_uid="1.2.3.4.1",
            prediction="Negative",
            confidence=0.88,
            processing_time=22.1,
            timestamp=datetime(2026, 4, 15, 10, 30),
            model_version="2.0.0",
        )

        fhir_report = report.to_fhir_diagnostic_report()

        assert isinstance(fhir_report, DiagnosticReportData)
        assert fhir_report.patient_id == "PAT001"
        assert fhir_report.imaging_study_id == "1.2.3.4"
        assert fhir_report.status == "final"
        assert fhir_report.primary_diagnosis == "Negative"
        assert fhir_report.confidence_score == 0.88
        assert fhir_report.model_version == "2.0.0"

    def test_report_with_attention_weights(self):
        """Test report with attention weights."""
        report = StreamingDiagnosticReport(
            patient_id="PAT001",
            study_uid="1.2.3.4",
            series_uid="1.2.3.4.1",
            prediction="Positive",
            confidence=0.92,
            processing_time=24.5,
            timestamp=datetime(2026, 4, 15, 10, 30),
            attention_weights={"region1": 0.7, "region2": 0.3},
        )

        assert report.attention_weights is not None
        assert len(report.attention_weights) == 2
        assert report.attention_weights["region1"] == 0.7
