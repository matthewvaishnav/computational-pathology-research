"""
Unit tests for clinical reporting system.

Tests report generation, export formats, and longitudinal summaries.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.clinical.reporting import (
    ClinicalReportGenerator,
    DiagnosisResult,
    ExportFormat,
    ReportData,
    ReportSpecialty,
)


@pytest.fixture
def sample_diagnosis_result():
    """Create sample diagnosis result."""
    return DiagnosisResult(
        disease_id="C50.9",
        disease_name="Malignant neoplasm of breast",
        probability=0.85,
        confidence_interval=(0.78, 0.92),
        uncertainty_score=0.12,
    )


@pytest.fixture
def sample_report_data(sample_diagnosis_result):
    """Create sample report data for testing."""
    return ReportData(
        patient_id="PATIENT_12345",
        scan_id="SCAN_67890",
        scan_date=datetime(2024, 1, 15, 10, 30, 0),
        primary_diagnosis=sample_diagnosis_result,
        alternative_diagnoses=[
            DiagnosisResult(
                disease_id="D05.9",
                disease_name="Carcinoma in situ of breast",
                probability=0.10,
                confidence_interval=(0.05, 0.15),
                uncertainty_score=0.18,
            ),
            DiagnosisResult(
                disease_id="N60.1",
                disease_name="Benign breast lesion",
                probability=0.05,
                confidence_interval=(0.02, 0.08),
                uncertainty_score=0.22,
            ),
        ],
        probability_distribution={
            "Malignant neoplasm of breast": 0.85,
            "Carcinoma in situ of breast": 0.10,
            "Benign breast lesion": 0.05,
        },
        uncertainty_explanation="Model confidence is high. No significant data quality issues detected.",
        ood_detected=False,
        ood_explanation=None,
        risk_scores={
            "Malignant neoplasm of breast": 0.82,
            "Carcinoma in situ of breast": 0.15,
        },
        risk_time_horizons=["1-year", "5-year", "10-year"],
        recommendations=[
            "Recommend biopsy for histological confirmation",
            "Consider additional imaging (MRI) for staging",
            "Refer to oncology for treatment planning",
        ],
        attention_heatmap_path=None,
        longitudinal_summary=None,
        previous_scan_comparison=None,
        model_version="1.2.3",
        report_timestamp=datetime(2024, 1, 15, 11, 0, 0),
        physician_notes=None,
        amendments=[],
    )


class TestClinicalReportGenerator:
    """Test suite for ClinicalReportGenerator."""

    def test_initialization(self):
        """Test report generator initialization."""
        generator = ClinicalReportGenerator()
        assert generator.default_specialty == ReportSpecialty.PATHOLOGY
        assert generator._builtin_templates is not None
        assert len(generator._builtin_templates) > 0

    def test_initialization_with_custom_specialty(self):
        """Test initialization with custom default specialty."""
        generator = ClinicalReportGenerator(default_specialty=ReportSpecialty.ONCOLOGY)
        assert generator.default_specialty == ReportSpecialty.ONCOLOGY

    def test_generate_report_basic(self, sample_report_data):
        """Test basic HTML report generation."""
        generator = ClinicalReportGenerator()
        html_report = generator.generate_report(sample_report_data)
        
        assert isinstance(html_report, str)
        assert len(html_report) > 0
        assert "<!DOCTYPE html>" in html_report
        assert sample_report_data.patient_id in html_report
        assert sample_report_data.scan_id in html_report
        assert sample_report_data.primary_diagnosis.disease_name in html_report

    def test_generate_report_with_different_specialties(self, sample_report_data):
        """Test report generation with different specialty templates."""
        generator = ClinicalReportGenerator()
        
        specialties = [
            ReportSpecialty.PATHOLOGY,
            ReportSpecialty.ONCOLOGY,
            ReportSpecialty.CARDIOLOGY,
            ReportSpecialty.RADIOLOGY,
        ]
        
        for specialty in specialties:
            html_report = generator.generate_report(sample_report_data, specialty=specialty)
            assert isinstance(html_report, str)
            assert len(html_report) > 0
            assert specialty.value.title() in html_report or "Pathology" in html_report

    def test_generate_report_includes_primary_diagnosis(self, sample_report_data):
        """Test that report includes primary diagnosis information."""
        generator = ClinicalReportGenerator()
        html_report = generator.generate_report(sample_report_data)
        
        assert sample_report_data.primary_diagnosis.disease_name in html_report
        assert f"{sample_report_data.primary_diagnosis.probability * 100:.1f}" in html_report

    def test_generate_report_includes_alternative_diagnoses(self, sample_report_data):
        """Test that report includes alternative diagnoses."""
        generator = ClinicalReportGenerator()
        html_report = generator.generate_report(sample_report_data)
        
        for alt_diag in sample_report_data.alternative_diagnoses:
            assert alt_diag.disease_name in html_report

    def test_generate_report_includes_probability_distribution(self, sample_report_data):
        """Test that report includes probability distribution table."""
        generator = ClinicalReportGenerator()
        html_report = generator.generate_report(sample_report_data)
        
        assert "Probability Distribution" in html_report
        for disease, prob in sample_report_data.probability_distribution.items():
            assert disease in html_report

    def test_generate_report_includes_uncertainty(self, sample_report_data):
        """Test that report includes uncertainty quantification."""
        generator = ClinicalReportGenerator()
        html_report = generator.generate_report(sample_report_data)
        
        assert "Uncertainty Quantification" in html_report
        assert sample_report_data.uncertainty_explanation in html_report

    def test_generate_report_includes_recommendations(self, sample_report_data):
        """Test that report includes clinical recommendations."""
        generator = ClinicalReportGenerator()
        html_report = generator.generate_report(sample_report_data)
        
        assert "Clinical Recommendations" in html_report
        for rec in sample_report_data.recommendations:
            assert rec in html_report

    def test_generate_report_with_ood_warning(self, sample_report_data):
        """Test report generation with OOD detection warning."""
        sample_report_data.ood_detected = True
        sample_report_data.ood_explanation = "Novel tissue pattern detected"
        
        generator = ClinicalReportGenerator()
        html_report = generator.generate_report(sample_report_data)
        
        assert "Out-of-Distribution Detection" in html_report
        assert sample_report_data.ood_explanation in html_report
        assert "Expert pathologist review recommended" in html_report

    def test_generate_report_with_longitudinal_summary(self, sample_report_data):
        """Test report generation with longitudinal progression summary."""
        sample_report_data.longitudinal_summary = {
            "num_scans": 3,
            "duration_days": 180,
            "progression_trend": "stable",
        }
        sample_report_data.previous_scan_comparison = {
            "previous_diagnosis": "Benign breast lesion",
            "current_diagnosis": "Malignant neoplasm of breast",
            "probability_change": "+0.75",
        }
        
        generator = ClinicalReportGenerator()
        html_report = generator.generate_report(sample_report_data)
        
        assert "Longitudinal Progression Summary" in html_report
        assert "3" in html_report  # num_scans
        assert "180" in html_report  # duration_days
        assert "stable" in html_report  # progression_trend
        assert "Change from Previous Scan" in html_report

    def test_generate_report_includes_model_version(self, sample_report_data):
        """Test that report includes model version information."""
        generator = ClinicalReportGenerator()
        html_report = generator.generate_report(sample_report_data)
        
        assert "Model Version" in html_report
        assert sample_report_data.model_version in html_report

    def test_generate_report_includes_timestamps(self, sample_report_data):
        """Test that report includes scan date and report generation timestamp."""
        generator = ClinicalReportGenerator()
        html_report = generator.generate_report(sample_report_data)
        
        assert "Scan Date" in html_report
        assert "Report Generated" in html_report
        assert "2024-01-15" in html_report


class TestReportExport:
    """Test suite for report export functionality."""

    def test_export_html(self, sample_report_data):
        """Test HTML export."""
        generator = ClinicalReportGenerator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            result_path = generator.export_report(
                sample_report_data,
                output_path,
                export_format=ExportFormat.HTML,
            )
            
            assert result_path.exists()
            assert result_path.suffix == ".html"
            
            content = result_path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content
            assert sample_report_data.patient_id in content

    def test_export_json(self, sample_report_data):
        """Test JSON export."""
        generator = ClinicalReportGenerator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            result_path = generator.export_report(
                sample_report_data,
                output_path,
                export_format=ExportFormat.JSON,
            )
            
            assert result_path.exists()
            assert result_path.suffix == ".json"
            
            content = json.loads(result_path.read_text(encoding="utf-8"))
            assert content["patient_id"] == sample_report_data.patient_id
            assert content["scan_id"] == sample_report_data.scan_id
            assert content["primary_diagnosis"]["disease_name"] == sample_report_data.primary_diagnosis.disease_name
            assert len(content["alternative_diagnoses"]) == len(sample_report_data.alternative_diagnoses)

    def test_export_fhir(self, sample_report_data):
        """Test FHIR DiagnosticReport export."""
        generator = ClinicalReportGenerator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report_fhir.json"
            result_path = generator.export_report(
                sample_report_data,
                output_path,
                export_format=ExportFormat.FHIR,
            )
            
            assert result_path.exists()
            
            fhir_report = json.loads(result_path.read_text(encoding="utf-8"))
            assert fhir_report["resourceType"] == "DiagnosticReport"
            assert fhir_report["status"] == "final"
            assert f"Patient/{sample_report_data.patient_id}" == fhir_report["subject"]["reference"]
            assert sample_report_data.primary_diagnosis.disease_name in fhir_report["conclusion"]

    def test_export_fhir_includes_alternative_diagnoses(self, sample_report_data):
        """Test FHIR export includes alternative diagnoses as observations."""
        generator = ClinicalReportGenerator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report_fhir.json"
            result_path = generator.export_report(
                sample_report_data,
                output_path,
                export_format=ExportFormat.FHIR,
            )
            
            fhir_report = json.loads(result_path.read_text(encoding="utf-8"))
            assert "contained" in fhir_report
            assert len(fhir_report["contained"]) == len(sample_report_data.alternative_diagnoses)
            
            for obs in fhir_report["contained"]:
                assert obs["resourceType"] == "Observation"
                assert obs["status"] == "final"

    def test_export_fhir_includes_model_version(self, sample_report_data):
        """Test FHIR export includes model version in extensions."""
        generator = ClinicalReportGenerator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report_fhir.json"
            result_path = generator.export_report(
                sample_report_data,
                output_path,
                export_format=ExportFormat.FHIR,
            )
            
            fhir_report = json.loads(result_path.read_text(encoding="utf-8"))
            assert "extension" in fhir_report
            
            model_version_ext = next(
                (ext for ext in fhir_report["extension"] if "model-version" in ext["url"]),
                None,
            )
            assert model_version_ext is not None
            assert model_version_ext["valueString"] == sample_report_data.model_version

    def test_export_fhir_with_longitudinal_summary(self, sample_report_data):
        """Test FHIR export includes longitudinal summary when available."""
        sample_report_data.longitudinal_summary = {
            "num_scans": 3,
            "duration_days": 180,
            "progression_trend": "stable",
        }
        
        generator = ClinicalReportGenerator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report_fhir.json"
            result_path = generator.export_report(
                sample_report_data,
                output_path,
                export_format=ExportFormat.FHIR,
            )
            
            fhir_report = json.loads(result_path.read_text(encoding="utf-8"))
            
            long_ext = next(
                (ext for ext in fhir_report["extension"] if "longitudinal-summary" in ext["url"]),
                None,
            )
            assert long_ext is not None
            long_data = json.loads(long_ext["valueString"])
            assert long_data["num_scans"] == 3
            assert long_data["duration_days"] == 180

    def test_export_dicom_sr(self, sample_report_data):
        """Test DICOM Structured Report export."""
        generator = ClinicalReportGenerator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report_dicom_sr.json"
            result_path = generator.export_report(
                sample_report_data,
                output_path,
                export_format=ExportFormat.DICOM_SR,
            )
            
            assert result_path.exists()
            
            dicom_sr = json.loads(result_path.read_text(encoding="utf-8"))
            assert dicom_sr["SOPClassUID"] == "1.2.840.10008.5.1.4.1.1.88.11"
            assert dicom_sr["Modality"] == "SR"
            assert dicom_sr["PatientID"] == sample_report_data.patient_id
            assert dicom_sr["CompletionFlag"] == "COMPLETE"

    def test_export_dicom_sr_includes_findings(self, sample_report_data):
        """Test DICOM SR export includes primary and alternative findings."""
        generator = ClinicalReportGenerator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report_dicom_sr.json"
            result_path = generator.export_report(
                sample_report_data,
                output_path,
                export_format=ExportFormat.DICOM_SR,
            )
            
            dicom_sr = json.loads(result_path.read_text(encoding="utf-8"))
            assert "ContentSequence" in dicom_sr
            
            # Check for primary finding
            findings = [
                item for item in dicom_sr["ContentSequence"]
                if item.get("ValueType") == "CODE"
            ]
            assert len(findings) >= 1  # At least primary diagnosis

    def test_export_pdf_requires_reportlab(self, sample_report_data):
        """Test PDF export (requires reportlab)."""
        generator = ClinicalReportGenerator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.pdf"
            
            try:
                result_path = generator.export_report(
                    sample_report_data,
                    output_path,
                    export_format=ExportFormat.PDF,
                )
                
                # If reportlab is installed, check the file was created
                assert result_path.exists()
                assert result_path.suffix == ".pdf"
                assert result_path.stat().st_size > 0
                
            except ImportError as e:
                # If reportlab is not installed, expect ImportError
                assert "reportlab" in str(e)

    def test_export_creates_parent_directories(self, sample_report_data):
        """Test that export creates parent directories if they don't exist."""
        generator = ClinicalReportGenerator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir1" / "subdir2" / "report.html"
            result_path = generator.export_report(
                sample_report_data,
                output_path,
                export_format=ExportFormat.HTML,
            )
            
            assert result_path.exists()
            assert result_path.parent.exists()

    def test_export_unsupported_format_raises_error(self, sample_report_data):
        """Test that unsupported export format raises ValueError."""
        generator = ClinicalReportGenerator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.txt"
            
            # Create a fake unsupported format
            class UnsupportedFormat:
                pass
            
            with pytest.raises((ValueError, AttributeError)):
                generator.export_report(
                    sample_report_data,
                    output_path,
                    export_format=UnsupportedFormat(),  # type: ignore
                )


class TestPhysicianAnnotations:
    """Test suite for physician annotations and amendments."""

    def test_add_physician_annotation(self, sample_report_data):
        """Test adding physician annotation to report."""
        generator = ClinicalReportGenerator()
        
        updated_data = generator.add_physician_annotation(
            sample_report_data,
            user="Dr. Smith",
            note="Confirmed with additional immunohistochemistry staining.",
        )
        
        assert len(updated_data.amendments) == 1
        assert updated_data.amendments[0]["user"] == "Dr. Smith"
        assert "immunohistochemistry" in updated_data.amendments[0]["note"]
        assert "timestamp" in updated_data.amendments[0]

    def test_multiple_annotations(self, sample_report_data):
        """Test adding multiple physician annotations."""
        generator = ClinicalReportGenerator()
        
        updated_data = generator.add_physician_annotation(
            sample_report_data,
            user="Dr. Smith",
            note="First annotation",
        )
        updated_data = generator.add_physician_annotation(
            updated_data,
            user="Dr. Jones",
            note="Second annotation",
        )
        
        assert len(updated_data.amendments) == 2
        assert updated_data.amendments[0]["user"] == "Dr. Smith"
        assert updated_data.amendments[1]["user"] == "Dr. Jones"

    def test_annotations_included_in_report(self, sample_report_data):
        """Test that annotations are included in generated report."""
        generator = ClinicalReportGenerator()
        
        updated_data = generator.add_physician_annotation(
            sample_report_data,
            user="Dr. Smith",
            note="Confirmed diagnosis",
        )
        
        html_report = generator.generate_report(updated_data)
        
        assert "Report Amendments" in html_report
        assert "Dr. Smith" in html_report
        assert "Confirmed diagnosis" in html_report


class TestReportDataStructure:
    """Test suite for ReportData and DiagnosisResult data structures."""

    def test_diagnosis_result_creation(self):
        """Test DiagnosisResult creation."""
        diag = DiagnosisResult(
            disease_id="C50.9",
            disease_name="Malignant neoplasm of breast",
            probability=0.85,
            confidence_interval=(0.78, 0.92),
            uncertainty_score=0.12,
        )
        
        assert diag.disease_id == "C50.9"
        assert diag.disease_name == "Malignant neoplasm of breast"
        assert diag.probability == 0.85
        assert diag.confidence_interval == (0.78, 0.92)
        assert diag.uncertainty_score == 0.12

    def test_diagnosis_result_optional_fields(self):
        """Test DiagnosisResult with optional fields as None."""
        diag = DiagnosisResult(
            disease_id="C50.9",
            disease_name="Malignant neoplasm of breast",
            probability=0.85,
        )
        
        assert diag.confidence_interval is None
        assert diag.uncertainty_score is None

    def test_report_data_creation(self, sample_diagnosis_result):
        """Test ReportData creation with required fields."""
        report = ReportData(
            patient_id="PATIENT_123",
            scan_id="SCAN_456",
            scan_date=datetime(2024, 1, 15, 10, 30, 0),
            primary_diagnosis=sample_diagnosis_result,
        )
        
        assert report.patient_id == "PATIENT_123"
        assert report.scan_id == "SCAN_456"
        assert report.primary_diagnosis == sample_diagnosis_result
        assert report.alternative_diagnoses == []
        assert report.probability_distribution == {}
        assert report.amendments == []

    def test_report_data_default_timestamp(self, sample_diagnosis_result):
        """Test that ReportData has default report_timestamp."""
        report = ReportData(
            patient_id="PATIENT_123",
            scan_id="SCAN_456",
            scan_date=datetime(2024, 1, 15, 10, 30, 0),
            primary_diagnosis=sample_diagnosis_result,
        )
        
        assert isinstance(report.report_timestamp, datetime)
        # Should be recent (within last minute)
        time_diff = (datetime.now() - report.report_timestamp).total_seconds()
        assert time_diff < 60


class TestReportSpecialtyEnum:
    """Test suite for ReportSpecialty enum."""

    def test_specialty_values(self):
        """Test that all specialty values are defined."""
        assert ReportSpecialty.CARDIOLOGY.value == "cardiology"
        assert ReportSpecialty.ONCOLOGY.value == "oncology"
        assert ReportSpecialty.RADIOLOGY.value == "radiology"
        assert ReportSpecialty.PATHOLOGY.value == "pathology"
        assert ReportSpecialty.GENERAL.value == "general"


class TestExportFormatEnum:
    """Test suite for ExportFormat enum."""

    def test_export_format_values(self):
        """Test that all export format values are defined."""
        assert ExportFormat.PDF.value == "pdf"
        assert ExportFormat.HTML.value == "html"
        assert ExportFormat.FHIR.value == "fhir"
        assert ExportFormat.DICOM_SR.value == "dicom_sr"
        assert ExportFormat.JSON.value == "json"
