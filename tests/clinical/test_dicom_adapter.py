"""
Integration tests for DICOM adapter.

Tests DICOM file reading and metadata extraction, SR generation with sample
predictions, and multi-series handling.

Requirements: 6.1, 6.2, 6.3, 6.6
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, JPEG2000Lossless, generate_uid

from src.clinical.dicom_adapter import (
    DICOMAdapter,
    DICOMMetadata,
    PredictionResult,
    TransferSyntax,
)


class TestDICOMAdapter:
    """Test cases for DICOMAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create DICOM adapter instance."""
        return DICOMAdapter(
            institution_name="Test Hospital",
            manufacturer="Test Manufacturer",
            manufacturer_model="Test Model v1.0",
        )

    @pytest.fixture
    def sample_dicom_file(self, tmp_path):
        """Create a sample DICOM file for testing."""
        # Create file meta information
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.77.1.6"  # VL WSI
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        # Create dataset
        filename = tmp_path / "test_wsi.dcm"
        ds = FileDataset(
            str(filename),
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )

        # Set required DICOM fields
        ds.PatientID = "TEST001"
        ds.PatientName = "Test^Patient"
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.Modality = "SM"  # Slide Microscopy

        # Set optional fields
        ds.StudyDate = "20240101"
        ds.SeriesDate = "20240101"
        ds.AcquisitionDate = "20240101"
        ds.InstitutionName = "Test Institution"
        ds.Manufacturer = "Test Scanner Manufacturer"
        ds.ManufacturerModelName = "Scanner Model X"
        ds.ImageType = ["ORIGINAL", "PRIMARY", "VOLUME"]
        ds.Rows = 1024
        ds.Columns = 1024
        ds.NumberOfFrames = 100

        # Save file
        ds.save_as(str(filename), write_like_original=False)

        return filename

    @pytest.fixture
    def sample_prediction(self):
        """Create sample prediction result."""
        return PredictionResult(
            primary_diagnosis="malignant",
            primary_diagnosis_probability=0.85,
            probability_distribution={
                "benign": 0.10,
                "malignant": 0.85,
                "uncertain": 0.05,
            },
            confidence=0.85,
            uncertainty_estimate=0.15,
            risk_scores={"1_year": 0.75, "5_year": 0.90},
            model_version="v1.2.3",
            processing_timestamp=datetime(2024, 1, 15, 10, 30, 0),
        )

    def test_adapter_initialization(self, adapter):
        """Test DICOM adapter initialization."""
        assert adapter.institution_name == "Test Hospital"
        assert adapter.manufacturer == "Test Manufacturer"
        assert adapter.manufacturer_model == "Test Model v1.0"

    def test_read_wsi_success(self, adapter, sample_dicom_file):
        """
        Test successful WSI file reading and metadata extraction.

        **Validates: Requirements 6.1**
        """
        dataset, metadata = adapter.read_wsi(sample_dicom_file)

        # Verify dataset is valid
        assert isinstance(dataset, Dataset)
        assert hasattr(dataset, "PatientID")
        assert hasattr(dataset, "StudyInstanceUID")

        # Verify metadata extraction
        assert isinstance(metadata, DICOMMetadata)
        assert metadata.patient_id == "TEST001"
        assert metadata.patient_name == "Test^Patient"
        assert metadata.modality == "SM"
        assert metadata.study_date == "20240101"
        assert metadata.institution_name == "Test Institution"
        assert metadata.manufacturer == "Test Scanner Manufacturer"
        assert metadata.rows == 1024
        assert metadata.columns == 1024
        assert metadata.number_of_frames == 100

    def test_read_wsi_file_not_found(self, adapter):
        """Test error handling for missing DICOM file."""
        with pytest.raises(FileNotFoundError, match="DICOM file not found"):
            adapter.read_wsi("nonexistent_file.dcm")

    def test_read_wsi_invalid_dicom(self, adapter, tmp_path):
        """Test error handling for invalid DICOM file."""
        # Create non-DICOM file
        invalid_file = tmp_path / "invalid.dcm"
        invalid_file.write_text("This is not a DICOM file")

        with pytest.raises(Exception):  # pydicom.errors.InvalidDicomError
            adapter.read_wsi(invalid_file)

    def test_extract_metadata_required_fields(self, adapter, sample_dicom_file):
        """
        Test metadata extraction includes all required fields.

        **Validates: Requirements 6.1**
        """
        dataset, metadata = adapter.read_wsi(sample_dicom_file)

        # Verify all required fields are present
        assert metadata.patient_id is not None
        assert metadata.patient_name is not None
        assert metadata.study_instance_uid is not None
        assert metadata.series_instance_uid is not None
        assert metadata.sop_instance_uid is not None
        assert metadata.modality is not None

    def test_extract_metadata_optional_fields(self, adapter, sample_dicom_file):
        """Test metadata extraction handles optional fields."""
        dataset, metadata = adapter.read_wsi(sample_dicom_file)

        # Verify optional fields are extracted when present
        assert metadata.study_date == "20240101"
        assert metadata.institution_name == "Test Institution"
        assert metadata.image_type == ["ORIGINAL", "PRIMARY", "VOLUME"]

    def test_extract_metadata_missing_required_field(self, adapter, tmp_path):
        """Test error handling for missing required metadata fields."""
        # Create DICOM file missing required field
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.77.1.6"
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        filename = tmp_path / "incomplete.dcm"
        ds = FileDataset(str(filename), {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Missing PatientID and other required fields
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "SM"

        ds.save_as(str(filename), write_like_original=False)

        with pytest.raises(ValueError, match="DICOM validation failed: missing required attribute"):
            adapter.read_wsi(filename)

    def test_metadata_to_dict(self, adapter, sample_dicom_file):
        """Test metadata conversion to dictionary."""
        _, metadata = adapter.read_wsi(sample_dicom_file)

        metadata_dict = metadata.to_dict()

        assert isinstance(metadata_dict, dict)
        assert metadata_dict["patient_id"] == "TEST001"
        assert metadata_dict["modality"] == "SM"
        assert metadata_dict["rows"] == 1024
        assert "additional_metadata" in metadata_dict

    def test_validate_dicom_integrity_success(self, adapter, sample_dicom_file):
        """
        Test DICOM file integrity validation passes for valid file.

        **Validates: Requirements 6.5**
        """
        dataset, _ = adapter.read_wsi(sample_dicom_file)

        # Validation should pass
        result = adapter.validate_dicom_integrity(dataset)
        assert result is True

    def test_validate_dicom_integrity_missing_attribute(self, adapter):
        """Test validation fails for missing required attributes."""
        # Create incomplete dataset
        ds = Dataset()
        ds.PatientID = "TEST001"
        ds.StudyInstanceUID = generate_uid()
        # Missing SeriesInstanceUID, SOPInstanceUID, Modality

        with pytest.raises(ValueError, match="missing required attribute"):
            adapter.validate_dicom_integrity(ds)

    def test_validate_dicom_integrity_empty_uid(self, adapter):
        """Test validation fails for empty UIDs."""
        ds = Dataset()
        ds.PatientID = "TEST001"
        ds.StudyInstanceUID = ""  # Empty UID
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = generate_uid()
        ds.Modality = "SM"

        with pytest.raises(ValueError, match="StudyInstanceUID is empty"):
            adapter.validate_dicom_integrity(ds)

    def test_write_structured_report_success(
        self, adapter, sample_dicom_file, sample_prediction, tmp_path
    ):
        """
        Test writing prediction results to DICOM SR format.

        **Validates: Requirements 6.2, 6.3**
        """
        # Read source DICOM
        _, source_metadata = adapter.read_wsi(sample_dicom_file)

        # Write SR
        sr_path = tmp_path / "report.dcm"
        sr_dataset = adapter.write_structured_report(
            prediction=sample_prediction,
            source_metadata=source_metadata,
            output_path=sr_path,
        )

        # Verify SR file was created
        assert sr_path.exists()

        # Verify SR dataset structure
        assert isinstance(sr_dataset, FileDataset)
        assert sr_dataset.Modality == "SR"
        assert hasattr(sr_dataset, "ContentSequence")

        # Verify preserved metadata
        assert sr_dataset.PatientID == source_metadata.patient_id
        assert sr_dataset.PatientName == source_metadata.patient_name
        assert sr_dataset.StudyInstanceUID == source_metadata.study_instance_uid

    def test_write_structured_report_preserves_metadata(
        self, adapter, sample_dicom_file, sample_prediction, tmp_path
    ):
        """
        Test SR preserves required DICOM metadata fields.

        **Validates: Requirements 6.3**
        """
        _, source_metadata = adapter.read_wsi(sample_dicom_file)

        sr_path = tmp_path / "report.dcm"
        sr_dataset = adapter.write_structured_report(
            prediction=sample_prediction,
            source_metadata=source_metadata,
            output_path=sr_path,
        )

        # Verify all required metadata is preserved
        assert sr_dataset.PatientID == source_metadata.patient_id
        assert sr_dataset.PatientName == source_metadata.patient_name
        assert sr_dataset.StudyInstanceUID == source_metadata.study_instance_uid
        assert sr_dataset.SOPInstanceUID is not None
        assert sr_dataset.SeriesInstanceUID is not None

        # Verify institution information
        assert sr_dataset.InstitutionName == adapter.institution_name
        assert sr_dataset.Manufacturer == adapter.manufacturer
        assert sr_dataset.ManufacturerModelName == adapter.manufacturer_model

    def test_write_structured_report_content(
        self, adapter, sample_dicom_file, sample_prediction, tmp_path
    ):
        """
        Test SR contains prediction results in content sequence.

        **Validates: Requirements 6.2**
        """
        _, source_metadata = adapter.read_wsi(sample_dicom_file)

        sr_path = tmp_path / "report.dcm"
        sr_dataset = adapter.write_structured_report(
            prediction=sample_prediction,
            source_metadata=source_metadata,
            output_path=sr_path,
        )

        # Verify content sequence exists
        assert hasattr(sr_dataset, "ContentSequence")
        assert len(sr_dataset.ContentSequence) > 0

        # Verify primary diagnosis is in content
        content_texts = []
        for item in sr_dataset.ContentSequence:
            if hasattr(item, "TextValue"):
                content_texts.append(item.TextValue)

        # Check for primary diagnosis
        diagnosis_found = any("malignant" in text for text in content_texts)
        assert diagnosis_found, "Primary diagnosis not found in SR content"

    def test_write_structured_report_can_be_read(
        self, adapter, sample_dicom_file, sample_prediction, tmp_path
    ):
        """Test generated SR can be read back as valid DICOM."""
        _, source_metadata = adapter.read_wsi(sample_dicom_file)

        sr_path = tmp_path / "report.dcm"
        adapter.write_structured_report(
            prediction=sample_prediction,
            source_metadata=source_metadata,
            output_path=sr_path,
        )

        # Read back the SR
        sr_dataset = pydicom.dcmread(str(sr_path))

        # Verify it's a valid SR
        assert sr_dataset.Modality == "SR"
        assert hasattr(sr_dataset, "ContentSequence")

    def test_read_multi_series_success(self, adapter, tmp_path):
        """
        Test reading multiple DICOM series with correct linking.

        **Validates: Requirements 6.6**
        """
        # Create multiple DICOM files in same study
        study_uid = generate_uid()
        dicom_files = []

        for i in range(3):
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.77.1.6"
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            file_meta.ImplementationClassUID = generate_uid()

            filename = tmp_path / f"series_{i}.dcm"
            ds = FileDataset(str(filename), {}, file_meta=file_meta, preamble=b"\0" * 128)

            ds.PatientID = "TEST001"
            ds.PatientName = "Test^Patient"
            ds.StudyInstanceUID = study_uid  # Same study
            ds.SeriesInstanceUID = generate_uid()  # Different series
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
            ds.Modality = "SM"

            ds.save_as(str(filename), write_like_original=False)
            dicom_files.append(filename)

        # Read multi-series
        series_data = adapter.read_multi_series(dicom_files)

        # Verify all series were read
        assert len(series_data) == 3

        # Verify each series has dataset and metadata
        for dataset, metadata in series_data:
            assert isinstance(dataset, Dataset)
            assert isinstance(metadata, DICOMMetadata)

        # Verify all belong to same study
        study_uids = {metadata.study_instance_uid for _, metadata in series_data}
        assert len(study_uids) == 1

        # Verify different series UIDs
        series_uids = {metadata.series_instance_uid for _, metadata in series_data}
        assert len(series_uids) == 3

    def test_read_multi_series_different_studies_error(self, adapter, tmp_path):
        """
        Test error when reading series from different studies.

        **Validates: Requirements 6.6**
        """
        dicom_files = []

        for i in range(2):
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.77.1.6"
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            file_meta.ImplementationClassUID = generate_uid()

            filename = tmp_path / f"study_{i}.dcm"
            ds = FileDataset(str(filename), {}, file_meta=file_meta, preamble=b"\0" * 128)

            ds.PatientID = "TEST001"
            ds.PatientName = "Test^Patient"
            ds.StudyInstanceUID = generate_uid()  # Different study UIDs
            ds.SeriesInstanceUID = generate_uid()
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
            ds.Modality = "SM"

            ds.save_as(str(filename), write_like_original=False)
            dicom_files.append(filename)

        # Should raise error for multiple studies
        with pytest.raises(ValueError, match="Multiple study UIDs found"):
            adapter.read_multi_series(dicom_files)

    def test_supports_transfer_syntax(self, adapter):
        """
        Test transfer syntax support checking.

        **Validates: Requirements 6.7**
        """
        # Test supported syntaxes
        assert adapter.supports_transfer_syntax(JPEG2000Lossless)
        assert adapter.supports_transfer_syntax(ExplicitVRLittleEndian)

        # Test unsupported syntax
        fake_uid = "1.2.3.4.5.6.7.8.9"
        assert not adapter.supports_transfer_syntax(fake_uid)

    def test_get_supported_transfer_syntaxes(self, adapter):
        """
        Test getting list of supported transfer syntaxes.

        **Validates: Requirements 6.7**
        """
        syntaxes = adapter.get_supported_transfer_syntaxes()

        assert isinstance(syntaxes, list)
        assert len(syntaxes) > 0

        # Verify common pathology syntaxes are included
        assert JPEG2000Lossless in syntaxes
        assert ExplicitVRLittleEndian in syntaxes

    def test_query_pacs_not_implemented(self, adapter):
        """Test PACS query raises NotImplementedError (placeholder)."""
        with pytest.raises(NotImplementedError, match="PACS query/retrieve requires"):
            adapter.query_pacs({"PatientID": "TEST001"})

    def test_retrieve_from_pacs_not_implemented(self, adapter):
        """Test PACS retrieve raises NotImplementedError (placeholder)."""
        with pytest.raises(NotImplementedError, match="PACS query/retrieve requires"):
            adapter.retrieve_from_pacs(study_uid="1.2.3.4.5")


class TestDICOMMetadata:
    """Test cases for DICOMMetadata dataclass."""

    def test_metadata_initialization(self):
        """Test metadata initialization with required fields."""
        metadata = DICOMMetadata(
            patient_id="TEST001",
            patient_name="Test^Patient",
            study_instance_uid="1.2.3.4.5",
            series_instance_uid="1.2.3.4.6",
            sop_instance_uid="1.2.3.4.7",
            modality="SM",
        )

        assert metadata.patient_id == "TEST001"
        assert metadata.patient_name == "Test^Patient"
        assert metadata.modality == "SM"

    def test_metadata_with_optional_fields(self):
        """Test metadata with optional fields."""
        metadata = DICOMMetadata(
            patient_id="TEST001",
            patient_name="Test^Patient",
            study_instance_uid="1.2.3.4.5",
            series_instance_uid="1.2.3.4.6",
            sop_instance_uid="1.2.3.4.7",
            modality="SM",
            study_date="20240101",
            institution_name="Test Hospital",
            rows=1024,
            columns=1024,
        )

        assert metadata.study_date == "20240101"
        assert metadata.institution_name == "Test Hospital"
        assert metadata.rows == 1024

    def test_metadata_to_dict(self):
        """Test metadata conversion to dictionary."""
        metadata = DICOMMetadata(
            patient_id="TEST001",
            patient_name="Test^Patient",
            study_instance_uid="1.2.3.4.5",
            series_instance_uid="1.2.3.4.6",
            sop_instance_uid="1.2.3.4.7",
            modality="SM",
        )

        metadata_dict = metadata.to_dict()

        assert isinstance(metadata_dict, dict)
        assert metadata_dict["patient_id"] == "TEST001"
        assert metadata_dict["modality"] == "SM"
        assert "additional_metadata" in metadata_dict


class TestPredictionResult:
    """Test cases for PredictionResult dataclass."""

    def test_prediction_result_initialization(self):
        """Test prediction result initialization."""
        prediction = PredictionResult(
            primary_diagnosis="malignant",
            primary_diagnosis_probability=0.85,
            probability_distribution={"benign": 0.15, "malignant": 0.85},
            confidence=0.85,
        )

        assert prediction.primary_diagnosis == "malignant"
        assert prediction.primary_diagnosis_probability == 0.85
        assert prediction.confidence == 0.85

    def test_prediction_result_with_optional_fields(self):
        """Test prediction result with optional fields."""
        prediction = PredictionResult(
            primary_diagnosis="malignant",
            primary_diagnosis_probability=0.85,
            probability_distribution={"benign": 0.15, "malignant": 0.85},
            confidence=0.85,
            uncertainty_estimate=0.15,
            risk_scores={"1_year": 0.75},
            model_version="v1.0.0",
        )

        assert prediction.uncertainty_estimate == 0.15
        assert prediction.risk_scores == {"1_year": 0.75}
        assert prediction.model_version == "v1.0.0"


class TestTransferSyntax:
    """Test cases for TransferSyntax enum."""

    def test_transfer_syntax_enum_values(self):
        """Test transfer syntax enum contains expected values."""
        # Verify enum has expected members
        assert hasattr(TransferSyntax, "JPEG2000_LOSSLESS")
        assert hasattr(TransferSyntax, "JPEG2000")
        assert hasattr(TransferSyntax, "JPEG_LS_LOSSLESS")
        assert hasattr(TransferSyntax, "EXPLICIT_VR_LITTLE_ENDIAN")

    def test_transfer_syntax_values_are_uids(self):
        """Test transfer syntax values are valid UIDs."""
        for syntax in TransferSyntax:
            # UIDs should be strings with dots
            assert isinstance(syntax.value, str)
            assert "." in syntax.value
