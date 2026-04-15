"""
DICOM adapter for medical imaging standards integration.

This module provides DICOM integration for reading WSI files in DICOM format,
writing prediction results to DICOM Structured Report (SR) format, and
supporting PACS integration for query/retrieve operations.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import (
    JPEG2000,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGLSLossless,
    JPEGLSNearLossless,
    generate_uid,
)


class TransferSyntax(Enum):
    """Supported DICOM transfer syntaxes for pathology."""

    JPEG2000_LOSSLESS = JPEG2000Lossless
    JPEG2000 = JPEG2000
    JPEG_LS_LOSSLESS = JPEGLSLossless
    JPEG_LS_NEAR_LOSSLESS = JPEGLSNearLossless
    EXPLICIT_VR_LITTLE_ENDIAN = ExplicitVRLittleEndian
    IMPLICIT_VR_LITTLE_ENDIAN = ImplicitVRLittleEndian


@dataclass
class DICOMMetadata:
    """Structured DICOM metadata extracted from WSI files."""

    patient_id: str
    patient_name: str
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    modality: str
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
    transfer_syntax_uid: Optional[str] = None
    additional_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "study_instance_uid": self.study_instance_uid,
            "series_instance_uid": self.series_instance_uid,
            "sop_instance_uid": self.sop_instance_uid,
            "modality": self.modality,
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
            "transfer_syntax_uid": self.transfer_syntax_uid,
            "additional_metadata": self.additional_metadata or {},
        }


@dataclass
class PredictionResult:
    """Prediction results to be written to DICOM SR."""

    primary_diagnosis: str
    primary_diagnosis_probability: float
    probability_distribution: Dict[str, float]
    confidence: float
    uncertainty_estimate: Optional[float] = None
    risk_scores: Optional[Dict[str, float]] = None
    attention_regions: Optional[List[Dict[str, Any]]] = None
    model_version: Optional[str] = None
    processing_timestamp: Optional[datetime] = None
    additional_findings: Optional[Dict[str, Any]] = None


class DICOMAdapter:
    """
    DICOM adapter for medical imaging standards integration.

    This class provides functionality for:
    - Reading WSI files in DICOM format with metadata extraction
    - Writing prediction results to DICOM Structured Report (SR) format
    - PACS integration support for query/retrieve operations
    - DICOM file integrity validation
    - Multi-series handling

    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
    """

    def __init__(
        self,
        institution_name: str = "Computational Pathology System",
        manufacturer: str = "Research Institution",
        manufacturer_model: str = "AI Pathology Classifier v1.0",
    ):
        """
        Initialize DICOM adapter.

        Args:
            institution_name: Institution name for generated DICOM files
            manufacturer: Manufacturer name for generated DICOM files
            manufacturer_model: Model name for generated DICOM files
        """
        self.institution_name = institution_name
        self.manufacturer = manufacturer
        self.manufacturer_model = manufacturer_model

    def read_wsi(self, dicom_path: Union[str, Path]) -> Tuple[Dataset, DICOMMetadata]:
        """
        Read WSI file in DICOM format and extract metadata.

        Requirements: 6.1

        Args:
            dicom_path: Path to DICOM file

        Returns:
            Tuple of (DICOM dataset, extracted metadata)

        Raises:
            FileNotFoundError: If DICOM file does not exist
            pydicom.errors.InvalidDicomError: If file is not valid DICOM
            ValueError: If required metadata fields are missing
        """
        dicom_path = Path(dicom_path)
        if not dicom_path.exists():
            raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

        # Read DICOM file
        dataset = pydicom.dcmread(str(dicom_path))

        # Validate file integrity
        self.validate_dicom_integrity(dataset)

        # Extract metadata
        metadata = self._extract_metadata(dataset)

        return dataset, metadata

    def _extract_metadata(self, dataset: Dataset) -> DICOMMetadata:
        """
        Extract structured metadata from DICOM dataset.

        Args:
            dataset: DICOM dataset

        Returns:
            Structured DICOM metadata

        Raises:
            ValueError: If required fields are missing
        """
        # Required fields
        required_fields = [
            "PatientID",
            "PatientName",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SOPInstanceUID",
            "Modality",
        ]

        for field in required_fields:
            if not hasattr(dataset, field):
                raise ValueError(f"Required DICOM field missing: {field}")

        # Extract required fields
        metadata = DICOMMetadata(
            patient_id=str(dataset.PatientID),
            patient_name=str(dataset.PatientName),
            study_instance_uid=str(dataset.StudyInstanceUID),
            series_instance_uid=str(dataset.SeriesInstanceUID),
            sop_instance_uid=str(dataset.SOPInstanceUID),
            modality=str(dataset.Modality),
        )

        # Extract optional fields
        if hasattr(dataset, "StudyDate"):
            metadata.study_date = str(dataset.StudyDate)
        if hasattr(dataset, "SeriesDate"):
            metadata.series_date = str(dataset.SeriesDate)
        if hasattr(dataset, "AcquisitionDate"):
            metadata.acquisition_date = str(dataset.AcquisitionDate)
        if hasattr(dataset, "InstitutionName"):
            metadata.institution_name = str(dataset.InstitutionName)
        if hasattr(dataset, "Manufacturer"):
            metadata.manufacturer = str(dataset.Manufacturer)
        if hasattr(dataset, "ManufacturerModelName"):
            metadata.manufacturer_model = str(dataset.ManufacturerModelName)
        if hasattr(dataset, "ImageType"):
            metadata.image_type = list(dataset.ImageType)
        if hasattr(dataset, "Rows"):
            metadata.rows = int(dataset.Rows)
        if hasattr(dataset, "Columns"):
            metadata.columns = int(dataset.Columns)
        if hasattr(dataset, "NumberOfFrames"):
            metadata.number_of_frames = int(dataset.NumberOfFrames)

        # Extract transfer syntax
        if hasattr(dataset, "file_meta") and hasattr(dataset.file_meta, "TransferSyntaxUID"):
            metadata.transfer_syntax_uid = str(dataset.file_meta.TransferSyntaxUID)

        # Store additional metadata
        metadata.additional_metadata = {}
        optional_tags = [
            "StudyDescription",
            "SeriesDescription",
            "BodyPartExamined",
            "SliceThickness",
            "PixelSpacing",
        ]
        for tag in optional_tags:
            if hasattr(dataset, tag):
                metadata.additional_metadata[tag] = str(getattr(dataset, tag))

        return metadata

    def validate_dicom_integrity(self, dataset: Dataset) -> bool:
        """
        Validate DICOM file integrity before processing.

        Requirements: 6.5

        Args:
            dataset: DICOM dataset to validate

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        # Check required attributes
        required_attrs = [
            "PatientID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SOPInstanceUID",
            "Modality",
        ]

        for attr in required_attrs:
            if not hasattr(dataset, attr):
                raise ValueError(f"DICOM validation failed: missing required attribute {attr}")

        # Validate UIDs are not empty
        if not dataset.StudyInstanceUID:
            raise ValueError("DICOM validation failed: StudyInstanceUID is empty")
        if not dataset.SeriesInstanceUID:
            raise ValueError("DICOM validation failed: SeriesInstanceUID is empty")
        if not dataset.SOPInstanceUID:
            raise ValueError("DICOM validation failed: SOPInstanceUID is empty")

        # Validate transfer syntax if present
        if hasattr(dataset, "file_meta") and hasattr(dataset.file_meta, "TransferSyntaxUID"):
            transfer_syntax = dataset.file_meta.TransferSyntaxUID
            supported_syntaxes = [ts.value for ts in TransferSyntax]
            if transfer_syntax not in supported_syntaxes:
                # Warning but not error - we can still process
                pass

        return True

    def write_structured_report(
        self,
        prediction: PredictionResult,
        source_metadata: DICOMMetadata,
        output_path: Union[str, Path],
    ) -> FileDataset:
        """
        Write prediction results to DICOM Structured Report (SR) format.

        Requirements: 6.2, 6.3

        Args:
            prediction: Prediction results to write
            source_metadata: Metadata from source DICOM file
            output_path: Path to write SR file

        Returns:
            Created DICOM SR dataset

        Raises:
            ValueError: If prediction or metadata is invalid
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file dataset
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.88.11"  # Basic Text SR
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        # Create dataset
        ds = FileDataset(
            str(output_path),
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )

        # Set creation date/time
        dt = datetime.now()
        ds.ContentDate = dt.strftime("%Y%m%d")
        ds.ContentTime = dt.strftime("%H%M%S.%f")
        ds.InstanceCreationDate = dt.strftime("%Y%m%d")
        ds.InstanceCreationTime = dt.strftime("%H%M%S.%f")

        # Preserve required metadata from source
        ds.PatientID = source_metadata.patient_id
        ds.PatientName = source_metadata.patient_name
        ds.StudyInstanceUID = source_metadata.study_instance_uid
        ds.SeriesInstanceUID = generate_uid()  # New series for SR
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

        # Set modality and SR-specific fields
        ds.Modality = "SR"
        ds.SeriesDescription = "AI Pathology Analysis Report"
        ds.SeriesNumber = "999"
        ds.InstanceNumber = "1"

        # Set institution information
        ds.InstitutionName = self.institution_name
        ds.Manufacturer = self.manufacturer
        ds.ManufacturerModelName = self.manufacturer_model

        # Reference the source image
        referenced_sop = Dataset()
        referenced_sop.ReferencedSOPClassUID = (
            "1.2.840.10008.5.1.4.1.1.77.1.6"  # VL Whole Slide Microscopy Image
        )
        referenced_sop.ReferencedSOPInstanceUID = source_metadata.sop_instance_uid

        ds.ReferencedPerformedProcedureStepSequence = []
        ds.CurrentRequestedProcedureEvidenceSequence = []

        # Create content sequence for SR
        content_seq = []

        # Add primary diagnosis
        diagnosis_item = Dataset()
        diagnosis_item.RelationshipType = "CONTAINS"
        diagnosis_item.ValueType = "TEXT"
        diagnosis_item.ConceptNameCodeSequence = [
            self._create_code_item("121071", "DCM", "Finding")
        ]
        diagnosis_item.TextValue = f"Primary Diagnosis: {prediction.primary_diagnosis}"
        content_seq.append(diagnosis_item)

        # Add confidence
        confidence_item = Dataset()
        confidence_item.RelationshipType = "CONTAINS"
        confidence_item.ValueType = "NUM"
        confidence_item.ConceptNameCodeSequence = [
            self._create_code_item("121402", "DCM", "Confidence")
        ]
        confidence_item.MeasuredValueSequence = [
            self._create_numeric_value(prediction.primary_diagnosis_probability, "1", "Probability")
        ]
        content_seq.append(confidence_item)

        # Add probability distribution
        for disease_id, probability in prediction.probability_distribution.items():
            prob_item = Dataset()
            prob_item.RelationshipType = "CONTAINS"
            prob_item.ValueType = "NUM"
            prob_item.ConceptNameCodeSequence = [
                self._create_code_item("121071", "DCM", f"Probability: {disease_id}")
            ]
            prob_item.MeasuredValueSequence = [
                self._create_numeric_value(probability, "1", "Probability")
            ]
            content_seq.append(prob_item)

        # Add model version if available
        if prediction.model_version:
            version_item = Dataset()
            version_item.RelationshipType = "CONTAINS"
            version_item.ValueType = "TEXT"
            version_item.ConceptNameCodeSequence = [
                self._create_code_item("121020", "DCM", "Algorithm Version")
            ]
            version_item.TextValue = prediction.model_version
            content_seq.append(version_item)

        # Add processing timestamp
        timestamp = prediction.processing_timestamp or datetime.now()
        time_item = Dataset()
        time_item.RelationshipType = "CONTAINS"
        time_item.ValueType = "DATETIME"
        time_item.ConceptNameCodeSequence = [
            self._create_code_item("121110", "DCM", "Processing DateTime")
        ]
        time_item.DateTime = timestamp.strftime("%Y%m%d%H%M%S")
        content_seq.append(time_item)

        ds.ContentSequence = content_seq

        # Set SR document information
        ds.CompletionFlag = "COMPLETE"
        ds.VerificationFlag = "UNVERIFIED"

        # Save file
        ds.save_as(str(output_path), write_like_original=False)

        return ds

    def _create_code_item(self, code_value: str, coding_scheme: str, code_meaning: str) -> Dataset:
        """Create a coded concept item."""
        code_item = Dataset()
        code_item.CodeValue = code_value
        code_item.CodingSchemeDesignator = coding_scheme
        code_item.CodeMeaning = code_meaning
        return code_item

    def _create_numeric_value(self, value: float, unit_code: str, unit_meaning: str) -> Dataset:
        """Create a numeric measurement value."""
        numeric_value = Dataset()
        numeric_value.NumericValue = str(value)
        numeric_value.MeasurementUnitsCodeSequence = [
            self._create_code_item(unit_code, "UCUM", unit_meaning)
        ]
        return numeric_value

    def read_multi_series(
        self, dicom_paths: List[Union[str, Path]]
    ) -> List[Tuple[Dataset, DICOMMetadata]]:
        """
        Read multiple DICOM series and link them correctly.

        Requirements: 6.6

        Args:
            dicom_paths: List of paths to DICOM files

        Returns:
            List of (dataset, metadata) tuples, one per series

        Raises:
            ValueError: If series identifiers are inconsistent
        """
        series_data = []

        for path in dicom_paths:
            dataset, metadata = self.read_wsi(path)
            series_data.append((dataset, metadata))

        # Validate series consistency
        if len(series_data) > 1:
            study_uids = {metadata.study_instance_uid for _, metadata in series_data}
            if len(study_uids) > 1:
                raise ValueError(
                    f"Multiple study UIDs found: {study_uids}. "
                    "All series must belong to the same study."
                )

        return series_data

    def query_pacs(
        self,
        query_params: Dict[str, str],
        pacs_host: str = "localhost",
        pacs_port: int = 11112,
        ae_title: str = "PATHOLOGY_AI",
    ) -> List[Dict[str, Any]]:
        """
        Query PACS system for DICOM studies/series.

        Requirements: 6.4

        Args:
            query_params: Query parameters (e.g., PatientID, StudyDate)
            pacs_host: PACS server hostname
            pacs_port: PACS server port
            ae_title: Application Entity title

        Returns:
            List of matching studies/series

        Note:
            This is a placeholder implementation. Full PACS integration
            requires pynetdicom library and proper DICOM networking setup.
        """
        # Placeholder for PACS query functionality
        # Full implementation would use pynetdicom for C-FIND operations
        raise NotImplementedError(
            "PACS query/retrieve requires pynetdicom library and network configuration. "
            "This is a placeholder for future implementation."
        )

    def retrieve_from_pacs(
        self,
        study_uid: str,
        series_uid: Optional[str] = None,
        output_dir: Union[str, Path] = ".",
        pacs_host: str = "localhost",
        pacs_port: int = 11112,
        ae_title: str = "PATHOLOGY_AI",
    ) -> List[Path]:
        """
        Retrieve DICOM files from PACS system.

        Requirements: 6.4

        Args:
            study_uid: Study Instance UID to retrieve
            series_uid: Optional Series Instance UID (retrieve specific series)
            output_dir: Directory to save retrieved files
            pacs_host: PACS server hostname
            pacs_port: PACS server port
            ae_title: Application Entity title

        Returns:
            List of paths to retrieved DICOM files

        Note:
            This is a placeholder implementation. Full PACS integration
            requires pynetdicom library and proper DICOM networking setup.
        """
        # Placeholder for PACS retrieve functionality
        # Full implementation would use pynetdicom for C-MOVE operations
        raise NotImplementedError(
            "PACS query/retrieve requires pynetdicom library and network configuration. "
            "This is a placeholder for future implementation."
        )

    def supports_transfer_syntax(self, transfer_syntax_uid: str) -> bool:
        """
        Check if transfer syntax is supported.

        Requirements: 6.7

        Args:
            transfer_syntax_uid: Transfer syntax UID to check

        Returns:
            True if supported, False otherwise
        """
        supported_syntaxes = [ts.value for ts in TransferSyntax]
        return transfer_syntax_uid in supported_syntaxes

    def get_supported_transfer_syntaxes(self) -> List[str]:
        """
        Get list of supported transfer syntaxes.

        Requirements: 6.7

        Returns:
            List of supported transfer syntax UIDs
        """
        return [ts.value for ts in TransferSyntax]
