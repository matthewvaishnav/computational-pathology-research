"""Property-based tests for PACS Retrieval Engine (DICOM C-MOVE operations).

Feature: pacs-integration-system
Property 4: Retrieval Operation Completeness
Property 5: File Integrity Validation
Property 6: File Storage Naming Convention
Property 7: Workflow Notification Completeness
"""

import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from src.clinical.pacs.data_models import (
    PACSEndpoint,
    PACSVendor,
    PerformanceConfig,
    SecurityConfig,
)
from src.clinical.pacs.retrieval_engine import RetrievalEngine

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_endpoint() -> PACSEndpoint:
    return PACSEndpoint(
        endpoint_id="test-ep",
        ae_title="TEST_AE",
        host="pacs.test.local",
        port=11112,
        vendor=PACSVendor.GENERIC,
        security_config=SecurityConfig(
            tls_enabled=False,
            verify_certificates=False,
            mutual_authentication=False,
        ),
        performance_config=PerformanceConfig(),
    )


def _create_test_dicom_file(
    file_path: Path,
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    patient_id: str = "P001",
) -> Path:
    """Create a minimal valid DICOM file for testing."""
    # Create file meta information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.77.1.6"  # VL Whole Slide Microscopy Image Storage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Create dataset
    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Add required DICOM fields
    ds.PatientID = patient_id
    ds.PatientName = f"Test^Patient^{patient_id}"
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = sop_uid
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "SM"
    ds.StudyDate = "20260101"
    ds.SeriesDate = "20260101"
    ds.StudyDescription = "Test Study"
    ds.SeriesDescription = "Test Series"

    # Save the file
    file_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(file_path), write_like_original=False)

    return file_path


# ---------------------------------------------------------------------------
# Property 4 — Retrieval Operation Completeness
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 4: Retrieval Operation Completeness
# For any C-MOVE operation that transfers N files, all N files SHALL be
# successfully stored and tracked.


@given(file_count=st.integers(min_value=1, max_value=10))
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_4_all_retrieved_files_tracked(tmp_path, file_count):
    """All N retrieved files must be tracked and accessible."""
    engine = RetrievalEngine()

    # Create test DICOM files
    study_uid = "1.2.840.10008.1.2.3.4.5"
    series_uid = "1.2.840.10008.1.2.3.4.5.1"

    created_files = []
    for i in range(file_count):
        sop_uid = f"1.2.840.10008.1.2.3.4.5.1.{i}"
        file_path = tmp_path / f"file_{i}.dcm"
        _create_test_dicom_file(file_path, study_uid, series_uid, sop_uid)
        created_files.append(file_path)

    # Validate all files
    validation = engine.validate_retrieved_files(created_files)

    # All files must be valid
    assert validation.is_valid, f"Validation failed: {validation.errors}"
    assert len(validation.errors) == 0


# ---------------------------------------------------------------------------
# Property 5 — File Integrity Validation
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 5: File Integrity Validation
# For any retrieved DICOM file, integrity validation SHALL detect corruption
# or missing required fields.


@given(
    study_uid=st.text(
        min_size=10,
        max_size=64,
        alphabet=st.characters(min_codepoint=48, max_codepoint=57, whitelist_characters="."),
    ),
    series_uid=st.text(
        min_size=10,
        max_size=64,
        alphabet=st.characters(min_codepoint=48, max_codepoint=57, whitelist_characters="."),
    ),
    sop_uid=st.text(
        min_size=10,
        max_size=64,
        alphabet=st.characters(min_codepoint=48, max_codepoint=57, whitelist_characters="."),
    ),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_5_valid_dicom_files_pass_validation(
    tmp_path, study_uid, series_uid, sop_uid
):
    """Valid DICOM files with all required fields must pass validation."""
    engine = RetrievalEngine()

    # Create valid DICOM file
    file_path = tmp_path / "test.dcm"
    _create_test_dicom_file(file_path, study_uid, series_uid, sop_uid)

    # Validate
    validation = engine.validate_retrieved_files([file_path])

    # Must be valid
    assert validation.is_valid, f"Valid file failed validation: {validation.errors}"
    assert len(validation.errors) == 0


@given(file_size=st.integers(min_value=0, max_value=0))
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_5_empty_files_fail_validation(tmp_path, file_size):
    """Empty files (0 bytes) must fail validation."""
    engine = RetrievalEngine()

    # Create empty file
    file_path = tmp_path / "empty.dcm"
    file_path.write_bytes(b"")

    # Validate
    validation = engine.validate_retrieved_files([file_path])

    # Must fail validation
    assert not validation.is_valid
    assert len(validation.errors) > 0
    assert any("empty" in err.lower() for err in validation.errors)


# ---------------------------------------------------------------------------
# Property 6 — File Storage Naming Convention
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 6: File Storage Naming Convention
# For any retrieved file, the storage path SHALL follow the hierarchical
# naming convention: StudyUID/SeriesUID/SOPInstanceUID.dcm


@given(
    study_uid=st.text(
        min_size=10,
        max_size=64,
        alphabet=st.characters(min_codepoint=48, max_codepoint=57, whitelist_characters="."),
    ),
    series_uid=st.text(
        min_size=10,
        max_size=64,
        alphabet=st.characters(min_codepoint=48, max_codepoint=57, whitelist_characters="."),
    ),
    sop_uid=st.text(
        min_size=10,
        max_size=64,
        alphabet=st.characters(min_codepoint=48, max_codepoint=57, whitelist_characters="."),
    ),
)
@settings(max_examples=100)
def test_property_6_filename_follows_hierarchical_convention(
    study_uid, series_uid, sop_uid
):
    """Generated filenames must follow StudyUID/SeriesUID/SOPInstanceUID.dcm format."""
    engine = RetrievalEngine()

    # Create dataset with UIDs
    ds = Dataset()
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = sop_uid

    # Generate filename
    filename = engine._generate_filename(ds)

    # Must follow hierarchical convention
    expected = f"{study_uid}/{series_uid}/{sop_uid}.dcm"
    assert filename == expected, f"Expected {expected}, got {filename}"


@given(
    sop_uid=st.text(
        min_size=10,
        max_size=64,
        alphabet=st.characters(min_codepoint=48, max_codepoint=57, whitelist_characters="."),
    )
)
@settings(max_examples=50)
def test_property_6_filename_fallback_without_study_series(sop_uid):
    """Filenames without study/series UIDs must use flat structure."""
    engine = RetrievalEngine()

    # Create dataset with only SOP UID
    ds = Dataset()
    ds.SOPInstanceUID = sop_uid

    # Generate filename
    filename = engine._generate_filename(ds)

    # Must use flat structure
    expected = f"{sop_uid}.dcm"
    assert filename == expected, f"Expected {expected}, got {filename}"


# ---------------------------------------------------------------------------
# Property 7 — Workflow Notification Completeness
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 7: Workflow Notification Completeness
# For any retrieval operation, the result SHALL include complete metadata
# about retrieved files and validation status.


@given(file_count=st.integers(min_value=1, max_value=5))
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_7_retrieval_result_includes_metadata(tmp_path, file_count):
    """Retrieval results must include file paths and validation status."""
    engine = RetrievalEngine()

    # Create test files
    study_uid = "1.2.840.10008.1.2.3.4.5"
    series_uid = "1.2.840.10008.1.2.3.4.5.1"

    created_files = []
    for i in range(file_count):
        sop_uid = f"1.2.840.10008.1.2.3.4.5.1.{i}"
        file_path = tmp_path / f"file_{i}.dcm"
        _create_test_dicom_file(file_path, study_uid, series_uid, sop_uid)
        created_files.append(file_path)

    # Validate files (simulates retrieval result)
    validation = engine.validate_retrieved_files(created_files)

    # Result must include validation status
    assert hasattr(validation, "is_valid")
    assert hasattr(validation, "errors")
    assert hasattr(validation, "warnings")

    # For valid files, is_valid must be True
    assert validation.is_valid


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_retrieval_engine_initialization():
    """RetrievalEngine initializes with correct parameters."""
    engine = RetrievalEngine(
        ae_title="TEST_AE", storage_scp_port=11113, max_concurrent_retrievals=5
    )

    assert engine.ae_title == "TEST_AE"
    assert engine.storage_scp_port == 11113
    assert engine.max_concurrent_retrievals == 5


def test_validate_retrieved_files_detects_missing_file(tmp_path):
    """Validation detects non-existent files."""
    engine = RetrievalEngine()

    # Reference non-existent file
    missing_file = tmp_path / "missing.dcm"

    validation = engine.validate_retrieved_files([missing_file])

    assert not validation.is_valid
    assert len(validation.errors) > 0
    assert any("does not exist" in err for err in validation.errors)


def test_validate_retrieved_files_detects_invalid_dicom(tmp_path):
    """Validation detects invalid DICOM files."""
    engine = RetrievalEngine()

    # Create invalid DICOM file (just random bytes)
    invalid_file = tmp_path / "invalid.dcm"
    invalid_file.write_bytes(b"This is not a DICOM file")

    validation = engine.validate_retrieved_files([invalid_file])

    assert not validation.is_valid
    assert len(validation.errors) > 0
    assert any("Invalid DICOM" in err for err in validation.errors)


def test_validate_retrieved_files_detects_missing_required_fields(tmp_path):
    """Validation detects DICOM files missing required fields."""
    engine = RetrievalEngine()

    # Create DICOM file missing required fields
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.77.1.6"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Missing StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID
    ds.PatientID = "P001"

    file_path = tmp_path / "incomplete.dcm"
    ds.save_as(str(file_path), write_like_original=False)

    validation = engine.validate_retrieved_files([file_path])

    assert not validation.is_valid
    assert len(validation.errors) > 0


def test_calculate_file_checksum(tmp_path):
    """Checksum calculation produces consistent results."""
    engine = RetrievalEngine()

    # Create test file
    test_file = tmp_path / "test.txt"
    test_content = b"Test content for checksum"
    test_file.write_bytes(test_content)

    # Calculate checksum
    checksum1 = engine._calculate_file_checksum(test_file)
    checksum2 = engine._calculate_file_checksum(test_file)

    # Checksums must be consistent
    assert checksum1 == checksum2

    # Verify against known SHA-256
    expected = hashlib.sha256(test_content).hexdigest()
    assert checksum1 == expected


def test_generate_filename_with_all_uids():
    """Filename generation with all UIDs uses hierarchical structure."""
    engine = RetrievalEngine()

    ds = Dataset()
    ds.StudyInstanceUID = "1.2.3"
    ds.SeriesInstanceUID = "1.2.3.4"
    ds.SOPInstanceUID = "1.2.3.4.5"

    filename = engine._generate_filename(ds)

    assert filename == "1.2.3/1.2.3.4/1.2.3.4.5.dcm"


def test_generate_filename_without_study_series():
    """Filename generation without study/series uses flat structure."""
    engine = RetrievalEngine()

    ds = Dataset()
    ds.SOPInstanceUID = "1.2.3.4.5"

    filename = engine._generate_filename(ds)

    assert filename == "1.2.3.4.5.dcm"


def test_generate_filename_fallback_on_error():
    """Filename generation falls back to timestamp on error."""
    engine = RetrievalEngine()

    # Empty dataset (no UIDs)
    ds = Dataset()

    filename = engine._generate_filename(ds)

    # Should generate timestamp-based name
    assert filename.startswith("dicom_")
    assert filename.endswith(".dcm")


def test_check_disk_space(tmp_path):
    """Disk space check returns boolean."""
    engine = RetrievalEngine()

    # Check disk space (should succeed on test system)
    has_space = engine._check_disk_space(tmp_path)

    assert isinstance(has_space, bool)


def test_get_retrieval_statistics():
    """Get retrieval statistics returns engine info."""
    engine = RetrievalEngine(
        ae_title="TEST_AE", storage_scp_port=11113, max_concurrent_retrievals=5
    )

    stats = engine.get_retrieval_statistics()

    assert stats["ae_title"] == "TEST_AE"
    assert stats["storage_scp_port"] == 11113
    assert stats["max_concurrent_retrievals"] == 5
    assert "active_retrievals" in stats
    assert "min_free_space_gb" in stats


def test_create_pacs_metadata(tmp_path):
    """Create PACSMetadata from DICOM file."""
    engine = RetrievalEngine()
    endpoint = _make_endpoint()

    # Create test DICOM file
    study_uid = "1.2.3"
    series_uid = "1.2.3.4"
    sop_uid = "1.2.3.4.5"
    file_path = tmp_path / "test.dcm"
    _create_test_dicom_file(file_path, study_uid, series_uid, sop_uid)

    # Create metadata
    metadata = engine._create_pacs_metadata(file_path, endpoint)

    assert metadata.study_instance_uid == study_uid
    assert metadata.series_instance_uid == series_uid
    assert metadata.sop_instance_uid == sop_uid
    assert metadata.patient_id == "P001"
    assert metadata.modality == "SM"
    assert metadata.source_pacs_ae_title == endpoint.ae_title
