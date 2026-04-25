"""Property-based tests for PACS Query Engine (DICOM C-FIND operations).

Feature: pacs-integration-system
Property 1: DICOM Query Parameter Translation
Property 2: Query Result Completeness
Property 3: Date Range Filtering Correctness
"""

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.clinical.pacs.data_models import (
    DicomPriority,
    PACSEndpoint,
    PACSVendor,
    PerformanceConfig,
    SecurityConfig,
    StudyInfo,
)
from src.clinical.pacs.query_engine import QueryEngine

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


# ---------------------------------------------------------------------------
# Property 1 — DICOM Query Parameter Translation
# For any valid query parameter, the system SHALL correctly translate it to
# the corresponding DICOM tag.
# ---------------------------------------------------------------------------


@given(
    patient_id=st.text(
        min_size=1,
        max_size=64,
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"),
    )
)
@settings(max_examples=100)
def test_property_1_patient_id_translates_to_dicom_tag(patient_id):
    """PatientID query parameter must map to DICOM tag (0010,0020)."""
    engine = QueryEngine()

    # Build the DICOM dataset for C-FIND
    dataset = engine._build_study_query_dataset(patient_id=patient_id)

    # Verify the PatientID tag is present and matches
    assert hasattr(dataset, "PatientID"), "PatientID tag not found in dataset"
    assert dataset.PatientID == patient_id


@given(
    start_date=st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 12, 31)),
    days_offset=st.integers(min_value=1, max_value=365),
)
@settings(max_examples=100)
def test_property_1_study_date_range_translates_to_dicom_tag(start_date, days_offset):
    """StudyDate range query parameter must map to DICOM tag (0008,0020) with range format."""
    engine = QueryEngine()
    end_date = start_date + timedelta(days=days_offset)

    # Convert to datetime for the API
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())

    dataset = engine._build_study_query_dataset(study_date_range=(start_dt, end_dt))

    assert hasattr(dataset, "StudyDate"), "StudyDate tag not found in dataset"
    # Should be in YYYYMMDD-YYYYMMDD format
    expected = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    assert dataset.StudyDate == expected


@given(modality=st.sampled_from(["SM", "XC", "GM", "MG", "CT", "MR", "US", "CR", "DX"]))
@settings(max_examples=50)
def test_property_1_modality_translates_to_dicom_tag(modality):
    """Modality query parameter must map to DICOM tag (0008,0061)."""
    engine = QueryEngine()
    dataset = engine._build_study_query_dataset(modality=modality)

    assert hasattr(dataset, "ModalitiesInStudy"), "ModalitiesInStudy tag not found in dataset"
    assert dataset.ModalitiesInStudy == modality


# ---------------------------------------------------------------------------
# Property 2 — Query Result Completeness
# For any C-FIND query that returns N results, all N results SHALL be
# returned to the caller without loss.
# ---------------------------------------------------------------------------


@given(result_count=st.integers(min_value=0, max_value=50))
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_2_all_query_results_returned(monkeypatch, result_count):
    """Query returning N results must yield exactly N StudyInfo objects."""
    engine = QueryEngine()
    endpoint = _make_endpoint()

    # Mock the association and C-FIND responses
    def mock_associate(addr, port, ae_title):
        mock_assoc = SimpleNamespace()
        mock_assoc.is_established = True
        mock_assoc.release = lambda: None

        def mock_send_c_find(dataset, query_model):
            for i in range(result_count):
                # Create mock response dataset
                from pydicom.dataset import Dataset

                response_ds = Dataset()
                response_ds.StudyInstanceUID = f"1.2.840.10008.{i}"
                response_ds.PatientID = f"P{i:04d}"
                response_ds.PatientName = f"Patient^{i}"
                response_ds.StudyDate = "20260101"
                response_ds.StudyDescription = f"Study {i}"
                response_ds.AccessionNumber = f"ACC{i:04d}"
                response_ds.NumberOfStudyRelatedSeries = "1"

                # Yield pending status with dataset
                status = SimpleNamespace()
                status.Status = 0xFF00  # Pending
                yield (status, response_ds)

            # Final success status
            status = SimpleNamespace()
            status.Status = 0x0000  # Success
            yield (status, None)

        mock_assoc.send_c_find = mock_send_c_find
        return mock_assoc

    monkeypatch.setattr(engine.ae, "associate", mock_associate)

    results = engine.query_studies(endpoint, patient_id="P*")

    assert len(results) == result_count, f"Expected {result_count} results, got {len(results)}"


# ---------------------------------------------------------------------------
# Property 3 — Date Range Filtering Correctness
# For any date range query, only studies within the specified range SHALL
# be returned.
# ---------------------------------------------------------------------------


@given(
    start_date=st.dates(min_value=date(2020, 1, 1), max_value=date(2025, 12, 31)),
    days_offset=st.integers(min_value=1, max_value=365),
)
@settings(max_examples=100)
def test_property_3_date_range_format_correct(start_date, days_offset):
    """Date range query must format as YYYYMMDD-YYYYMMDD."""
    engine = QueryEngine()
    end_date = start_date + timedelta(days=days_offset)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())

    dataset = engine._build_study_query_dataset(study_date_range=(start_dt, end_dt))

    assert hasattr(dataset, "StudyDate"), "StudyDate tag not found"
    expected = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    assert dataset.StudyDate == expected


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_query_engine_initialization():
    """QueryEngine initializes with AE title."""
    engine = QueryEngine(ae_title="TEST_AE")
    assert engine.ae_title == "TEST_AE"


def test_build_study_query_dataset_includes_required_tags():
    """Study query dataset includes all required return tags."""
    engine = QueryEngine()
    dataset = engine._build_study_query_dataset(patient_id="P001")

    # Required return tags for study-level queries
    assert hasattr(dataset, "StudyInstanceUID")
    assert hasattr(dataset, "PatientID")
    assert hasattr(dataset, "PatientName")
    assert hasattr(dataset, "StudyDate")
    assert dataset.QueryRetrieveLevel == "STUDY"


def test_build_series_query_dataset_includes_study_uid():
    """Series query dataset includes StudyInstanceUID."""
    engine = QueryEngine()
    dataset = engine._build_series_query_dataset("1.2.840.10008.1")

    assert hasattr(dataset, "StudyInstanceUID")
    assert dataset.StudyInstanceUID == "1.2.840.10008.1"
    assert hasattr(dataset, "SeriesInstanceUID")
    assert hasattr(dataset, "Modality")
    assert dataset.QueryRetrieveLevel == "SERIES"


def test_validate_query_parameters_accepts_valid_params():
    """Validate query params accepts valid parameters."""
    engine = QueryEngine()

    result = engine.validate_query_parameters(
        {
            "patient_id": "P001",
            "modality": "SM",
            "max_results": 100,
        }
    )

    assert result.is_valid


def test_validate_query_parameters_rejects_invalid_patient_id():
    """Validate query params rejects empty patient ID."""
    engine = QueryEngine()

    result = engine.validate_query_parameters({"patient_id": ""})

    assert not result.is_valid
    assert len(result.errors) > 0


def test_validate_query_parameters_rejects_invalid_date_range():
    """Validate query params rejects invalid date range."""
    engine = QueryEngine()

    # End date before start date
    start = datetime(2026, 1, 31)
    end = datetime(2026, 1, 1)

    result = engine.validate_query_parameters({"study_date_range": (start, end)})

    assert not result.is_valid
    assert any("before" in err.lower() for err in result.errors)


def test_validate_query_parameters_warns_large_date_range():
    """Validate query params warns for large date ranges."""
    engine = QueryEngine()

    start = datetime(2020, 1, 1)
    end = datetime(2022, 1, 1)  # 2 years

    result = engine.validate_query_parameters({"study_date_range": (start, end)})

    assert result.is_valid
    assert len(result.warnings) > 0


def test_build_study_query_dataset_with_accession_number():
    """Query with AccessionNumber parameter."""
    engine = QueryEngine()
    dataset = engine._build_study_query_dataset(accession_number="ACC12345")

    assert hasattr(dataset, "AccessionNumber")
    assert dataset.AccessionNumber == "ACC12345"


def test_build_study_query_dataset_with_study_description():
    """Query with StudyDescription parameter adds wildcards."""
    engine = QueryEngine()
    dataset = engine._build_study_query_dataset(study_description="CT")

    assert hasattr(dataset, "StudyDescription")
    assert "*CT*" in dataset.StudyDescription


def test_build_series_query_dataset_with_modality():
    """Series query with modality filter."""
    engine = QueryEngine()
    dataset = engine._build_series_query_dataset("1.2.840.10008.1", modality="SM")

    assert hasattr(dataset, "Modality")
    assert dataset.Modality == "SM"


def test_parse_study_response_creates_study_info():
    """Parse study response creates StudyInfo object."""
    from pydicom.dataset import Dataset

    engine = QueryEngine()
    endpoint = _make_endpoint()

    response_ds = Dataset()
    response_ds.StudyInstanceUID = "1.2.840.10008.1"
    response_ds.PatientID = "P001"
    response_ds.PatientName = "Test^Patient"
    response_ds.StudyDate = "20260101"
    response_ds.StudyDescription = "Test Study"
    response_ds.AccessionNumber = "ACC001"
    response_ds.NumberOfStudyRelatedSeries = "5"

    study = engine._parse_study_response(response_ds, endpoint)

    assert study is not None
    assert study.study_instance_uid == "1.2.840.10008.1"
    assert study.patient_id == "P001"
    assert study.series_count == 5


def test_parse_study_response_handles_missing_fields():
    """Parse study response handles missing optional fields."""
    from pydicom.dataset import Dataset

    engine = QueryEngine()
    endpoint = _make_endpoint()

    response_ds = Dataset()
    response_ds.StudyInstanceUID = "1.2.840.10008.1"
    response_ds.PatientID = "P001"
    response_ds.PatientName = "Test^Patient"
    response_ds.StudyDate = "20260101"
    # Missing StudyDescription, AccessionNumber, etc.

    study = engine._parse_study_response(response_ds, endpoint)

    assert study is not None
    assert study.study_instance_uid == "1.2.840.10008.1"


def test_parse_study_response_returns_none_for_incomplete_data():
    """Parse study response returns None for incomplete data."""
    from pydicom.dataset import Dataset

    engine = QueryEngine()
    endpoint = _make_endpoint()

    response_ds = Dataset()
    # Missing required fields
    response_ds.StudyDate = "20260101"

    study = engine._parse_study_response(response_ds, endpoint)

    assert study is None


def test_parse_series_response_creates_series_info():
    """Parse series response creates SeriesInfo object."""
    from pydicom.dataset import Dataset

    engine = QueryEngine()

    response_ds = Dataset()
    response_ds.SeriesInstanceUID = "1.2.840.10008.1.1"
    response_ds.SeriesNumber = "1"
    response_ds.Modality = "SM"
    response_ds.SeriesDescription = "WSI Series"
    response_ds.NumberOfSeriesRelatedInstances = "10"

    series = engine._parse_series_response(response_ds, "1.2.840.10008.1")

    assert series is not None
    assert series.series_instance_uid == "1.2.840.10008.1.1"
    assert series.instance_count == 10


def test_apply_post_query_filters_limits_results():
    """Post-query filtering limits results to max_results."""
    engine = QueryEngine()

    # Create 10 mock studies
    studies = []
    for i in range(10):
        study = StudyInfo(
            study_instance_uid=f"1.2.840.10008.{i}",
            patient_id=f"P{i:04d}",
            patient_name=f"Patient^{i}",
            study_date=datetime(2026, 1, i + 1),
            study_description=f"Study {i}",
            modality="SM",
            series_count=1,
            priority=DicomPriority.MEDIUM,
        )
        studies.append(study)

    filtered = engine._apply_post_query_filters(studies, None, max_results=5)

    assert len(filtered) == 5


def test_apply_post_query_filters_sorts_by_date():
    """Post-query filtering sorts by study date (most recent first)."""
    engine = QueryEngine()

    studies = [
        StudyInfo(
            study_instance_uid="1.2.840.10008.1",
            patient_id="P001",
            patient_name="Patient^1",
            study_date=datetime(2026, 1, 1),
            study_description="Old Study",
            modality="SM",
            series_count=1,
            priority=DicomPriority.MEDIUM,
        ),
        StudyInfo(
            study_instance_uid="1.2.840.10008.2",
            patient_id="P002",
            patient_name="Patient^2",
            study_date=datetime(2026, 1, 15),
            study_description="Recent Study",
            modality="SM",
            series_count=1,
            priority=DicomPriority.MEDIUM,
        ),
    ]

    filtered = engine._apply_post_query_filters(studies, None, max_results=10)

    # Most recent should be first
    assert filtered[0].study_date > filtered[1].study_date


def test_get_query_statistics():
    """Get query statistics returns engine info."""
    engine = QueryEngine(ae_title="TEST_AE")

    stats = engine.get_query_statistics()

    assert stats["ae_title"] == "TEST_AE"
    assert "supported_contexts" in stats
