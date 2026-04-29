"""Tests for PACS workflow integration."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.clinical.pacs.data_models import OperationResult, SeriesInfo, StudyInfo
from src.streaming.pacs_wsi_client import AnalysisResult, PACSWSIStreamingClient, WorklistEntry


@pytest.fixture
def temp_cache_dir():
    """Create temp cache dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_pacs_adapter():
    """Mock PACS adapter."""
    with patch("src.streaming.pacs_wsi_client.PACSAdapter") as mock:
        adapter = Mock()
        mock.return_value = adapter
        yield adapter


@pytest.fixture
def sample_study():
    """Create sample study."""
    return StudyInfo(
        study_instance_uid="1.2.3.4",
        patient_id="PAT001",
        patient_name="Test^Patient",
        study_date=datetime(2026, 4, 15),
        study_description="WSI Study",
        modality="SM",
        series_count=2,
    )


@pytest.fixture
def sample_series():
    """Create sample series list."""
    return [
        SeriesInfo(
            series_instance_uid="1.2.3.4.1",
            study_instance_uid="1.2.3.4",
            series_number="1",
            series_description="Series 1",
            modality="SM",
            instance_count=1,
        ),
        SeriesInfo(
            series_instance_uid="1.2.3.4.2",
            study_instance_uid="1.2.3.4",
            series_number="2",
            series_description="Series 2",
            modality="SM",
            instance_count=1,
        ),
    ]


class TestSeriesProcessing:
    """Tests for series-level processing."""

    def test_query_series_for_study(self, temp_cache_dir, mock_pacs_adapter, sample_series):
        """Test query series for study."""
        mock_pacs_adapter.query_series.return_value = (
            sample_series,
            OperationResult.success_result("query", "Success"),
        )

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        series = client.query_series_for_study("1.2.3.4")

        assert len(series) == 2
        assert series[0].series_instance_uid == "1.2.3.4.1"
        assert series[1].series_instance_uid == "1.2.3.4.2"

    def test_query_series_failure(self, temp_cache_dir, mock_pacs_adapter):
        """Test series query failure."""
        mock_pacs_adapter.query_series.return_value = (
            [],
            OperationResult.error_result("query", "Failed", ["error"]),
        )

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        series = client.query_series_for_study("1.2.3.4")

        assert len(series) == 0

    def test_retrieve_series_cached(self, temp_cache_dir, mock_pacs_adapter):
        """Test retrieve series from cache."""
        # Create cached file
        cached_file = temp_cache_dir / "1.2.3.4_1.2.3.4.1.svs"
        cached_file.write_text("cached data")

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        file_path, metadata = client.retrieve_series("1.2.3.4", "1.2.3.4.1")

        assert file_path == cached_file
        assert metadata.study_uid == "1.2.3.4"
        assert metadata.series_uid == "1.2.3.4.1"
        assert metadata.bytes_downloaded > 0


class TestWorklistManagement:
    """Tests for worklist integration."""

    def test_add_to_worklist(self, temp_cache_dir, mock_pacs_adapter, sample_study, sample_series):
        """Test add study to worklist."""
        mock_pacs_adapter.query_series.return_value = (
            sample_series,
            OperationResult.success_result("query", "Success"),
        )

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        entry = client.add_to_worklist(sample_study, priority="URGENT")

        assert entry.study_uid == "1.2.3.4"
        assert entry.patient_id == "PAT001"
        assert entry.priority == "URGENT"
        assert entry.status == "PENDING"
        assert entry.series_count == 2

    def test_add_to_worklist_with_assignment(
        self, temp_cache_dir, mock_pacs_adapter, sample_study, sample_series
    ):
        """Test add study to worklist with assignment."""
        mock_pacs_adapter.query_series.return_value = (
            sample_series,
            OperationResult.success_result("query", "Success"),
        )

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        entry = client.add_to_worklist(sample_study, priority="STAT", assigned_to="pathologist1")

        assert entry.priority == "STAT"
        assert entry.assigned_to == "pathologist1"

    def test_update_worklist_status(
        self, temp_cache_dir, mock_pacs_adapter, sample_study, sample_series
    ):
        """Test update worklist status."""
        mock_pacs_adapter.query_series.return_value = (
            sample_series,
            OperationResult.success_result("query", "Success"),
        )

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        entry = client.add_to_worklist(sample_study)
        assert entry.status == "PENDING"

        client.update_worklist_status("1.2.3.4", "IN_PROGRESS")

        updated_entry = client.worklist["1.2.3.4"]
        assert updated_entry.status == "IN_PROGRESS"

    def test_get_worklist_all(self, temp_cache_dir, mock_pacs_adapter, sample_series):
        """Test get all worklist entries."""
        mock_pacs_adapter.query_series.return_value = (
            sample_series,
            OperationResult.success_result("query", "Success"),
        )

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        # Add multiple entries
        study1 = StudyInfo(
            study_instance_uid="1.2.3.4",
            patient_id="PAT001",
            patient_name="Test^Patient1",
            study_date=datetime(2026, 4, 15),
            study_description="Study 1",
            modality="SM",
            series_count=1,
        )
        study2 = StudyInfo(
            study_instance_uid="1.2.3.5",
            patient_id="PAT002",
            patient_name="Test^Patient2",
            study_date=datetime(2026, 4, 16),
            study_description="Study 2",
            modality="SM",
            series_count=1,
        )

        client.add_to_worklist(study1, priority="ROUTINE")
        client.add_to_worklist(study2, priority="URGENT")

        entries = client.get_worklist()

        assert len(entries) == 2
        # URGENT should come before ROUTINE
        assert entries[0].priority == "URGENT"
        assert entries[1].priority == "ROUTINE"

    def test_get_worklist_filtered_by_priority(
        self, temp_cache_dir, mock_pacs_adapter, sample_series
    ):
        """Test get worklist filtered by priority."""
        mock_pacs_adapter.query_series.return_value = (
            sample_series,
            OperationResult.success_result("query", "Success"),
        )

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        study1 = StudyInfo(
            study_instance_uid="1.2.3.4",
            patient_id="PAT001",
            patient_name="Test^Patient1",
            study_date=datetime(2026, 4, 15),
            study_description="Study 1",
            modality="SM",
            series_count=1,
        )
        study2 = StudyInfo(
            study_instance_uid="1.2.3.5",
            patient_id="PAT002",
            patient_name="Test^Patient2",
            study_date=datetime(2026, 4, 16),
            study_description="Study 2",
            modality="SM",
            series_count=1,
        )

        client.add_to_worklist(study1, priority="ROUTINE")
        client.add_to_worklist(study2, priority="URGENT")

        urgent_entries = client.get_worklist(priority="URGENT")

        assert len(urgent_entries) == 1
        assert urgent_entries[0].priority == "URGENT"

    def test_get_worklist_filtered_by_status(
        self, temp_cache_dir, mock_pacs_adapter, sample_series
    ):
        """Test get worklist filtered by status."""
        mock_pacs_adapter.query_series.return_value = (
            sample_series,
            OperationResult.success_result("query", "Success"),
        )

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        study1 = StudyInfo(
            study_instance_uid="1.2.3.4",
            patient_id="PAT001",
            patient_name="Test^Patient1",
            study_date=datetime(2026, 4, 15),
            study_description="Study 1",
            modality="SM",
            series_count=1,
        )

        client.add_to_worklist(study1)
        client.update_worklist_status("1.2.3.4", "COMPLETED")

        completed_entries = client.get_worklist(status="COMPLETED")

        assert len(completed_entries) == 1
        assert completed_entries[0].status == "COMPLETED"


class TestResultDelivery:
    """Tests for result delivery to PACS."""

    def test_deliver_result_to_pacs(
        self, temp_cache_dir, mock_pacs_adapter, sample_study, sample_series
    ):
        """Test deliver analysis result to PACS."""
        mock_pacs_adapter.query_series.return_value = (
            sample_series,
            OperationResult.success_result("query", "Success"),
        )

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        # Add to worklist first
        client.add_to_worklist(sample_study)
        client.update_worklist_status("1.2.3.4", "IN_PROGRESS")

        # Create result
        result = AnalysisResult(
            study_uid="1.2.3.4",
            series_uid="1.2.3.4.1",
            patient_id="PAT001",
            confidence=0.95,
            prediction="Positive",
            processing_time=25.3,
        )

        success = client.deliver_result_to_pacs(result)

        assert success is True

        # Check worklist updated
        entry = client.worklist["1.2.3.4"]
        assert entry.status == "COMPLETED"

    def test_analysis_result_to_dicom_sr(self):
        """Test convert analysis result to DICOM SR."""
        result = AnalysisResult(
            study_uid="1.2.3.4",
            series_uid="1.2.3.4.1",
            patient_id="PAT001",
            confidence=0.92,
            prediction="Negative",
            attention_weights={"region1": 0.8, "region2": 0.2},
        )

        sr_data = result.to_dicom_sr()

        assert sr_data["StudyInstanceUID"] == "1.2.3.4"
        assert sr_data["SeriesInstanceUID"] == "1.2.3.4.1"
        assert sr_data["PatientID"] == "PAT001"
        assert sr_data["CompletionFlag"] == "COMPLETE"
        assert len(sr_data["ContentSequence"]) == 2


class TestWorklistEntry:
    """Tests for WorklistEntry dataclass."""

    def test_worklist_entry_creation(self):
        """Test worklist entry creation."""
        entry = WorklistEntry(
            study_uid="1.2.3.4",
            patient_id="PAT001",
            patient_name="Test^Patient",
            study_date="20260415",
            modality="SM",
            priority="URGENT",
            status="PENDING",
            series_count=2,
        )

        assert entry.study_uid == "1.2.3.4"
        assert entry.priority == "URGENT"
        assert entry.created_at > 0
        assert entry.updated_at > 0

    def test_worklist_entry_timestamps(self):
        """Test worklist entry auto-timestamps."""
        import time

        before = time.time()
        entry = WorklistEntry(
            study_uid="1.2.3.4",
            patient_id="PAT001",
            patient_name="Test^Patient",
            study_date="20260415",
            modality="SM",
            priority="ROUTINE",
            status="PENDING",
        )
        after = time.time()

        assert before <= entry.created_at <= after
        assert before <= entry.updated_at <= after


class TestStatistics:
    """Tests for statistics with workflow data."""

    def test_statistics_includes_worklist(
        self, temp_cache_dir, mock_pacs_adapter, sample_study, sample_series
    ):
        """Test statistics include worklist data."""
        mock_pacs_adapter.query_series.return_value = (
            sample_series,
            OperationResult.success_result("query", "Success"),
        )
        mock_pacs_adapter.get_adapter_statistics.return_value = {"endpoints": 1}

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        # Add entries with different statuses
        client.add_to_worklist(sample_study, priority="URGENT")
        client.update_worklist_status("1.2.3.4", "IN_PROGRESS")

        stats = client.get_statistics()

        assert stats["worklist_entries"] == 1
        assert "worklist_by_status" in stats
        assert stats["worklist_by_status"]["IN_PROGRESS"] == 1
        assert stats["worklist_by_status"]["PENDING"] == 0
