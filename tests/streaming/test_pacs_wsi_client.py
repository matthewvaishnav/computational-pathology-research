"""Tests for PACS WSI streaming client."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.clinical.pacs.data_models import OperationResult, StudyInfo
from src.streaming.pacs_wsi_client import PACSWSIMetadata, PACSWSIStreamingClient


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


class TestPACSWSIStreamingClient:
    """Tests for PACSWSIStreamingClient."""

    def test_init(self, temp_cache_dir, mock_pacs_adapter):
        """Test client init."""
        client = PACSWSIStreamingClient(
            pacs_config_profile="development", cache_dir=str(temp_cache_dir), ae_title="TEST_AE"
        )

        assert client.cache_dir == temp_cache_dir
        assert client.cache_dir.exists()
        assert len(client.retrieved_studies) == 0

    def test_query_wsi_studies_success(self, temp_cache_dir, mock_pacs_adapter):
        """Test successful WSI query."""
        from datetime import datetime

        # Setup mock
        study1 = StudyInfo(
            study_instance_uid="1.2.3.4",
            patient_id="PAT001",
            patient_name="Test^Patient",
            study_date=datetime(2026, 4, 15),
            study_description="WSI Study 1",
            modality="SM",
            series_count=1,
        )
        study2 = StudyInfo(
            study_instance_uid="1.2.3.5",
            patient_id="PAT001",
            patient_name="Test^Patient",
            study_date=datetime(2026, 4, 16),
            study_description="WSI Study 2",
            modality="SM",
            series_count=1,
        )

        mock_pacs_adapter.query_studies.return_value = (
            [study1, study2],
            OperationResult.success_result("query_1", "Success"),
        )

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        # Query
        studies = client.query_wsi_studies(patient_id="PAT001")

        assert len(studies) == 2
        assert studies[0].study_instance_uid == "1.2.3.4"
        assert studies[1].study_instance_uid == "1.2.3.5"

    def test_query_wsi_studies_failure(self, temp_cache_dir, mock_pacs_adapter):
        """Test failed WSI query."""
        mock_pacs_adapter.query_studies.return_value = (
            [],
            OperationResult.error_result("query_1", "Connection failed", ["timeout"]),
        )

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        studies = client.query_wsi_studies(patient_id="PAT001")

        assert len(studies) == 0

    def test_test_pacs_connection_success(self, temp_cache_dir, mock_pacs_adapter):
        """Test successful PACS connection test."""
        mock_pacs_adapter.test_connection.return_value = OperationResult.success_result(
            "test_1", "Connection OK"
        )

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        result = client.test_pacs_connection()

        assert result is True

    def test_test_pacs_connection_failure(self, temp_cache_dir, mock_pacs_adapter):
        """Test failed PACS connection test."""
        mock_pacs_adapter.test_connection.return_value = OperationResult.error_result(
            "test_1", "Connection failed", ["timeout"]
        )

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        result = client.test_pacs_connection()

        assert result is False

    def test_get_endpoint_status(self, temp_cache_dir, mock_pacs_adapter):
        """Test get endpoint status."""
        mock_status = {
            "endpoint_1": {
                "host": "pacs.hospital.com",
                "port": 11112,
                "connection_status": "online",
            }
        }
        mock_pacs_adapter.get_endpoint_status.return_value = mock_status

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        status = client.get_endpoint_status()

        assert status == mock_status

    def test_clear_cache_all(self, temp_cache_dir, mock_pacs_adapter):
        """Test clear all cache."""
        # Create dummy files
        (temp_cache_dir / "file1.svs").touch()
        (temp_cache_dir / "file2.tiff").touch()

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        client.clear_cache()

        # Cache dir should exist but be empty
        assert client.cache_dir.exists()
        assert len(list(client.cache_dir.iterdir())) == 0

    def test_clear_cache_old_files(self, temp_cache_dir, mock_pacs_adapter):
        """Test clear old cache files."""
        import time

        # Create old file
        old_file = temp_cache_dir / "old.svs"
        old_file.touch()

        # Set old mtime (2 hours ago)
        old_time = time.time() - (2 * 3600)
        import os

        os.utime(old_file, (old_time, old_time))

        # Create new file
        new_file = temp_cache_dir / "new.svs"
        new_file.touch()

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        # Clear files older than 1 hour
        client.clear_cache(older_than_hours=1)

        # Old file should be deleted, new file should remain
        assert not old_file.exists()
        assert new_file.exists()

    def test_get_statistics(self, temp_cache_dir, mock_pacs_adapter):
        """Test get statistics."""
        mock_pacs_adapter.get_adapter_statistics.return_value = {
            "endpoints_configured": 2,
            "active_operations": 0,
        }

        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        stats = client.get_statistics()

        assert "cache_dir" in stats
        assert "retrieved_studies" in stats
        assert "pacs_adapter" in stats
        assert stats["retrieved_studies"] == 0

    def test_context_manager(self, temp_cache_dir, mock_pacs_adapter):
        """Test context manager."""
        with PACSWSIStreamingClient(cache_dir=str(temp_cache_dir)) as client:
            assert client is not None

        # Shutdown should be called
        mock_pacs_adapter.shutdown.assert_called_once()

    def test_shutdown(self, temp_cache_dir, mock_pacs_adapter):
        """Test shutdown."""
        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))

        client.shutdown()

        mock_pacs_adapter.shutdown.assert_called_once()


class TestPACSWSIMetadata:
    """Tests for PACSWSIMetadata."""

    def test_metadata_creation(self):
        """Test metadata dataclass."""
        metadata = PACSWSIMetadata(
            study_uid="1.2.3.4",
            series_uid="1.2.3.4.5",
            patient_id="PAT001",
            study_date="20260415",
            modality="SM",
            file_path=Path("/tmp/test.svs"),
            retrieval_time=5.2,
        )

        assert metadata.study_uid == "1.2.3.4"
        assert metadata.patient_id == "PAT001"
        assert metadata.retrieval_time == 5.2
