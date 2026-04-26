"""Tests for PACS WSI client network resilience."""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.streaming.pacs_wsi_client import (
    PACSWSIStreamingClient,
    DownloadProgress,
    retry_with_backoff
)
from src.clinical.pacs.data_models import StudyInfo, OperationResult


@pytest.fixture
def temp_cache_dir():
    """Create temp cache dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_pacs_adapter():
    """Mock PACS adapter."""
    with patch('src.streaming.pacs_wsi_client.PACSAdapter') as mock:
        adapter = Mock()
        mock.return_value = adapter
        yield adapter


class TestNetworkResilience:
    """Tests for network resilience features."""
    
    def test_retry_decorator_success_first_try(self):
        """Test retry decorator succeeds on first try."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.1)
        def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = test_func()
        
        assert result == "success"
        assert call_count == 1
    
    def test_retry_decorator_success_after_retries(self):
        """Test retry decorator succeeds after failures."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.1)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary failure")
            return "success"
        
        result = test_func()
        
        assert result == "success"
        assert call_count == 3
    
    def test_retry_decorator_max_retries_exceeded(self):
        """Test retry decorator fails after max retries."""
        call_count = 0
        
        @retry_with_backoff(max_retries=2, base_delay=0.1)
        def test_func():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Persistent failure")
        
        with pytest.raises(RuntimeError, match="Persistent failure"):
            test_func()
        
        assert call_count == 3  # Initial + 2 retries
    
    def test_query_with_retry_success(self, temp_cache_dir, mock_pacs_adapter):
        """Test query succeeds with retry."""
        from datetime import datetime
        
        study = StudyInfo(
            study_instance_uid="1.2.3.4",
            patient_id="PAT001",
            patient_name="Test^Patient",
            study_date=datetime(2026, 4, 15),
            study_description="WSI Study",
            modality="SM",
            series_count=1
        )
        
        mock_pacs_adapter.query_studies.return_value = (
            [study],
            OperationResult.success_result("query_1", "Success")
        )
        
        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))
        
        studies = client.query_wsi_studies(patient_id="PAT001")
        
        assert len(studies) == 1
        assert studies[0].study_instance_uid == "1.2.3.4"
    
    def test_query_with_retry_failure(self, temp_cache_dir, mock_pacs_adapter):
        """Test query fails after retries."""
        mock_pacs_adapter.query_studies.return_value = (
            [],
            OperationResult.error_result("query_1", "Network timeout", ["timeout"])
        )
        
        client = PACSWSIStreamingClient(
            cache_dir=str(temp_cache_dir),
            max_retries=2
        )
        
        # Should raise after retries
        with pytest.raises(RuntimeError, match="Query failed"):
            client.query_wsi_studies(patient_id="PAT001")
    
    def test_connection_test_with_retry(self, temp_cache_dir, mock_pacs_adapter):
        """Test connection test with retry."""
        # Fail twice, succeed third time
        call_count = 0
        
        def mock_test_connection():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return OperationResult.error_result("test", "Timeout", ["timeout"])
            return OperationResult.success_result("test", "OK")
        
        mock_pacs_adapter.test_connection.side_effect = mock_test_connection
        
        client = PACSWSIStreamingClient(
            cache_dir=str(temp_cache_dir),
            max_retries=3
        )
        
        result = client.test_pacs_connection()
        
        assert result is True
        assert call_count == 3
    
    def test_graceful_interruption(self, temp_cache_dir, mock_pacs_adapter):
        """Test graceful interruption handling."""
        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))
        
        # Interrupt operations
        client.interrupt()
        
        # Operations should return early
        studies = client.query_wsi_studies(patient_id="PAT001")
        assert len(studies) == 0
        
        result = client.test_pacs_connection()
        assert result is False
        
        # Resume operations
        client.resume_operations()
        
        # Now operations should work
        mock_pacs_adapter.query_studies.return_value = (
            [],
            OperationResult.success_result("query", "OK")
        )
        
        studies = client.query_wsi_studies(patient_id="PAT001")
        # Should attempt query (even if empty)
        mock_pacs_adapter.query_studies.assert_called()
    
    def test_partial_download_marker(self, temp_cache_dir, mock_pacs_adapter):
        """Test partial download marker creation."""
        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))
        
        study_uid = "1.2.3.4"
        dest_path = temp_cache_dir / study_uid
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Save partial state
        client._save_partial_state(study_uid, dest_path)
        
        # Check marker exists
        partial_marker = temp_cache_dir / f"{study_uid}.partial"
        assert partial_marker.exists()
        
        # Check marker content
        content = partial_marker.read_text()
        assert "interrupted_at=" in content
        assert "dest_path=" in content
    
    def test_statistics_includes_resilience_config(self, temp_cache_dir, mock_pacs_adapter):
        """Test statistics include resilience config."""
        mock_pacs_adapter.get_adapter_statistics.return_value = {
            "endpoints": 1
        }
        
        client = PACSWSIStreamingClient(
            cache_dir=str(temp_cache_dir),
            max_retries=5,
            retry_base_delay=2.0,
            retry_max_delay=120.0
        )
        
        stats = client.get_statistics()
        
        assert stats["max_retries"] == 5
        assert stats["retry_config"]["base_delay"] == 2.0
        assert stats["retry_config"]["max_delay"] == 120.0
        assert "interrupted" in stats
    
    def test_shutdown_sets_interrupted_flag(self, temp_cache_dir, mock_pacs_adapter):
        """Test shutdown sets interrupted flag."""
        client = PACSWSIStreamingClient(cache_dir=str(temp_cache_dir))
        
        assert client._interrupted is False
        
        client.shutdown()
        
        assert client._interrupted is True
        mock_pacs_adapter.shutdown.assert_called_once()


class TestDownloadProgress:
    """Tests for DownloadProgress dataclass."""
    
    def test_progress_calculation(self):
        """Test progress percentage calculation."""
        progress = DownloadProgress(
            study_uid="1.2.3.4",
            total_bytes=1000,
            downloaded_bytes=250
        )
        
        assert progress.progress_percent == 25.0
    
    def test_progress_zero_total(self):
        """Test progress with zero total bytes."""
        progress = DownloadProgress(
            study_uid="1.2.3.4",
            total_bytes=0,
            downloaded_bytes=0
        )
        
        assert progress.progress_percent == 0.0
    
    def test_is_complete(self):
        """Test completion check."""
        progress = DownloadProgress(
            study_uid="1.2.3.4",
            total_bytes=1000,
            downloaded_bytes=1000
        )
        
        assert progress.is_complete is True
    
    def test_is_not_complete(self):
        """Test incomplete check."""
        progress = DownloadProgress(
            study_uid="1.2.3.4",
            total_bytes=1000,
            downloaded_bytes=500
        )
        
        assert progress.is_complete is False
