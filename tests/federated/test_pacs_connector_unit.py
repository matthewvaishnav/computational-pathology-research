"""
Unit Tests for PACS Connector (without protobuf dependencies).

Tests WSI discovery, data loading, preprocessing, and incremental updates.

**Validates: Requirements 5.1-5.7**
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.features.clinical.workflow.pacs.data_models import DicomPriority, StudyInfo

# Direct import to avoid protobuf issues
from src.features.federated.pathology_fl.client.pacs_connector import PACSConnector

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_pacs_service():
    """Create mock PACS service."""
    service = MagicMock()
    service._is_running = False
    service.audit_logger = MagicMock()
    service.failover_manager = MagicMock()
    service.pacs_adapter = MagicMock()
    service.pacs_adapter.query_engine = MagicMock()
    service.pacs_adapter.retrieval_engine = MagicMock()
    return service


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def pacs_connector(mock_pacs_service, temp_cache_dir):
    """Create PACS connector with mocked service."""
    with patch("src.federated.client.pacs_connector.PACSService") as mock_service_class:
        mock_service_class.return_value = mock_pacs_service

        connector = PACSConnector(
            pacs_config_path=Path(".kiro/pacs/config.yaml"),
            profile="default",
            cache_dir=temp_cache_dir,
        )

        yield connector


def create_mock_study(
    study_uid: str,
    patient_id: str,
    study_date: datetime,
    modality: str = "SM",
) -> StudyInfo:
    """Create mock StudyInfo object."""
    return StudyInfo(
        study_instance_uid=study_uid,
        patient_id=patient_id,
        patient_name=f"Patient_{patient_id}",
        study_date=study_date,
        study_description="WSI Study",
        modality=modality,
        series_count=1,
        priority=DicomPriority.MEDIUM,
    )


# ============================================================================
# Unit Tests: Basic Functionality
# ============================================================================


class TestBasicFunctionality:
    """Unit tests for basic PACS connector functionality."""

    def test_initialization(self, temp_cache_dir):
        """Test basic initialization."""
        with patch("src.federated.client.pacs_connector.PACSService") as mock_service:
            connector = PACSConnector(
                pacs_config_path=Path(".kiro/pacs/config.yaml"),
                profile="default",
                cache_dir=temp_cache_dir,
            )

            assert connector.profile == "default"
            assert connector.cache_dir == temp_cache_dir
            assert connector.last_query_timestamp is None
            assert mock_service.called

    def test_context_manager(self, pacs_connector, mock_pacs_service):
        """Test context manager protocol."""
        with pacs_connector as conn:
            assert mock_pacs_service.start.called

        assert mock_pacs_service.shutdown.called

    def test_clear_cache(self, pacs_connector, temp_cache_dir):
        """Test cache clearing."""
        # Create some cache files
        cache_file = temp_cache_dir / "test.pt"
        torch.save(torch.randn(3, 224, 224), cache_file)

        assert cache_file.exists()

        # Clear cache
        pacs_connector.clear_cache()

        # Verify cache is empty
        assert not cache_file.exists()
        assert temp_cache_dir.exists()

    def test_get_statistics(self, pacs_connector, mock_pacs_service):
        """Test statistics retrieval."""
        stats = pacs_connector.get_statistics()

        assert "pacs_profile" in stats
        assert "cache_dir" in stats
        assert "cached_studies" in stats
        assert "last_query_timestamp" in stats
        assert stats["pacs_profile"] == "default"


# ============================================================================
# Unit Tests: Study Discovery
# ============================================================================


class TestStudyDiscovery:
    """Unit tests for WSI study discovery."""

    def test_discover_wsi_studies_basic(self, pacs_connector, mock_pacs_service):
        """Test basic study discovery."""
        # Generate mock studies
        mock_studies = [
            create_mock_study(
                study_uid=f"1.2.3.{i}",
                patient_id=f"P{i:04d}",
                study_date=datetime.now() - timedelta(days=i),
            )
            for i in range(5)
        ]

        # Mock query response
        mock_pacs_service.pacs_adapter.query_engine.query_studies.return_value = mock_studies
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        # Execute discovery
        study_uids = pacs_connector.discover_wsi_studies(max_results=10)

        # Verify results
        assert len(study_uids) == 5
        assert all(uid.startswith("1.2.3.") for uid in study_uids)

        # Verify query was called with correct parameters
        call_args = mock_pacs_service.pacs_adapter.query_engine.query_studies.call_args
        assert call_args[1]["modality"] == "SM"
        assert call_args[1]["max_results"] == 10

    def test_discover_wsi_studies_respects_max_results(self, pacs_connector, mock_pacs_service):
        """Test that max_results parameter is respected."""
        # Generate 20 mock studies
        mock_studies = [
            create_mock_study(
                study_uid=f"1.2.3.{i}",
                patient_id=f"P{i:04d}",
                study_date=datetime.now() - timedelta(days=i),
            )
            for i in range(20)
        ]

        mock_pacs_service.pacs_adapter.query_engine.query_studies.return_value = mock_studies
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        # Execute discovery with max_results=10
        study_uids = pacs_connector.discover_wsi_studies(max_results=10)

        # Verify result count respects max_results
        assert len(study_uids) <= 10

    def test_discover_wsi_studies_with_date_range(self, pacs_connector, mock_pacs_service):
        """Test study discovery with date range."""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        mock_studies = [
            create_mock_study(
                study_uid=f"1.2.3.{i}",
                patient_id=f"P{i:04d}",
                study_date=datetime.now() - timedelta(days=i),
            )
            for i in range(10)
        ]

        mock_pacs_service.pacs_adapter.query_engine.query_studies.return_value = mock_studies
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        # Execute discovery
        study_uids = pacs_connector.discover_wsi_studies(
            start_date=start_date,
            end_date=end_date,
        )

        # Verify query was called with date range
        call_args = mock_pacs_service.pacs_adapter.query_engine.query_studies.call_args
        assert call_args[1]["study_date_range"] == (start_date, end_date)

    def test_date_range_validation(self, pacs_connector):
        """Test that invalid date ranges raise ValueError."""
        now = datetime.now()
        start_date = now
        end_date = now - timedelta(days=1)

        with pytest.raises(ValueError, match="start_date must be before end_date"):
            pacs_connector.discover_wsi_studies(
                start_date=start_date,
                end_date=end_date,
            )

    def test_audit_logging_for_discovery(self, pacs_connector, mock_pacs_service):
        """Test that study discovery is logged to audit system."""
        mock_studies = [
            create_mock_study(
                study_uid=f"1.2.3.{i}",
                patient_id=f"P{i:04d}",
                study_date=datetime.now() - timedelta(days=i),
            )
            for i in range(3)
        ]

        mock_pacs_service.pacs_adapter.query_engine.query_studies.return_value = mock_studies
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        # Execute discovery
        pacs_connector.discover_wsi_studies()

        # Verify audit log was called
        assert mock_pacs_service.audit_logger.log_query.called

        # Verify audit log parameters
        call_args = mock_pacs_service.audit_logger.log_query.call_args
        assert call_args[1]["query_type"] == "study_discovery"
        assert call_args[1]["parameters"]["modality"] == "SM"
        assert call_args[1]["result_count"] == 3


# ============================================================================
# Unit Tests: Incremental Updates
# ============================================================================


class TestIncrementalUpdates:
    """Unit tests for incremental data updates."""

    def test_incremental_query_uses_last_timestamp(self, pacs_connector, mock_pacs_service):
        """Test that incremental query uses last query timestamp."""
        now = datetime.now()

        # First query
        initial_studies = [
            create_mock_study(
                study_uid=f"1.2.3.{i}",
                patient_id=f"P{i:04d}",
                study_date=now - timedelta(days=i),
            )
            for i in range(5)
        ]

        mock_pacs_service.pacs_adapter.query_engine.query_studies.return_value = initial_studies
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        pacs_connector.discover_wsi_studies()
        last_timestamp = pacs_connector.last_query_timestamp
        assert last_timestamp is not None

        # Second query (incremental)
        new_studies = [
            create_mock_study(
                study_uid=f"1.2.3.new.{i}",
                patient_id=f"P_NEW_{i:04d}",
                study_date=now + timedelta(hours=i),
            )
            for i in range(3)
        ]

        mock_pacs_service.pacs_adapter.query_engine.query_studies.return_value = new_studies

        pacs_connector.discover_wsi_studies(incremental=True)

        # Verify query used last timestamp
        call_args = mock_pacs_service.pacs_adapter.query_engine.query_studies.call_args
        assert call_args[1]["study_date_range"][0] == last_timestamp

    def test_get_incremental_updates_convenience_method(self, pacs_connector, mock_pacs_service):
        """Test get_incremental_updates convenience method."""
        # Set up last query timestamp
        pacs_connector.last_query_timestamp = datetime.now() - timedelta(days=1)

        mock_studies = [
            create_mock_study(
                study_uid=f"1.2.3.{i}",
                patient_id=f"P{i:04d}",
                study_date=datetime.now(),
            )
            for i in range(2)
        ]

        mock_pacs_service.pacs_adapter.query_engine.query_studies.return_value = mock_studies
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        # Get incremental updates
        study_uids = pacs_connector.get_incremental_updates()

        assert len(study_uids) == 2


# ============================================================================
# Unit Tests: Data Loading and Preprocessing
# ============================================================================


class TestDataLoadingAndPreprocessing:
    """Unit tests for WSI data loading and preprocessing."""

    def test_load_wsi_data_basic(self, pacs_connector, mock_pacs_service, temp_cache_dir):
        """Test basic WSI data loading."""
        study_uid = "1.2.3.test"
        target_size = (224, 224)

        # Create mock image file
        mock_image = Image.new("RGB", (512, 512), color="red")
        image_path = temp_cache_dir / study_uid / "image.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mock_image.save(image_path)

        # Mock retrieval result
        mock_result = MagicMock()
        mock_result.file_paths = [image_path]
        mock_result.total_size_bytes = 1024

        mock_pacs_service.pacs_adapter.retrieval_engine.retrieve_study.return_value = mock_result
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        # Load and preprocess
        tensor = pacs_connector.load_wsi_data(study_uid, target_size=target_size)

        # Verify tensor properties
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_preprocessed_tensor_shape(self, pacs_connector, mock_pacs_service, temp_cache_dir):
        """Test that preprocessed tensor has correct shape."""
        study_uid = "1.2.3.shape_test"
        target_size = (128, 256)  # (height, width)

        # Create mock image
        mock_image = Image.new("RGB", (512, 512), color="blue")
        image_path = temp_cache_dir / study_uid / "image.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mock_image.save(image_path)

        # Mock retrieval
        mock_result = MagicMock()
        mock_result.file_paths = [image_path]
        mock_result.total_size_bytes = 1024

        mock_pacs_service.pacs_adapter.retrieval_engine.retrieve_study.return_value = mock_result
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        # Load with custom target size
        tensor = pacs_connector.load_wsi_data(study_uid, target_size=target_size)

        # Verify shape matches target_size [C, H, W]
        # Note: PIL resize expects (width, height), so we need to swap
        assert tensor.shape == (3, target_size[0], target_size[1])

    def test_normalization_applied(self, pacs_connector, mock_pacs_service, temp_cache_dir):
        """Test that normalization is applied correctly."""
        study_uid = "1.2.3.norm_test"

        # Create mock image
        mock_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        image_path = temp_cache_dir / study_uid / "image.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mock_image.save(image_path)

        # Mock retrieval
        mock_result = MagicMock()
        mock_result.file_paths = [image_path]
        mock_result.total_size_bytes = 1024

        mock_pacs_service.pacs_adapter.retrieval_engine.retrieve_study.return_value = mock_result
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        # Load with normalization
        tensor_normalized = pacs_connector.load_wsi_data(study_uid, normalize=True)

        # Normalized values should be centered around 0
        mean_val = tensor_normalized.mean().item()
        assert -2.0 < mean_val < 2.0

    def test_caching_mechanism(self, pacs_connector, mock_pacs_service, temp_cache_dir):
        """Test that caching works correctly."""
        study_uid = "1.2.3.cache_test"

        # Create mock image
        mock_image = Image.new("RGB", (224, 224), color="green")
        image_path = temp_cache_dir / study_uid / "image.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mock_image.save(image_path)

        # Mock retrieval
        mock_result = MagicMock()
        mock_result.file_paths = [image_path]
        mock_result.total_size_bytes = 1024

        mock_pacs_service.pacs_adapter.retrieval_engine.retrieve_study.return_value = mock_result
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        # First load - should retrieve from PACS
        tensor1 = pacs_connector.load_wsi_data(study_uid)
        assert mock_pacs_service.pacs_adapter.retrieval_engine.retrieve_study.call_count == 1

        # Second load - should use cache
        tensor2 = pacs_connector.load_wsi_data(study_uid)
        assert mock_pacs_service.pacs_adapter.retrieval_engine.retrieve_study.call_count == 1

        # Verify cached tensor equals original
        assert torch.allclose(tensor1, tensor2)

    def test_batch_loading(self, pacs_connector, mock_pacs_service, temp_cache_dir):
        """Test batch loading of multiple studies."""
        study_uids = [f"1.2.3.batch.{i}" for i in range(3)]

        # Create mock images for each study
        for study_uid in study_uids:
            mock_image = Image.new("RGB", (224, 224), color="yellow")
            image_path = temp_cache_dir / study_uid / "image.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mock_image.save(image_path)

            # Mock retrieval for this study
            mock_result = MagicMock()
            mock_result.file_paths = [image_path]
            mock_result.total_size_bytes = 1024

            mock_pacs_service.pacs_adapter.retrieval_engine.retrieve_study.return_value = (
                mock_result
            )

        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        # Batch load
        tensors = pacs_connector.load_batch_wsi_data(study_uids)

        # Verify results
        assert len(tensors) == 3
        for tensor in tensors:
            assert tensor.shape == (3, 224, 224)

    def test_audit_logging_for_retrieval(self, pacs_connector, mock_pacs_service, temp_cache_dir):
        """Test that data retrieval is logged to audit system."""
        study_uid = "1.2.3.audit_test"

        # Create mock image
        mock_image = Image.new("RGB", (224, 224), color="purple")
        image_path = temp_cache_dir / study_uid / "image.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mock_image.save(image_path)

        # Mock retrieval
        mock_result = MagicMock()
        mock_result.file_paths = [image_path]
        mock_result.total_size_bytes = 2048

        mock_pacs_service.pacs_adapter.retrieval_engine.retrieve_study.return_value = mock_result
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        # Load data
        pacs_connector.load_wsi_data(study_uid)

        # Verify audit log was called
        assert mock_pacs_service.audit_logger.log_retrieval.called

        # Verify audit log parameters
        call_args = mock_pacs_service.audit_logger.log_retrieval.call_args
        assert call_args[1]["study_instance_uid"] == study_uid
        assert call_args[1]["file_count"] == 1
        assert call_args[1]["total_size_bytes"] == 2048
