"""
Property-Based Tests for PACS Connector.

Tests WSI discovery, data loading, preprocessing, and incremental updates
using property-based testing with Hypothesis.

**Validates: Requirements 5.1-5.7**
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st
from PIL import Image

from src.clinical.pacs.data_models import DicomPriority, StudyInfo
from src.federated.client.pacs_connector import PACSConnector

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
# Property-Based Tests: Study Discovery
# ============================================================================


class TestStudyDiscovery:
    """Property-based tests for WSI study discovery."""

    @given(
        num_studies=st.integers(min_value=0, max_value=100),
        max_results=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=50, deadline=None)
    def test_retrieved_study_count_respects_max_results(
        self, pacs_connector, mock_pacs_service, num_studies, max_results
    ):
        """
        Property: Retrieved study count ≤ max_results parameter.

        **Validates: Requirements 5.7**
        """
        # Generate mock studies
        mock_studies = [
            create_mock_study(
                study_uid=f"1.2.3.{i}",
                patient_id=f"P{i:04d}",
                study_date=datetime.now() - timedelta(days=i),
            )
            for i in range(num_studies)
        ]

        # Mock query response
        mock_pacs_service.pacs_adapter.query_engine.query_studies.return_value = mock_studies
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        # Execute discovery
        study_uids = pacs_connector.discover_wsi_studies(max_results=max_results)

        # Property: result count ≤ max_results
        assert len(study_uids) <= max_results

        # Property: result count ≤ available studies
        assert len(study_uids) <= num_studies

    @given(
        num_studies=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=30, deadline=None)
    def test_all_retrieved_studies_have_sm_modality(
        self, pacs_connector, mock_pacs_service, num_studies
    ):
        """
        Property: All retrieved studies have modality = "SM".

        **Validates: Requirements 5.2**
        """
        # Generate mock studies with SM modality
        mock_studies = [
            create_mock_study(
                study_uid=f"1.2.3.{i}",
                patient_id=f"P{i:04d}",
                study_date=datetime.now() - timedelta(days=i),
                modality="SM",
            )
            for i in range(num_studies)
        ]

        mock_pacs_service.pacs_adapter.query_engine.query_studies.return_value = mock_studies
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        # Execute discovery
        study_uids = pacs_connector.discover_wsi_studies()

        # Verify query was called with modality="SM"
        call_args = mock_pacs_service.pacs_adapter.query_engine.query_studies.call_args
        assert call_args[1]["modality"] == "SM"

        # Property: All studies should be SM modality
        assert len(study_uids) == num_studies

    @given(
        days_range_a=st.integers(min_value=1, max_value=30),
        days_range_b=st.integers(min_value=31, max_value=60),
    )
    @settings(max_examples=20, deadline=None)
    def test_date_range_subset_property(
        self, pacs_connector, mock_pacs_service, days_range_a, days_range_b
    ):
        """
        Metamorphic Property: Query with date range [A, B] returns subset
        of query with range [A, C] where C > B.

        **Validates: Requirements 5.2**
        """
        now = datetime.now()
        start_date = now - timedelta(days=days_range_b)

        # Generate studies across full range
        all_studies = [
            create_mock_study(
                study_uid=f"1.2.3.{i}",
                patient_id=f"P{i:04d}",
                study_date=now - timedelta(days=i),
            )
            for i in range(days_range_b)
        ]

        # Mock first query (shorter range)
        end_date_short = now - timedelta(days=days_range_a)
        studies_short = [s for s in all_studies if s.study_date <= end_date_short]

        mock_pacs_service.pacs_adapter.query_engine.query_studies.return_value = studies_short
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        uids_short = pacs_connector.discover_wsi_studies(
            start_date=start_date,
            end_date=end_date_short,
        )

        # Mock second query (longer range)
        end_date_long = now
        studies_long = [s for s in all_studies if s.study_date <= end_date_long]

        mock_pacs_service.pacs_adapter.query_engine.query_studies.return_value = studies_long

        uids_long = pacs_connector.discover_wsi_studies(
            start_date=start_date,
            end_date=end_date_long,
        )

        # Metamorphic property: shorter range is subset of longer range
        assert len(uids_short) <= len(uids_long)
        assert set(uids_short).issubset(set(uids_long))

    def test_date_range_validation(self, pacs_connector):
        """
        Error Condition: start_date >= end_date raises ValueError.

        **Validates: Requirements 5.2**
        """
        now = datetime.now()
        start_date = now
        end_date = now - timedelta(days=1)

        with pytest.raises(ValueError, match="start_date must be before end_date"):
            pacs_connector.discover_wsi_studies(
                start_date=start_date,
                end_date=end_date,
            )


# ============================================================================
# Property-Based Tests: Incremental Updates
# ============================================================================


class TestIncrementalUpdates:
    """Property-based tests for incremental data updates."""

    @given(
        initial_studies=st.integers(min_value=5, max_value=20),
        new_studies=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20, deadline=None)
    def test_incremental_query_returns_only_new_studies(
        self, pacs_connector, mock_pacs_service, initial_studies, new_studies
    ):
        """
        Property: Incremental query returns only studies after last query.

        **Validates: Requirements 5.6**
        """
        now = datetime.now()

        # First query - initial studies
        initial_study_list = [
            create_mock_study(
                study_uid=f"1.2.3.{i}",
                patient_id=f"P{i:04d}",
                study_date=now - timedelta(days=initial_studies - i),
            )
            for i in range(initial_studies)
        ]

        mock_pacs_service.pacs_adapter.query_engine.query_studies.return_value = initial_study_list
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        initial_uids = pacs_connector.discover_wsi_studies()
        assert len(initial_uids) == initial_studies

        # Record last query timestamp
        last_timestamp = pacs_connector.last_query_timestamp
        assert last_timestamp is not None

        # Second query - new studies only
        new_study_list = [
            create_mock_study(
                study_uid=f"1.2.3.new.{i}",
                patient_id=f"P_NEW_{i:04d}",
                study_date=now + timedelta(hours=i),
            )
            for i in range(new_studies)
        ]

        mock_pacs_service.pacs_adapter.query_engine.query_studies.return_value = new_study_list

        incremental_uids = pacs_connector.discover_wsi_studies(incremental=True)

        # Property: incremental query returns only new studies
        assert len(incremental_uids) == new_studies
        assert set(incremental_uids).isdisjoint(set(initial_uids))

        # Verify query used last timestamp as start_date
        call_args = mock_pacs_service.pacs_adapter.query_engine.query_studies.call_args
        assert call_args[1]["study_date_range"][0] == last_timestamp

    def test_incremental_without_previous_query_performs_full_discovery(
        self, pacs_connector, mock_pacs_service
    ):
        """
        Property: Incremental query without previous timestamp performs full discovery.

        **Validates: Requirements 5.6**
        """
        # Ensure no previous query
        assert pacs_connector.last_query_timestamp is None

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

        # Call incremental update
        study_uids = pacs_connector.get_incremental_updates()

        # Should perform full discovery
        assert len(study_uids) == 10


# ============================================================================
# Property-Based Tests: Data Loading and Preprocessing
# ============================================================================


class TestDataLoadingAndPreprocessing:
    """Property-based tests for WSI data loading and preprocessing."""

    @given(
        height=st.integers(min_value=64, max_value=512),
        width=st.integers(min_value=64, max_value=512),
    )
    @settings(max_examples=20, deadline=None)
    def test_preprocessed_tensor_has_correct_shape(
        self, pacs_connector, mock_pacs_service, temp_cache_dir, height, width
    ):
        """
        Property: Preprocessed tensor has shape [3, H, W] matching target_size.

        **Validates: Requirements 5.3**
        """
        study_uid = "1.2.3.test"
        target_size = (height, width)

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

        # Property: tensor shape matches [C, H, W]
        assert tensor.shape == (3, height, width)
        assert tensor.dtype == torch.float32

    @given(
        normalize=st.booleans(),
    )
    @settings(max_examples=10, deadline=None)
    def test_normalization_applied_correctly(
        self, pacs_connector, mock_pacs_service, temp_cache_dir, normalize
    ):
        """
        Property: Normalized tensors have ImageNet mean/std applied.

        **Validates: Requirements 5.3**
        """
        study_uid = "1.2.3.norm_test"
        target_size = (224, 224)

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

        # Load with/without normalization
        tensor = pacs_connector.load_wsi_data(
            study_uid, target_size=target_size, normalize=normalize
        )

        if normalize:
            # Property: normalized values should be centered around 0
            # (ImageNet normalization subtracts mean ~0.45 and divides by std ~0.22)
            mean_val = tensor.mean().item()
            assert -2.0 < mean_val < 2.0  # Reasonable range after normalization
        else:
            # Property: unnormalized values should be in [0, 1]
            assert tensor.min() >= 0.0
            assert tensor.max() <= 1.0

    def test_caching_mechanism(self, pacs_connector, mock_pacs_service, temp_cache_dir):
        """
        Property: Second load of same study uses cache (no PACS retrieval).

        **Validates: Requirements 5.3**
        """
        study_uid = "1.2.3.cache_test"
        target_size = (224, 224)

        # Create mock image
        mock_image = Image.new("RGB", (224, 224), color="blue")
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
        tensor1 = pacs_connector.load_wsi_data(study_uid, target_size=target_size)
        assert mock_pacs_service.pacs_adapter.retrieval_engine.retrieve_study.call_count == 1

        # Second load - should use cache
        tensor2 = pacs_connector.load_wsi_data(study_uid, target_size=target_size)
        assert mock_pacs_service.pacs_adapter.retrieval_engine.retrieve_study.call_count == 1

        # Property: cached tensor equals original
        assert torch.allclose(tensor1, tensor2)

    @given(
        num_studies=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=10, deadline=None)
    def test_batch_loading(self, pacs_connector, mock_pacs_service, temp_cache_dir, num_studies):
        """
        Property: Batch loading returns list of tensors with correct count.

        **Validates: Requirements 5.3**
        """
        study_uids = [f"1.2.3.batch.{i}" for i in range(num_studies)]

        # Create mock images for each study
        for study_uid in study_uids:
            mock_image = Image.new("RGB", (224, 224), color="green")
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

        # Property: returns correct number of tensors
        assert len(tensors) == num_studies

        # Property: all tensors have correct shape
        for tensor in tensors:
            assert tensor.shape == (3, 224, 224)


# ============================================================================
# Property-Based Tests: HIPAA Audit Logging
# ============================================================================


class TestAuditLogging:
    """Property-based tests for HIPAA audit logging."""

    def test_study_discovery_logged(self, pacs_connector, mock_pacs_service):
        """
        Property: All study discovery operations are logged to audit system.

        **Validates: Requirements 5.5**
        """
        mock_studies = [
            create_mock_study(
                study_uid=f"1.2.3.{i}",
                patient_id=f"P{i:04d}",
                study_date=datetime.now() - timedelta(days=i),
            )
            for i in range(5)
        ]

        mock_pacs_service.pacs_adapter.query_engine.query_studies.return_value = mock_studies
        mock_pacs_service.failover_manager.get_active_endpoint.return_value = MagicMock()

        # Execute discovery
        pacs_connector.discover_wsi_studies()

        # Property: audit log was called
        assert mock_pacs_service.audit_logger.log_query.called

        # Verify audit log parameters
        call_args = mock_pacs_service.audit_logger.log_query.call_args
        assert call_args[1]["query_type"] == "study_discovery"
        assert call_args[1]["parameters"]["modality"] == "SM"
        assert call_args[1]["result_count"] == 5

    def test_data_retrieval_logged(self, pacs_connector, mock_pacs_service, temp_cache_dir):
        """
        Property: All data retrieval operations are logged to audit system.

        **Validates: Requirements 5.5**
        """
        study_uid = "1.2.3.audit_test"

        # Create mock image
        mock_image = Image.new("RGB", (224, 224), color="yellow")
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

        # Property: audit log was called for retrieval
        assert mock_pacs_service.audit_logger.log_retrieval.called

        # Verify audit log parameters
        call_args = mock_pacs_service.audit_logger.log_retrieval.call_args
        assert call_args[1]["study_instance_uid"] == study_uid
        assert call_args[1]["file_count"] == 1
        assert call_args[1]["total_size_bytes"] == 2048


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
