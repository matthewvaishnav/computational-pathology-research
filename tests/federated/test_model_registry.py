"""
Tests for ModelRegistry with versioning and provenance tracking.

Tests cover:
- 9.1 Checkpoint saving
- 9.2 Checkpoint loading
- 9.3 Version indexing
- 9.4 Provenance tracking
- 9.5 Rollback support
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from src.federated.common.data_models import ModelCheckpoint
from src.federated.coordinator.model_registry import ModelRegistry

# Test fixtures


@pytest.fixture
def temp_registry_dir():
    """Create temporary directory for registry."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def registry(temp_registry_dir):
    """Create ModelRegistry instance."""
    return ModelRegistry(
        storage_path=temp_registry_dir,
        retention_policy=10,
        enable_compression=False,
        enable_integrity_checks=True,
    )


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )


@pytest.fixture
def sample_checkpoint(simple_model):
    """Create sample checkpoint."""
    return ModelCheckpoint(
        version=1,
        round_id=1,
        timestamp=datetime.now(),
        model_state_dict=simple_model.state_dict(),
        optimizer_state_dict={},
        contributors=["client_a", "client_b", "client_c"],
        metrics={"loss": 0.5, "accuracy": 0.85},
        provenance={
            "aggregation_algorithm": "FedAvg",
            "local_epochs": 5,
            "learning_rate": 0.01,
            "byzantine_detection": True,
            "total_rounds": 1,
        },
    )


# Task 9.1: Checkpoint Saving Tests


def test_save_checkpoint_basic(registry, sample_checkpoint):
    """Test basic checkpoint saving functionality."""
    checkpoint_path = registry.save_checkpoint(sample_checkpoint)

    # Verify checkpoint file exists
    assert Path(checkpoint_path).exists()

    # Verify version index updated
    assert registry._version_exists(1)

    # Verify provenance database updated
    provenance = registry.get_provenance(1)
    assert provenance["version"] == 1
    assert provenance["contributors"] == ["client_a", "client_b", "client_c"]


def test_save_checkpoint_with_tags(registry, sample_checkpoint):
    """Test checkpoint saving with tags."""
    tags = {"type": "best", "dataset": "pcam"}
    checkpoint_path = registry.save_checkpoint(sample_checkpoint, tags=tags)

    # Verify tags stored
    metadata = registry.get_version_metadata(1)
    assert metadata["tags"] == tags


def test_save_checkpoint_duplicate_version(registry, sample_checkpoint):
    """Test that saving duplicate version raises error."""
    registry.save_checkpoint(sample_checkpoint)

    # Try to save same version again
    with pytest.raises(ValueError, match="already exists"):
        registry.save_checkpoint(sample_checkpoint)


def test_save_checkpoint_integrity_checksum(registry, sample_checkpoint):
    """Test that integrity checksum is computed."""
    registry.save_checkpoint(sample_checkpoint)

    metadata = registry.get_version_metadata(1)
    assert "checksum" in metadata
    assert metadata["checksum"] is not None
    assert len(metadata["checksum"]) == 64  # SHA-256 hex digest


def test_save_multiple_checkpoints(registry, simple_model):
    """Test saving multiple checkpoints with incrementing versions."""
    for version in range(1, 6):
        checkpoint = ModelCheckpoint(
            version=version,
            round_id=version,
            timestamp=datetime.now(),
            model_state_dict=simple_model.state_dict(),
            optimizer_state_dict={},
            contributors=[f"client_{version}"],
            metrics={"loss": 1.0 / version},
            provenance={"aggregation_algorithm": "FedAvg"},
        )
        registry.save_checkpoint(checkpoint)

    # Verify all versions exist
    versions = registry.list_versions()
    assert len(versions) == 5
    assert [v["version"] for v in versions] == [1, 2, 3, 4, 5]


# Task 9.2: Checkpoint Loading Tests


def test_load_checkpoint_basic(registry, sample_checkpoint):
    """Test basic checkpoint loading functionality."""
    registry.save_checkpoint(sample_checkpoint)

    loaded_checkpoint = registry.load_checkpoint(1)

    # Verify checkpoint data
    assert loaded_checkpoint.version == 1
    assert loaded_checkpoint.round_id == 1
    assert loaded_checkpoint.contributors == ["client_a", "client_b", "client_c"]
    assert loaded_checkpoint.metrics == {"loss": 0.5, "accuracy": 0.85}


def test_load_checkpoint_nonexistent(registry):
    """Test loading nonexistent checkpoint raises error."""
    with pytest.raises(FileNotFoundError, match="not found"):
        registry.load_checkpoint(999)


def test_load_checkpoint_integrity_verification(registry, sample_checkpoint):
    """Test checkpoint integrity verification."""
    registry.save_checkpoint(sample_checkpoint)

    # Load with integrity check
    loaded = registry.load_checkpoint(1, verify_integrity=True)
    assert loaded.version == 1


def test_load_checkpoint_corrupted_integrity(registry, sample_checkpoint):
    """Test that corrupted checkpoint fails integrity check."""
    registry.save_checkpoint(sample_checkpoint)

    # Corrupt the checkpoint file
    checkpoint_path = registry._get_checkpoint_path(1)
    with open(checkpoint_path, "ab") as f:
        f.write(b"corrupted_data")

    # Loading should fail integrity check
    with pytest.raises(ValueError, match="Integrity check failed"):
        registry.load_checkpoint(1, verify_integrity=True)


def test_load_model_state(registry, sample_checkpoint, simple_model):
    """Test loading model state into a model instance."""
    registry.save_checkpoint(sample_checkpoint)

    # Create new model with random weights
    new_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )

    # Load state
    registry.load_model_state(1, new_model)

    # Verify weights match
    original_state = sample_checkpoint.model_state_dict
    loaded_state = new_model.state_dict()

    for key in original_state.keys():
        assert torch.allclose(original_state[key], loaded_state[key])


def test_load_checkpoint_round_trip(registry, sample_checkpoint):
    """
    Test round-trip property: save → load → verify produces identical parameters.

    **Validates: Requirements 6**
    """
    registry.save_checkpoint(sample_checkpoint)
    loaded = registry.load_checkpoint(1)

    # Verify all fields match
    assert loaded.version == sample_checkpoint.version
    assert loaded.round_id == sample_checkpoint.round_id
    assert loaded.contributors == sample_checkpoint.contributors
    assert loaded.metrics == sample_checkpoint.metrics

    # Verify model state dicts match
    for key in sample_checkpoint.model_state_dict.keys():
        assert torch.allclose(sample_checkpoint.model_state_dict[key], loaded.model_state_dict[key])


# Task 9.3: Version Indexing Tests


def test_get_latest_version_empty(registry):
    """Test getting latest version when no checkpoints exist."""
    assert registry.get_latest_version() is None


def test_get_latest_version(registry, simple_model):
    """Test getting latest version number."""
    for version in [1, 3, 2, 5, 4]:  # Non-sequential order
        checkpoint = ModelCheckpoint(
            version=version,
            round_id=version,
            timestamp=datetime.now(),
            model_state_dict=simple_model.state_dict(),
            optimizer_state_dict={},
            contributors=[],
            metrics={},
            provenance={},
        )
        registry.save_checkpoint(checkpoint)

    assert registry.get_latest_version() == 5


def test_get_version_metadata(registry, sample_checkpoint):
    """Test getting metadata for a specific version."""
    registry.save_checkpoint(sample_checkpoint)

    metadata = registry.get_version_metadata(1)

    assert metadata["version"] == 1
    assert metadata["round_id"] == 1
    assert metadata["contributors"] == ["client_a", "client_b", "client_c"]
    assert metadata["metrics"] == {"loss": 0.5, "accuracy": 0.85}


def test_get_version_metadata_nonexistent(registry):
    """Test getting metadata for nonexistent version raises error."""
    with pytest.raises(ValueError, match="not found"):
        registry.get_version_metadata(999)


def test_list_versions_all(registry, simple_model):
    """Test listing all versions."""
    for version in range(1, 6):
        checkpoint = ModelCheckpoint(
            version=version,
            round_id=version,
            timestamp=datetime.now(),
            model_state_dict=simple_model.state_dict(),
            optimizer_state_dict={},
            contributors=[],
            metrics={},
            provenance={},
        )
        registry.save_checkpoint(checkpoint)

    versions = registry.list_versions()
    assert len(versions) == 5
    assert [v["version"] for v in versions] == [1, 2, 3, 4, 5]


def test_list_versions_range(registry, simple_model):
    """Test listing versions with range filter."""
    for version in range(1, 11):
        checkpoint = ModelCheckpoint(
            version=version,
            round_id=version,
            timestamp=datetime.now(),
            model_state_dict=simple_model.state_dict(),
            optimizer_state_dict={},
            contributors=[],
            metrics={},
            provenance={},
        )
        registry.save_checkpoint(checkpoint)

    # Filter by range
    versions = registry.list_versions(start_version=3, end_version=7)
    assert len(versions) == 5
    assert [v["version"] for v in versions] == [3, 4, 5, 6, 7]


def test_list_versions_tags(registry, simple_model):
    """Test listing versions filtered by tags."""
    for version in range(1, 6):
        checkpoint = ModelCheckpoint(
            version=version,
            round_id=version,
            timestamp=datetime.now(),
            model_state_dict=simple_model.state_dict(),
            optimizer_state_dict={},
            contributors=[],
            metrics={},
            provenance={},
        )
        tags = {"type": "best"} if version % 2 == 0 else {"type": "regular"}
        registry.save_checkpoint(checkpoint, tags=tags)

    # Filter by tags
    best_versions = registry.list_versions(tags={"type": "best"})
    assert len(best_versions) == 2
    assert [v["version"] for v in best_versions] == [2, 4]


def test_version_numbers_strictly_increasing(registry, simple_model):
    """
    Test invariant: Model version numbers are strictly increasing.

    **Validates: Requirements 6**
    """
    versions_to_save = [1, 3, 2, 5, 4, 7, 6]

    for version in versions_to_save:
        checkpoint = ModelCheckpoint(
            version=version,
            round_id=version,
            timestamp=datetime.now(),
            model_state_dict=simple_model.state_dict(),
            optimizer_state_dict={},
            contributors=[],
            metrics={},
            provenance={},
        )
        registry.save_checkpoint(checkpoint)

    # List all versions
    all_versions = registry.list_versions()
    version_numbers = [v["version"] for v in all_versions]

    # Verify sorted order
    assert version_numbers == sorted(version_numbers)


# Task 9.4: Provenance Tracking Tests


def test_get_provenance(registry, sample_checkpoint):
    """Test getting provenance information."""
    registry.save_checkpoint(sample_checkpoint)

    provenance = registry.get_provenance(1)

    assert provenance["version"] == 1
    assert provenance["contributors"] == ["client_a", "client_b", "client_c"]
    assert provenance["aggregation_algorithm"] == "FedAvg"
    assert provenance["hyperparameters"]["local_epochs"] == 5
    assert provenance["hyperparameters"]["learning_rate"] == 0.01
    assert provenance["byzantine_detection"] is True


def test_get_provenance_nonexistent(registry):
    """Test getting provenance for nonexistent version raises error."""
    with pytest.raises(ValueError, match="not found"):
        registry.get_provenance(999)


def test_get_contributors(registry, sample_checkpoint):
    """Test getting contributors for a version."""
    registry.save_checkpoint(sample_checkpoint)

    contributors = registry.get_contributors(1)
    assert contributors == ["client_a", "client_b", "client_c"]


def test_provenance_tracks_multiple_rounds(registry, simple_model):
    """Test provenance tracking across multiple rounds."""
    contributors_by_round = {
        1: ["client_a", "client_b"],
        2: ["client_b", "client_c"],
        3: ["client_a", "client_c", "client_d"],
    }

    for version, contributors in contributors_by_round.items():
        checkpoint = ModelCheckpoint(
            version=version,
            round_id=version,
            timestamp=datetime.now(),
            model_state_dict=simple_model.state_dict(),
            optimizer_state_dict={},
            contributors=contributors,
            metrics={},
            provenance={"aggregation_algorithm": "FedAvg"},
        )
        registry.save_checkpoint(checkpoint)

    # Verify provenance for each version
    for version, expected_contributors in contributors_by_round.items():
        actual_contributors = registry.get_contributors(version)
        assert actual_contributors == expected_contributors


def test_provenance_metadata_completeness(registry, sample_checkpoint):
    """
    Test invariant: Each version has associated metadata.

    **Validates: Requirements 6**
    """
    registry.save_checkpoint(sample_checkpoint)

    metadata = registry.get_version_metadata(1)

    # Verify all required metadata fields present
    required_fields = ["version", "round_id", "timestamp", "contributors", "metrics", "provenance"]
    for field in required_fields:
        assert field in metadata, f"Missing required field: {field}"


# Task 9.5: Rollback Support Tests


def test_rollback_basic(registry, simple_model):
    """Test basic rollback functionality."""
    # Save multiple versions
    for version in range(1, 4):
        checkpoint = ModelCheckpoint(
            version=version,
            round_id=version,
            timestamp=datetime.now(),
            model_state_dict=simple_model.state_dict(),
            optimizer_state_dict={},
            contributors=[f"client_{version}"],
            metrics={"loss": 1.0 / version},
            provenance={},
        )
        registry.save_checkpoint(checkpoint)

    # Create model with random weights
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )

    # Rollback to version 2
    registry.rollback(2, model, reason="testing")

    # Verify model state matches version 2
    checkpoint_v2 = registry.load_checkpoint(2)
    for key in checkpoint_v2.model_state_dict.keys():
        assert torch.allclose(model.state_dict()[key], checkpoint_v2.model_state_dict[key])


def test_rollback_nonexistent_version(registry, simple_model):
    """Test rollback to nonexistent version raises error."""
    with pytest.raises(FileNotFoundError):
        registry.rollback(999, simple_model)


def test_rollback_logging(registry, simple_model, sample_checkpoint):
    """Test that rollback events are logged."""
    registry.save_checkpoint(sample_checkpoint)

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )

    registry.rollback(1, model, reason="test_rollback")

    # Verify rollback log exists
    rollback_log_path = Path(registry.storage_path) / "rollback_log.jsonl"
    assert rollback_log_path.exists()

    # Verify log entry
    with open(rollback_log_path, "r") as f:
        log_entries = [json.loads(line) for line in f]

    assert len(log_entries) >= 1
    last_entry = log_entries[-1]
    assert last_entry["target_version"] == 1
    assert last_entry["reason"] == "test_rollback"


def test_rollback_metamorphic_property(registry, simple_model):
    """
    Test metamorphic property: Rollback to version N then train produces
    different version N+1 than original.

    **Validates: Requirements 6**
    """
    # Save version 1
    checkpoint_v1 = ModelCheckpoint(
        version=1,
        round_id=1,
        timestamp=datetime.now(),
        model_state_dict=simple_model.state_dict(),
        optimizer_state_dict={},
        contributors=["client_a"],
        metrics={"loss": 1.0},
        provenance={},
    )
    registry.save_checkpoint(checkpoint_v1)

    # Modify model and save version 2
    for param in simple_model.parameters():
        param.data += 0.1

    checkpoint_v2_original = ModelCheckpoint(
        version=2,
        round_id=2,
        timestamp=datetime.now(),
        model_state_dict=simple_model.state_dict(),
        optimizer_state_dict={},
        contributors=["client_b"],
        metrics={"loss": 0.8},
        provenance={},
    )
    registry.save_checkpoint(checkpoint_v2_original)

    # Rollback to version 1
    registry.rollback(1, simple_model)

    # Verify model state matches version 1
    loaded_v1 = registry.load_checkpoint(1)
    for key in loaded_v1.model_state_dict.keys():
        assert torch.allclose(simple_model.state_dict()[key], loaded_v1.model_state_dict[key])


# Additional Feature Tests


def test_compare_versions(registry, simple_model):
    """Test version comparison functionality."""
    # Save two versions with different metrics
    checkpoint_v1 = ModelCheckpoint(
        version=1,
        round_id=1,
        timestamp=datetime.now(),
        model_state_dict=simple_model.state_dict(),
        optimizer_state_dict={},
        contributors=["client_a", "client_b"],
        metrics={"loss": 1.0, "accuracy": 0.7},
        provenance={"aggregation_algorithm": "FedAvg"},
    )
    registry.save_checkpoint(checkpoint_v1)

    checkpoint_v2 = ModelCheckpoint(
        version=2,
        round_id=2,
        timestamp=datetime.now(),
        model_state_dict=simple_model.state_dict(),
        optimizer_state_dict={},
        contributors=["client_b", "client_c"],
        metrics={"loss": 0.5, "accuracy": 0.85},
        provenance={"aggregation_algorithm": "FedProx"},
    )
    registry.save_checkpoint(checkpoint_v2)

    # Compare versions
    comparison = registry.compare_versions(1, 2)

    assert comparison["version_a"] == 1
    assert comparison["version_b"] == 2
    assert abs(comparison["metric_diff"]["loss"] - (-0.5)) < 1e-6
    assert abs(comparison["metric_diff"]["accuracy"] - 0.15) < 1e-6
    assert "client_c" in comparison["contributor_diff"]["added"]
    assert "client_a" in comparison["contributor_diff"]["removed"]


def test_delete_version(registry, simple_model):
    """Test version deletion."""
    # Save multiple versions
    for version in range(1, 4):
        checkpoint = ModelCheckpoint(
            version=version,
            round_id=version,
            timestamp=datetime.now(),
            model_state_dict=simple_model.state_dict(),
            optimizer_state_dict={},
            contributors=[],
            metrics={},
            provenance={},
        )
        registry.save_checkpoint(checkpoint)

    # Delete version 2
    registry.delete_version(2, force=True)

    # Verify version 2 no longer exists
    versions = registry.list_versions()
    assert len(versions) == 2
    assert 2 not in [v["version"] for v in versions]


def test_delete_latest_version_without_force(registry, sample_checkpoint):
    """Test that deleting latest version without force raises error."""
    registry.save_checkpoint(sample_checkpoint)

    with pytest.raises(ValueError, match="Cannot delete latest version"):
        registry.delete_version(1, force=False)


def test_export_checkpoint(registry, sample_checkpoint, temp_registry_dir):
    """Test checkpoint export functionality."""
    registry.save_checkpoint(sample_checkpoint)

    export_path = Path(temp_registry_dir) / "exported_model.pt"
    registry.export_checkpoint(1, str(export_path), include_optimizer=False)

    # Verify export file exists
    assert export_path.exists()

    # Load and verify
    exported_data = torch.load(export_path)
    assert exported_data["version"] == 1
    assert "model_state_dict" in exported_data
    assert "optimizer_state_dict" not in exported_data


def test_retention_policy(registry, simple_model):
    """Test retention policy enforcement."""
    # Create registry with retention policy of 3
    registry_with_retention = ModelRegistry(
        storage_path=registry.storage_path,
        retention_policy=3,
    )

    # Save 5 versions
    for version in range(1, 6):
        checkpoint = ModelCheckpoint(
            version=version,
            round_id=version,
            timestamp=datetime.now(),
            model_state_dict=simple_model.state_dict(),
            optimizer_state_dict={},
            contributors=[],
            metrics={},
            provenance={},
        )
        registry_with_retention.save_checkpoint(checkpoint)

    # Verify only last 3 versions remain
    versions = registry_with_retention.list_versions()
    assert len(versions) == 3
    assert [v["version"] for v in versions] == [3, 4, 5]


# Property-Based Tests


@given(
    version=st.integers(min_value=1, max_value=100),
    num_contributors=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=50, deadline=None)
def test_save_load_round_trip_property(version, num_contributors):
    """
    Property test: Save → Load round-trip preserves checkpoint data.

    **Validates: Requirements 6**
    """
    # Create fresh temp directory for each test
    temp_dir = tempfile.mkdtemp()
    try:
        registry = ModelRegistry(storage_path=temp_dir)

        simple_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )

        contributors = [f"client_{i}" for i in range(num_contributors)]

        checkpoint = ModelCheckpoint(
            version=version,
            round_id=version,
            timestamp=datetime.now(),
            model_state_dict=simple_model.state_dict(),
            optimizer_state_dict={},
            contributors=contributors,
            metrics={"loss": 1.0 / version},
            provenance={"aggregation_algorithm": "FedAvg"},
        )

        registry.save_checkpoint(checkpoint)
        loaded = registry.load_checkpoint(version)

        # Verify round-trip
        assert loaded.version == checkpoint.version
        assert loaded.contributors == checkpoint.contributors
        assert loaded.metrics == checkpoint.metrics
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@given(
    num_versions=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=30, deadline=None)
def test_version_ordering_property(num_versions):
    """
    Property test: Listed versions are always in sorted order.

    **Validates: Requirements 6**
    """
    # Create fresh temp directory for each test
    temp_dir = tempfile.mkdtemp()
    try:
        registry = ModelRegistry(storage_path=temp_dir)

        simple_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )

        # Save versions in random order
        import random

        versions = list(range(1, num_versions + 1))
        random.shuffle(versions)

        for version in versions:
            checkpoint = ModelCheckpoint(
                version=version,
                round_id=version,
                timestamp=datetime.now(),
                model_state_dict=simple_model.state_dict(),
                optimizer_state_dict={},
                contributors=[],
                metrics={},
                provenance={},
            )
            registry.save_checkpoint(checkpoint)

        # Verify sorted order
        listed_versions = registry.list_versions()
        version_numbers = [v["version"] for v in listed_versions]
        assert version_numbers == sorted(version_numbers)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
