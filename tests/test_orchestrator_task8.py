"""
Tests for Task 8: Training Orchestrator Enhancements

Tests cover:
- 8.1 Round initialization with client selection
- 8.2 Model broadcasting
- 8.3 Update collection and validation
- 8.4 Aggregation trigger with Byzantine detection
- 8.5 Model versioning and provenance tracking
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Direct imports to avoid protobuf issues
from src.federated.aggregator.fedavg import FedAvgAggregator
from src.federated.common.data_models import ClientUpdate
from src.federated.coordinator.orchestrator import TrainingOrchestrator


class TinyModel(nn.Module):
    """Tiny model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


# ============================================================================
# Task 8.1: Round Initialization Tests
# ============================================================================


def test_round_initialization_with_client_selection():
    """Test round initialization with automatic client selection."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(
        model, min_clients_per_round=2, client_selection_fraction=0.5
    )

    # Register 4 clients
    for i in range(4):
        orchestrator.register_client(f"client_{i}")

    # Start round with automatic selection
    round_metadata = orchestrator.start_round()

    assert round_metadata.round_id == 1
    assert round_metadata.status == "in_progress"
    assert len(round_metadata.participants) >= 2  # At least min_clients
    assert len(round_metadata.participants) <= 4  # At most all clients
    assert orchestrator.current_round == 1


def test_round_initialization_insufficient_clients():
    """Test round initialization fails with insufficient clients."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model, min_clients_per_round=3)

    # Register only 2 clients
    orchestrator.register_client("client_0")
    orchestrator.register_client("client_1")

    # Should fail due to insufficient clients
    with pytest.raises(ValueError, match="Insufficient clients"):
        orchestrator.start_round()


def test_client_registration_and_unregistration():
    """Test client registration and unregistration."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model)

    # Register clients
    orchestrator.register_client("client_a")
    orchestrator.register_client("client_b")
    assert len(orchestrator.available_clients) == 2

    # Unregister client
    orchestrator.unregister_client("client_a")
    assert len(orchestrator.available_clients) == 1
    assert "client_b" in orchestrator.available_clients


def test_client_selection_fraction():
    """Test client selection respects selection fraction."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(
        model, min_clients_per_round=2, client_selection_fraction=0.5
    )

    # Register 10 clients
    for i in range(10):
        orchestrator.register_client(f"client_{i}")

    # Select clients
    selected = orchestrator.select_clients()

    # Should select approximately 50% (5 clients)
    assert 2 <= len(selected) <= 10
    # With 10 clients and 0.5 fraction, should select 5
    assert len(selected) == max(2, int(10 * 0.5))


# ============================================================================
# Task 8.2: Model Broadcasting Tests
# ============================================================================


def test_model_broadcasting():
    """Test model broadcasting returns correct state dict."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model)

    # Broadcast model
    model_state = orchestrator.broadcast_model()

    # Verify state dict
    assert isinstance(model_state, dict)
    assert "fc.weight" in model_state
    assert "fc.bias" in model_state
    assert model_state["fc.weight"].shape == (2, 10)
    assert model_state["fc.bias"].shape == (2,)


def test_get_global_model():
    """Test getting global model state dict."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model)

    model_state = orchestrator.get_global_model()

    # Verify state dict structure
    assert isinstance(model_state, dict)
    assert "fc.weight" in model_state
    assert "fc.bias" in model_state
    
    # Note: state_dict() returns references to parameters, not copies
    # This is expected PyTorch behavior


# ============================================================================
# Task 8.3: Update Collection Tests
# ============================================================================


def test_update_collection_and_validation():
    """Test update collection with validation."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model, min_clients_per_round=2)

    # Register and start round
    orchestrator.register_client("client_0")
    orchestrator.register_client("client_1")
    orchestrator.start_round(["client_0", "client_1"])

    # Create valid updates
    updates = []
    for i in range(2):
        gradients = {name: torch.randn_like(param) for name, param in model.named_parameters()}
        update = ClientUpdate(
            client_id=f"client_{i}",
            round_id=1,
            model_version=0,
            gradients=gradients,
            dataset_size=100,
            training_time_seconds=10.0,
        )
        updates.append(update)

    # Collect updates
    valid_updates = orchestrator.collect_updates(updates, validate=True)

    assert len(valid_updates) == 2


def test_update_validation_wrong_round():
    """Test update validation rejects wrong round ID."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model, min_clients_per_round=1)

    orchestrator.register_client("client_0")
    orchestrator.start_round(["client_0"])

    # Create update with wrong round ID
    gradients = {name: torch.randn_like(param) for name, param in model.named_parameters()}
    update = ClientUpdate(
        client_id="client_0",
        round_id=999,  # Wrong round
        model_version=0,
        gradients=gradients,
        dataset_size=100,
        training_time_seconds=10.0,
    )

    # Should be filtered out
    valid_updates = orchestrator.collect_updates([update], validate=True)
    assert len(valid_updates) == 0


def test_update_validation_inactive_client():
    """Test update validation rejects inactive client."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model, min_clients_per_round=1)

    orchestrator.register_client("client_0")
    orchestrator.register_client("client_1")
    orchestrator.start_round(["client_0"])  # Only client_0 active

    # Create update from inactive client
    gradients = {name: torch.randn_like(param) for name, param in model.named_parameters()}
    update = ClientUpdate(
        client_id="client_1",  # Not in active set
        round_id=1,
        model_version=0,
        gradients=gradients,
        dataset_size=100,
        training_time_seconds=10.0,
    )

    # Should be filtered out
    valid_updates = orchestrator.collect_updates([update], validate=True)
    assert len(valid_updates) == 0


def test_update_validation_shape_mismatch():
    """Test update validation rejects shape mismatch."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model, min_clients_per_round=1)

    orchestrator.register_client("client_0")
    orchestrator.start_round(["client_0"])

    # Create update with wrong shape
    gradients = {
        "fc.weight": torch.randn(5, 5),  # Wrong shape
        "fc.bias": torch.randn(2),
    }
    update = ClientUpdate(
        client_id="client_0",
        round_id=1,
        model_version=0,
        gradients=gradients,
        dataset_size=100,
        training_time_seconds=10.0,
    )

    # Should be filtered out
    valid_updates = orchestrator.collect_updates([update], validate=True)
    assert len(valid_updates) == 0


# ============================================================================
# Task 8.4: Aggregation Trigger Tests
# ============================================================================


def test_aggregation_with_byzantine_detection():
    """Test aggregation with Byzantine detection enabled."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model, enable_byzantine_detection=True)

    orchestrator.register_client("client_0")
    orchestrator.register_client("client_1")
    orchestrator.register_client("client_2")
    orchestrator.start_round(["client_0", "client_1", "client_2"])

    # Create normal updates
    updates = []
    for i in range(2):
        gradients = {
            name: torch.randn_like(param) * 0.1 for name, param in model.named_parameters()
        }
        update = ClientUpdate(
            client_id=f"client_{i}",
            round_id=1,
            model_version=0,
            gradients=gradients,
            dataset_size=100,
            training_time_seconds=10.0,
        )
        updates.append(update)

    # Create Byzantine update (10x larger)
    byzantine_gradients = {
        name: torch.randn_like(param) * 10.0 for name, param in model.named_parameters()
    }
    byzantine_update = ClientUpdate(
        client_id="client_2",
        round_id=1,
        model_version=0,
        gradients=byzantine_gradients,
        dataset_size=100,
        training_time_seconds=10.0,
    )
    updates.append(byzantine_update)

    # Aggregate with Byzantine detection
    aggregated = orchestrator.aggregate_updates(updates, apply_byzantine_detection=True)

    # Should successfully aggregate (Byzantine update may be filtered)
    assert isinstance(aggregated, dict)
    assert "fc.weight" in aggregated


def test_aggregation_without_byzantine_detection():
    """Test aggregation without Byzantine detection."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model, enable_byzantine_detection=False, min_clients_per_round=2)

    orchestrator.register_client("client_0")
    orchestrator.register_client("client_1")
    orchestrator.start_round(["client_0", "client_1"])

    # Create updates
    updates = []
    for i in range(2):
        gradients = {name: torch.randn_like(param) for name, param in model.named_parameters()}
        update = ClientUpdate(
            client_id=f"client_{i}",
            round_id=1,
            model_version=0,
            gradients=gradients,
            dataset_size=100,
            training_time_seconds=10.0,
        )
        updates.append(update)

    # Aggregate without Byzantine detection
    aggregated = orchestrator.aggregate_updates(updates, apply_byzantine_detection=False)

    assert isinstance(aggregated, dict)
    assert "fc.weight" in aggregated


def test_aggregation_empty_updates_fails():
    """Test aggregation fails with empty updates."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model)

    with pytest.raises(ValueError, match="No client updates"):
        orchestrator.aggregate_updates([])


# ============================================================================
# Task 8.5: Model Versioning Tests
# ============================================================================


def test_model_versioning_increment():
    """Test model version increments correctly."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model)

    assert orchestrator.current_version == 0

    # Update model
    aggregated_update = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    orchestrator.update_global_model(aggregated_update, increment_version=True)

    assert orchestrator.current_version == 1

    # Update again
    orchestrator.update_global_model(aggregated_update, increment_version=True)
    assert orchestrator.current_version == 2


def test_model_versioning_no_increment():
    """Test model update without version increment."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model)

    assert orchestrator.current_version == 0

    # Update without incrementing
    aggregated_update = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    orchestrator.update_global_model(aggregated_update, increment_version=False)

    assert orchestrator.current_version == 0


def test_checkpoint_save_and_load():
    """Test checkpoint saving and loading with versioning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        orchestrator = TrainingOrchestrator(model, checkpoint_dir=tmpdir)

        # Save initial checkpoint
        orchestrator.save_checkpoint(metrics={"loss": 0.5, "accuracy": 0.8})

        # Verify checkpoint file exists
        checkpoint_path = Path(tmpdir) / "model_v0.pt"
        assert checkpoint_path.exists()

        # Modify model
        with torch.no_grad():
            model.fc.weight.fill_(999.0)

        # Load checkpoint
        orchestrator.load_checkpoint(version=0)

        # Verify model restored
        assert not torch.allclose(model.fc.weight, torch.full_like(model.fc.weight, 999.0))


def test_checkpoint_provenance_tracking():
    """Test checkpoint includes provenance metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        orchestrator = TrainingOrchestrator(model, checkpoint_dir=tmpdir, min_clients_per_round=1)

        # Start round and save checkpoint
        orchestrator.register_client("client_0")
        orchestrator.start_round(["client_0"])
        orchestrator.save_checkpoint(metrics={"loss": 0.5})

        # Load checkpoint and verify provenance
        checkpoint_path = Path(tmpdir) / "model_v0.pt"
        checkpoint = torch.load(checkpoint_path)

        assert "provenance" in checkpoint
        assert "aggregation_algorithm" in checkpoint["provenance"]
        assert "local_epochs" in checkpoint["provenance"]
        assert "learning_rate" in checkpoint["provenance"]


def test_version_index_creation():
    """Test version index file is created and updated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        orchestrator = TrainingOrchestrator(model, checkpoint_dir=tmpdir)

        # Save multiple checkpoints
        for i in range(3):
            orchestrator.save_checkpoint(metrics={"loss": 0.5 - i * 0.1})
            aggregated_update = {
                name: torch.zeros_like(param) for name, param in model.named_parameters()
            }
            orchestrator.update_global_model(aggregated_update)

        # Verify index file
        index_path = Path(tmpdir) / "version_index.json"
        assert index_path.exists()

        with open(index_path, "r") as f:
            index = json.load(f)

        assert "versions" in index
        assert len(index["versions"]) == 3
        assert index["versions"][0]["version"] == 0
        assert index["versions"][2]["version"] == 2


# ============================================================================
# Audit Logging Tests
# ============================================================================


def test_audit_logging_enabled():
    """Test audit logging is enabled and records events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        orchestrator = TrainingOrchestrator(
            model, audit_log_dir=tmpdir, enable_audit_logging=True, min_clients_per_round=1
        )

        # Perform actions
        orchestrator.register_client("client_0")
        orchestrator.start_round(["client_0"])

        # Verify audit log
        assert len(orchestrator.audit_log) > 0
        assert any(e.event_type == "client_registered" for e in orchestrator.audit_log)
        assert any(e.event_type == "round_started" for e in orchestrator.audit_log)


def test_audit_logging_disabled():
    """Test audit logging can be disabled."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model, enable_audit_logging=False)

    # Perform actions
    orchestrator.register_client("client_0")

    # Verify no audit log
    assert len(orchestrator.audit_log) == 0


def test_audit_log_integrity_verification():
    """Test audit log integrity verification with hash chain."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model, enable_audit_logging=True)

    # Generate some events
    orchestrator.register_client("client_0")
    orchestrator.register_client("client_1")

    # Verify integrity
    assert orchestrator.verify_audit_log() is True

    # Tamper with log
    if orchestrator.audit_log:
        orchestrator.audit_log[0].details["tampered"] = True

    # Verification should fail
    assert orchestrator.verify_audit_log() is False


def test_audit_log_filtering():
    """Test audit log filtering by round and event type."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model, enable_audit_logging=True, min_clients_per_round=1)

    # Generate events across multiple rounds
    for i in range(3):
        orchestrator.register_client(f"client_{i}")
        orchestrator.start_round([f"client_{i}"])
        orchestrator.complete_round()

    # Filter by event type
    round_events = orchestrator.get_audit_log(event_type="round_started")
    assert len(round_events) == 3
    assert all(e.event_type == "round_started" for e in round_events)

    # Filter by round range
    round_1_events = orchestrator.get_audit_log(start_round=1, end_round=1)
    assert all(e.round_id == 1 for e in round_1_events)


# ============================================================================
# Integration Tests
# ============================================================================


def test_complete_training_round_workflow():
    """Test complete training round workflow (8.1-8.5)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        orchestrator = TrainingOrchestrator(
            model,
            checkpoint_dir=tmpdir,
            audit_log_dir=tmpdir,
            enable_byzantine_detection=True,
            enable_audit_logging=True,
            min_clients_per_round=2,
        )

        # 8.1: Round initialization
        orchestrator.register_client("client_0")
        orchestrator.register_client("client_1")
        round_metadata = orchestrator.start_round(["client_0", "client_1"])
        assert round_metadata.round_id == 1

        # 8.2: Model broadcasting
        model_state = orchestrator.broadcast_model()
        assert "fc.weight" in model_state

        # 8.3: Update collection
        updates = []
        for i in range(2):
            gradients = {
                name: torch.randn_like(param) * 0.1 for name, param in model.named_parameters()
            }
            update = ClientUpdate(
                client_id=f"client_{i}",
                round_id=1,
                model_version=0,
                gradients=gradients,
                dataset_size=100,
                training_time_seconds=10.0,
            )
            updates.append(update)

        valid_updates = orchestrator.collect_updates(updates, validate=True)
        assert len(valid_updates) == 2

        # 8.4: Aggregation trigger
        aggregated = orchestrator.aggregate_updates(valid_updates, apply_byzantine_detection=False)
        assert isinstance(aggregated, dict)

        # 8.5: Model versioning
        old_version = orchestrator.current_version
        orchestrator.update_global_model(aggregated, increment_version=True)
        assert orchestrator.current_version == old_version + 1

        # Complete round and save checkpoint
        orchestrator.complete_round(
            convergence_metrics={"loss": 0.5, "accuracy": 0.8}, save_checkpoint=True
        )

        # Verify checkpoint exists
        checkpoint_path = Path(tmpdir) / f"model_v{orchestrator.current_version}.pt"
        assert checkpoint_path.exists()

        # Verify audit log
        assert len(orchestrator.audit_log) > 0
        assert orchestrator.verify_audit_log() is True


def test_multiple_rounds_version_consistency():
    """Test version consistency across multiple rounds."""
    model = TinyModel()
    orchestrator = TrainingOrchestrator(model, min_clients_per_round=1)

    orchestrator.register_client("client_0")

    # Run 5 rounds
    for expected_round in range(1, 6):
        orchestrator.start_round(["client_0"])
        assert orchestrator.current_round == expected_round

        # Create and aggregate update
        gradients = {name: torch.randn_like(param) for name, param in model.named_parameters()}
        update = ClientUpdate(
            client_id="client_0",
            round_id=expected_round,
            model_version=orchestrator.current_version,
            gradients=gradients,
            dataset_size=100,
            training_time_seconds=10.0,
        )

        aggregated = orchestrator.aggregate_updates([update], apply_byzantine_detection=False)
        orchestrator.update_global_model(aggregated)
        orchestrator.complete_round()

        # Verify version incremented
        assert orchestrator.current_version == expected_round


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
