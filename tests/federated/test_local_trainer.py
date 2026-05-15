"""
Property-based tests for Local Trainer.

Tests Task 12: Local trainer implementation
- 12.1 Model initialization from global
- 12.2 Local training loop
- 12.3 Gradient computation
- 12.4 Privacy engine integration
- 12.5 Update serialization
"""

import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st
from hypothesis import assume

from src.federated.client.trainer import LocalTrainer
from src.federated.privacy.dp_sgd import DPSGDEngine

# ============================================================================
# Test Fixtures
# ============================================================================


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel(input_dim=10, hidden_dim=20, output_dim=2)


@pytest.fixture
def local_trainer(simple_model):
    """Create a local trainer without privacy."""
    return LocalTrainer(model=simple_model, device="cpu")


@pytest.fixture
def local_trainer_with_privacy(simple_model):
    """Create a local trainer with privacy engine."""
    privacy_engine = DPSGDEngine(
        max_grad_norm=1.0,
        noise_multiplier=1.0,
        sample_rate=0.01,
        target_delta=1e-5,
        device="cpu",
    )
    return LocalTrainer(model=simple_model, privacy_engine=privacy_engine, device="cpu")


@pytest.fixture
def sample_data():
    """Create sample training data."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    return X, y


@pytest.fixture
def global_model_state(simple_model):
    """Create a global model state dict."""
    return {name: param.clone().detach() for name, param in simple_model.named_parameters()}


# ============================================================================
# Task 12.1: Model initialization from global
# ============================================================================


def test_initialize_from_global_loads_parameters(local_trainer, global_model_state):
    """
    Test that initialize_from_global correctly loads global model parameters.

    **Validates: Requirements 1.2**
    """
    # Initialize from global
    local_trainer.initialize_from_global(global_model_state)

    # Verify all parameters match
    for name, param in local_trainer.model.named_parameters():
        assert name in global_model_state
        assert torch.allclose(param.data, global_model_state[name])


def test_initialize_from_global_stores_initial_state(local_trainer, global_model_state):
    """
    Test that initialize_from_global stores initial state for gradient computation.

    **Validates: Requirements 1.2**
    """
    # Initialize from global
    local_trainer.initialize_from_global(global_model_state)

    # Verify initial state is stored
    assert local_trainer.initial_model_state is not None
    assert local_trainer.global_model_state is not None

    # Verify initial state matches global state
    for name in global_model_state:
        assert name in local_trainer.initial_model_state
        assert torch.allclose(local_trainer.initial_model_state[name], global_model_state[name])


@given(
    input_dim=st.integers(min_value=5, max_value=20),
    hidden_dim=st.integers(min_value=10, max_value=50),
    output_dim=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=10, deadline=None)
def test_initialize_from_global_property_idempotent(input_dim, hidden_dim, output_dim):
    """
    Property: Initializing twice with same state produces same result.

    **Validates: Requirements 1.2**
    """
    # Create model and trainer
    model = SimpleModel(input_dim, hidden_dim, output_dim)
    trainer = LocalTrainer(model=model, device="cpu")

    # Create global state
    global_state = {name: param.clone().detach() for name, param in model.named_parameters()}

    # Initialize twice
    trainer.initialize_from_global(global_state)
    state_after_first = {
        name: param.clone().detach() for name, param in trainer.model.named_parameters()
    }

    trainer.initialize_from_global(global_state)
    state_after_second = {
        name: param.clone().detach() for name, param in trainer.model.named_parameters()
    }

    # Verify idempotence
    for name in state_after_first:
        assert torch.allclose(state_after_first[name], state_after_second[name])


def test_initialize_from_global_raises_on_incompatible_state(local_trainer):
    """
    Test that initialize_from_global raises error on incompatible state.

    **Validates: Requirements 1.2**
    """
    # Create incompatible state (wrong shape)
    incompatible_state = {
        "fc1.weight": torch.randn(5, 5),  # Wrong shape
        "fc1.bias": torch.randn(5),
    }

    # Should raise ValueError
    with pytest.raises(ValueError):
        local_trainer.initialize_from_global(incompatible_state)


# ============================================================================
# Task 12.2: Local training loop
# ============================================================================


def test_train_local_epochs_runs_successfully(local_trainer, sample_data, global_model_state):
    """
    Test that train_local_epochs runs without errors.

    **Validates: Requirements 1.2, 2.1-2.4**
    """
    X, y = sample_data

    # Initialize and set data
    local_trainer.initialize_from_global(global_model_state)
    local_trainer.set_data(X, y)

    # Train
    metrics = local_trainer.train_local_epochs(
        num_epochs=2,
        batch_size=32,
        learning_rate=0.01,
    )

    # Verify metrics returned
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "training_time" in metrics
    assert "samples_trained" in metrics
    assert metrics["samples_trained"] == len(X)


def test_train_local_epochs_updates_model_parameters(
    local_trainer, sample_data, global_model_state
):
    """
    Test that training actually updates model parameters.

    **Validates: Requirements 1.2**
    """
    X, y = sample_data

    # Initialize and set data
    local_trainer.initialize_from_global(global_model_state)
    local_trainer.set_data(X, y)

    # Store initial parameters
    initial_params = {
        name: param.clone().detach() for name, param in local_trainer.model.named_parameters()
    }

    # Train
    local_trainer.train_local_epochs(num_epochs=2, batch_size=32, learning_rate=0.01)

    # Verify parameters changed
    parameters_changed = False
    for name, param in local_trainer.model.named_parameters():
        if not torch.allclose(param.data, initial_params[name]):
            parameters_changed = True
            break

    assert parameters_changed, "Model parameters should change after training"


@given(
    num_epochs=st.integers(min_value=1, max_value=5),
    batch_size=st.integers(min_value=8, max_value=64),
    learning_rate=st.floats(min_value=0.001, max_value=0.1),
)
@settings(max_examples=10, deadline=None)
def test_train_local_epochs_property_loss_decreases(num_epochs, batch_size, learning_rate):
    """
    Property: Training should generally decrease loss (on simple data).

    **Validates: Requirements 1.2**
    """
    # Create simple separable data
    X = torch.randn(100, 10)
    y = (X[:, 0] > 0).long()  # Simple linear separation

    # Create model and trainer
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=2)
    trainer = LocalTrainer(model=model, device="cpu")

    # Initialize
    global_state = {name: param.clone().detach() for name, param in model.named_parameters()}
    trainer.initialize_from_global(global_state)
    trainer.set_data(X, y)

    # Compute initial loss
    model.eval()
    with torch.no_grad():
        initial_output = model(X)
        initial_loss = nn.CrossEntropyLoss()(initial_output, y).item()

    # Train
    metrics = trainer.train_local_epochs(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    final_loss = metrics["loss"]

    # Loss should decrease (or at least not increase significantly)
    # Allow some tolerance for stochastic training
    assert (
        final_loss <= initial_loss * 1.5
    ), f"Loss increased too much: {initial_loss} -> {final_loss}"


def test_train_local_epochs_raises_without_data(local_trainer, global_model_state):
    """
    Test that train_local_epochs raises error if data not set.

    **Validates: Requirements 1.2**
    """
    # Initialize but don't set data
    local_trainer.initialize_from_global(global_model_state)

    # Should raise ValueError
    with pytest.raises(ValueError, match="Training data not set"):
        local_trainer.train_local_epochs(num_epochs=1)


def test_train_local_epochs_raises_without_initialization(local_trainer, sample_data):
    """
    Test that train_local_epochs raises error if model not initialized.

    **Validates: Requirements 1.2**
    """
    X, y = sample_data

    # Set data but don't initialize
    local_trainer.set_data(X, y)

    # Should raise ValueError
    with pytest.raises(ValueError, match="Model not initialized"):
        local_trainer.train_local_epochs(num_epochs=1)


# ============================================================================
# Task 12.3: Gradient computation
# ============================================================================


def test_compute_model_update_returns_gradients(local_trainer, sample_data, global_model_state):
    """
    Test that compute_model_update returns gradient dictionary.

    **Validates: Requirements 1.3**
    """
    X, y = sample_data

    # Initialize, set data, and train
    local_trainer.initialize_from_global(global_model_state)
    local_trainer.set_data(X, y)
    local_trainer.train_local_epochs(num_epochs=1, batch_size=32)

    # Compute update
    model_update = local_trainer.compute_model_update()

    # Verify structure
    assert isinstance(model_update, dict)
    assert len(model_update) > 0

    # Verify all parameters present
    for name, param in local_trainer.model.named_parameters():
        assert name in model_update
        assert model_update[name].shape == param.shape


def test_compute_model_update_is_difference(local_trainer, sample_data, global_model_state):
    """
    Test that model update is difference between trained and initial model.

    **Validates: Requirements 1.3**
    """
    X, y = sample_data

    # Initialize, set data, and train
    local_trainer.initialize_from_global(global_model_state)
    local_trainer.set_data(X, y)
    local_trainer.train_local_epochs(num_epochs=1, batch_size=32)

    # Compute update
    model_update = local_trainer.compute_model_update()

    # Verify: update = current - initial
    for name, param in local_trainer.model.named_parameters():
        expected_update = param.data - local_trainer.initial_model_state[name]
        assert torch.allclose(model_update[name], expected_update, atol=1e-6)


@given(
    num_epochs=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=5, deadline=None)
def test_compute_model_update_property_nonzero_after_training(num_epochs):
    """
    Property: Model update should be non-zero after training.

    **Validates: Requirements 1.3**
    """
    # Create data
    X = torch.randn(50, 10)
    y = torch.randint(0, 2, (50,))

    # Create model and trainer
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=2)
    trainer = LocalTrainer(model=model, device="cpu")

    # Initialize and train
    global_state = {name: param.clone().detach() for name, param in model.named_parameters()}
    trainer.initialize_from_global(global_state)
    trainer.set_data(X, y)
    trainer.train_local_epochs(num_epochs=num_epochs, batch_size=16)

    # Compute update
    model_update = trainer.compute_model_update()

    # At least one parameter should have changed
    has_nonzero_update = False
    for grad in model_update.values():
        if grad.abs().sum() > 1e-6:
            has_nonzero_update = True
            break

    assert has_nonzero_update, "Model update should be non-zero after training"


def test_compute_model_update_raises_without_initialization(local_trainer):
    """
    Test that compute_model_update raises error if not initialized.

    **Validates: Requirements 1.3**
    """
    # Should raise ValueError
    with pytest.raises(ValueError, match="No initial model state"):
        local_trainer.compute_model_update()


# ============================================================================
# Task 12.4: Privacy engine integration
# ============================================================================


def test_train_with_privacy_engine_applies_dp(
    local_trainer_with_privacy, sample_data, global_model_state
):
    """
    Test that training with privacy engine applies differential privacy.

    **Validates: Requirements 2.1-2.4**
    """
    X, y = sample_data

    # Initialize and set data
    local_trainer_with_privacy.initialize_from_global(global_model_state)
    local_trainer_with_privacy.set_data(X, y)

    # Train
    metrics = local_trainer_with_privacy.train_local_epochs(
        num_epochs=2,
        batch_size=32,
        learning_rate=0.01,
    )

    # Verify privacy metrics present
    assert "epsilon_used" in metrics
    assert "delta_used" in metrics
    assert "clipping_rate" in metrics

    # Verify epsilon is positive (privacy budget consumed)
    assert metrics["epsilon_used"] > 0


def test_train_with_privacy_engine_tracks_budget(
    local_trainer_with_privacy, sample_data, global_model_state
):
    """
    Test that privacy engine tracks privacy budget across training.

    **Validates: Requirements 2.4, 2.5**
    """
    X, y = sample_data

    # Initialize and set data
    local_trainer_with_privacy.initialize_from_global(global_model_state)
    local_trainer_with_privacy.set_data(X, y)

    # Train first round
    metrics1 = local_trainer_with_privacy.train_local_epochs(num_epochs=1, batch_size=32)
    epsilon1 = metrics1["epsilon_used"]

    # Train second round (without resetting budget)
    local_trainer_with_privacy.initialize_from_global(global_model_state)
    metrics2 = local_trainer_with_privacy.train_local_epochs(num_epochs=1, batch_size=32)
    epsilon2 = metrics2["epsilon_used"]

    # Epsilon should increase (monotonically)
    assert epsilon2 > epsilon1, "Privacy budget should increase across rounds"


@given(
    max_grad_norm=st.floats(min_value=0.5, max_value=2.0),
    noise_multiplier=st.floats(min_value=0.5, max_value=2.0),
)
@settings(max_examples=5, deadline=None)
def test_privacy_engine_property_epsilon_increases(max_grad_norm, noise_multiplier):
    """
    Property: Privacy budget (epsilon) increases with training steps.

    **Validates: Requirements 2.4**
    """
    # Create model and privacy engine
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=2)
    privacy_engine = DPSGDEngine(
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        sample_rate=0.01,
        target_delta=1e-5,
        device="cpu",
    )
    trainer = LocalTrainer(model=model, privacy_engine=privacy_engine, device="cpu")

    # Create data
    X = torch.randn(50, 10)
    y = torch.randint(0, 2, (50,))

    # Initialize and train
    global_state = {name: param.clone().detach() for name, param in model.named_parameters()}
    trainer.initialize_from_global(global_state)
    trainer.set_data(X, y)

    # Get initial epsilon
    epsilon_before, _ = privacy_engine.get_privacy_spent()

    # Train
    trainer.train_local_epochs(num_epochs=1, batch_size=16)

    # Get final epsilon
    epsilon_after, _ = privacy_engine.get_privacy_spent()

    # Epsilon should increase
    assert epsilon_after > epsilon_before


# ============================================================================
# Task 12.5: Update serialization
# ============================================================================


def test_serialize_update_returns_dict(local_trainer, sample_data, global_model_state):
    """
    Test that serialize_update returns properly structured dictionary.

    **Validates: Requirements 1.3**
    """
    X, y = sample_data

    # Initialize, set data, and train
    local_trainer.initialize_from_global(global_model_state)
    local_trainer.set_data(X, y)
    local_trainer.train_local_epochs(num_epochs=1, batch_size=32)

    # Serialize update
    serialized = local_trainer.serialize_update()

    # Verify structure
    assert isinstance(serialized, dict)
    assert "model_update" in serialized
    assert "round_id" in serialized
    assert "metadata" in serialized
    assert "update_info" in serialized


def test_serialize_update_includes_metadata(local_trainer, sample_data, global_model_state):
    """
    Test that serialize_update includes training metadata.

    **Validates: Requirements 1.3**
    """
    X, y = sample_data

    # Initialize, set data, and train
    local_trainer.initialize_from_global(global_model_state)
    local_trainer.set_data(X, y)
    local_trainer.train_local_epochs(num_epochs=2, batch_size=32, learning_rate=0.01)

    # Serialize update
    serialized = local_trainer.serialize_update(include_metadata=True)

    # Verify metadata
    metadata = serialized["metadata"]
    assert "dataset_size" in metadata
    assert "training_time" in metadata
    assert "loss" in metadata
    assert "accuracy" in metadata
    assert "epochs" in metadata
    assert metadata["dataset_size"] == len(X)
    assert metadata["epochs"] == 2


def test_serialize_update_includes_privacy_metrics(
    local_trainer_with_privacy, sample_data, global_model_state
):
    """
    Test that serialize_update includes privacy metrics when using DP.

    **Validates: Requirements 2.4**
    """
    X, y = sample_data

    # Initialize, set data, and train
    local_trainer_with_privacy.initialize_from_global(global_model_state)
    local_trainer_with_privacy.set_data(X, y)
    local_trainer_with_privacy.train_local_epochs(num_epochs=1, batch_size=32)

    # Serialize update
    serialized = local_trainer_with_privacy.serialize_update(include_metadata=True)

    # Verify privacy metrics
    assert "metadata" in serialized
    assert "privacy" in serialized["metadata"]
    privacy_metrics = serialized["metadata"]["privacy"]
    assert "epsilon_used" in privacy_metrics
    assert "delta_used" in privacy_metrics
    assert "clipping_rate" in privacy_metrics


@given(
    num_epochs=st.integers(min_value=1, max_value=3),
    batch_size=st.integers(min_value=16, max_value=64),
)
@settings(max_examples=5, deadline=None)
def test_serialize_deserialize_roundtrip(num_epochs, batch_size):
    """
    Property: Serialize then deserialize should preserve model update.

    **Validates: Requirements 1.3**
    """
    # Create model and trainer
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=2)
    trainer = LocalTrainer(model=model, device="cpu")

    # Create data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))

    # Initialize and train
    global_state = {name: param.clone().detach() for name, param in model.named_parameters()}
    trainer.initialize_from_global(global_state)
    trainer.set_data(X, y)
    trainer.train_local_epochs(num_epochs=num_epochs, batch_size=batch_size)

    # Get original update
    original_update = trainer.compute_model_update()

    # Serialize and deserialize
    serialized = trainer.serialize_update(model_update=original_update)
    deserialized_update = trainer.deserialize_update(serialized)

    # Verify round-trip preserves data
    for name in original_update:
        assert name in deserialized_update
        assert torch.allclose(original_update[name], deserialized_update[name], atol=1e-6)


def test_serialize_update_computes_size_info(local_trainer, sample_data, global_model_state):
    """
    Test that serialize_update computes update size information.

    **Validates: Requirements 1.3**
    """
    X, y = sample_data

    # Initialize, set data, and train
    local_trainer.initialize_from_global(global_model_state)
    local_trainer.set_data(X, y)
    local_trainer.train_local_epochs(num_epochs=1, batch_size=32)

    # Serialize update
    serialized = local_trainer.serialize_update()

    # Verify size info
    update_info = serialized["update_info"]
    assert "num_parameters" in update_info
    assert "size_bytes" in update_info
    assert "size_mb" in update_info
    assert update_info["num_parameters"] > 0
    assert update_info["size_bytes"] > 0
    assert update_info["size_mb"] > 0


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_training_round_workflow(local_trainer, sample_data, global_model_state):
    """
    Integration test: Full training round workflow.

    Tests the complete workflow:
    1. Initialize from global model
    2. Set training data
    3. Train local epochs
    4. Compute model update
    5. Serialize update

    **Validates: Requirements 1.2, 1.3, 2.1-2.4**
    """
    X, y = sample_data

    # Step 1: Initialize from global
    local_trainer.initialize_from_global(global_model_state)

    # Step 2: Set training data
    local_trainer.set_data(X, y)

    # Step 3: Train local epochs
    training_metrics = local_trainer.train_local_epochs(
        num_epochs=2,
        batch_size=32,
        learning_rate=0.01,
    )

    assert training_metrics["loss"] >= 0
    assert 0 <= training_metrics["accuracy"] <= 1

    # Step 4: Compute model update
    model_update = local_trainer.compute_model_update()

    assert len(model_update) > 0

    # Step 5: Serialize update
    serialized_update = local_trainer.serialize_update(model_update=model_update)

    assert "model_update" in serialized_update
    assert "metadata" in serialized_update
    assert serialized_update["metadata"]["dataset_size"] == len(X)


def test_full_training_round_with_privacy(
    local_trainer_with_privacy, sample_data, global_model_state
):
    """
    Integration test: Full training round with differential privacy.

    **Validates: Requirements 1.2, 1.3, 2.1-2.4**
    """
    X, y = sample_data

    # Initialize from global
    local_trainer_with_privacy.initialize_from_global(global_model_state)

    # Set training data
    local_trainer_with_privacy.set_data(X, y)

    # Train with privacy
    training_metrics = local_trainer_with_privacy.train_local_epochs(
        num_epochs=2,
        batch_size=32,
        learning_rate=0.01,
    )

    # Verify privacy metrics
    assert "epsilon_used" in training_metrics
    assert training_metrics["epsilon_used"] > 0

    # Compute and serialize update
    model_update = local_trainer_with_privacy.compute_model_update()
    serialized_update = local_trainer_with_privacy.serialize_update(model_update=model_update)

    # Verify privacy info in serialized update
    assert "privacy" in serialized_update["metadata"]
    assert serialized_update["metadata"]["privacy"]["epsilon_used"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
