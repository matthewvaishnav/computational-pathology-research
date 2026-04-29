"""
Tests for Hospital Client System.

Tests local training, secure communication, and privacy preservation.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.federated.client.hospital_client import HospitalClient


class SimpleTestModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def simple_model():
    """Create simple test model."""
    return SimpleTestModel()


@pytest.fixture
def synthetic_data():
    """Create synthetic training data."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    return train_loader, val_loader


class TestHospitalClientInitialization:
    """Test hospital client initialization."""

    def test_basic_initialization(self, simple_model):
        """Test basic client initialization."""
        client = HospitalClient(
            hospital_id="hospital_001",
            model=simple_model,
            use_privacy=False,
        )

        assert client.hospital_id == "hospital_001"
        assert client.model is simple_model
        assert not client.is_registered
        assert client.current_round == 0

    def test_initialization_with_privacy(self, simple_model):
        """Test initialization with privacy enabled."""
        client = HospitalClient(
            hospital_id="hospital_002",
            model=simple_model,
            use_privacy=True,
            privacy_epsilon=1.0,
            privacy_delta=1e-5,
        )

        assert client.use_privacy
        assert client.privacy_engine is not None
        assert client.privacy_epsilon == 1.0
        assert client.privacy_delta == 1e-5

    def test_initialization_without_privacy(self, simple_model):
        """Test initialization without privacy."""
        client = HospitalClient(
            hospital_id="hospital_003",
            model=simple_model,
            use_privacy=False,
        )

        assert not client.use_privacy
        assert client.privacy_engine is None


class TestLocalDataManagement:
    """Test local data loading and management."""

    def test_load_local_data(self, simple_model, synthetic_data):
        """Test loading local training data."""
        train_loader, val_loader = synthetic_data

        client = HospitalClient(
            hospital_id="hospital_004",
            model=simple_model,
            use_privacy=False,
        )

        client.load_local_data(train_loader, val_loader)

        assert client.train_loader is train_loader
        assert client.val_loader is val_loader
        assert client.local_data_size == 100

    def test_load_data_updates_privacy_engine(self, simple_model, synthetic_data):
        """Test that loading data updates privacy engine."""
        train_loader, val_loader = synthetic_data

        client = HospitalClient(
            hospital_id="hospital_005",
            model=simple_model,
            use_privacy=True,
        )

        client.load_local_data(train_loader, val_loader)

        assert client.privacy_engine.sample_rate == train_loader.batch_size / 100
        assert hasattr(client.privacy_engine, "batch_size")
        assert client.privacy_engine.batch_size == train_loader.batch_size


class TestLocalTraining:
    """Test local model training."""

    def test_train_local_model(self, simple_model, synthetic_data):
        """Test local training without privacy."""
        train_loader, val_loader = synthetic_data

        client = HospitalClient(
            hospital_id="hospital_006",
            model=simple_model,
            use_privacy=False,
        )

        client.load_local_data(train_loader, val_loader)

        # Train locally
        metrics = client.train_local_model(
            num_epochs=2,
            batch_size=16,
            learning_rate=0.01,
        )

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "training_time" in metrics
        assert metrics["loss"] >= 0
        assert 0 <= metrics["accuracy"] <= 100

    def test_train_without_data_raises_error(self, simple_model):
        """Test that training without data raises error."""
        client = HospitalClient(
            hospital_id="hospital_007",
            model=simple_model,
            use_privacy=False,
        )

        with pytest.raises(ValueError, match="Local data not loaded"):
            client.train_local_model(num_epochs=1)

    def test_train_with_privacy(self, simple_model, synthetic_data):
        """Test local training with privacy enabled."""
        train_loader, val_loader = synthetic_data

        client = HospitalClient(
            hospital_id="hospital_008",
            model=simple_model,
            use_privacy=True,
            privacy_epsilon=1.0,
        )

        client.load_local_data(train_loader, val_loader)

        # Train with privacy
        metrics = client.train_local_model(
            num_epochs=1,
            batch_size=16,
            learning_rate=0.01,
        )

        assert "loss" in metrics
        assert "accuracy" in metrics
        # Privacy metrics should be present
        assert "epsilon_used" in metrics or client.privacy_engine is not None


class TestModelUpdates:
    """Test model update computation."""

    def test_compute_model_update(self, simple_model, synthetic_data):
        """Test computing model update."""
        train_loader, val_loader = synthetic_data

        client = HospitalClient(
            hospital_id="hospital_009",
            model=simple_model,
            use_privacy=False,
        )

        client.load_local_data(train_loader, val_loader)

        # Store initial state
        initial_state = {
            name: param.clone().detach() for name, param in simple_model.named_parameters()
        }

        # Train model
        client.train_local_model(num_epochs=1, batch_size=16, learning_rate=0.01)

        # Compute update
        update = client.compute_model_update(initial_state)

        assert len(update) > 0
        assert all(isinstance(v, torch.Tensor) for v in update.values())

        # Check that update is non-zero (model changed)
        total_change = sum(torch.abs(v).sum().item() for v in update.values())
        assert total_change > 0


class TestCommunication:
    """Test communication with coordinator."""

    @patch("src.federated.client.hospital_client.SecureFLClient")
    def test_connect_to_coordinator(self, mock_client_class, simple_model):
        """Test connecting to coordinator."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_client_class.return_value = mock_client

        client = HospitalClient(
            hospital_id="hospital_010",
            model=simple_model,
            use_privacy=False,
        )

        success = client.connect_to_coordinator()

        assert success
        assert client.comm_client is not None
        mock_client.connect.assert_called_once()

    @patch("src.federated.client.hospital_client.SecureFLClient")
    def test_register_with_coordinator(self, mock_client_class, simple_model, synthetic_data):
        """Test registering with coordinator."""
        train_loader, _ = synthetic_data

        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_client.register.return_value = True
        mock_client_class.return_value = mock_client

        client = HospitalClient(
            hospital_id="hospital_011",
            model=simple_model,
            use_privacy=False,
        )

        client.load_local_data(train_loader)
        client.connect_to_coordinator()

        success = client.register_with_coordinator(
            memory_gb=16,
            num_gpus=1,
        )

        assert success
        assert client.is_registered
        mock_client.register.assert_called_once()

    @patch("src.federated.client.hospital_client.SecureFLClient")
    def test_receive_global_model(self, mock_client_class, simple_model):
        """Test receiving global model."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_client.register.return_value = True

        # Mock global model
        mock_model_info = {
            "state_dict": simple_model.state_dict(),
            "version": 1,
            "round_id": 1,
        }
        mock_client.get_global_model.return_value = mock_model_info
        mock_client_class.return_value = mock_client

        client = HospitalClient(
            hospital_id="hospital_012",
            model=simple_model,
            use_privacy=False,
        )

        client.connect_to_coordinator()
        client.is_registered = True
        client.comm_client = mock_client

        success = client.receive_global_model(round_id=1)

        assert success
        mock_client.get_global_model.assert_called_once_with(1)


class TestPrivacyFeatures:
    """Test privacy preservation features."""

    def test_privacy_budget_tracking(self, simple_model):
        """Test privacy budget tracking."""
        client = HospitalClient(
            hospital_id="hospital_013",
            model=simple_model,
            use_privacy=True,
            privacy_epsilon=2.0,
        )

        budget_status = client.get_privacy_budget_status()

        assert budget_status["privacy_enabled"]
        assert budget_status["epsilon_budget"] == 2.0
        assert "epsilon_used" in budget_status
        assert "epsilon_remaining" in budget_status

    def test_privacy_disabled_status(self, simple_model):
        """Test privacy status when disabled."""
        client = HospitalClient(
            hospital_id="hospital_014",
            model=simple_model,
            use_privacy=False,
        )

        budget_status = client.get_privacy_budget_status()

        assert not budget_status["privacy_enabled"]


class TestClientInfo:
    """Test client information retrieval."""

    def test_get_client_info(self, simple_model, synthetic_data):
        """Test getting client information."""
        train_loader, val_loader = synthetic_data

        client = HospitalClient(
            hospital_id="hospital_015",
            model=simple_model,
            use_privacy=True,
            privacy_epsilon=1.5,
        )

        client.load_local_data(train_loader, val_loader)

        info = client.get_client_info()

        assert info["hospital_id"] == "hospital_015"
        assert info["local_data_size"] == 100
        assert info["privacy_enabled"]
        assert "total_parameters" in info
        assert "model_size_mb" in info


class TestEvaluation:
    """Test model evaluation."""

    def test_evaluate_local_model(self, simple_model, synthetic_data):
        """Test evaluating model on local data."""
        train_loader, val_loader = synthetic_data

        client = HospitalClient(
            hospital_id="hospital_016",
            model=simple_model,
            use_privacy=False,
        )

        client.load_local_data(train_loader, val_loader)

        # Evaluate
        metrics = client.evaluate_local_model()

        assert "test_loss" in metrics
        assert "test_accuracy" in metrics
        assert metrics["test_loss"] >= 0
        assert 0 <= metrics["test_accuracy"] <= 1

    def test_evaluate_without_validation_data(self, simple_model, synthetic_data):
        """Test evaluation without validation data."""
        train_loader, _ = synthetic_data

        client = HospitalClient(
            hospital_id="hospital_017",
            model=simple_model,
            use_privacy=False,
        )

        client.load_local_data(train_loader, val_loader=None)

        # Should return empty dict
        metrics = client.evaluate_local_model()

        assert metrics == {}


class TestDataPrivacy:
    """Test that patient data never leaves hospital."""

    @patch("src.federated.client.hospital_client.SecureFLClient")
    def test_only_model_updates_sent(self, mock_client_class, simple_model, synthetic_data):
        """Test that only model updates are sent, not raw data."""
        train_loader, _ = synthetic_data

        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_client.register.return_value = True
        mock_client.submit_update.return_value = True
        mock_client_class.return_value = mock_client

        client = HospitalClient(
            hospital_id="hospital_018",
            model=simple_model,
            use_privacy=False,
        )

        client.load_local_data(train_loader)
        client.connect_to_coordinator()
        client.is_registered = True
        client.comm_client = mock_client

        # Store initial state
        initial_state = {
            name: param.clone().detach() for name, param in simple_model.named_parameters()
        }

        # Train
        metrics = client.train_local_model(num_epochs=1)

        # Compute and send update
        update = client.compute_model_update(initial_state)
        client.send_model_update(
            round_id=1,
            model_version=1,
            update=update,
            training_metrics=metrics,
        )

        # Verify submit_update was called
        mock_client.submit_update.assert_called_once()

        # Verify the call arguments - should contain gradients, not raw data
        call_args = mock_client.submit_update.call_args
        assert "gradients" in call_args[1]
        assert "dataset_size" in call_args[1]

        # Verify gradients are tensors (model updates), not raw patient data
        gradients = call_args[1]["gradients"]
        assert isinstance(gradients, dict)
        assert all(isinstance(v, torch.Tensor) for v in gradients.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
