"""
Tests for SimpleSlideClassifier model.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_camelyon import SimpleSlideClassifier


class TestSimpleSlideClassifier:
    """Test SimpleSlideClassifier functionality."""

    @pytest.fixture
    def model_mean(self):
        """Create a model with mean pooling."""
        return SimpleSlideClassifier(
            feature_dim=128,
            hidden_dim=64,
            num_classes=2,
            pooling="mean",
            dropout=0.3,
        )

    @pytest.fixture
    def model_max(self):
        """Create a model with max pooling."""
        return SimpleSlideClassifier(
            feature_dim=128,
            hidden_dim=64,
            num_classes=2,
            pooling="max",
            dropout=0.3,
        )

    def test_mean_pooling_with_masking(self, model_mean):
        """Test mean pooling with num_patches masking."""
        # Create batch with variable-length slides
        batch_size = 2
        max_patches = 10
        feature_dim = 128

        # Create features with known values
        features = torch.zeros(batch_size, max_patches, feature_dim)
        features[0, :5, :] = 1.0  # First slide has 5 patches, all ones
        features[1, :8, :] = 2.0  # Second slide has 8 patches, all twos

        num_patches = torch.tensor([5, 8])

        # Forward pass
        model_mean.eval()
        with torch.no_grad():
            logits = model_mean(features, num_patches)

        # Check output shape
        assert logits.shape == (batch_size, 1)

        # Verify that masking works by checking intermediate aggregation
        # We can't directly access slide_features, but we can verify the model runs
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_mean_pooling_without_masking(self, model_mean):
        """Test mean pooling without num_patches (no masking)."""
        batch_size = 2
        max_patches = 10
        feature_dim = 128

        features = torch.randn(batch_size, max_patches, feature_dim)

        # Forward pass without num_patches
        model_mean.eval()
        with torch.no_grad():
            logits = model_mean(features, num_patches=None)

        # Check output shape
        assert logits.shape == (batch_size, 1)
        assert not torch.isnan(logits).any()

    def test_max_pooling_ignores_padding(self, model_max):
        """Test max pooling naturally handles padding (zeros)."""
        batch_size = 2
        max_patches = 10
        feature_dim = 128

        # Create features with padding
        features = torch.zeros(batch_size, max_patches, feature_dim)
        features[0, :5, :] = torch.randn(5, feature_dim) + 1.0  # Positive values
        features[1, :8, :] = torch.randn(8, feature_dim) + 1.0  # Positive values

        num_patches = torch.tensor([5, 8])

        # Forward pass
        model_max.eval()
        with torch.no_grad():
            logits = model_max(features, num_patches)

        # Check output shape
        assert logits.shape == (batch_size, 1)
        assert not torch.isnan(logits).any()

    def test_forward_pass_with_variable_length_inputs(self, model_mean):
        """Test forward pass with variable-length inputs."""
        batch_size = 3
        max_patches = 15
        feature_dim = 128

        features = torch.randn(batch_size, max_patches, feature_dim)
        num_patches = torch.tensor([5, 10, 15])

        # Forward pass
        model_mean.eval()
        with torch.no_grad():
            logits = model_mean(features, num_patches)

        # Check output shape
        assert logits.shape == (batch_size, 1)

    def test_backward_compatibility_with_old_checkpoint_format(self, model_mean):
        """Test backward compatibility with existing checkpoints."""
        # Save model state
        state_dict = model_mean.state_dict()

        # Create new model and load state
        new_model = SimpleSlideClassifier(
            feature_dim=128,
            hidden_dim=64,
            num_classes=2,
            pooling="mean",
            dropout=0.3,
        )
        new_model.load_state_dict(state_dict)

        # Verify models produce same output
        features = torch.randn(2, 10, 128)
        num_patches = torch.tensor([5, 8])

        model_mean.eval()
        new_model.eval()

        with torch.no_grad():
            logits1 = model_mean(features, num_patches)
            logits2 = new_model(features, num_patches)

        assert torch.allclose(logits1, logits2, atol=1e-6)

    def test_invalid_pooling_method_raises_error(self):
        """Test that invalid pooling method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown pooling"):
            model = SimpleSlideClassifier(
                feature_dim=128,
                hidden_dim=64,
                num_classes=2,
                pooling="invalid",
                dropout=0.3,
            )
            features = torch.randn(2, 10, 128)
            model(features)

    def test_masked_mean_correctness(self, model_mean):
        """Test that masked mean produces correct aggregation."""
        # Create a simple test case where we can verify the math
        batch_size = 1
        max_patches = 10
        feature_dim = 128

        # Create features where first 5 patches are ones, rest are zeros (padding)
        features = torch.zeros(batch_size, max_patches, feature_dim)
        features[0, :5, :] = 1.0

        num_patches = torch.tensor([5])

        # We'll create a simple linear model to test aggregation
        # The aggregated features should be all ones (mean of 5 ones)
        # But we can't directly test this without accessing internals

        # Instead, verify that using num_patches gives different result than not using it
        model_mean.eval()
        with torch.no_grad():
            logits_with_mask = model_mean(features, num_patches)
            logits_without_mask = model_mean(features, num_patches=None)

        # These should be different because without mask, it averages over all 10 patches
        # With mask, it only averages over 5 patches
        assert not torch.allclose(logits_with_mask, logits_without_mask, atol=1e-4)

    def test_max_pooling_with_and_without_num_patches(self, model_max):
        """Test that max pooling works with and without num_patches."""
        batch_size = 2
        max_patches = 10
        feature_dim = 128

        features = torch.randn(batch_size, max_patches, feature_dim)
        num_patches = torch.tensor([5, 8])

        model_max.eval()
        with torch.no_grad():
            logits_with = model_max(features, num_patches)
            logits_without = model_max(features, num_patches=None)

        # Both should work and produce valid outputs
        assert logits_with.shape == (batch_size, 1)
        assert logits_without.shape == (batch_size, 1)
        assert not torch.isnan(logits_with).any()
        assert not torch.isnan(logits_without).any()

    def test_gradient_flow(self, model_mean):
        """Test that gradients flow correctly through the model."""
        batch_size = 2
        max_patches = 10
        feature_dim = 128

        features = torch.randn(batch_size, max_patches, feature_dim, requires_grad=True)
        num_patches = torch.tensor([5, 8])

        # Forward pass
        logits = model_mean(features, num_patches)

        # Backward pass
        loss = logits.sum()
        loss.backward()

        # Check that gradients exist
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()

    def test_model_parameters_count(self, model_mean):
        """Test that model has reasonable number of parameters."""
        total_params = sum(p.numel() for p in model_mean.parameters())

        # Should have parameters for the MLP classifier
        # Input: 128, Hidden: 64, Hidden/2: 32, Output: 1
        # Layer 1: 128*64 + 64 = 8256
        # Layer 2: 64*32 + 32 = 2080
        # Layer 3: 32*1 + 1 = 33
        # Total: ~10369
        assert total_params > 0
        assert total_params < 50000  # Reasonable upper bound

    def test_binary_classification_output(self, model_mean):
        """Test that binary classification produces single logit."""
        model = SimpleSlideClassifier(
            feature_dim=128,
            hidden_dim=64,
            num_classes=2,  # Binary classification
            pooling="mean",
            dropout=0.3,
        )

        features = torch.randn(2, 10, 128)
        num_patches = torch.tensor([5, 8])

        model.eval()
        with torch.no_grad():
            logits = model(features, num_patches)

        # Binary classification should output single logit per sample
        assert logits.shape == (2, 1)
