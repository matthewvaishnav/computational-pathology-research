"""
Tests for model factory function.

This module tests the create_attention_model factory function to ensure it
correctly creates all supported model types with proper configurations.
"""

import pytest
import torch

from src.models.factory import create_attention_model


class TestModelFactory:
    """Test suite for model factory function."""

    def test_create_attention_mil_model(self):
        """Test creating AttentionMIL model."""
        config = {
            "model_type": "attention_mil",
            "hidden_dim": 256,
            "num_classes": 2,
            "dropout": 0.1,
            "attention_mil": {"gated": True, "attention_mode": "instance"},
        }

        model = create_attention_model(config, feature_dim=1024)

        # Check model was created
        assert model is not None
        assert hasattr(model, "forward")

        # Test forward pass
        features = torch.randn(2, 50, 1024)
        num_patches = torch.tensor([50, 40])
        logits = model(features, num_patches)

        assert logits.shape == (2, 2)

    def test_create_clam_model(self):
        """Test creating CLAM model."""
        config = {
            "model_type": "clam",
            "hidden_dim": 256,
            "num_classes": 2,
            "dropout": 0.1,
            "clam": {
                "num_clusters": 10,
                "multi_branch": True,
                "instance_loss_weight": 0.3,
            },
        }

        model = create_attention_model(config, feature_dim=1024)

        # Check model was created
        assert model is not None
        assert hasattr(model, "forward")

        # Test forward pass
        features = torch.randn(2, 50, 1024)
        num_patches = torch.tensor([50, 40])
        logits = model(features, num_patches)

        assert logits.shape == (2, 2)

    def test_create_transmil_model(self):
        """Test creating TransMIL model."""
        config = {
            "model_type": "transmil",
            "hidden_dim": 256,
            "num_classes": 2,
            "dropout": 0.1,
            "transmil": {
                "num_layers": 2,
                "num_heads": 8,
                "use_pos_encoding": True,
            },
        }

        model = create_attention_model(config, feature_dim=1024)

        # Check model was created
        assert model is not None
        assert hasattr(model, "forward")

        # Test forward pass
        features = torch.randn(2, 50, 1024)
        num_patches = torch.tensor([50, 40])
        logits = model(features, num_patches)

        assert logits.shape == (2, 2)

    def test_create_mean_pooling_model(self):
        """Test creating mean pooling baseline model."""
        config = {
            "model_type": "mean",
            "hidden_dim": 256,
            "num_classes": 2,
            "dropout": 0.1,
        }

        model = create_attention_model(config, feature_dim=1024)

        # Check model was created
        assert model is not None
        assert hasattr(model, "forward")

        # Test forward pass
        features = torch.randn(2, 50, 1024)
        num_patches = torch.tensor([50, 40])
        logits = model(features, num_patches)

        assert logits.shape == (2, 2)

    def test_create_max_pooling_model(self):
        """Test creating max pooling baseline model."""
        config = {
            "model_type": "max",
            "hidden_dim": 256,
            "num_classes": 2,
            "dropout": 0.1,
        }

        model = create_attention_model(config, feature_dim=1024)

        # Check model was created
        assert model is not None
        assert hasattr(model, "forward")

        # Test forward pass
        features = torch.randn(2, 50, 1024)
        num_patches = torch.tensor([50, 40])
        logits = model(features, num_patches)

        assert logits.shape == (2, 2)

    def test_invalid_model_type(self):
        """Test that invalid model type raises ValueError."""
        config = {
            "model_type": "invalid_model",
            "hidden_dim": 256,
            "num_classes": 2,
        }

        with pytest.raises(ValueError, match="Invalid model_type"):
            create_attention_model(config, feature_dim=1024)

    def test_default_config_values(self):
        """Test that default config values are used when not specified."""
        config = {"model_type": "attention_mil"}

        model = create_attention_model(config)

        # Should create model with defaults
        assert model is not None

        # Test forward pass with default feature_dim=1024
        features = torch.randn(2, 50, 1024)
        logits = model(features)

        assert logits.shape == (2, 2)  # Default num_classes=2

    def test_attention_mil_with_return_attention(self):
        """Test AttentionMIL returns attention weights when requested."""
        config = {
            "model_type": "attention_mil",
            "hidden_dim": 256,
            "num_classes": 2,
        }

        model = create_attention_model(config, feature_dim=1024)

        features = torch.randn(2, 50, 1024)
        num_patches = torch.tensor([50, 40])
        logits, attention = model(features, num_patches, return_attention=True)

        assert logits.shape == (2, 2)
        assert attention.shape == (2, 50)

    def test_clam_with_return_attention(self):
        """Test CLAM returns attention weights when requested."""
        config = {
            "model_type": "clam",
            "hidden_dim": 256,
            "num_classes": 2,
            "clam": {"num_clusters": 10, "multi_branch": True},
        }

        model = create_attention_model(config, feature_dim=1024)

        features = torch.randn(2, 50, 1024)
        num_patches = torch.tensor([50, 40])
        result = model(features, num_patches, return_attention=True)

        # CLAM returns (logits, attention_weights, instance_preds)
        assert len(result) == 3
        logits, attention, instance_preds = result

        assert logits.shape == (2, 2)
        assert instance_preds.shape == (2, 50, 10)  # num_clusters=10

    def test_transmil_with_return_attention(self):
        """Test TransMIL returns attention weights when requested."""
        config = {
            "model_type": "transmil",
            "hidden_dim": 256,
            "num_classes": 2,
            "transmil": {"num_layers": 2, "num_heads": 8},
        }

        model = create_attention_model(config, feature_dim=1024)

        features = torch.randn(2, 50, 1024)
        num_patches = torch.tensor([50, 40])
        logits, attention = model(features, num_patches, return_attention=True)

        assert logits.shape == (2, 2)
        assert attention.shape == (2, 50)

    def test_pooling_with_return_attention(self):
        """Test pooling models return uniform attention when requested."""
        config = {
            "model_type": "mean",
            "hidden_dim": 256,
            "num_classes": 2,
        }

        model = create_attention_model(config, feature_dim=1024)

        features = torch.randn(2, 50, 1024)
        num_patches = torch.tensor([50, 40])
        logits, attention = model(features, num_patches, return_attention=True)

        assert logits.shape == (2, 2)
        assert attention.shape == (2, 50)

        # Check attention is normalized
        assert torch.allclose(attention.sum(dim=1), torch.ones(2), atol=1e-6)

    def test_custom_feature_dim(self):
        """Test creating model with custom feature dimension."""
        config = {
            "model_type": "attention_mil",
            "hidden_dim": 128,
            "num_classes": 3,
        }

        model = create_attention_model(config, feature_dim=512)

        # Test with custom feature_dim
        features = torch.randn(2, 50, 512)
        logits = model(features)

        assert logits.shape == (2, 3)

    def test_custom_num_classes(self):
        """Test creating model with custom number of classes."""
        config = {
            "model_type": "attention_mil",
            "hidden_dim": 256,
            "num_classes": 5,
        }

        model = create_attention_model(config, feature_dim=1024)

        features = torch.randn(2, 50, 1024)
        logits = model(features)

        assert logits.shape == (2, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
