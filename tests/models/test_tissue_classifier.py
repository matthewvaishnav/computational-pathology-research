"""
Tests for tissue classifier module.
"""

import pytest
import torch

from src.models.tissue_classifier import PretrainedTissueClassifier, TissueClassifier


class TestTissueClassifier:
    """Test TissueClassifier functionality."""

    def test_initialization(self):
        """Test classifier initialization."""
        classifier = TissueClassifier(feature_dim=1024)

        assert classifier.feature_dim == 1024
        assert classifier.hidden_dim == 128
        assert classifier.num_tissue_types == 4
        assert classifier.dropout == 0.1

    def test_forward_2d(self):
        """Test forward pass with 2D input."""
        classifier = TissueClassifier(feature_dim=1024)
        features = torch.randn(100, 1024)

        logits = classifier(features)

        assert logits.shape == (100, 4)

    def test_forward_3d(self):
        """Test forward pass with 3D input."""
        classifier = TissueClassifier(feature_dim=1024)
        features = torch.randn(4, 100, 1024)

        logits = classifier(features)

        assert logits.shape == (4, 100, 4)

    def test_get_importance_weights_2d(self):
        """Test importance weight computation with 2D input."""
        classifier = TissueClassifier(feature_dim=1024)
        features = torch.randn(100, 1024)

        importance = classifier.get_importance_weights(features)

        assert importance.shape == (100,)
        assert torch.all(importance >= 0)

    def test_get_importance_weights_3d(self):
        """Test importance weight computation with 3D input."""
        classifier = TissueClassifier(feature_dim=1024)
        features = torch.randn(4, 100, 1024)

        importance = classifier.get_importance_weights(features)

        assert importance.shape == (4, 100)
        assert torch.all(importance >= 0)

    def test_temperature_scaling(self):
        """Test temperature parameter affects importance distribution."""
        classifier = TissueClassifier(feature_dim=1024)
        features = torch.randn(100, 1024)

        # Low temperature = more peaked distribution
        importance_low = classifier.get_importance_weights(features, temperature=0.1)

        # High temperature = more uniform distribution
        importance_high = classifier.get_importance_weights(features, temperature=10.0)

        # Low temp should have higher variance
        assert importance_low.std() > importance_high.std()

    def test_set_importance_weights(self):
        """Test setting custom importance weights."""
        classifier = TissueClassifier(feature_dim=1024)

        # Set custom weights
        custom_weights = torch.tensor([0.05, 0.2, 0.2, 2.0])
        classifier.set_importance_weights(custom_weights)

        assert torch.allclose(classifier.importance_weights, custom_weights)

    def test_predict_tissue_types(self):
        """Test tissue type prediction."""
        classifier = TissueClassifier(feature_dim=1024)
        features = torch.randn(100, 1024)

        tissue_types = classifier.predict_tissue_types(features)

        assert tissue_types.shape == (100,)
        assert torch.all(tissue_types >= 0)
        assert torch.all(tissue_types < 4)

    def test_get_tissue_distribution(self):
        """Test tissue distribution computation."""
        classifier = TissueClassifier(feature_dim=1024)
        features = torch.randn(100, 1024)

        probs = classifier.get_tissue_distribution(features)

        assert probs.shape == (100, 4)
        assert torch.allclose(probs.sum(dim=1), torch.ones(100))
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)

    def test_invalid_feature_dim(self):
        """Test initialization with invalid feature_dim."""
        with pytest.raises(ValueError, match="feature_dim must be positive"):
            TissueClassifier(feature_dim=-1)

    def test_invalid_hidden_dim(self):
        """Test initialization with invalid hidden_dim."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            TissueClassifier(feature_dim=1024, hidden_dim=0)

    def test_invalid_num_tissue_types(self):
        """Test initialization with invalid num_tissue_types."""
        with pytest.raises(ValueError, match="num_tissue_types must be positive"):
            TissueClassifier(feature_dim=1024, num_tissue_types=0)

    def test_invalid_dropout(self):
        """Test initialization with invalid dropout."""
        with pytest.raises(ValueError, match="dropout must be in"):
            TissueClassifier(feature_dim=1024, dropout=1.5)

    def test_invalid_input_dim(self):
        """Test forward pass with invalid input dimensions."""
        classifier = TissueClassifier(feature_dim=1024)
        features = torch.randn(10, 20, 30, 1024)  # 4D input

        with pytest.raises(ValueError, match="Expected 2D"):
            classifier(features)

    def test_invalid_importance_weights_shape(self):
        """Test setting importance weights with wrong shape."""
        classifier = TissueClassifier(feature_dim=1024, num_tissue_types=4)

        with pytest.raises(ValueError, match="Expected 4 weights"):
            classifier.set_importance_weights(torch.tensor([1.0, 2.0]))


class TestPretrainedTissueClassifier:
    """Test PretrainedTissueClassifier functionality."""

    def test_initialization_without_checkpoint(self):
        """Test initialization without checkpoint."""
        classifier = PretrainedTissueClassifier(feature_dim=1024)

        assert classifier.feature_dim == 1024

    def test_initialization_with_nonexistent_checkpoint(self):
        """Test initialization with nonexistent checkpoint."""
        # Should not raise, just warn
        classifier = PretrainedTissueClassifier(feature_dim=1024, checkpoint_path="nonexistent.pth")

        assert classifier.feature_dim == 1024

    def test_forward_pass(self):
        """Test forward pass works same as base class."""
        classifier = PretrainedTissueClassifier(feature_dim=1024)
        features = torch.randn(100, 1024)

        logits = classifier(features)

        assert logits.shape == (100, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
