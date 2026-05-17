"""
Basic tests for nnMIL model implementation.

This test file verifies the core functionality of the nnMIL model class.
"""

import pytest
import torch

from src.models.nnmil import nnMIL


class TestnnMILBasic:
    """Basic functionality tests for nnMIL model."""

    def test_single_scale_forward(self):
        """Test single-scale forward pass."""
        model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        features = torch.randn(4, 100, 1024)
        num_patches = torch.tensor([100, 80, 90, 100])

        logits = model(features, num_patches)

        assert logits.shape == (4, 2), f"Expected shape (4, 2), got {logits.shape}"

    def test_single_scale_with_attention(self):
        """Test single-scale forward pass with attention weights."""
        model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        features = torch.randn(4, 100, 1024)
        num_patches = torch.tensor([100, 80, 90, 100])

        logits, attention = model(features, num_patches, return_attention=True)

        assert logits.shape == (4, 2), f"Expected logits shape (4, 2), got {logits.shape}"
        assert attention.shape == (
            4,
            100,
        ), f"Expected attention shape (4, 100), got {attention.shape}"

    def test_feature_projection_when_dims_differ(self):
        """Test that feature projection is created when feature_dim != hidden_dim."""
        model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        assert model.feature_proj is not None, "Feature projection should exist when dims differ"

    def test_no_feature_projection_when_dims_match(self):
        """Test that feature projection is not created when feature_dim == hidden_dim."""
        model = nnMIL(feature_dim=256, hidden_dim=256, num_classes=2)
        assert model.feature_proj is None, "Feature projection should not exist when dims match"

    def test_multi_scale_early_fusion(self):
        """Test multi-scale forward pass with early fusion."""
        model = nnMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            multi_scale=True,
            num_scales=3,
            fusion_strategy="early",
        )

        features_scale1 = torch.randn(4, 100, 1024)
        features_scale2 = torch.randn(4, 100, 1024)
        features_scale3 = torch.randn(4, 100, 1024)
        multi_scale_features = [features_scale1, features_scale2, features_scale3]
        num_patches = torch.tensor([100, 80, 90, 100])

        logits = model(multi_scale_features, num_patches)

        assert logits.shape == (4, 2), f"Expected shape (4, 2), got {logits.shape}"

    def test_multi_scale_late_fusion(self):
        """Test multi-scale forward pass with late fusion."""
        model = nnMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            multi_scale=True,
            num_scales=3,
            fusion_strategy="late",
        )

        features_scale1 = torch.randn(4, 100, 1024)
        features_scale2 = torch.randn(4, 100, 1024)
        features_scale3 = torch.randn(4, 100, 1024)
        multi_scale_features = [features_scale1, features_scale2, features_scale3]
        num_patches = torch.tensor([100, 80, 90, 100])

        logits = model(multi_scale_features, num_patches)

        assert logits.shape == (4, 2), f"Expected shape (4, 2), got {logits.shape}"

    def test_attention_masking(self):
        """Test that attention weights respect masking."""
        model = nnMIL(feature_dim=256, hidden_dim=256, num_classes=2)
        features = torch.randn(2, 100, 256)
        num_patches = torch.tensor([50, 80])  # First sample has 50 valid patches

        logits, attention = model(features, num_patches, return_attention=True)

        # Check that masked patches have near-zero attention
        # First sample: patches 50-99 should be masked
        masked_attention = attention[0, 50:].sum().item()
        assert (
            masked_attention < 0.01
        ), f"Masked patches should have ~0 attention, got {masked_attention}"

    def test_invalid_fusion_strategy(self):
        """Test that invalid fusion strategy raises error."""
        with pytest.raises(ValueError, match="fusion_strategy must be 'early' or 'late'"):
            nnMIL(feature_dim=1024, fusion_strategy="invalid")

    def test_invalid_hidden_dim(self):
        """Test that invalid hidden_dim raises error."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            nnMIL(feature_dim=1024, hidden_dim=-1)

    def test_invalid_dropout(self):
        """Test that invalid dropout raises error."""
        with pytest.raises(ValueError, match="dropout must be in"):
            nnMIL(feature_dim=1024, dropout=1.5)

    def test_multi_scale_mismatch_error(self):
        """Test that providing list features to single-scale model raises error."""
        model = nnMIL(feature_dim=1024, multi_scale=False)
        features = [torch.randn(4, 100, 1024), torch.randn(4, 100, 1024)]

        with pytest.raises(ValueError, match="not initialized with multi_scale=True"):
            model(features)

    def test_num_scales_mismatch_error(self):
        """Test that providing wrong number of scales raises error."""
        model = nnMIL(feature_dim=1024, multi_scale=True, num_scales=3)
        features = [torch.randn(4, 100, 1024), torch.randn(4, 100, 1024)]  # Only 2 scales

        with pytest.raises(ValueError, match="Expected 3 scales but received 2"):
            model(features)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = nnMIL(feature_dim=256, hidden_dim=256, num_classes=2)
        features = torch.randn(2, 50, 256, requires_grad=True)
        num_patches = torch.tensor([50, 40])

        logits = model(features, num_patches)
        loss = logits.sum()
        loss.backward()

        assert features.grad is not None, "Gradients should flow to input features"
        assert features.grad.abs().sum() > 0, "Gradients should be non-zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
