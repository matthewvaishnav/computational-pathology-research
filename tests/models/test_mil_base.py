"""
Unit tests for MIL Base Class.

Tests the MILBase class that provides common functionality for MIL models:
- Attention computation
- Feature aggregation
- Multimodal fusion application
"""

import pytest
import torch

from src.models.components.attention_mechanisms import (
    GatedAttention,
    SimpleAttention,
    TransformerAttention,
)
from src.models.components.fusion_strategies import EarlyFusion, LateFusion
from src.models.mil.mil_base import MILBase


class TestMILBaseInitialization:
    """Tests for MILBase initialization."""

    def test_initialization_without_fusion(self):
        """Test MILBase can be initialized without fusion strategy."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=1024, num_classes=2, attention=attention, fusion=None)

        assert model.feature_dim == 1024
        assert model.num_classes == 2
        assert model.attention is attention
        assert model.fusion is None

    def test_initialization_with_early_fusion(self):
        """Test MILBase can be initialized with early fusion."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
        model = MILBase(feature_dim=1024, num_classes=2, attention=attention, fusion=fusion)

        assert model.feature_dim == 1024
        assert model.num_classes == 2
        assert model.attention is attention
        assert model.fusion is fusion

    def test_initialization_with_late_fusion(self):
        """Test MILBase can be initialized with late fusion."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        fusion = LateFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
        model = MILBase(feature_dim=1024, num_classes=2, attention=attention, fusion=fusion)

        assert model.feature_dim == 1024
        assert model.num_classes == 2
        assert model.attention is attention
        assert model.fusion is fusion

    def test_initialization_with_different_attention_mechanisms(self):
        """Test MILBase works with different attention mechanisms."""
        # GatedAttention
        gated = GatedAttention(feature_dim=256, hidden_dim=256)
        model1 = MILBase(feature_dim=256, num_classes=2, attention=gated)
        assert model1.attention is gated

        # SimpleAttention
        simple = SimpleAttention(feature_dim=256, hidden_dim=128)
        model2 = MILBase(feature_dim=256, num_classes=2, attention=simple)
        assert model2.attention is simple

        # TransformerAttention
        transformer = TransformerAttention(feature_dim=256, num_heads=8, num_layers=2)
        model3 = MILBase(feature_dim=256, num_classes=2, attention=transformer)
        assert model3.attention is transformer


class TestComputeAttention:
    """Tests for compute_attention method."""

    def test_compute_attention_with_gated_attention(self):
        """Test compute_attention with GatedAttention mechanism."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        features = torch.randn(4, 100, 256)
        attention_weights = model.compute_attention(features)

        assert attention_weights.shape == (4, 100)
        assert (attention_weights >= 0).all()
        # Check weights sum to 1 for each sample
        for i in range(4):
            assert torch.allclose(attention_weights[i].sum(), torch.tensor(1.0), atol=1e-6)

    def test_compute_attention_with_simple_attention(self):
        """Test compute_attention with SimpleAttention mechanism."""
        attention = SimpleAttention(feature_dim=256, hidden_dim=128)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        features = torch.randn(4, 100, 256)
        attention_weights = model.compute_attention(features)

        assert attention_weights.shape == (4, 100)
        assert (attention_weights >= 0).all()

    def test_compute_attention_with_transformer_attention(self):
        """Test compute_attention with TransformerAttention mechanism."""
        attention = TransformerAttention(feature_dim=256, num_heads=8, num_layers=2)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        features = torch.randn(4, 100, 256)
        output = model.compute_attention(features)

        # Transformer returns transformed features (includes CLS token)
        assert output.shape == (4, 101, 256)

    def test_compute_attention_with_mask(self):
        """Test compute_attention respects mask."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        features = torch.randn(4, 100, 256)
        mask = torch.ones(4, 100, dtype=torch.bool)
        mask[0, 80:] = False  # First sample has only 80 valid patches

        attention_weights = model.compute_attention(features, mask)

        # Masked patches should have ~0 weight
        assert torch.allclose(attention_weights[0, 80:].sum(), torch.tensor(0.0), atol=1e-6)
        # Valid patches should sum to ~1.0
        assert torch.allclose(attention_weights[0, :80].sum(), torch.tensor(1.0), atol=1e-6)

    def test_compute_attention_different_batch_sizes(self):
        """Test compute_attention with different batch sizes."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        for batch_size in [1, 2, 8, 16]:
            features = torch.randn(batch_size, 100, 256)
            attention_weights = model.compute_attention(features)
            assert attention_weights.shape == (batch_size, 100)

    def test_compute_attention_different_num_patches(self):
        """Test compute_attention with different numbers of patches."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        for num_patches in [10, 50, 100, 500]:
            features = torch.randn(4, num_patches, 256)
            attention_weights = model.compute_attention(features)
            assert attention_weights.shape == (4, num_patches)


class TestAggregateFeatures:
    """Tests for aggregate_features method."""

    def test_aggregate_features_basic(self):
        """Test basic feature aggregation."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        features = torch.randn(4, 100, 256)
        attention_weights = torch.softmax(torch.randn(4, 100), dim=1)

        aggregated = model.aggregate_features(features, attention_weights)

        assert aggregated.shape == (4, 256)
        assert not torch.isnan(aggregated).any()

    def test_aggregate_features_uniform_weights(self):
        """Test aggregation with uniform weights equals mean."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        features = torch.randn(4, 100, 256)
        # Uniform weights: each patch gets equal weight
        attention_weights = torch.ones(4, 100) / 100

        aggregated = model.aggregate_features(features, attention_weights)
        expected = features.mean(dim=1)

        assert torch.allclose(aggregated, expected, atol=1e-5)

    def test_aggregate_features_single_patch_weight(self):
        """Test aggregation with all weight on single patch."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        features = torch.randn(4, 100, 256)
        # All weight on patch 42
        attention_weights = torch.zeros(4, 100)
        attention_weights[:, 42] = 1.0

        aggregated = model.aggregate_features(features, attention_weights)
        expected = features[:, 42, :]

        assert torch.allclose(aggregated, expected, atol=1e-6)

    def test_aggregate_features_different_batch_sizes(self):
        """Test aggregation with different batch sizes."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        for batch_size in [1, 2, 8, 16]:
            features = torch.randn(batch_size, 100, 256)
            attention_weights = torch.softmax(torch.randn(batch_size, 100), dim=1)
            aggregated = model.aggregate_features(features, attention_weights)
            assert aggregated.shape == (batch_size, 256)

    def test_aggregate_features_different_feature_dims(self):
        """Test aggregation with different feature dimensions."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        for feature_dim in [128, 256, 512, 1024]:
            features = torch.randn(4, 100, feature_dim)
            attention_weights = torch.softmax(torch.randn(4, 100), dim=1)
            aggregated = model.aggregate_features(features, attention_weights)
            assert aggregated.shape == (4, feature_dim)

    def test_aggregate_features_gradient_flow(self):
        """Test gradients flow through aggregation."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        features = torch.randn(4, 100, 256, requires_grad=True)
        attention_weights = torch.softmax(torch.randn(4, 100), dim=1)

        aggregated = model.aggregate_features(features, attention_weights)
        loss = aggregated.sum()
        loss.backward()

        assert features.grad is not None
        assert features.grad.abs().sum() > 0


class TestApplyFusion:
    """Tests for apply_fusion method."""

    def test_apply_fusion_without_fusion_strategy(self):
        """Test apply_fusion returns input unchanged when no fusion configured."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=1024, num_classes=2, attention=attention, fusion=None)

        features = torch.randn(4, 100, 1024)
        output = model.apply_fusion(features)

        assert torch.equal(output, features)

    def test_apply_fusion_with_early_fusion(self):
        """Test apply_fusion with early fusion strategy."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
        model = MILBase(feature_dim=1024, num_classes=2, attention=attention, fusion=fusion)

        scale1 = torch.randn(4, 100, 1024)
        scale2 = torch.randn(4, 100, 1024)
        features = [scale1, scale2]

        fused = model.apply_fusion(features)

        assert isinstance(fused, torch.Tensor)
        assert fused.shape == (4, 100, 256)

    def test_apply_fusion_with_late_fusion(self):
        """Test apply_fusion with late fusion strategy."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        fusion = LateFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
        model = MILBase(feature_dim=1024, num_classes=2, attention=attention, fusion=fusion)

        scale1 = torch.randn(4, 100, 1024)
        scale2 = torch.randn(4, 100, 1024)
        features = [scale1, scale2]

        projected = model.apply_fusion(features)

        assert isinstance(projected, list)
        assert len(projected) == 2
        assert projected[0].shape == (4, 100, 256)
        assert projected[1].shape == (4, 100, 256)

    def test_apply_fusion_single_tensor_wrapped(self):
        """Test apply_fusion wraps single tensor in list for fusion."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=1)
        model = MILBase(feature_dim=1024, num_classes=2, attention=attention, fusion=fusion)

        # Pass single tensor instead of list
        features = torch.randn(4, 100, 1024)
        fused = model.apply_fusion(features)

        assert isinstance(fused, torch.Tensor)
        assert fused.shape == (4, 100, 256)

    def test_apply_fusion_with_mask(self):
        """Test apply_fusion passes mask to fusion strategy."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
        model = MILBase(feature_dim=1024, num_classes=2, attention=attention, fusion=fusion)

        scale1 = torch.randn(4, 100, 1024)
        scale2 = torch.randn(4, 100, 1024)
        features = [scale1, scale2]

        mask = torch.ones(4, 100, dtype=torch.bool)
        mask[0, 80:] = False

        fused = model.apply_fusion(features, mask)

        assert fused.shape == (4, 100, 256)


class TestForwardPass:
    """Tests for forward method."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        features = torch.randn(4, 100, 256)
        output = model(features)

        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()

    def test_forward_return_attention(self):
        """Test forward pass returns attention weights when requested."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        features = torch.randn(4, 100, 256)
        output = model(features, return_attention=True)

        assert isinstance(output, dict)
        assert "features" in output
        assert "attention_weights" in output
        assert output["features"].shape == (4, 256)
        assert output["attention_weights"].shape == (4, 100)

    def test_forward_with_mask(self):
        """Test forward pass with mask."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        features = torch.randn(4, 100, 256)
        mask = torch.ones(4, 100, dtype=torch.bool)
        mask[0, 80:] = False

        output = model(features, mask=mask, return_attention=True)

        # Masked patches should have ~0 attention weight
        assert torch.allclose(
            output["attention_weights"][0, 80:].sum(), torch.tensor(0.0), atol=1e-6
        )

    def test_forward_with_early_fusion(self):
        """Test forward pass with early fusion."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
        model = MILBase(feature_dim=1024, num_classes=2, attention=attention, fusion=fusion)

        # For this test, we pass a single tensor (will be wrapped in list)
        features = torch.randn(4, 100, 1024)
        output = model(features)

        assert output.shape == (4, 256)

    def test_forward_gradient_flow(self):
        """Test gradients flow through forward pass."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        features = torch.randn(4, 100, 256, requires_grad=True)
        output = model(features)
        loss = output.sum()
        loss.backward()

        assert features.grad is not None
        assert features.grad.abs().sum() > 0

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        for batch_size in [1, 2, 8, 16]:
            features = torch.randn(batch_size, 100, 256)
            output = model(features)
            assert output.shape == (batch_size, 256)

    def test_forward_different_num_patches(self):
        """Test forward pass with different numbers of patches."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        for num_patches in [10, 50, 100, 500]:
            features = torch.randn(4, num_patches, 256)
            output = model(features)
            assert output.shape == (4, 256)

    def test_forward_deterministic_with_eval(self):
        """Test forward pass is deterministic in eval mode."""
        torch.manual_seed(42)
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)
        model.eval()

        features = torch.randn(4, 100, 256)

        # Run twice
        output1 = model(features)
        output2 = model(features)

        assert torch.allclose(output1, output2)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_gated_attention_early_fusion(self):
        """Test full pipeline with gated attention and early fusion."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
        model = MILBase(feature_dim=1024, num_classes=2, attention=attention, fusion=fusion)

        scale1 = torch.randn(4, 100, 1024)
        scale2 = torch.randn(4, 100, 1024)

        # Apply fusion
        fused = model.apply_fusion([scale1, scale2])
        assert fused.shape == (4, 100, 256)

        # Compute attention
        attention_weights = model.compute_attention(fused)
        assert attention_weights.shape == (4, 100)

        # Aggregate features
        aggregated = model.aggregate_features(fused, attention_weights)
        assert aggregated.shape == (4, 256)

    def test_full_pipeline_simple_attention_late_fusion(self):
        """Test full pipeline with simple attention and late fusion."""
        attention = SimpleAttention(feature_dim=256, hidden_dim=128)
        fusion = LateFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
        model = MILBase(feature_dim=1024, num_classes=2, attention=attention, fusion=fusion)

        scale1 = torch.randn(4, 100, 1024)
        scale2 = torch.randn(4, 100, 1024)

        # Apply fusion
        projected = model.apply_fusion([scale1, scale2])
        assert len(projected) == 2

        # Use first scale for attention
        attention_weights = model.compute_attention(projected[0])
        assert attention_weights.shape == (4, 100)

        # Aggregate features
        aggregated = model.aggregate_features(projected[0], attention_weights)
        assert aggregated.shape == (4, 256)

    def test_full_pipeline_no_fusion(self):
        """Test full pipeline without fusion."""
        attention = GatedAttention(feature_dim=1024, hidden_dim=512)
        model = MILBase(feature_dim=1024, num_classes=2, attention=attention)

        features = torch.randn(4, 100, 1024)

        # Apply fusion (should return unchanged)
        fused = model.apply_fusion(features)
        assert torch.equal(fused, features)

        # Compute attention
        attention_weights = model.compute_attention(fused)
        assert attention_weights.shape == (4, 100)

        # Aggregate features
        aggregated = model.aggregate_features(fused, attention_weights)
        assert aggregated.shape == (4, 1024)

    def test_end_to_end_with_mask(self):
        """Test end-to-end pipeline with masking."""
        attention = GatedAttention(feature_dim=256, hidden_dim=256)
        model = MILBase(feature_dim=256, num_classes=2, attention=attention)

        features = torch.randn(4, 100, 256)
        mask = torch.ones(4, 100, dtype=torch.bool)
        # Variable length bags
        mask[0, 80:] = False  # 80 patches
        mask[1, 60:] = False  # 60 patches
        mask[2, 90:] = False  # 90 patches
        # mask[3] all True - 100 patches

        output = model(features, mask=mask, return_attention=True)

        # Check masked patches have ~0 weight
        assert torch.allclose(
            output["attention_weights"][0, 80:].sum(), torch.tensor(0.0), atol=1e-6
        )
        assert torch.allclose(
            output["attention_weights"][1, 60:].sum(), torch.tensor(0.0), atol=1e-6
        )
        assert torch.allclose(
            output["attention_weights"][2, 90:].sum(), torch.tensor(0.0), atol=1e-6
        )

        # Check valid patches sum to ~1.0
        assert torch.allclose(
            output["attention_weights"][0, :80].sum(), torch.tensor(1.0), atol=1e-6
        )
        assert torch.allclose(
            output["attention_weights"][1, :60].sum(), torch.tensor(1.0), atol=1e-6
        )
        assert torch.allclose(
            output["attention_weights"][2, :90].sum(), torch.tensor(1.0), atol=1e-6
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
