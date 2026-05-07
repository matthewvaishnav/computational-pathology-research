"""
Unit tests for fusion strategies module.

Tests the FusionStrategy base class, EarlyFusion, and LateFusion implementations
to ensure they correctly combine multi-scale features.
"""

import pytest
import torch

from src.models.fusion_strategies import EarlyFusion, FusionStrategy, LateFusion


class TestFusionStrategyBase:
    """Tests for FusionStrategy base class."""

    def test_base_class_is_abstract(self):
        """Test that FusionStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            FusionStrategy(feature_dim=1024, hidden_dim=256, num_scales=2)

    def test_base_class_stores_parameters(self):
        """Test that base class stores initialization parameters."""

        class ConcreteFusion(FusionStrategy):
            def forward(self, multi_scale_features, mask=None):
                return multi_scale_features[0]

        fusion = ConcreteFusion(feature_dim=1024, hidden_dim=256, num_scales=3)
        assert fusion.feature_dim == 1024
        assert fusion.hidden_dim == 256
        assert fusion.num_scales == 3


class TestEarlyFusion:
    """Tests for EarlyFusion strategy."""

    def test_initialization(self):
        """Test EarlyFusion initialization."""
        fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2, dropout=0.1)

        assert fusion.feature_dim == 1024
        assert fusion.hidden_dim == 256
        assert fusion.num_scales == 2
        assert len(fusion.projections) == 2

    def test_forward_two_scales(self):
        """Test early fusion with 2 scales."""
        fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2)

        scale1 = torch.randn(4, 100, 1024)
        scale2 = torch.randn(4, 100, 1024)
        features = [scale1, scale2]

        fused = fusion(features)

        assert fused.shape == (4, 100, 256)
        assert not torch.isnan(fused).any()

    def test_forward_three_scales(self):
        """Test early fusion with 3 scales."""
        fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=3)

        scale1 = torch.randn(2, 50, 1024)
        scale2 = torch.randn(2, 50, 1024)
        scale3 = torch.randn(2, 50, 1024)
        features = [scale1, scale2, scale3]

        fused = fusion(features)

        assert fused.shape == (2, 50, 256)
        assert not torch.isnan(fused).any()

    def test_forward_with_mask(self):
        """Test early fusion with mask (mask is accepted but not used)."""
        fusion = EarlyFusion(feature_dim=512, hidden_dim=128, num_scales=2)

        scale1 = torch.randn(3, 80, 512)
        scale2 = torch.randn(3, 80, 512)
        features = [scale1, scale2]

        # Create mask: first 60 patches valid, rest padding
        mask = torch.zeros(3, 80, dtype=torch.bool)
        mask[:, :60] = True

        fused = fusion(features, mask=mask)

        assert fused.shape == (3, 80, 128)
        assert not torch.isnan(fused).any()

    def test_forward_with_missing_scale(self):
        """Test early fusion with one missing scale (None)."""
        fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=3)

        scale1 = torch.randn(4, 100, 1024)
        scale2 = None  # Missing scale
        scale3 = torch.randn(4, 100, 1024)
        features = [scale1, scale2, scale3]

        fused = fusion(features)

        # Should still work with 2 valid scales
        assert fused.shape == (4, 100, 256)
        assert not torch.isnan(fused).any()

    def test_forward_all_scales_none_raises_error(self):
        """Test that early fusion raises error when all scales are None."""
        fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2)

        features = [None, None]

        with pytest.raises(ValueError, match="All scales are None"):
            fusion(features)

    def test_gradient_flow(self):
        """Test that gradients flow through early fusion."""
        fusion = EarlyFusion(feature_dim=512, hidden_dim=128, num_scales=2)

        scale1 = torch.randn(2, 50, 512, requires_grad=True)
        scale2 = torch.randn(2, 50, 512, requires_grad=True)
        features = [scale1, scale2]

        fused = fusion(features)
        loss = fused.sum()
        loss.backward()

        assert scale1.grad is not None
        assert scale2.grad is not None
        assert scale1.grad.abs().sum() > 0
        assert scale2.grad.abs().sum() > 0

    def test_different_batch_sizes(self):
        """Test early fusion with different batch sizes."""
        fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2)

        # Batch size 1
        features = [torch.randn(1, 100, 1024), torch.randn(1, 100, 1024)]
        fused = fusion(features)
        assert fused.shape == (1, 100, 256)

        # Batch size 8
        features = [torch.randn(8, 100, 1024), torch.randn(8, 100, 1024)]
        fused = fusion(features)
        assert fused.shape == (8, 100, 256)

    def test_different_num_patches(self):
        """Test early fusion with different numbers of patches."""
        fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2)

        # 50 patches
        features = [torch.randn(4, 50, 1024), torch.randn(4, 50, 1024)]
        fused = fusion(features)
        assert fused.shape == (4, 50, 256)

        # 200 patches
        features = [torch.randn(4, 200, 1024), torch.randn(4, 200, 1024)]
        fused = fusion(features)
        assert fused.shape == (4, 200, 256)

    def test_output_deterministic(self):
        """Test that early fusion produces deterministic output."""
        torch.manual_seed(42)
        fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
        fusion.eval()  # Set to eval mode to disable dropout

        scale1 = torch.randn(2, 50, 1024)
        scale2 = torch.randn(2, 50, 1024)
        features = [scale1, scale2]

        # Run twice with same input
        fused1 = fusion(features)
        fused2 = fusion(features)

        assert torch.allclose(fused1, fused2)


class TestLateFusion:
    """Tests for LateFusion strategy."""

    def test_initialization(self):
        """Test LateFusion initialization."""
        fusion = LateFusion(feature_dim=1024, hidden_dim=256, num_scales=2, dropout=0.1)

        assert fusion.feature_dim == 1024
        assert fusion.hidden_dim == 256
        assert fusion.num_scales == 2
        assert len(fusion.projections) == 2

    def test_forward_two_scales(self):
        """Test late fusion with 2 scales."""
        fusion = LateFusion(feature_dim=1024, hidden_dim=256, num_scales=2)

        scale1 = torch.randn(4, 100, 1024)
        scale2 = torch.randn(4, 100, 1024)
        features = [scale1, scale2]

        projected = fusion(features)

        assert len(projected) == 2
        assert projected[0].shape == (4, 100, 256)
        assert projected[1].shape == (4, 100, 256)
        assert not torch.isnan(projected[0]).any()
        assert not torch.isnan(projected[1]).any()

    def test_forward_three_scales(self):
        """Test late fusion with 3 scales."""
        fusion = LateFusion(feature_dim=1024, hidden_dim=256, num_scales=3)

        scale1 = torch.randn(2, 50, 1024)
        scale2 = torch.randn(2, 50, 1024)
        scale3 = torch.randn(2, 50, 1024)
        features = [scale1, scale2, scale3]

        projected = fusion(features)

        assert len(projected) == 3
        for i in range(3):
            assert projected[i].shape == (2, 50, 256)
            assert not torch.isnan(projected[i]).any()

    def test_forward_with_mask(self):
        """Test late fusion with mask (mask is accepted but not used)."""
        fusion = LateFusion(feature_dim=512, hidden_dim=128, num_scales=2)

        scale1 = torch.randn(3, 80, 512)
        scale2 = torch.randn(3, 80, 512)
        features = [scale1, scale2]

        # Create mask
        mask = torch.zeros(3, 80, dtype=torch.bool)
        mask[:, :60] = True

        projected = fusion(features, mask=mask)

        assert len(projected) == 2
        assert projected[0].shape == (3, 80, 128)
        assert projected[1].shape == (3, 80, 128)

    def test_forward_with_missing_scale(self):
        """Test late fusion with one missing scale (None)."""
        fusion = LateFusion(feature_dim=1024, hidden_dim=256, num_scales=3)

        scale1 = torch.randn(4, 100, 1024)
        scale2 = None  # Missing scale
        scale3 = torch.randn(4, 100, 1024)
        features = [scale1, scale2, scale3]

        projected = fusion(features)

        assert len(projected) == 3
        assert projected[0].shape == (4, 100, 256)
        assert projected[1] is None  # Preserved None
        assert projected[2].shape == (4, 100, 256)

    def test_forward_all_scales_none(self):
        """Test late fusion with all scales None (returns list of Nones)."""
        fusion = LateFusion(feature_dim=1024, hidden_dim=256, num_scales=2)

        features = [None, None]
        projected = fusion(features)

        assert len(projected) == 2
        assert projected[0] is None
        assert projected[1] is None

    def test_gradient_flow(self):
        """Test that gradients flow through late fusion."""
        fusion = LateFusion(feature_dim=512, hidden_dim=128, num_scales=2)

        scale1 = torch.randn(2, 50, 512, requires_grad=True)
        scale2 = torch.randn(2, 50, 512, requires_grad=True)
        features = [scale1, scale2]

        projected = fusion(features)
        loss = projected[0].sum() + projected[1].sum()
        loss.backward()

        assert scale1.grad is not None
        assert scale2.grad is not None
        assert scale1.grad.abs().sum() > 0
        assert scale2.grad.abs().sum() > 0

    def test_different_batch_sizes(self):
        """Test late fusion with different batch sizes."""
        fusion = LateFusion(feature_dim=1024, hidden_dim=256, num_scales=2)

        # Batch size 1
        features = [torch.randn(1, 100, 1024), torch.randn(1, 100, 1024)]
        projected = fusion(features)
        assert projected[0].shape == (1, 100, 256)
        assert projected[1].shape == (1, 100, 256)

        # Batch size 8
        features = [torch.randn(8, 100, 1024), torch.randn(8, 100, 1024)]
        projected = fusion(features)
        assert projected[0].shape == (8, 100, 256)
        assert projected[1].shape == (8, 100, 256)

    def test_different_num_patches(self):
        """Test late fusion with different numbers of patches."""
        fusion = LateFusion(feature_dim=1024, hidden_dim=256, num_scales=2)

        # 50 patches
        features = [torch.randn(4, 50, 1024), torch.randn(4, 50, 1024)]
        projected = fusion(features)
        assert projected[0].shape == (4, 50, 256)
        assert projected[1].shape == (4, 50, 256)

        # 200 patches
        features = [torch.randn(4, 200, 1024), torch.randn(4, 200, 1024)]
        projected = fusion(features)
        assert projected[0].shape == (4, 200, 256)
        assert projected[1].shape == (4, 200, 256)

    def test_output_deterministic(self):
        """Test that late fusion produces deterministic output."""
        torch.manual_seed(42)
        fusion = LateFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
        fusion.eval()  # Set to eval mode to disable dropout

        scale1 = torch.randn(2, 50, 1024)
        scale2 = torch.randn(2, 50, 1024)
        features = [scale1, scale2]

        # Run twice with same input
        projected1 = fusion(features)
        projected2 = fusion(features)

        assert torch.allclose(projected1[0], projected2[0])
        assert torch.allclose(projected1[1], projected2[1])

    def test_scales_processed_independently(self):
        """Test that scales are processed independently in late fusion."""
        fusion = LateFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
        fusion.eval()  # Set to eval mode to disable dropout

        scale1 = torch.randn(2, 50, 1024)
        scale2 = torch.randn(2, 50, 1024)

        # Process together
        projected_together = fusion([scale1, scale2])

        # Process separately
        projected_separate1 = fusion([scale1, None])
        projected_separate2 = fusion([None, scale2])

        # Results should match (scales are independent)
        assert torch.allclose(projected_together[0], projected_separate1[0])
        assert torch.allclose(projected_together[1], projected_separate2[1])


class TestFusionComparison:
    """Tests comparing early and late fusion behaviors."""

    def test_early_vs_late_output_shapes(self):
        """Test that early and late fusion produce different output shapes."""
        early = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
        late = LateFusion(feature_dim=1024, hidden_dim=256, num_scales=2)

        scale1 = torch.randn(4, 100, 1024)
        scale2 = torch.randn(4, 100, 1024)
        features = [scale1, scale2]

        early_output = early(features)
        late_output = late(features)

        # Early fusion returns single tensor
        assert isinstance(early_output, torch.Tensor)
        assert early_output.shape == (4, 100, 256)

        # Late fusion returns list of tensors
        assert isinstance(late_output, list)
        assert len(late_output) == 2
        assert late_output[0].shape == (4, 100, 256)
        assert late_output[1].shape == (4, 100, 256)

    def test_parameter_count_comparison(self):
        """Test that early and late fusion have same number of parameters."""
        early = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
        late = LateFusion(feature_dim=1024, hidden_dim=256, num_scales=2)

        early_params = sum(p.numel() for p in early.parameters())
        late_params = sum(p.numel() for p in late.parameters())

        # Both should have same number of projection parameters
        assert early_params == late_params
