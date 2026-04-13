"""
Tests for multi-scale feature support in AttentionMIL and CLAM.

This module tests the multi-scale functionality added to AttentionMIL and CLAM,
including early fusion and late fusion strategies.
"""

import pytest
import torch

from src.models.attention_mil import AttentionMIL, CLAM


class TestAttentionMILMultiScale:
    """Test multi-scale feature support in AttentionMIL."""

    def test_single_scale_backward_compatibility(self):
        """Test that single-scale input still works (backward compatibility)."""
        model = AttentionMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            dropout=0.1,
            multi_scale=False,
        )

        # Single-scale input
        batch_size, num_patches, feature_dim = 4, 100, 1024
        features = torch.randn(batch_size, num_patches, feature_dim)
        num_patches_actual = torch.tensor([80, 90, 100, 75])

        # Forward pass
        logits, attention = model(features, num_patches_actual, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert attention.shape == (batch_size, num_patches), "Attention shape mismatch"

        # Attention weights should sum to 1 for each slide
        attention_sums = attention.sum(dim=1)
        assert torch.allclose(attention_sums, torch.ones(batch_size), atol=1e-5), \
            "Attention weights don't sum to 1"

    def test_early_fusion_two_scales(self):
        """Test early fusion with 2 scales."""
        model = AttentionMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            dropout=0.1,
            multi_scale=True,
            num_scales=2,
            fusion_strategy="early",
        )

        # Multi-scale input: 2 scales
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features]
        num_patches_actual = torch.tensor([80, 90, 100, 75])

        # Forward pass
        logits, attention = model(multi_scale_features, num_patches_actual, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert attention.shape == (batch_size, num_patches), "Attention shape mismatch"

        # Attention weights should sum to 1 for each slide
        attention_sums = attention.sum(dim=1)
        assert torch.allclose(attention_sums, torch.ones(batch_size), atol=1e-5), \
            "Attention weights don't sum to 1"

    def test_late_fusion_three_scales(self):
        """Test late fusion with 3 scales."""
        model = AttentionMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            dropout=0.1,
            multi_scale=True,
            num_scales=3,
            fusion_strategy="late",
        )

        # Multi-scale input: 3 scales
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        scale3_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features, scale3_features]
        num_patches_actual = torch.tensor([80, 90, 100, 75])

        # Forward pass
        logits, attention = model(multi_scale_features, num_patches_actual, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert attention.shape == (batch_size, num_patches), "Attention shape mismatch"

        # Attention weights should sum to 1 for each slide
        attention_sums = attention.sum(dim=1)
        assert torch.allclose(attention_sums, torch.ones(batch_size), atol=1e-5), \
            "Attention weights don't sum to 1"

    def test_missing_scale_handling(self):
        """Test that missing scales are handled gracefully (late fusion)."""
        model = AttentionMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            dropout=0.1,
            multi_scale=True,
            num_scales=3,
            fusion_strategy="late",
        )

        # Multi-scale input with one missing scale (None)
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = None  # Missing scale
        scale3_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features, scale3_features]
        num_patches_actual = torch.tensor([80, 90, 100, 75])

        # Forward pass should not raise an error
        logits, attention = model(multi_scale_features, num_patches_actual, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert attention.shape == (batch_size, num_patches), "Attention shape mismatch"

    def test_gradient_flow_multi_scale(self):
        """Test that gradients flow through multi-scale attention mechanism."""
        model = AttentionMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            dropout=0.1,
            multi_scale=True,
            num_scales=2,
            fusion_strategy="early",
        )

        # Multi-scale input
        batch_size, num_patches, feature_dim = 2, 50, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim, requires_grad=True)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim, requires_grad=True)
        multi_scale_features = [scale1_features, scale2_features]
        labels = torch.tensor([0, 1], dtype=torch.long)

        # Forward pass
        logits = model(multi_scale_features, return_attention=False)

        # Compute loss and backward
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        # Check gradients exist
        assert scale1_features.grad is not None, "No gradient for scale 1 features"
        assert scale1_features.grad.abs().sum() > 0, "Gradient is zero for scale 1"
        assert scale2_features.grad is not None, "No gradient for scale 2 features"
        assert scale2_features.grad.abs().sum() > 0, "Gradient is zero for scale 2"

    def test_invalid_fusion_strategy(self):
        """Test that invalid fusion strategy raises error."""
        with pytest.raises(ValueError, match="fusion_strategy must be"):
            AttentionMIL(
                feature_dim=1024,
                hidden_dim=256,
                num_classes=2,
                multi_scale=True,
                num_scales=2,
                fusion_strategy="invalid",
            )

    def test_multi_scale_without_flag_raises_error(self):
        """Test that passing list of features to non-multi-scale model raises error."""
        model = AttentionMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            multi_scale=False,  # Not multi-scale
        )

        # Try to pass multi-scale input
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features]

        with pytest.raises(ValueError, match="Model was not initialized with multi_scale=True"):
            model(multi_scale_features)

    def test_early_fusion_with_gated_attention(self):
        """Test early fusion with gated attention mechanism."""
        model = AttentionMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            dropout=0.1,
            gated=True,
            multi_scale=True,
            num_scales=2,
            fusion_strategy="early",
        )

        # Multi-scale input
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features]

        # Forward pass
        logits, attention = model(multi_scale_features, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert attention.shape == (batch_size, num_patches), "Attention shape mismatch"

    def test_late_fusion_with_simple_attention(self):
        """Test late fusion with simple (non-gated) attention mechanism."""
        model = AttentionMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            dropout=0.1,
            gated=False,
            multi_scale=True,
            num_scales=2,
            fusion_strategy="late",
        )

        # Multi-scale input
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features]

        # Forward pass
        logits, attention = model(multi_scale_features, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert attention.shape == (batch_size, num_patches), "Attention shape mismatch"


class TestCLAMMultiScale:
    """Test multi-scale feature support in CLAM."""

    def test_single_scale_backward_compatibility(self):
        """Test that single-scale input still works (backward compatibility)."""
        model = CLAM(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_clusters=10,
            dropout=0.1,
            multi_branch=True,
            multi_scale=False,
        )

        # Single-scale input
        batch_size, num_patches, feature_dim = 4, 100, 1024
        features = torch.randn(batch_size, num_patches, feature_dim)
        num_patches_actual = torch.tensor([80, 90, 100, 75])

        # Forward pass
        logits, attention, instance_preds = model(features, num_patches_actual, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert isinstance(attention, dict), "Multi-branch should return dict"
        assert "positive" in attention and "negative" in attention
        assert attention["positive"].shape == (batch_size, num_patches), "Attention shape mismatch"
        assert instance_preds.shape == (batch_size, num_patches, 10), "Instance preds shape mismatch"

        # Attention weights should sum to 1 for each slide
        attention_sums = attention["positive"].sum(dim=1)
        assert torch.allclose(attention_sums, torch.ones(batch_size), atol=1e-5), \
            "Attention weights don't sum to 1"

    def test_early_fusion_two_scales(self):
        """Test early fusion with 2 scales."""
        model = CLAM(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_clusters=10,
            dropout=0.1,
            multi_branch=True,
            multi_scale=True,
            num_scales=2,
            fusion_strategy="early",
        )

        # Multi-scale input: 2 scales
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features]
        num_patches_actual = torch.tensor([80, 90, 100, 75])

        # Forward pass
        logits, attention, instance_preds = model(multi_scale_features, num_patches_actual, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert isinstance(attention, dict), "Multi-branch should return dict"
        assert attention["positive"].shape == (batch_size, num_patches), "Attention shape mismatch"
        assert instance_preds.shape == (batch_size, num_patches, 10), "Instance preds shape mismatch"

        # Attention weights should sum to 1 for each slide
        attention_sums = attention["positive"].sum(dim=1)
        assert torch.allclose(attention_sums, torch.ones(batch_size), atol=1e-5), \
            "Attention weights don't sum to 1"

    def test_late_fusion_three_scales(self):
        """Test late fusion with 3 scales."""
        model = CLAM(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_clusters=10,
            dropout=0.1,
            multi_branch=True,
            multi_scale=True,
            num_scales=3,
            fusion_strategy="late",
        )

        # Multi-scale input: 3 scales
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        scale3_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features, scale3_features]
        num_patches_actual = torch.tensor([80, 90, 100, 75])

        # Forward pass
        logits, attention, instance_preds = model(multi_scale_features, num_patches_actual, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert isinstance(attention, dict), "Multi-branch should return dict"
        assert attention["positive"].shape == (batch_size, num_patches), "Attention shape mismatch"
        assert instance_preds.shape == (batch_size, num_patches, 10), "Instance preds shape mismatch"

        # Attention weights should sum to 1 for each slide
        attention_sums = attention["positive"].sum(dim=1)
        assert torch.allclose(attention_sums, torch.ones(batch_size), atol=1e-5), \
            "Attention weights don't sum to 1"

    def test_single_branch_multi_scale(self):
        """Test single-branch CLAM with multi-scale features."""
        model = CLAM(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_clusters=10,
            dropout=0.1,
            multi_branch=False,  # Single branch
            multi_scale=True,
            num_scales=2,
            fusion_strategy="early",
        )

        # Multi-scale input
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features]
        num_patches_actual = torch.tensor([80, 90, 100, 75])

        # Forward pass
        logits, attention, instance_preds = model(multi_scale_features, num_patches_actual, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert not isinstance(attention, dict), "Single-branch should return tensor"
        assert attention.shape == (batch_size, num_patches), "Attention shape mismatch"
        assert instance_preds.shape == (batch_size, num_patches, 10), "Instance preds shape mismatch"

    def test_missing_scale_handling(self):
        """Test that missing scales are handled gracefully (late fusion)."""
        model = CLAM(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_clusters=10,
            dropout=0.1,
            multi_branch=True,
            multi_scale=True,
            num_scales=3,
            fusion_strategy="late",
        )

        # Multi-scale input with one missing scale (None)
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = None  # Missing scale
        scale3_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features, scale3_features]
        num_patches_actual = torch.tensor([80, 90, 100, 75])

        # Forward pass should not raise an error
        logits, attention, instance_preds = model(multi_scale_features, num_patches_actual, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert isinstance(attention, dict), "Multi-branch should return dict"
        assert attention["positive"].shape == (batch_size, num_patches), "Attention shape mismatch"

    def test_gradient_flow_multi_scale(self):
        """Test that gradients flow through multi-scale CLAM attention mechanism."""
        model = CLAM(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_clusters=10,
            dropout=0.1,
            multi_branch=True,
            multi_scale=True,
            num_scales=2,
            fusion_strategy="early",
        )

        # Multi-scale input
        batch_size, num_patches, feature_dim = 2, 50, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim, requires_grad=True)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim, requires_grad=True)
        multi_scale_features = [scale1_features, scale2_features]
        labels = torch.tensor([0, 1], dtype=torch.long)

        # Forward pass
        logits = model(multi_scale_features, return_attention=False)

        # Compute loss and backward
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        # Check gradients exist
        assert scale1_features.grad is not None, "No gradient for scale 1 features"
        assert scale1_features.grad.abs().sum() > 0, "Gradient is zero for scale 1"
        assert scale2_features.grad is not None, "No gradient for scale 2 features"
        assert scale2_features.grad.abs().sum() > 0, "Gradient is zero for scale 2"

    def test_invalid_fusion_strategy(self):
        """Test that invalid fusion strategy raises error."""
        with pytest.raises(ValueError, match="fusion_strategy must be"):
            CLAM(
                feature_dim=1024,
                hidden_dim=256,
                num_classes=2,
                num_clusters=10,
                multi_scale=True,
                num_scales=2,
                fusion_strategy="invalid",
            )

    def test_multi_scale_without_flag_raises_error(self):
        """Test that passing list of features to non-multi-scale model raises error."""
        model = CLAM(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_clusters=10,
            multi_scale=False,  # Not multi-scale
        )

        # Try to pass multi-scale input
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features]

        with pytest.raises(ValueError, match="Model was not initialized with multi_scale=True"):
            model(multi_scale_features)

    def test_late_fusion_single_branch(self):
        """Test late fusion with single-branch CLAM."""
        model = CLAM(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_clusters=10,
            dropout=0.1,
            multi_branch=False,  # Single branch
            multi_scale=True,
            num_scales=2,
            fusion_strategy="late",
        )

        # Multi-scale input
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features]

        # Forward pass
        logits, attention, instance_preds = model(multi_scale_features, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert not isinstance(attention, dict), "Single-branch should return tensor"
        assert attention.shape == (batch_size, num_patches), "Attention shape mismatch"



class TestTransMILMultiScale:
    """Test multi-scale feature support in TransMIL."""

    def test_single_scale_backward_compatibility(self):
        """Test that single-scale input still works (backward compatibility)."""
        from src.models.attention_mil import TransMIL

        model = TransMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
            multi_scale=False,
        )

        # Single-scale input
        batch_size, num_patches, feature_dim = 4, 100, 1024
        features = torch.randn(batch_size, num_patches, feature_dim)
        num_patches_actual = torch.tensor([80, 90, 100, 75])

        # Forward pass
        logits, attention = model(features, num_patches_actual, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert attention.shape == (batch_size, num_patches), "Attention shape mismatch"

        # Attention weights should sum to 1 for each slide (uniform for TransMIL)
        attention_sums = attention.sum(dim=1)
        assert torch.allclose(attention_sums, torch.ones(batch_size), atol=1e-5), \
            "Attention weights don't sum to 1"

    def test_early_fusion_two_scales(self):
        """Test early fusion with 2 scales."""
        from src.models.attention_mil import TransMIL

        model = TransMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
            multi_scale=True,
            num_scales=2,
            fusion_strategy="early",
        )

        # Multi-scale input: 2 scales
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features]
        num_patches_actual = torch.tensor([80, 90, 100, 75])

        # Forward pass
        logits, attention = model(multi_scale_features, num_patches_actual, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert attention.shape == (batch_size, num_patches), "Attention shape mismatch"

        # Attention weights should sum to 1 for each slide
        attention_sums = attention.sum(dim=1)
        assert torch.allclose(attention_sums, torch.ones(batch_size), atol=1e-5), \
            "Attention weights don't sum to 1"

    def test_late_fusion_three_scales(self):
        """Test late fusion with 3 scales."""
        from src.models.attention_mil import TransMIL

        model = TransMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
            multi_scale=True,
            num_scales=3,
            fusion_strategy="late",
        )

        # Multi-scale input: 3 scales
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        scale3_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features, scale3_features]
        num_patches_actual = torch.tensor([80, 90, 100, 75])

        # Forward pass
        logits, attention = model(multi_scale_features, num_patches_actual, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert attention.shape == (batch_size, num_patches), "Attention shape mismatch"

        # Attention weights should sum to 1 for each slide
        attention_sums = attention.sum(dim=1)
        assert torch.allclose(attention_sums, torch.ones(batch_size), atol=1e-5), \
            "Attention weights don't sum to 1"

    def test_missing_scale_handling(self):
        """Test that missing scales are handled gracefully (late fusion)."""
        from src.models.attention_mil import TransMIL

        model = TransMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
            multi_scale=True,
            num_scales=3,
            fusion_strategy="late",
        )

        # Multi-scale input with one missing scale (None)
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = None  # Missing scale
        scale3_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features, scale3_features]
        num_patches_actual = torch.tensor([80, 90, 100, 75])

        # Forward pass should not raise an error
        logits, attention = model(multi_scale_features, num_patches_actual, return_attention=True)

        # Assertions
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
        assert attention.shape == (batch_size, num_patches), "Attention shape mismatch"

    def test_gradient_flow_multi_scale(self):
        """Test that gradients flow through multi-scale TransMIL mechanism."""
        from src.models.attention_mil import TransMIL

        model = TransMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
            multi_scale=True,
            num_scales=2,
            fusion_strategy="early",
        )

        # Multi-scale input
        batch_size, num_patches, feature_dim = 2, 50, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim, requires_grad=True)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim, requires_grad=True)
        multi_scale_features = [scale1_features, scale2_features]
        labels = torch.tensor([0, 1], dtype=torch.long)

        # Forward pass
        logits = model(multi_scale_features, return_attention=False)

        # Compute loss and backward
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        # Check gradients exist
        assert scale1_features.grad is not None, "No gradient for scale 1 features"
        assert scale1_features.grad.abs().sum() > 0, "Gradient is zero for scale 1"
        assert scale2_features.grad is not None, "No gradient for scale 2 features"
        assert scale2_features.grad.abs().sum() > 0, "Gradient is zero for scale 2"

    def test_invalid_fusion_strategy(self):
        """Test that invalid fusion strategy raises error."""
        from src.models.attention_mil import TransMIL

        with pytest.raises(ValueError, match="fusion_strategy must be"):
            TransMIL(
                feature_dim=1024,
                hidden_dim=256,
                num_classes=2,
                num_layers=2,
                num_heads=8,
                multi_scale=True,
                num_scales=2,
                fusion_strategy="invalid",
            )

    def test_multi_scale_without_flag_raises_error(self):
        """Test that passing list of features to non-multi-scale model raises error."""
        from src.models.attention_mil import TransMIL

        model = TransMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_layers=2,
            num_heads=8,
            multi_scale=False,  # Not multi-scale
        )

        # Try to pass multi-scale input
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features]

        with pytest.raises(ValueError, match="Model was not initialized with multi_scale=True"):
            model(multi_scale_features)

    def test_scale_specific_positional_encodings(self):
        """Test that scale-specific positional encodings are used in late fusion."""
        from src.models.attention_mil import TransMIL

        model = TransMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
            use_pos_encoding=True,
            multi_scale=True,
            num_scales=2,
            fusion_strategy="late",
        )

        # Check that scale-specific positional encodings exist
        assert isinstance(model.pos_encoding, torch.nn.ParameterList), \
            "Late fusion should have scale-specific positional encodings"
        assert len(model.pos_encoding) == 2, \
            "Should have 2 positional encodings for 2 scales"

        # Multi-scale input
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features]

        # Forward pass should work
        logits = model(multi_scale_features, return_attention=False)
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"

    def test_scale_specific_transformers_late_fusion(self):
        """Test that scale-specific transformers are used in late fusion."""
        from src.models.attention_mil import TransMIL

        model = TransMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
            multi_scale=True,
            num_scales=2,
            fusion_strategy="late",
        )

        # Check that scale-specific transformers exist
        assert isinstance(model.transformer, torch.nn.ModuleList), \
            "Late fusion should have scale-specific transformers"
        assert len(model.transformer) == 2, \
            "Should have 2 transformers for 2 scales"

        # Multi-scale input
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features]

        # Forward pass should work
        logits = model(multi_scale_features, return_attention=False)
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"

    def test_early_fusion_shared_transformer(self):
        """Test that early fusion uses a shared transformer."""
        from src.models.attention_mil import TransMIL

        model = TransMIL(
            feature_dim=1024,
            hidden_dim=256,
            num_classes=2,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
            multi_scale=True,
            num_scales=2,
            fusion_strategy="early",
        )

        # Check that transformer is NOT a ModuleList for early fusion
        assert not isinstance(model.transformer, torch.nn.ModuleList), \
            "Early fusion should use a shared transformer"

        # Multi-scale input
        batch_size, num_patches, feature_dim = 4, 100, 1024
        scale1_features = torch.randn(batch_size, num_patches, feature_dim)
        scale2_features = torch.randn(batch_size, num_patches, feature_dim)
        multi_scale_features = [scale1_features, scale2_features]

        # Forward pass should work
        logits = model(multi_scale_features, return_attention=False)
        assert logits.shape == (batch_size, 2), "Logits shape mismatch"
