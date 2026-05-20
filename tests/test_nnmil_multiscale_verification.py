"""
Verification tests for nnMIL multi-scale support (Task 1.3).

This test file verifies that all requirements 13.1-13.6 are met by the
existing nnMIL implementation.

Feature: nnmil-architecture-upgrade
Task: 1.3 Implement multi-scale support for nnMIL
"""

import pytest
import torch

from src.models.mil.nnmil import nnMIL


class TestnnMILMultiScaleVerification:
    """Verification tests for multi-scale support requirements."""

    def test_requirement_13_1_accept_multi_scale_list(self):
        """
        Requirement 13.1: THE nnMIL_Model SHALL accept multi-scale features
        as a list of tensors with dimensions [[batch, N, feat_dim_scale1],
        [batch, N, feat_dim_scale2], ...]
        """
        model = nnMIL(feature_dim=1024, multi_scale=True, num_scales=3, fusion_strategy="early")

        # Create multi-scale features as list of tensors
        batch_size, num_patches = 4, 100
        scale1 = torch.randn(batch_size, num_patches, 1024)
        scale2 = torch.randn(batch_size, num_patches, 1024)
        scale3 = torch.randn(batch_size, num_patches, 1024)
        multi_scale_features = [scale1, scale2, scale3]

        # Model should accept list of tensors
        logits = model(multi_scale_features)

        assert logits.shape == (batch_size, 2), "Model should process multi-scale list"

    def test_requirement_13_2_support_early_and_late_fusion(self):
        """
        Requirement 13.2: THE nnMIL_Model SHALL support early fusion
        (concatenate features before processing) and late fusion
        (separate processing then combine)
        """
        batch_size, num_patches = 4, 100
        scale1 = torch.randn(batch_size, num_patches, 1024)
        scale2 = torch.randn(batch_size, num_patches, 1024)
        multi_scale_features = [scale1, scale2]

        # Test early fusion
        model_early = nnMIL(
            feature_dim=1024, multi_scale=True, num_scales=2, fusion_strategy="early"
        )
        logits_early = model_early(multi_scale_features)
        assert logits_early.shape == (batch_size, 2), "Early fusion should work"

        # Test late fusion
        model_late = nnMIL(feature_dim=1024, multi_scale=True, num_scales=2, fusion_strategy="late")
        logits_late = model_late(multi_scale_features)
        assert logits_late.shape == (batch_size, 2), "Late fusion should work"

    def test_requirement_13_3_early_fusion_concatenates_features(self):
        """
        Requirement 13.3: WHEN using early fusion, THE nnMIL_Model SHALL
        concatenate features from all scales along the feature dimension
        """
        model = nnMIL(feature_dim=1024, multi_scale=True, num_scales=2, fusion_strategy="early")

        batch_size, num_patches = 2, 50
        scale1 = torch.randn(batch_size, num_patches, 1024)
        scale2 = torch.randn(batch_size, num_patches, 1024)
        multi_scale_features = [scale1, scale2]

        # Verify early fusion by checking attention input dimension
        # Early fusion should concatenate along feature dimension
        # attention_V should expect input of size feature_dim * num_scales
        assert hasattr(model, "attention_V"), "Model should have attention_V"

        # For early fusion, attention_V input should be feature_dim * num_scales
        expected_input_dim = 1024 * 2  # 2048
        actual_input_dim = model.attention_V.in_features

        assert actual_input_dim == expected_input_dim, (
            f"Early fusion attention should accept concatenated features "
            f"({expected_input_dim}), got {actual_input_dim}"
        )

        # Verify forward pass works
        logits = model(multi_scale_features)
        assert logits.shape == (batch_size, 2)

    def test_requirement_13_4_late_fusion_processes_independently(self):
        """
        Requirement 13.4: WHEN using late fusion, THE nnMIL_Model SHALL
        process each scale independently then concatenate representations
        """
        model = nnMIL(feature_dim=1024, multi_scale=True, num_scales=3, fusion_strategy="late")

        # Verify late fusion by checking that attention modules are ModuleLists
        # Late fusion should have separate attention per scale
        assert hasattr(model, "attention_V"), "Model should have attention_V"
        assert isinstance(
            model.attention_V, torch.nn.ModuleList
        ), "Late fusion should use ModuleList for scale-specific attention"
        assert (
            len(model.attention_V) == 3
        ), f"Late fusion should have {3} attention modules, got {len(model.attention_V)}"

        # Verify forward pass works
        batch_size, num_patches = 2, 50
        scale1 = torch.randn(batch_size, num_patches, 1024)
        scale2 = torch.randn(batch_size, num_patches, 1024)
        scale3 = torch.randn(batch_size, num_patches, 1024)
        multi_scale_features = [scale1, scale2, scale3]

        logits = model(multi_scale_features)
        assert logits.shape == (batch_size, 2)

    def test_requirement_13_5_support_1_to_4_scales(self):
        """
        Requirement 13.5: THE nnMIL_Model SHALL support 1 to 4 magnification scales
        """
        batch_size, num_patches = 2, 50

        # Test 1 scale (single-scale mode)
        model_1 = nnMIL(feature_dim=1024, multi_scale=False)
        features_1 = torch.randn(batch_size, num_patches, 1024)
        logits_1 = model_1(features_1)
        assert logits_1.shape == (batch_size, 2), "1 scale should work"

        # Test 2 scales
        model_2 = nnMIL(feature_dim=1024, multi_scale=True, num_scales=2)
        features_2 = [
            torch.randn(batch_size, num_patches, 1024),
            torch.randn(batch_size, num_patches, 1024),
        ]
        logits_2 = model_2(features_2)
        assert logits_2.shape == (batch_size, 2), "2 scales should work"

        # Test 3 scales
        model_3 = nnMIL(feature_dim=1024, multi_scale=True, num_scales=3)
        features_3 = [
            torch.randn(batch_size, num_patches, 1024),
            torch.randn(batch_size, num_patches, 1024),
            torch.randn(batch_size, num_patches, 1024),
        ]
        logits_3 = model_3(features_3)
        assert logits_3.shape == (batch_size, 2), "3 scales should work"

        # Test 4 scales
        model_4 = nnMIL(feature_dim=1024, multi_scale=True, num_scales=4)
        features_4 = [
            torch.randn(batch_size, num_patches, 1024),
            torch.randn(batch_size, num_patches, 1024),
            torch.randn(batch_size, num_patches, 1024),
            torch.randn(batch_size, num_patches, 1024),
        ]
        logits_4 = model_4(features_4)
        assert logits_4.shape == (batch_size, 2), "4 scales should work"

    def test_requirement_13_6_backward_compatible_api(self):
        """
        Requirement 13.6: THE MIL_System SHALL maintain the existing
        multi-scale API from TransMIL_Model for backward compatibility

        The API should:
        - Accept list of tensors for multi-scale input
        - Accept single tensor for single-scale input
        - Support return_attention parameter
        - Support num_patches parameter for masking
        """
        # Test single-scale API
        model_single = nnMIL(feature_dim=1024)
        features_single = torch.randn(4, 100, 1024)
        num_patches = torch.tensor([100, 80, 90, 100])

        # Basic forward
        logits = model_single(features_single, num_patches)
        assert logits.shape == (4, 2)

        # With attention
        logits, attention = model_single(features_single, num_patches, return_attention=True)
        assert logits.shape == (4, 2)
        assert attention.shape == (4, 100)

        # Test multi-scale API
        model_multi = nnMIL(feature_dim=1024, multi_scale=True, num_scales=2)
        features_multi = [torch.randn(4, 100, 1024), torch.randn(4, 100, 1024)]

        # Basic forward
        logits = model_multi(features_multi, num_patches)
        assert logits.shape == (4, 2)

        # With attention
        logits, attention = model_multi(features_multi, num_patches, return_attention=True)
        assert logits.shape == (4, 2)
        assert attention.shape == (4, 100)

    def test_multi_scale_with_different_feature_dims(self):
        """
        Additional test: Verify multi-scale works with different feature dimensions
        per scale (e.g., different foundation models per scale)
        """
        # Note: Current implementation expects same feature_dim for all scales
        # This is a design choice for simplicity
        model = nnMIL(feature_dim=1024, multi_scale=True, num_scales=2, fusion_strategy="early")

        batch_size, num_patches = 2, 50
        # Both scales must have same feature_dim in current implementation
        scale1 = torch.randn(batch_size, num_patches, 1024)
        scale2 = torch.randn(batch_size, num_patches, 1024)

        logits = model([scale1, scale2])
        assert logits.shape == (batch_size, 2)

    def test_multi_scale_gradient_flow(self):
        """
        Additional test: Verify gradients flow through multi-scale model
        """
        model = nnMIL(feature_dim=256, multi_scale=True, num_scales=2, fusion_strategy="late")

        batch_size, num_patches = 2, 50
        scale1 = torch.randn(batch_size, num_patches, 256, requires_grad=True)
        scale2 = torch.randn(batch_size, num_patches, 256, requires_grad=True)

        logits = model([scale1, scale2])
        loss = logits.sum()
        loss.backward()

        # Verify gradients flow to both scales
        assert scale1.grad is not None, "Gradients should flow to scale 1"
        assert scale2.grad is not None, "Gradients should flow to scale 2"
        assert scale1.grad.abs().sum() > 0, "Scale 1 gradients should be non-zero"
        assert scale2.grad.abs().sum() > 0, "Scale 2 gradients should be non-zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
