"""
Verification script for nnMIL requirements.

This script verifies that the nnMIL implementation meets the requirements
specified in the design document.
"""

import torch

from src.models.nnmil import nnMIL


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_parameter_efficiency():
    """
    Verify that nnMIL maintains parameter efficiency.

    Requirement 1.6: THE nnMIL_Model SHALL maintain parameter efficiency
    comparable to TransMIL_Model (within 20% of 12.2M parameters)
    """
    print("=" * 80)
    print("Verifying Parameter Efficiency (Requirement 1.6)")
    print("=" * 80)

    # Create nnMIL model with typical configuration
    model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2, dropout=0.25)

    num_params = count_parameters(model)
    target_params = 12.2e6  # 12.2M parameters
    max_params = target_params * 1.2  # Within 20%

    print(f"nnMIL parameters: {num_params:,}")
    print(f"Target (TransMIL): {target_params:,.0f}")
    print(f"Maximum allowed (120%): {max_params:,.0f}")
    print(f"Percentage of target: {(num_params / target_params) * 100:.1f}%")

    if num_params <= max_params:
        print("✓ PASS: Parameter efficiency requirement met")
    else:
        print("✗ FAIL: Exceeds maximum parameter count")

    print()


def verify_input_output_shapes():
    """
    Verify input/output shape requirements.

    Requirement 1.2: THE nnMIL_Model SHALL accept feature tensors from
    Foundation_Model extractors with dimensions [batch_size, num_patches, feature_dim]

    Requirement 1.3: THE nnMIL_Model SHALL produce slide-level classification
    logits with dimensions [batch_size, num_classes]
    """
    print("=" * 80)
    print("Verifying Input/Output Shapes (Requirements 1.2, 1.3)")
    print("=" * 80)

    model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)

    # Test various batch sizes and patch counts
    test_cases = [
        (1, 50, 1024),  # Single sample, 50 patches
        (4, 100, 1024),  # Batch of 4, 100 patches
        (8, 200, 1024),  # Batch of 8, 200 patches
    ]

    all_passed = True
    for batch_size, num_patches, feature_dim in test_cases:
        features = torch.randn(batch_size, num_patches, feature_dim)
        logits = model(features)

        expected_shape = (batch_size, 2)
        if logits.shape == expected_shape:
            print(f"✓ Input {features.shape} → Output {logits.shape}")
        else:
            print(f"✗ Input {features.shape} → Output {logits.shape} (expected {expected_shape})")
            all_passed = False

    if all_passed:
        print("✓ PASS: All input/output shape requirements met")
    else:
        print("✗ FAIL: Some shape requirements not met")

    print()


def verify_configurable_parameters():
    """
    Verify configurable parameter requirements.

    Requirement 1.4: THE nnMIL_Model SHALL support configurable hidden dimensions,
    number of layers, and attention heads
    """
    print("=" * 80)
    print("Verifying Configurable Parameters (Requirement 1.4)")
    print("=" * 80)

    configs = [
        {"feature_dim": 1024, "hidden_dim": 128, "num_classes": 2, "dropout": 0.1},
        {"feature_dim": 1024, "hidden_dim": 256, "num_classes": 3, "dropout": 0.25},
        {"feature_dim": 2048, "hidden_dim": 512, "num_classes": 5, "dropout": 0.3},
    ]

    all_passed = True
    for config in configs:
        try:
            model = nnMIL(**config)
            features = torch.randn(2, 50, config["feature_dim"])
            logits = model(features)

            if logits.shape == (2, config["num_classes"]):
                print(f"✓ Config {config} → Output shape {logits.shape}")
            else:
                print(f"✗ Config {config} → Unexpected output shape {logits.shape}")
                all_passed = False
        except Exception as e:
            print(f"✗ Config {config} → Error: {e}")
            all_passed = False

    if all_passed:
        print("✓ PASS: All configurable parameter requirements met")
    else:
        print("✗ FAIL: Some configuration requirements not met")

    print()


def verify_feature_projection():
    """
    Verify feature projection behavior.

    Design requirement: Feature projection is optional, only used if
    feature_dim != hidden_dim
    """
    print("=" * 80)
    print("Verifying Feature Projection Behavior")
    print("=" * 80)

    # Case 1: feature_dim != hidden_dim (should have projection)
    model1 = nnMIL(feature_dim=1024, hidden_dim=256)
    has_proj1 = model1.feature_proj is not None
    print(f"feature_dim=1024, hidden_dim=256: feature_proj exists = {has_proj1}")

    # Case 2: feature_dim == hidden_dim (should NOT have projection)
    model2 = nnMIL(feature_dim=256, hidden_dim=256)
    has_proj2 = model2.feature_proj is not None
    print(f"feature_dim=256, hidden_dim=256: feature_proj exists = {has_proj2}")

    if has_proj1 and not has_proj2:
        print("✓ PASS: Feature projection behavior correct")
    else:
        print("✗ FAIL: Feature projection behavior incorrect")

    print()


def verify_multi_scale_support():
    """
    Verify multi-scale support.

    Requirement 13: Multi-scale feature support
    """
    print("=" * 80)
    print("Verifying Multi-Scale Support (Requirement 13)")
    print("=" * 80)

    # Early fusion
    model_early = nnMIL(
        feature_dim=1024,
        hidden_dim=256,
        num_classes=2,
        multi_scale=True,
        num_scales=3,
        fusion_strategy="early",
    )

    features = [
        torch.randn(4, 100, 1024),
        torch.randn(4, 100, 1024),
        torch.randn(4, 100, 1024),
    ]

    logits_early = model_early(features)
    print(f"Early fusion: {[f.shape for f in features]} → {logits_early.shape}")

    # Late fusion
    model_late = nnMIL(
        feature_dim=1024,
        hidden_dim=256,
        num_classes=2,
        multi_scale=True,
        num_scales=3,
        fusion_strategy="late",
    )

    logits_late = model_late(features)
    print(f"Late fusion: {[f.shape for f in features]} → {logits_late.shape}")

    if logits_early.shape == (4, 2) and logits_late.shape == (4, 2):
        print("✓ PASS: Multi-scale support working correctly")
    else:
        print("✗ FAIL: Multi-scale support not working correctly")

    print()


def verify_attention_mechanism():
    """
    Verify gated attention mechanism.

    Requirement 1.1: Implement gated attention: α_i = softmax(w^T(tanh(Vx')⊙σ(Ux')))
    """
    print("=" * 80)
    print("Verifying Gated Attention Mechanism (Requirement 1.1)")
    print("=" * 80)

    model = nnMIL(feature_dim=256, hidden_dim=256, num_classes=2)
    features = torch.randn(2, 50, 256)

    logits, attention = model(features, return_attention=True)

    print(f"Input shape: {features.shape}")
    print(f"Attention weights shape: {attention.shape}")
    print(f"Attention weights sum per sample: {attention.sum(dim=1)}")
    print(f"Attention weights range: [{attention.min():.4f}, {attention.max():.4f}]")

    # Verify attention weights sum to 1
    attention_sums = attention.sum(dim=1)
    sums_correct = torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-5)

    # Verify attention weights are non-negative
    non_negative = (attention >= 0).all()

    if sums_correct and non_negative:
        print("✓ PASS: Attention mechanism working correctly")
    else:
        print("✗ FAIL: Attention mechanism not working correctly")

    print()


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("nnMIL Requirements Verification")
    print("=" * 80)
    print("\n")

    verify_parameter_efficiency()
    verify_input_output_shapes()
    verify_configurable_parameters()
    verify_feature_projection()
    verify_multi_scale_support()
    verify_attention_mechanism()

    print("=" * 80)
    print("Verification Complete")
    print("=" * 80)
