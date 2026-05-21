"""
Property-based tests for foundation model compatibility.

This test file validates correctness properties for the FoundationModelAdapter
and nnMIL compatibility with various foundation models using property-based
testing with Hypothesis. Each property test runs a minimum of 100 iterations
to verify universal invariants.

Feature: nnmil-architecture-upgrade
"""


import pytest
import torch

from hypothesis import given, settings
from hypothesis import strategies as st
from src.models.foundation_adapter import FoundationModelAdapter
from src.models.mil.nnmil import nnMIL

# Foundation model dimensions (real-world values)
FOUNDATION_MODELS = {"UNI": 1024, "CONCH": 512, "Phikon": 768, "ResNet50": 2048}


# ============================================================================
# Property 24: Foundation Model Compatibility
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    foundation_model=st.sampled_from(list(FOUNDATION_MODELS.keys())),
    batch_size=st.integers(min_value=1, max_value=16),
    num_patches=st.integers(min_value=10, max_value=500),
    hidden_dim=st.integers(min_value=64, max_value=512),
    num_classes=st.integers(min_value=2, max_value=10),
)
def test_property_24_foundation_model_compatibility(
    foundation_model, batch_size, num_patches, hidden_dim, num_classes
):
    """
    Feature: nnmil-architecture-upgrade, Property 24: For any feature tensor
    with dimensions matching UNI (1024), CONCH (512), Phikon (768), or
    ResNet50 (2048), the nnMIL model SHALL successfully process the features.

    **Validates: Requirements 7.1**
    """
    feature_dim = FOUNDATION_MODELS[foundation_model]

    # Create nnMIL model with foundation model features
    model = nnMIL(feature_dim=feature_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    model.eval()

    # Create features from the foundation model
    features = torch.randn(batch_size, num_patches, feature_dim)

    # Should successfully process features without error
    try:
        with torch.no_grad():
            logits = model(features)

        # Verify output shape
        expected_shape = (batch_size, num_classes)
        assert logits.shape == expected_shape, (
            f"Expected output shape {expected_shape}, got {logits.shape} "
            f"for {foundation_model} features"
        )

        # Verify output is valid (no NaN/Inf)
        assert not torch.isnan(logits).any(), f"NaN detected in {foundation_model} output"
        assert not torch.isinf(logits).any(), f"Inf detected in {foundation_model} output"

    except Exception as e:
        pytest.fail(f"nnMIL failed to process {foundation_model} features: {e}")


# ============================================================================
# Property 25: Automatic Dimension Detection
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    input_feature_dim=st.sampled_from([256, 512, 768, 1024, 1536, 2048]),
    batch_size=st.integers(min_value=1, max_value=8),
    num_patches=st.integers(min_value=10, max_value=200),
)
def test_property_25_automatic_dimension_detection(input_feature_dim, batch_size, num_patches):
    """
    Feature: nnmil-architecture-upgrade, Property 25: For any input feature
    tensor, the model SHALL correctly detect the feature dimension from the
    tensor shape.

    **Validates: Requirements 7.2**
    """
    # Create foundation model adapter
    adapter = FoundationModelAdapter()

    # Create input features with specific dimension
    features = torch.randn(batch_size, num_patches, input_feature_dim)

    # Adapter should automatically detect feature dimension
    detected_dim = adapter.detect_feature_dimension(features)

    # Verify correct detection
    assert (
        detected_dim == input_feature_dim
    ), f"Expected detected dimension {input_feature_dim}, got {detected_dim}"


# ============================================================================
# Property 26: Adaptive Projection
# ============================================================================


@settings(max_examples=50, deadline=None)
@given(
    input_feature_dim=st.integers(min_value=256, max_value=2048),
    target_hidden_dim=st.integers(min_value=64, max_value=512),
    batch_size=st.integers(min_value=1, max_value=8),
    num_patches=st.integers(min_value=10, max_value=200),
)
def test_property_26_adaptive_projection(
    input_feature_dim, target_hidden_dim, batch_size, num_patches
):
    """
    Feature: nnmil-architecture-upgrade, Property 26: For any input with
    feature_dim != configured hidden_dim, the model SHALL apply a learned
    projection layer to match dimensions.

    **Validates: Requirements 7.3**
    """
    # Only test when dimensions differ (projection needed)
    if input_feature_dim == target_hidden_dim:
        return  # Skip when no projection needed

    # Create nnMIL model with different input and hidden dimensions
    model = nnMIL(feature_dim=input_feature_dim, hidden_dim=target_hidden_dim, num_classes=2)

    # Verify projection layer exists when dimensions differ
    assert model.feature_proj is not None, (
        f"Feature projection should exist when feature_dim ({input_feature_dim}) "
        f"!= hidden_dim ({target_hidden_dim})"
    )

    # Create input features
    features = torch.randn(batch_size, num_patches, input_feature_dim)

    # Forward pass should work with projection
    with torch.no_grad():
        logits = model(features)

    # Verify output shape is correct
    assert logits.shape == (batch_size, 2)

    # Verify projection transforms to correct dimension
    if hasattr(model.feature_proj, "__getitem__"):
        # Sequential projection layer
        proj_layer = model.feature_proj[0]  # First layer should be Linear
        if hasattr(proj_layer, "out_features"):
            assert proj_layer.out_features == target_hidden_dim


# ============================================================================
# Property 27: Weight Freezing
# ============================================================================


def test_property_27_weight_freezing():
    """
    Feature: nnmil-architecture-upgrade, Property 27: For any foundation model
    with frozen weights, training SHALL not update those weights (gradient
    should be None or zero).

    **Validates: Requirements 7.4**
    """
    # Create foundation model adapter with freezing capability
    adapter = FoundationModelAdapter()

    # Create mock foundation model (simple linear layer)
    foundation_model = torch.nn.Linear(1024, 1024)

    # Freeze the foundation model
    adapter.freeze_foundation_model(foundation_model)

    # Verify all parameters have requires_grad=False
    for param in foundation_model.parameters():
        assert (
            not param.requires_grad
        ), "Frozen foundation model parameters should not require gradients"

    # Test that gradients are not computed during backward pass
    input_tensor = torch.randn(4, 100, 1024, requires_grad=True)
    output = foundation_model(input_tensor)
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Verify foundation model parameters have no gradients
    for param in foundation_model.parameters():
        assert param.grad is None, "Frozen parameters should not accumulate gradients"


def test_foundation_model_unfreezing():
    """Test unfreezing foundation model weights for fine-tuning."""
    adapter = FoundationModelAdapter()

    # Create and freeze foundation model
    foundation_model = torch.nn.Linear(1024, 1024)
    adapter.freeze_foundation_model(foundation_model)

    # Verify frozen
    for param in foundation_model.parameters():
        assert not param.requires_grad

    # Unfreeze for fine-tuning
    adapter.unfreeze_foundation_model(foundation_model)

    # Verify unfrozen
    for param in foundation_model.parameters():
        assert param.requires_grad, "Unfrozen parameters should require gradients"


def test_selective_layer_freezing():
    """Test freezing only specific layers of foundation model."""
    adapter = FoundationModelAdapter()

    # Create multi-layer foundation model
    foundation_model = torch.nn.Sequential(
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1024),
    )

    # Freeze only first two layers
    layers_to_freeze = [0, 1]  # First Linear and ReLU
    adapter.freeze_specific_layers(foundation_model, layers_to_freeze)

    # Verify selective freezing
    for i, layer in enumerate(foundation_model):
        if hasattr(layer, "parameters"):
            for param in layer.parameters():
                if i in layers_to_freeze:
                    assert not param.requires_grad, f"Layer {i} should be frozen"
                else:
                    assert param.requires_grad, f"Layer {i} should not be frozen"


# ============================================================================
# Integration Tests with Real Foundation Model Patterns
# ============================================================================


def test_uni_integration():
    """Test integration with UNI foundation model pattern."""
    # UNI: 1024-dimensional features
    model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)

    # Simulate UNI features (typical patch count and batch size)
    features = torch.randn(4, 256, 1024)  # 4 slides, 256 patches each

    with torch.no_grad():
        logits = model(features)

    assert logits.shape == (4, 2)
    assert not torch.isnan(logits).any()


def test_conch_integration():
    """Test integration with CONCH foundation model pattern."""
    # CONCH: 512-dimensional features
    model = nnMIL(feature_dim=512, hidden_dim=256, num_classes=3)

    # Simulate CONCH features
    features = torch.randn(2, 512, 512)  # 2 slides, 512 patches each

    with torch.no_grad():
        logits = model(features)

    assert logits.shape == (2, 3)
    assert not torch.isnan(logits).any()


def test_phikon_integration():
    """Test integration with Phikon foundation model pattern."""
    # Phikon: 768-dimensional features
    model = nnMIL(feature_dim=768, hidden_dim=256, num_classes=5)

    # Simulate Phikon features
    features = torch.randn(3, 128, 768)  # 3 slides, 128 patches each

    with torch.no_grad():
        logits = model(features)

    assert logits.shape == (3, 5)
    assert not torch.isnan(logits).any()


def test_resnet50_integration():
    """Test integration with ResNet50 foundation model pattern."""
    # ResNet50: 2048-dimensional features
    model = nnMIL(feature_dim=2048, hidden_dim=512, num_classes=2)

    # Simulate ResNet50 features
    features = torch.randn(1, 1024, 2048)  # 1 slide, 1024 patches

    with torch.no_grad():
        logits = model(features)

    assert logits.shape == (1, 2)
    assert not torch.isnan(logits).any()


def test_multi_scale_foundation_compatibility():
    """Test multi-scale compatibility with different foundation models."""
    # Multi-scale with different foundation models per scale
    model = nnMIL(
        feature_dim=1024,  # Will be overridden by multi-scale
        hidden_dim=256,
        num_classes=2,
        multi_scale=True,
        num_scales=3,
        fusion_strategy="early",
    )

    # Different foundation model features per scale
    scale1_features = torch.randn(2, 100, 1024)  # UNI
    scale2_features = torch.randn(2, 100, 512)  # CONCH
    scale3_features = torch.randn(2, 100, 768)  # Phikon

    multi_scale_features = [scale1_features, scale2_features, scale3_features]

    with torch.no_grad():
        logits = model(multi_scale_features)

    assert logits.shape == (2, 2)
    assert not torch.isnan(logits).any()


def test_foundation_model_performance_consistency():
    """Test that performance is consistent across foundation models."""
    batch_size, num_patches, num_classes = 4, 100, 2

    results = {}

    for model_name, feature_dim in FOUNDATION_MODELS.items():
        # Create model for this foundation model
        model = nnMIL(feature_dim=feature_dim, hidden_dim=256, num_classes=num_classes)
        model.eval()

        # Create features
        features = torch.randn(batch_size, num_patches, feature_dim)

        # Forward pass
        with torch.no_grad():
            logits = model(features)

        # Store results
        results[model_name] = {
            "shape": logits.shape,
            "mean": logits.mean().item(),
            "std": logits.std().item(),
        }

    # Verify all models produce same output shape
    expected_shape = (batch_size, num_classes)
    for model_name, result in results.items():
        assert (
            result["shape"] == expected_shape
        ), f"{model_name} produced wrong shape: {result['shape']}"

    # Verify outputs are in reasonable range (not all zeros or extreme values)
    for model_name, result in results.items():
        assert abs(result["mean"]) < 10, f"{model_name} mean too extreme: {result['mean']}"
        assert 0.1 < result["std"] < 10, f"{model_name} std too extreme: {result['std']}"


def test_adapter_memory_efficiency():
    """Test that adapter doesn't significantly increase memory usage."""
    adapter = FoundationModelAdapter()

    # Test with large feature tensors
    large_features = torch.randn(8, 1000, 2048)  # Large batch

    # Memory usage should be reasonable
    import os

    import psutil

    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    # Process features through adapter
    adapter.detect_feature_dimension(large_features)

    memory_after = process.memory_info().rss
    memory_increase = memory_after - memory_before

    # Memory increase should be minimal (< 100MB for this test)
    assert (
        memory_increase < 100 * 1024 * 1024
    ), f"Adapter used too much memory: {memory_increase / 1024 / 1024:.1f} MB"


def test_adapter_batch_processing():
    """Test adapter handles different batch sizes efficiently."""
    adapter = FoundationModelAdapter()

    # Test various batch sizes
    batch_sizes = [1, 4, 8, 16, 32]
    feature_dim = 1024
    num_patches = 200

    for batch_size in batch_sizes:
        features = torch.randn(batch_size, num_patches, feature_dim)

        # Should handle any batch size
        detected_dim = adapter.detect_feature_dimension(features)
        assert detected_dim == feature_dim

        # Processing time should scale reasonably with batch size
        import time

        start_time = time.time()

        # Simulate some processing
        _ = adapter.detect_feature_dimension(features)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete quickly (< 1 second even for large batches)
        assert (
            processing_time < 1.0
        ), f"Processing took too long for batch_size={batch_size}: {processing_time:.3f}s"
