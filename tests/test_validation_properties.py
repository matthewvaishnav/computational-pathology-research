"""
Property-based tests for validation module using Hypothesis.

Tests universal correctness properties for tensor validation, shape checking,
and input validation across a wide range of inputs.
"""

import pytest
import torch
from hypothesis import given, settings, strategies as st
from hypothesis import assume
import hypothesis.extra.numpy as npst

from src.utils.validation import (
    validate_tensor_shape,
    validate_wsi_features,
    validate_genomic_features,
    validate_clinical_text,
    validate_multimodal_batch,
    is_validation_enabled,
    set_validation_enabled,
    ValidationError,
)


# ============================================================================
# Hypothesis Strategies
# ============================================================================


@st.composite
def tensor_with_shape(draw, min_dims=1, max_dims=3, min_size=1, max_size=20):
    """Generate a tensor with a random shape (simplified for speed)."""
    ndims = draw(st.integers(min_value=min_dims, max_value=max_dims))
    shape = tuple(draw(st.integers(min_value=min_size, max_value=max_size)) for _ in range(ndims))
    tensor = torch.randn(*shape)
    return tensor, shape


@st.composite
def valid_wsi_features(draw):
    """Generate valid WSI feature tensors."""
    batch_size = draw(st.integers(min_value=1, max_value=32))
    num_instances = draw(st.integers(min_value=1, max_value=500))
    feature_dim = draw(st.sampled_from([512, 1024, 2048]))
    return torch.randn(batch_size, num_instances, feature_dim)


@st.composite
def valid_genomic_features(draw):
    """Generate valid genomic feature tensors."""
    batch_size = draw(st.integers(min_value=1, max_value=32))
    num_genes = draw(st.integers(min_value=100, max_value=20000))
    return torch.randn(batch_size, num_genes)


@st.composite
def valid_clinical_features(draw):
    """Generate valid clinical text tokens."""
    batch_size = draw(st.integers(min_value=1, max_value=32))
    seq_length = draw(st.integers(min_value=10, max_value=512))
    return torch.randint(0, 30000, (batch_size, seq_length))


# ============================================================================
# Property Tests: Tensor Shape Validation
# ============================================================================


class TestTensorShapeProperties:
    """Property-based tests for tensor shape validation."""

    @given(tensor_with_shape())
    @settings(max_examples=20, deadline=5000)
    def test_valid_shape_always_passes(self, tensor_and_shape):
        """Property: Validating a tensor with its actual shape always succeeds."""
        tensor, shape = tensor_and_shape
        # Should not raise
        validate_tensor_shape(tensor, shape, "test_tensor")

    @given(tensor_with_shape())
    @settings(max_examples=20, deadline=5000)
    def test_none_dimensions_allow_any_size(self, tensor_and_shape):
        """Property: None in expected shape allows any size for that dimension."""
        tensor, shape = tensor_and_shape
        # Replace all dimensions with None
        flexible_shape = tuple(None for _ in shape)
        # Should not raise
        validate_tensor_shape(tensor, flexible_shape, "test_tensor")

    @given(tensor_with_shape(min_dims=2, max_dims=4))
    @settings(max_examples=20, deadline=5000)
    def test_wrong_ndims_always_fails(self, tensor_and_shape):
        """Property: Wrong number of dimensions always raises ValidationError."""
        tensor, shape = tensor_and_shape
        assume(len(shape) > 1)  # Ensure we can add a dimension
        
        # Add an extra dimension to expected shape
        wrong_shape = shape + (10,)
        
        with pytest.raises(ValidationError, match="Shape mismatch"):
            validate_tensor_shape(tensor, wrong_shape, "test_tensor")

    @given(tensor_with_shape(min_dims=2, max_dims=4))
    @settings(max_examples=20, deadline=5000)
    def test_wrong_dimension_size_fails(self, tensor_and_shape):
        """Property: Wrong dimension size raises ValidationError."""
        tensor, shape = tensor_and_shape
        assume(len(shape) >= 2)
        
        # Change last dimension size
        wrong_shape = shape[:-1] + (shape[-1] + 1,)
        
        with pytest.raises(ValidationError, match="Shape mismatch"):
            validate_tensor_shape(tensor, wrong_shape, "test_tensor")

    @given(st.integers(), st.floats(), st.text(), st.lists(st.integers()))
    @settings(max_examples=10, deadline=5000)
    def test_non_tensor_input_fails(self, non_tensor):
        """Property: Non-tensor inputs always raise ValidationError."""
        with pytest.raises(ValidationError, match="Expected .* to be a torch.Tensor"):
            validate_tensor_shape(non_tensor, (10,), "test_tensor")


# ============================================================================
# Property Tests: WSI Feature Validation
# ============================================================================


class TestWSIFeatureProperties:
    """Property-based tests for WSI feature validation."""

    @given(valid_wsi_features())
    @settings(max_examples=20, deadline=5000)
    def test_valid_wsi_features_pass(self, features):
        """Property: Valid WSI features always pass validation."""
        # Should not raise
        validate_wsi_features(features)

    @given(
        st.integers(min_value=1, max_value=32),
        st.integers(min_value=1, max_value=500),
        st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=20, deadline=5000)
    def test_wrong_ndims_fails(self, batch_size, num_instances, wrong_dim):
        """Property: Wrong number of dimensions always fails."""
        # Create 2D tensor instead of 3D
        features = torch.randn(batch_size, num_instances)
        
        with pytest.raises(ValidationError, match="Expected 3D tensor"):
            validate_wsi_features(features)

    @given(
        st.integers(min_value=1, max_value=32),
        st.integers(min_value=1, max_value=500),
        st.integers(min_value=1, max_value=511),  # Invalid feature dim
    )
    @settings(max_examples=20, deadline=5000)
    def test_invalid_feature_dim_fails(self, batch_size, num_instances, feature_dim):
        """Property: Invalid feature dimensions fail validation."""
        assume(feature_dim not in [512, 1024, 2048, 4096])
        
        features = torch.randn(batch_size, num_instances, feature_dim)
        
        with pytest.raises(ValidationError, match="feature dimension"):
            validate_wsi_features(features)


# ============================================================================
# Property Tests: Genomic Feature Validation
# ============================================================================


class TestGenomicFeatureProperties:
    """Property-based tests for genomic feature validation."""

    @given(valid_genomic_features())
    @settings(max_examples=20, deadline=5000)
    def test_valid_genomic_features_pass(self, features):
        """Property: Valid genomic features always pass validation."""
        # Should not raise
        validate_genomic_features(features)

    @given(
        st.integers(min_value=1, max_value=32),
        st.integers(min_value=1, max_value=99),  # Too few genes
    )
    @settings(max_examples=20, deadline=5000)
    def test_too_few_genes_fails(self, batch_size, num_genes):
        """Property: Too few genes always fails validation."""
        features = torch.randn(batch_size, num_genes)
        
        with pytest.raises(ValidationError, match="at least 100 genes"):
            validate_genomic_features(features)

    @given(st.integers(min_value=1, max_value=32))
    @settings(max_examples=20, deadline=5000)
    def test_wrong_ndims_fails(self, batch_size):
        """Property: Wrong number of dimensions always fails."""
        # Create 3D tensor instead of 2D
        features = torch.randn(batch_size, 1000, 10)
        
        with pytest.raises(ValidationError, match="Expected 2D tensor"):
            validate_genomic_features(features)


# ============================================================================
# Property Tests: Clinical Feature Validation
# ============================================================================


class TestClinicalFeatureProperties:
    """Property-based tests for clinical text validation."""

    @given(valid_clinical_features())
    @settings(max_examples=20, deadline=5000)
    def test_valid_clinical_text_passes(self, tokens):
        """Property: Valid clinical text tokens always pass validation."""
        # Should not raise
        validate_clinical_text(tokens)

    @given(st.integers(min_value=1, max_value=32))
    @settings(max_examples=20, deadline=5000)
    def test_wrong_ndims_fails(self, batch_size):
        """Property: Wrong number of dimensions always fails."""
        # Create 3D tensor instead of 2D
        tokens = torch.randint(0, 30000, (batch_size, 100, 10))
        
        with pytest.raises(ValidationError, match="Expected 2D tensor"):
            validate_clinical_text(tokens)


# ============================================================================
# Property Tests: Batch Validation
# ============================================================================


class TestBatchValidationProperties:
    """Property-based tests for multimodal batch validation."""

    @given(
        valid_wsi_features(),
        valid_genomic_features(),
        valid_clinical_features(),
    )
    @settings(max_examples=20, deadline=5000)
    def test_consistent_batch_sizes_pass(self, wsi, genomic, clinical):
        """Property: Batches with consistent batch sizes always pass."""
        # Ensure all have same batch size
        batch_size = wsi.shape[0]
        genomic = genomic[:batch_size]
        clinical = clinical[:batch_size]
        
        batch = {
            "wsi_features": wsi,
            "genomic_features": genomic,
            "clinical_text": clinical,
            "labels": torch.randint(0, 2, (batch_size,)),
        }
        
        # Should not raise
        validate_multimodal_batch(batch)

    @given(
        valid_wsi_features(),
        valid_genomic_features(),
    )
    @settings(max_examples=20, deadline=5000)
    def test_inconsistent_batch_sizes_fail(self, wsi, genomic):
        """Property: Inconsistent batch sizes always fail validation."""
        assume(wsi.shape[0] != genomic.shape[0])
        
        batch = {
            "wsi_features": wsi,
            "genomic_features": genomic,
            "labels": torch.randint(0, 2, (wsi.shape[0],)),
        }
        
        with pytest.raises(ValidationError, match="Inconsistent batch sizes"):
            validate_multimodal_batch(batch)

    @given(valid_wsi_features())
    @settings(max_examples=20, deadline=5000)
    def test_missing_labels_fails(self, wsi):
        """Property: Missing labels always fails validation."""
        batch = {"wsi_features": wsi}
        
        with pytest.raises(ValidationError, match="Missing required key"):
            validate_multimodal_batch(batch)


# ============================================================================
# Property Tests: Validation Enable/Disable
# ============================================================================


class TestValidationToggleProperties:
    """Property-based tests for validation enable/disable."""

    @given(tensor_with_shape())
    @settings(max_examples=20, deadline=5000)
    def test_disabled_validation_never_raises(self, tensor_and_shape):
        """Property: Disabled validation never raises errors."""
        tensor, shape = tensor_and_shape
        
        # Disable validation
        original_state = is_validation_enabled()
        set_validation_enabled(False)
        
        try:
            # Use wrong shape - should not raise
            wrong_shape = shape + (999,)
            validate_tensor_shape(tensor, wrong_shape, "test_tensor")
        finally:
            # Restore original state
            set_validation_enabled(original_state)

    @given(st.booleans())
    @settings(max_examples=20, deadline=5000)
    def test_toggle_is_idempotent(self, enabled):
        """Property: Setting validation state is idempotent."""
        original_state = is_validation_enabled()
        
        try:
            set_validation_enabled(enabled)
            assert is_validation_enabled() == enabled
            
            # Set again - should still be the same
            set_validation_enabled(enabled)
            assert is_validation_enabled() == enabled
        finally:
            set_validation_enabled(original_state)


# ============================================================================
# Property Tests: NaN and Inf Handling
# ============================================================================


class TestNaNInfProperties:
    """Property-based tests for NaN and Inf handling."""

    @given(
        st.integers(min_value=1, max_value=32),
        st.integers(min_value=10, max_value=100),
        st.integers(min_value=512, max_value=2048),
    )
    @settings(max_examples=20, deadline=5000)
    def test_nan_values_detected(self, batch_size, num_instances, feature_dim):
        """Property: NaN values are always detected in validation."""
        features = torch.randn(batch_size, num_instances, feature_dim)
        # Inject NaN
        features[0, 0, 0] = float('nan')
        
        with pytest.raises(ValidationError, match="contains NaN"):
            validate_wsi_features(features)

    @given(
        st.integers(min_value=1, max_value=32),
        st.integers(min_value=10, max_value=100),
        st.integers(min_value=512, max_value=2048),
    )
    @settings(max_examples=20, deadline=5000)
    def test_inf_values_detected(self, batch_size, num_instances, feature_dim):
        """Property: Inf values are always detected in validation."""
        features = torch.randn(batch_size, num_instances, feature_dim)
        # Inject Inf
        features[0, 0, 0] = float('inf')
        
        with pytest.raises(ValidationError, match="contains Inf"):
            validate_wsi_features(features)


