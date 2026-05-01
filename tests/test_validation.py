"""
Tests for input validation utilities.

Tests tensor validation, modality-specific validation, and batch validation.
"""

import pytest
import torch
import os

from src.utils.validation import (
    is_validation_enabled,
    set_validation_enabled,
    ValidationError,
    validate_tensor_shape,
    validate_tensor_range,
    validate_no_nan_inf,
    validate_batch_size,
    validate_wsi_features,
    validate_genomic_features,
    validate_clinical_text,
    validate_multimodal_batch,
    validate_inputs,
    get_validation_summary,
)


@pytest.fixture(autouse=True)
def enable_validation():
    """Ensure validation is enabled for all tests."""
    set_validation_enabled(True)
    yield
    set_validation_enabled(True)  # Reset after test


class TestValidationControl:
    """Test validation enable/disable functionality."""

    def test_validation_enabled_by_default(self):
        """Test validation is enabled by default."""
        assert is_validation_enabled() is True

    def test_set_validation_enabled(self):
        """Test enabling/disabling validation."""
        set_validation_enabled(False)
        assert is_validation_enabled() is False
        
        set_validation_enabled(True)
        assert is_validation_enabled() is True

    def test_validation_disabled_skips_checks(self):
        """Test that disabled validation skips checks."""
        set_validation_enabled(False)
        
        # Should not raise even with invalid tensor
        invalid_tensor = torch.randn(10, 20)
        validate_tensor_shape(invalid_tensor, (5, 10), "test")  # Wrong shape
        
        set_validation_enabled(True)


class TestTensorShapeValidation:
    """Test tensor shape validation."""

    def test_valid_shape_exact_match(self):
        """Test validation passes with exact shape match."""
        tensor = torch.randn(16, 100, 1024)
        validate_tensor_shape(tensor, (16, 100, 1024), "test_tensor")
        # Should not raise

    def test_valid_shape_with_none_dimensions(self):
        """Test validation passes with None for variable dimensions."""
        tensor = torch.randn(16, 100, 1024)
        validate_tensor_shape(tensor, (None, 100, 1024), "test_tensor")
        validate_tensor_shape(tensor, (16, None, 1024), "test_tensor")
        validate_tensor_shape(tensor, (None, None, None), "test_tensor")
        # Should not raise

    def test_invalid_shape_dimension_count(self):
        """Test validation fails with wrong number of dimensions."""
        tensor = torch.randn(16, 100)
        
        with pytest.raises(ValidationError, match="Shape mismatch"):
            validate_tensor_shape(tensor, (16, 100, 1024), "test_tensor")

    def test_invalid_shape_dimension_size(self):
        """Test validation fails with wrong dimension size."""
        tensor = torch.randn(16, 100, 1024)
        
        with pytest.raises(ValidationError, match="Shape mismatch.*dimension"):
            validate_tensor_shape(tensor, (16, 100, 512), "test_tensor")

    def test_non_tensor_input(self):
        """Test validation fails with non-tensor input."""
        with pytest.raises(ValidationError, match="Expected.*torch.Tensor"):
            validate_tensor_shape([1, 2, 3], (3,), "test_tensor")


class TestTensorRangeValidation:
    """Test tensor value range validation."""

    def test_valid_range(self):
        """Test validation passes with values in range."""
        tensor = torch.randn(16, 1024) * 0.1  # Small values
        validate_tensor_range(tensor, -1.0, 1.0, "test_tensor")
        # Should not raise

    def test_invalid_range_below_min(self):
        """Test validation fails with values below minimum."""
        tensor = torch.tensor([-2.0, 0.5, 0.8])
        
        with pytest.raises(ValidationError, match="Value range error"):
            validate_tensor_range(tensor, -1.0, 1.0, "test_tensor")

    def test_invalid_range_above_max(self):
        """Test validation fails with values above maximum."""
        tensor = torch.tensor([0.5, 0.8, 2.0])
        
        with pytest.raises(ValidationError, match="Value range error"):
            validate_tensor_range(tensor, -1.0, 1.0, "test_tensor")

    def test_range_with_only_min(self):
        """Test validation with only minimum specified."""
        tensor = torch.tensor([0.5, 1.0, 2.0])
        validate_tensor_range(tensor, 0.0, None, "test_tensor")
        # Should not raise

    def test_range_with_only_max(self):
        """Test validation with only maximum specified."""
        tensor = torch.tensor([-1.0, 0.0, 0.5])
        validate_tensor_range(tensor, None, 1.0, "test_tensor")
        # Should not raise


class TestNaNInfValidation:
    """Test NaN and Inf validation."""

    def test_valid_no_nan_inf(self):
        """Test validation passes with no NaN/Inf."""
        tensor = torch.randn(16, 1024)
        validate_no_nan_inf(tensor, "test_tensor")
        # Should not raise

    def test_invalid_with_nan(self):
        """Test validation fails with NaN values."""
        tensor = torch.randn(16, 1024)
        tensor[0, 0] = float('nan')
        
        with pytest.raises(ValidationError, match="Invalid values"):
            validate_no_nan_inf(tensor, "test_tensor")

    def test_invalid_with_inf(self):
        """Test validation fails with Inf values."""
        tensor = torch.randn(16, 1024)
        tensor[0, 0] = float('inf')
        
        with pytest.raises(ValidationError, match="Invalid values"):
            validate_no_nan_inf(tensor, "test_tensor")

    def test_invalid_with_both_nan_and_inf(self):
        """Test validation fails with both NaN and Inf."""
        tensor = torch.randn(16, 1024)
        tensor[0, 0] = float('nan')
        tensor[1, 0] = float('inf')
        
        with pytest.raises(ValidationError, match="Invalid values"):
            validate_no_nan_inf(tensor, "test_tensor")


class TestBatchSizeValidation:
    """Test batch size validation."""

    def test_valid_batch_size(self):
        """Test validation passes with correct batch size."""
        tensor = torch.randn(16, 1024)
        validate_batch_size(tensor, 16, "test_tensor")
        # Should not raise

    def test_invalid_batch_size(self):
        """Test validation fails with wrong batch size."""
        tensor = torch.randn(16, 1024)
        
        with pytest.raises(ValidationError, match="Batch size mismatch"):
            validate_batch_size(tensor, 32, "test_tensor")

    def test_scalar_tensor(self):
        """Test validation fails with scalar tensor."""
        tensor = torch.tensor(5.0)
        
        with pytest.raises(ValidationError, match="scalar"):
            validate_batch_size(tensor, 1, "test_tensor")


class TestWSIFeaturesValidation:
    """Test WSI features validation."""

    def test_valid_wsi_features(self):
        """Test validation passes with valid WSI features."""
        features = torch.randn(16, 100, 1024)
        validate_wsi_features(features)
        # Should not raise

    def test_valid_wsi_features_custom_dim(self):
        """Test validation passes with custom feature dimension."""
        features = torch.randn(16, 100, 512)
        validate_wsi_features(features, expected_feature_dim=512)
        # Should not raise

    def test_invalid_wsi_features_wrong_dims(self):
        """Test validation fails with wrong number of dimensions."""
        features = torch.randn(16, 1024)  # 2D instead of 3D
        
        with pytest.raises(ValidationError, match="Invalid shape"):
            validate_wsi_features(features)

    def test_invalid_wsi_features_wrong_feature_dim(self):
        """Test validation fails with wrong feature dimension."""
        features = torch.randn(16, 100, 512)
        
        with pytest.raises(ValidationError, match="Feature dimension mismatch"):
            validate_wsi_features(features, expected_feature_dim=1024)

    def test_invalid_wsi_features_zero_patches(self):
        """Test validation fails with zero patches."""
        features = torch.randn(16, 0, 1024)
        
        with pytest.raises(ValidationError, match="Invalid number of patches"):
            validate_wsi_features(features)

    def test_invalid_wsi_features_with_nan(self):
        """Test validation fails with NaN values."""
        features = torch.randn(16, 100, 1024)
        features[0, 0, 0] = float('nan')
        
        with pytest.raises(ValidationError, match="Invalid values"):
            validate_wsi_features(features)


class TestGenomicFeaturesValidation:
    """Test genomic features validation."""

    def test_valid_genomic_features(self):
        """Test validation passes with valid genomic features."""
        features = torch.randn(16, 2000)
        validate_genomic_features(features)
        # Should not raise

    def test_valid_genomic_features_custom_dim(self):
        """Test validation passes with custom gene count."""
        features = torch.randn(16, 1000)
        validate_genomic_features(features, expected_feature_dim=1000)
        # Should not raise

    def test_invalid_genomic_features_wrong_dims(self):
        """Test validation fails with wrong number of dimensions."""
        features = torch.randn(16, 100, 2000)  # 3D instead of 2D
        
        with pytest.raises(ValidationError, match="Invalid shape"):
            validate_genomic_features(features)

    def test_invalid_genomic_features_wrong_gene_count(self):
        """Test validation fails with wrong gene count."""
        features = torch.randn(16, 1000)
        
        with pytest.raises(ValidationError, match="Feature dimension mismatch"):
            validate_genomic_features(features, expected_feature_dim=2000)


class TestClinicalTextValidation:
    """Test clinical text validation."""

    def test_valid_clinical_text(self):
        """Test validation passes with valid clinical text."""
        tokens = torch.randint(0, 30000, (16, 128))
        validate_clinical_text(tokens)
        # Should not raise

    def test_valid_clinical_text_with_max_length(self):
        """Test validation passes with max length check."""
        tokens = torch.randint(0, 30000, (16, 128))
        validate_clinical_text(tokens, max_seq_length=512)
        # Should not raise

    def test_invalid_clinical_text_wrong_dims(self):
        """Test validation fails with wrong number of dimensions."""
        tokens = torch.randint(0, 30000, (16,))  # 1D instead of 2D
        
        with pytest.raises(ValidationError, match="Invalid shape"):
            validate_clinical_text(tokens)

    def test_invalid_clinical_text_exceeds_max_length(self):
        """Test validation fails when sequence exceeds max length."""
        tokens = torch.randint(0, 30000, (16, 1000))
        
        with pytest.raises(ValidationError, match="Sequence length exceeds maximum"):
            validate_clinical_text(tokens, max_seq_length=512)

    def test_invalid_clinical_text_wrong_dtype(self):
        """Test validation fails with wrong dtype."""
        tokens = torch.randn(16, 128)  # Float instead of int
        
        with pytest.raises(ValidationError, match="Invalid dtype"):
            validate_clinical_text(tokens)

    def test_invalid_clinical_text_negative_tokens(self):
        """Test validation fails with negative token IDs."""
        tokens = torch.randint(-10, 30000, (16, 128))
        
        with pytest.raises(ValidationError, match="negative token IDs"):
            validate_clinical_text(tokens)


class TestMultimodalBatchValidation:
    """Test multimodal batch validation."""

    def test_valid_complete_batch(self):
        """Test validation passes with complete batch."""
        batch = {
            'wsi_features': torch.randn(16, 100, 1024),
            'genomic': torch.randn(16, 2000),
            'clinical_text': torch.randint(0, 30000, (16, 128)),
            'labels': torch.randint(0, 2, (16,))
        }
        validate_multimodal_batch(batch)
        # Should not raise

    def test_valid_partial_batch(self):
        """Test validation passes with partial modalities."""
        batch = {
            'wsi_features': torch.randn(16, 100, 1024),
            'labels': torch.randint(0, 2, (16,))
        }
        validate_multimodal_batch(batch, require_all_modalities=False)
        # Should not raise

    def test_invalid_batch_not_dict(self):
        """Test validation fails with non-dict batch."""
        with pytest.raises(ValidationError, match="Expected batch to be a dictionary"):
            validate_multimodal_batch([1, 2, 3])

    def test_invalid_batch_size_mismatch(self):
        """Test validation fails with mismatched batch sizes."""
        batch = {
            'wsi_features': torch.randn(16, 100, 1024),
            'genomic': torch.randn(32, 2000),  # Wrong batch size
        }
        
        with pytest.raises(ValidationError, match="Batch size mismatch"):
            validate_multimodal_batch(batch)

    def test_invalid_batch_missing_required_modality(self):
        """Test validation fails when required modality is missing."""
        batch = {
            'wsi_features': torch.randn(16, 100, 1024),
        }
        
        with pytest.raises(ValidationError, match="Missing required modality"):
            validate_multimodal_batch(batch, require_all_modalities=True)

    def test_valid_batch_with_variable_length_wsi(self):
        """Test validation passes with variable length WSI features."""
        batch = {
            'wsi_features': [
                torch.randn(50, 1024),
                torch.randn(100, 1024),
                torch.randn(75, 1024),
            ] * 5 + [torch.randn(80, 1024)],  # 16 samples
            'labels': torch.randint(0, 2, (16,))
        }
        validate_multimodal_batch(batch)
        # Should not raise


class TestValidateInputsDecorator:
    """Test validate_inputs decorator."""

    def test_decorator_validates_batch(self):
        """Test decorator validates batch argument."""
        @validate_inputs
        def forward(self, batch):
            return batch
        
        valid_batch = {
            'wsi_features': torch.randn(16, 100, 1024),
            'labels': torch.randint(0, 2, (16,))
        }
        
        # Should not raise
        result = forward(None, valid_batch)
        assert result == valid_batch

    def test_decorator_catches_invalid_batch(self):
        """Test decorator catches invalid batch."""
        @validate_inputs
        def forward(self, batch):
            return batch
        
        invalid_batch = {
            'wsi_features': torch.randn(16, 100, 512),  # Wrong feature dim
        }
        
        with pytest.raises(ValidationError, match="Validation failed"):
            forward(None, invalid_batch)

    def test_decorator_skips_when_disabled(self):
        """Test decorator skips validation when disabled."""
        set_validation_enabled(False)
        
        @validate_inputs
        def forward(self, batch):
            return batch
        
        invalid_batch = {
            'wsi_features': torch.randn(16, 100, 512),  # Wrong feature dim
        }
        
        # Should not raise when validation disabled
        result = forward(None, invalid_batch)
        assert result == invalid_batch
        
        set_validation_enabled(True)


class TestValidationSummary:
    """Test validation summary utility."""

    def test_summary_with_tensors(self):
        """Test summary generation with tensor batch."""
        batch = {
            'wsi_features': torch.randn(16, 100, 1024),
            'genomic': torch.randn(16, 2000),
            'labels': torch.randint(0, 2, (16,))
        }
        
        summary = get_validation_summary(batch)
        
        assert "Batch Summary:" in summary
        assert "wsi_features" in summary
        assert "genomic" in summary
        assert "labels" in summary
        assert "[16, 100, 1024]" in summary

    def test_summary_with_none_values(self):
        """Test summary generation with None values."""
        batch = {
            'wsi_features': torch.randn(16, 100, 1024),
            'genomic': None,
        }
        
        summary = get_validation_summary(batch)
        
        assert "wsi_features" in summary
        assert "genomic: None" in summary

    def test_summary_with_list_of_tensors(self):
        """Test summary generation with list of tensors."""
        batch = {
            'wsi_features': [
                torch.randn(50, 1024),
                torch.randn(100, 1024),
            ],
        }
        
        summary = get_validation_summary(batch)
        
        assert "List[Tensor]" in summary
        assert "length=2" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
