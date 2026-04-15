"""
Unit tests for input validation module.

Tests cover:
- Core tensor validation functions
- Modality-specific validation
- Batch validation
- Decorator functionality
- Error messages and suggestions
"""

import unittest

import torch
import torch.nn as nn

from src.utils.validation import (
    ValidationError,
    get_validation_summary,
    is_validation_enabled,
    set_validation_enabled,
    validate_batch_size,
    validate_clinical_text,
    validate_genomic_features,
    validate_inputs,
    validate_multimodal_batch,
    validate_no_nan_inf,
    validate_tensor_range,
    validate_tensor_shape,
    validate_wsi_features,
)


class TestValidationEnabled(unittest.TestCase):
    """Test validation enable/disable functionality."""

    def setUp(self):
        """Ensure validation is enabled for tests."""
        set_validation_enabled(True)

    def test_validation_enabled_by_default(self):
        """Test that validation is enabled by default."""
        self.assertTrue(is_validation_enabled())

    def test_disable_validation(self):
        """Test disabling validation."""
        set_validation_enabled(False)
        self.assertFalse(is_validation_enabled())
        set_validation_enabled(True)  # Re-enable for other tests

    def test_validation_skipped_when_disabled(self):
        """Test that validation is skipped when disabled."""
        set_validation_enabled(False)

        # This should not raise even though shape is wrong
        tensor = torch.randn(10, 20)
        validate_tensor_shape(tensor, (5, 10), "test")

        set_validation_enabled(True)  # Re-enable


class TestTensorShapeValidation(unittest.TestCase):
    """Test validate_tensor_shape function."""

    def setUp(self):
        set_validation_enabled(True)

    def test_valid_shape_exact(self):
        """Test validation passes for exact shape match."""
        tensor = torch.randn(16, 100, 1024)
        validate_tensor_shape(tensor, (16, 100, 1024), "test_tensor")

    def test_valid_shape_with_none(self):
        """Test validation passes with None for variable dimensions."""
        tensor = torch.randn(16, 100, 1024)
        validate_tensor_shape(tensor, (None, 100, 1024), "test_tensor")
        validate_tensor_shape(tensor, (16, None, 1024), "test_tensor")
        validate_tensor_shape(tensor, (None, None, None), "test_tensor")

    def test_invalid_shape_wrong_dims(self):
        """Test validation fails for wrong number of dimensions."""
        tensor = torch.randn(16, 100)
        with self.assertRaises(ValidationError) as cm:
            validate_tensor_shape(tensor, (16, 100, 1024), "test_tensor")

        self.assertIn("2 dimensions", str(cm.exception))
        self.assertIn("3 dimensions", str(cm.exception))

    def test_invalid_shape_wrong_size(self):
        """Test validation fails for wrong dimension size."""
        tensor = torch.randn(16, 100, 512)
        with self.assertRaises(ValidationError) as cm:
            validate_tensor_shape(tensor, (16, 100, 1024), "test_tensor")

        self.assertIn("dimension 2", str(cm.exception))
        self.assertIn("expected 1024", str(cm.exception))
        self.assertIn("got 512", str(cm.exception))

    def test_not_a_tensor(self):
        """Test validation fails for non-tensor input."""
        not_tensor = [1, 2, 3]
        with self.assertRaises(ValidationError) as cm:
            validate_tensor_shape(not_tensor, (3,), "test_tensor")

        self.assertIn("torch.Tensor", str(cm.exception))
        self.assertIn("list", str(cm.exception))


class TestTensorRangeValidation(unittest.TestCase):
    """Test validate_tensor_range function."""

    def setUp(self):
        set_validation_enabled(True)

    def test_valid_range(self):
        """Test validation passes for values in range."""
        tensor = torch.randn(16, 1024) * 0.1  # Small values
        validate_tensor_range(tensor, -1.0, 1.0, "test_tensor")

    def test_valid_range_min_only(self):
        """Test validation with only minimum specified."""
        tensor = torch.randn(16, 1024).abs()  # Positive values
        validate_tensor_range(tensor, min_val=0.0, name="test_tensor")

    def test_valid_range_max_only(self):
        """Test validation with only maximum specified."""
        tensor = -torch.randn(16, 1024).abs()  # Negative values
        validate_tensor_range(tensor, max_val=0.0, name="test_tensor")

    def test_invalid_range_too_small(self):
        """Test validation fails for values below minimum."""
        tensor = torch.ones(16, 1024) * 0.5  # Start with valid values
        tensor[0, 0] = -5.0  # Set one value below minimum
        with self.assertRaises(ValidationError) as cm:
            validate_tensor_range(tensor, -1.0, 1.0, "test_tensor")

        self.assertIn("minimum", str(cm.exception))
        self.assertIn("-5", str(cm.exception))

    def test_invalid_range_too_large(self):
        """Test validation fails for values above maximum."""
        tensor = torch.randn(16, 1024) * 0.1
        tensor[0, 0] = 5.0
        with self.assertRaises(ValidationError) as cm:
            validate_tensor_range(tensor, -1.0, 1.0, "test_tensor")

        self.assertIn("maximum", str(cm.exception))
        self.assertIn("5", str(cm.exception))


class TestNaNInfValidation(unittest.TestCase):
    """Test validate_no_nan_inf function."""

    def setUp(self):
        set_validation_enabled(True)

    def test_valid_no_nan_inf(self):
        """Test validation passes for clean tensor."""
        tensor = torch.randn(16, 1024)
        validate_no_nan_inf(tensor, "test_tensor")

    def test_invalid_with_nan(self):
        """Test validation fails for tensor with NaN."""
        tensor = torch.randn(16, 1024)
        tensor[0, 0] = float("nan")
        with self.assertRaises(ValidationError) as cm:
            validate_no_nan_inf(tensor, "test_tensor")

        self.assertIn("NaN", str(cm.exception))
        self.assertIn("1 /", str(cm.exception))

    def test_invalid_with_inf(self):
        """Test validation fails for tensor with Inf."""
        tensor = torch.randn(16, 1024)
        tensor[0, 0] = float("inf")
        with self.assertRaises(ValidationError) as cm:
            validate_no_nan_inf(tensor, "test_tensor")

        self.assertIn("Inf", str(cm.exception))

    def test_invalid_with_both(self):
        """Test validation fails for tensor with both NaN and Inf."""
        tensor = torch.randn(16, 1024)
        tensor[0, 0] = float("nan")
        tensor[0, 1] = float("inf")
        with self.assertRaises(ValidationError) as cm:
            validate_no_nan_inf(tensor, "test_tensor")

        error_msg = str(cm.exception)
        self.assertIn("NaN", error_msg)
        self.assertIn("Inf", error_msg)


class TestBatchSizeValidation(unittest.TestCase):
    """Test validate_batch_size function."""

    def setUp(self):
        set_validation_enabled(True)

    def test_valid_batch_size(self):
        """Test validation passes for correct batch size."""
        tensor = torch.randn(16, 1024)
        validate_batch_size(tensor, 16, "test_tensor")

    def test_invalid_batch_size(self):
        """Test validation fails for wrong batch size."""
        tensor = torch.randn(16, 1024)
        with self.assertRaises(ValidationError) as cm:
            validate_batch_size(tensor, 32, "test_tensor")

        self.assertIn("Expected: 32", str(cm.exception))
        self.assertIn("Received: 16", str(cm.exception))

    def test_scalar_tensor(self):
        """Test validation fails for scalar tensor."""
        tensor = torch.tensor(5.0)
        with self.assertRaises(ValidationError) as cm:
            validate_batch_size(tensor, 1, "test_tensor")

        self.assertIn("scalar", str(cm.exception))


class TestWSIFeaturesValidation(unittest.TestCase):
    """Test validate_wsi_features function."""

    def setUp(self):
        set_validation_enabled(True)

    def test_valid_wsi_features(self):
        """Test validation passes for correct WSI features."""
        features = torch.randn(16, 100, 1024)
        validate_wsi_features(features)

    def test_valid_wsi_features_variable_patches(self):
        """Test validation passes for variable number of patches."""
        features = torch.randn(16, 50, 1024)
        validate_wsi_features(features)

    def test_invalid_wsi_wrong_dims(self):
        """Test validation fails for wrong number of dimensions."""
        features = torch.randn(16, 1024)  # Missing patches dimension
        with self.assertRaises(ValidationError) as cm:
            validate_wsi_features(features)

        self.assertIn("3 dimensions", str(cm.exception))
        self.assertIn("2 dimensions", str(cm.exception))

    def test_invalid_wsi_wrong_feature_dim(self):
        """Test validation fails for wrong feature dimension."""
        features = torch.randn(16, 100, 512)
        with self.assertRaises(ValidationError) as cm:
            validate_wsi_features(features)

        self.assertIn("expected 1024", str(cm.exception))
        self.assertIn("got 512", str(cm.exception))

    def test_invalid_wsi_zero_patches(self):
        """Test validation fails for zero patches."""
        features = torch.randn(16, 0, 1024)
        with self.assertRaises(ValidationError) as cm:
            validate_wsi_features(features)

        self.assertIn("0 patches", str(cm.exception))

    def test_invalid_wsi_with_nan(self):
        """Test validation fails for WSI features with NaN."""
        features = torch.randn(16, 100, 1024)
        features[0, 0, 0] = float("nan")
        with self.assertRaises(ValidationError) as cm:
            validate_wsi_features(features)

        self.assertIn("NaN", str(cm.exception))


class TestGenomicFeaturesValidation(unittest.TestCase):
    """Test validate_genomic_features function."""

    def setUp(self):
        set_validation_enabled(True)

    def test_valid_genomic_features(self):
        """Test validation passes for correct genomic features."""
        features = torch.randn(16, 2000)
        validate_genomic_features(features)

    def test_invalid_genomic_wrong_dims(self):
        """Test validation fails for wrong number of dimensions."""
        features = torch.randn(16, 100, 2000)  # Extra dimension
        with self.assertRaises(ValidationError) as cm:
            validate_genomic_features(features)

        self.assertIn("2 dimensions", str(cm.exception))
        self.assertIn("3 dimensions", str(cm.exception))

    def test_invalid_genomic_wrong_feature_dim(self):
        """Test validation fails for wrong feature dimension."""
        features = torch.randn(16, 1000)
        with self.assertRaises(ValidationError) as cm:
            validate_genomic_features(features)

        self.assertIn("expected 2000", str(cm.exception))
        self.assertIn("got 1000", str(cm.exception))

    def test_invalid_genomic_with_nan(self):
        """Test validation fails for genomic features with NaN."""
        features = torch.randn(16, 2000)
        features[0, 0] = float("nan")
        with self.assertRaises(ValidationError) as cm:
            validate_genomic_features(features)

        self.assertIn("NaN", str(cm.exception))


class TestClinicalTextValidation(unittest.TestCase):
    """Test validate_clinical_text function."""

    def setUp(self):
        set_validation_enabled(True)

    def test_valid_clinical_text(self):
        """Test validation passes for correct clinical text."""
        tokens = torch.randint(0, 30000, (16, 128))
        validate_clinical_text(tokens)

    def test_valid_clinical_text_with_max_length(self):
        """Test validation passes with max length check."""
        tokens = torch.randint(0, 30000, (16, 128))
        validate_clinical_text(tokens, max_seq_length=512)

    def test_invalid_clinical_wrong_dims(self):
        """Test validation fails for wrong number of dimensions."""
        tokens = torch.randint(0, 30000, (16,))  # Missing sequence dimension
        with self.assertRaises(ValidationError) as cm:
            validate_clinical_text(tokens)

        self.assertIn("2 dimensions", str(cm.exception))
        self.assertIn("1 dimensions", str(cm.exception))

    def test_invalid_clinical_exceeds_max_length(self):
        """Test validation fails for sequence exceeding max length."""
        tokens = torch.randint(0, 30000, (16, 1000))
        with self.assertRaises(ValidationError) as cm:
            validate_clinical_text(tokens, max_seq_length=512)

        self.assertIn("exceeds maximum", str(cm.exception))
        self.assertIn("512", str(cm.exception))
        self.assertIn("1000", str(cm.exception))

    def test_invalid_clinical_wrong_dtype(self):
        """Test validation fails for non-integer dtype."""
        tokens = torch.randn(16, 128)  # Float instead of int
        with self.assertRaises(ValidationError) as cm:
            validate_clinical_text(tokens)

        self.assertIn("integer type", str(cm.exception))

    def test_invalid_clinical_negative_tokens(self):
        """Test validation fails for negative token IDs."""
        tokens = torch.randint(0, 30000, (16, 128))
        tokens[0, 0] = -10  # Set one negative token
        with self.assertRaises(ValidationError) as cm:
            validate_clinical_text(tokens)

        self.assertIn("negative token IDs", str(cm.exception))


class TestMultimodalBatchValidation(unittest.TestCase):
    """Test validate_multimodal_batch function."""

    def setUp(self):
        set_validation_enabled(True)

    def test_valid_complete_batch(self):
        """Test validation passes for complete batch with all modalities."""
        batch = {
            "wsi_features": torch.randn(16, 100, 1024),
            "genomic": torch.randn(16, 2000),
            "clinical_text": torch.randint(0, 30000, (16, 128)),
            "labels": torch.randint(0, 2, (16,)),
        }
        validate_multimodal_batch(batch)

    def test_valid_partial_batch(self):
        """Test validation passes for batch with missing modalities."""
        batch = {
            "wsi_features": torch.randn(16, 100, 1024),
            "genomic": None,
            "clinical_text": torch.randint(0, 30000, (16, 128)),
            "labels": torch.randint(0, 2, (16,)),
        }
        validate_multimodal_batch(batch, require_all_modalities=False)

    def test_valid_batch_with_list_wsi(self):
        """Test validation passes for batch with list of WSI features."""
        batch = {
            "wsi_features": [
                torch.randn(100, 1024),
                torch.randn(50, 1024),
                torch.randn(75, 1024),
            ],
            "genomic": torch.randn(3, 2000),
            "clinical_text": torch.randint(0, 30000, (3, 128)),
            "labels": torch.randint(0, 2, (3,)),
        }
        validate_multimodal_batch(batch)

    def test_invalid_not_dict(self):
        """Test validation fails for non-dict batch."""
        batch = [torch.randn(16, 1024)]
        with self.assertRaises(ValidationError) as cm:
            validate_multimodal_batch(batch)

        self.assertIn("dictionary", str(cm.exception))

    def test_invalid_batch_size_mismatch(self):
        """Test validation fails for mismatched batch sizes."""
        batch = {
            "wsi_features": torch.randn(16, 100, 1024),
            "genomic": torch.randn(8, 2000),  # Wrong batch size
            "labels": torch.randint(0, 2, (16,)),
        }
        with self.assertRaises(ValidationError) as cm:
            validate_multimodal_batch(batch)

        self.assertIn("Batch size mismatch", str(cm.exception))

    def test_invalid_missing_required_modality(self):
        """Test validation fails when required modality is missing."""
        batch = {
            "wsi_features": torch.randn(16, 100, 1024),
            "genomic": None,
            "labels": torch.randint(0, 2, (16,)),
        }
        with self.assertRaises(ValidationError) as cm:
            validate_multimodal_batch(batch, require_all_modalities=True)

        self.assertIn("Missing required modality", str(cm.exception))

    def test_invalid_empty_batch(self):
        """Test validation fails for empty batch."""
        batch = {}
        with self.assertRaises(ValidationError) as cm:
            validate_multimodal_batch(batch)

        self.assertIn("Cannot determine batch size", str(cm.exception))

    def test_invalid_wsi_wrong_shape_in_list(self):
        """Test validation fails for wrong WSI shape in list."""
        batch = {
            "wsi_features": [
                torch.randn(100, 1024),
                torch.randn(50, 512),  # Wrong feature dim
            ],
            "labels": torch.randint(0, 2, (2,)),
        }
        with self.assertRaises(ValidationError) as cm:
            validate_multimodal_batch(batch)

        self.assertIn("feature dimension", str(cm.exception))


class TestValidateInputsDecorator(unittest.TestCase):
    """Test @validate_inputs decorator."""

    def setUp(self):
        set_validation_enabled(True)

    def test_decorator_validates_batch(self):
        """Test decorator validates batch input."""

        class TestModel(nn.Module):
            @validate_inputs
            def forward(self, batch):
                return batch["wsi_features"]

        model = TestModel()

        # Valid batch should work
        valid_batch = {
            "wsi_features": torch.randn(16, 100, 1024),
            "genomic": torch.randn(16, 2000),
        }
        output = model(valid_batch)
        self.assertIsNotNone(output)

        # Invalid batch should raise
        invalid_batch = {
            "wsi_features": torch.randn(16, 100, 512),  # Wrong feature dim
        }
        with self.assertRaises(ValidationError):
            model(invalid_batch)

    def test_decorator_with_kwargs(self):
        """Test decorator works with keyword arguments."""

        class TestModel(nn.Module):
            @validate_inputs
            def forward(self, batch=None):
                return batch["wsi_features"]

        model = TestModel()

        valid_batch = {
            "wsi_features": torch.randn(16, 100, 1024),
        }
        output = model(batch=valid_batch)
        self.assertIsNotNone(output)

    def test_decorator_skipped_when_disabled(self):
        """Test decorator is skipped when validation disabled."""
        set_validation_enabled(False)

        class TestModel(nn.Module):
            @validate_inputs
            def forward(self, batch):
                return batch["wsi_features"]

        model = TestModel()

        # Invalid batch should not raise when validation disabled
        invalid_batch = {
            "wsi_features": torch.randn(16, 100, 512),
        }
        output = model(invalid_batch)
        self.assertIsNotNone(output)

        set_validation_enabled(True)

    def test_decorator_preserves_function_name(self):
        """Test decorator preserves original function name."""

        class TestModel(nn.Module):
            @validate_inputs
            def forward(self, batch):
                return batch

        model = TestModel()
        self.assertEqual(model.forward.__name__, "forward")


class TestValidationSummary(unittest.TestCase):
    """Test get_validation_summary function."""

    def test_summary_with_tensors(self):
        """Test summary generation for batch with tensors."""
        batch = {
            "wsi_features": torch.randn(16, 100, 1024),
            "genomic": torch.randn(16, 2000),
            "clinical_text": torch.randint(0, 30000, (16, 128)),
            "labels": torch.randint(0, 2, (16,)),
        }
        summary = get_validation_summary(batch)

        self.assertIn("Batch Summary", summary)
        self.assertIn("wsi_features", summary)
        self.assertIn("[16, 100, 1024]", summary)
        self.assertIn("genomic", summary)
        self.assertIn("[16, 2000]", summary)

    def test_summary_with_none(self):
        """Test summary generation for batch with None values."""
        batch = {
            "wsi_features": torch.randn(16, 100, 1024),
            "genomic": None,
            "clinical_text": None,
        }
        summary = get_validation_summary(batch)

        self.assertIn("genomic: None", summary)
        self.assertIn("clinical_text: None", summary)

    def test_summary_with_list(self):
        """Test summary generation for batch with list of tensors."""
        batch = {
            "wsi_features": [
                torch.randn(100, 1024),
                torch.randn(50, 1024),
            ],
        }
        summary = get_validation_summary(batch)

        self.assertIn("List[Tensor]", summary)
        self.assertIn("length=2", summary)


class TestErrorMessages(unittest.TestCase):
    """Test that error messages are helpful and actionable."""

    def setUp(self):
        set_validation_enabled(True)

    def test_error_includes_expected_and_received(self):
        """Test error messages include both expected and received values."""
        tensor = torch.randn(16, 100, 512)
        with self.assertRaises(ValidationError) as cm:
            validate_wsi_features(tensor)

        error_msg = str(cm.exception)
        self.assertIn("Expected", error_msg)
        self.assertIn("Received", error_msg)
        self.assertIn("1024", error_msg)
        self.assertIn("512", error_msg)

    def test_error_includes_suggestions(self):
        """Test error messages include helpful suggestions."""
        tensor = torch.randn(16, 100, 512)
        with self.assertRaises(ValidationError) as cm:
            validate_wsi_features(tensor)

        error_msg = str(cm.exception)
        self.assertIn("Suggestion", error_msg)

    def test_nan_error_includes_statistics(self):
        """Test NaN error includes count and percentage."""
        tensor = torch.randn(100, 100)
        tensor[:10, :10] = float("nan")  # 100 NaN values out of 10000

        with self.assertRaises(ValidationError) as cm:
            validate_no_nan_inf(tensor)

        error_msg = str(cm.exception)
        self.assertIn("100 /", error_msg)
        self.assertIn("1.00%", error_msg)


if __name__ == "__main__":
    unittest.main()
