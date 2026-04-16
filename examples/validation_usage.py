"""
Example usage of the validation module for computational pathology framework.

This script demonstrates how to use the validation functions and decorator
to ensure data integrity in multimodal pathology models.
"""

import torch
import torch.nn as nn

from src.utils.validation import (
    ValidationError,
    get_validation_summary,
    set_validation_enabled,
    validate_clinical_text,
    validate_genomic_features,
    validate_inputs,
    validate_multimodal_batch,
    validate_no_nan_inf,
    validate_tensor_range,
    validate_tensor_shape,
    validate_wsi_features,
)


def example_1_basic_tensor_validation():
    """Example 1: Basic tensor validation."""
    print("=" * 70)
    print("Example 1: Basic Tensor Validation")
    print("=" * 70)

    # Create a sample tensor
    tensor = torch.randn(16, 100, 1024)

    # Validate shape (with None for variable dimensions)
    try:
        validate_tensor_shape(tensor, (None, 100, 1024), "wsi_features")
        print("✓ Shape validation passed")
    except ValidationError as e:
        print(f"✗ Shape validation failed: {e}")

    # Validate no NaN/Inf values
    try:
        validate_no_nan_inf(tensor, "wsi_features")
        print("✓ NaN/Inf validation passed")
    except ValidationError as e:
        print(f"✗ NaN/Inf validation failed: {e}")

    # Validate value range
    try:
        validate_tensor_range(tensor, -10.0, 10.0, "wsi_features")
        print("✓ Range validation passed")
    except ValidationError as e:
        print(f"✗ Range validation failed: {e}")

    print()


def example_2_modality_specific_validation():
    """Example 2: Modality-specific validation."""
    print("=" * 70)
    print("Example 2: Modality-Specific Validation")
    print("=" * 70)

    # WSI features
    wsi_features = torch.randn(16, 100, 1024)
    try:
        validate_wsi_features(wsi_features)
        print("✓ WSI features validation passed")
    except ValidationError as e:
        print(f"✗ WSI features validation failed: {e}")

    # Genomic features
    genomic_features = torch.randn(16, 2000)
    try:
        validate_genomic_features(genomic_features)
        print("✓ Genomic features validation passed")
    except ValidationError as e:
        print(f"✗ Genomic features validation failed: {e}")

    # Clinical text
    clinical_text = torch.randint(0, 30000, (16, 128))
    try:
        validate_clinical_text(clinical_text, max_seq_length=512)
        print("✓ Clinical text validation passed")
    except ValidationError as e:
        print(f"✗ Clinical text validation failed: {e}")

    print()


def example_3_batch_validation():
    """Example 3: Complete batch validation."""
    print("=" * 70)
    print("Example 3: Complete Batch Validation")
    print("=" * 70)

    # Create a complete multimodal batch
    batch = {
        "wsi_features": torch.randn(16, 100, 1024),
        "genomic": torch.randn(16, 2000),
        "clinical_text": torch.randint(0, 30000, (16, 128)),
        "labels": torch.randint(0, 2, (16,)),
    }

    # Validate the entire batch
    try:
        validate_multimodal_batch(batch)
        print("✓ Batch validation passed")
    except ValidationError as e:
        print(f"✗ Batch validation failed: {e}")

    # Print batch summary
    print("\n" + get_validation_summary(batch))
    print()


def example_4_handling_missing_modalities():
    """Example 4: Handling missing modalities."""
    print("=" * 70)
    print("Example 4: Handling Missing Modalities")
    print("=" * 70)

    # Batch with missing genomic data
    batch = {
        "wsi_features": torch.randn(16, 100, 1024),
        "genomic": None,
        "clinical_text": torch.randint(0, 30000, (16, 128)),
        "labels": torch.randint(0, 2, (16,)),
    }

    # Validate without requiring all modalities
    try:
        validate_multimodal_batch(batch, require_all_modalities=False)
        print("✓ Batch validation passed (missing modalities allowed)")
    except ValidationError as e:
        print(f"✗ Batch validation failed: {e}")

    # Try to validate requiring all modalities (should fail)
    try:
        validate_multimodal_batch(batch, require_all_modalities=True)
        print("✓ Batch validation passed (all modalities required)")
    except ValidationError as e:
        print(f"✗ Batch validation failed (expected): Missing genomic data")

    print()


def example_5_decorator_usage():
    """Example 5: Using the @validate_inputs decorator."""
    print("=" * 70)
    print("Example 5: Using @validate_inputs Decorator")
    print("=" * 70)

    class SimpleModel(nn.Module):
        """Simple model with automatic input validation."""

        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(256, 2)

        @validate_inputs
        def forward(self, batch):
            """Forward pass with automatic validation."""
            # Validation happens automatically before this code runs
            wsi = batch["wsi_features"]
            # Process features...
            return torch.randn(wsi.shape[0], 2)

    model = SimpleModel()

    # Valid batch
    valid_batch = {
        "wsi_features": torch.randn(16, 100, 1024),
        "genomic": torch.randn(16, 2000),
    }

    try:
        output = model(valid_batch)
        print(f"✓ Model forward pass succeeded with output shape: {output.shape}")
    except ValidationError as e:
        print(f"✗ Model forward pass failed: {e}")

    # Invalid batch (wrong feature dimension)
    invalid_batch = {
        "wsi_features": torch.randn(16, 100, 512),  # Wrong feature dim
    }

    try:
        output = model(invalid_batch)
        print(f"✓ Model forward pass succeeded (unexpected)")
    except ValidationError as e:
        print(f"✗ Model forward pass failed (expected): Feature dimension mismatch")

    print()


def example_6_error_messages():
    """Example 6: Demonstrating helpful error messages."""
    print("=" * 70)
    print("Example 6: Helpful Error Messages")
    print("=" * 70)

    # Wrong shape
    print("Testing wrong shape error:")
    try:
        tensor = torch.randn(16, 100, 512)
        validate_wsi_features(tensor)
    except ValidationError as e:
        print(f"{e}\n")

    # NaN values
    print("Testing NaN detection:")
    try:
        tensor = torch.randn(16, 2000)
        tensor[0, 0] = float("nan")
        validate_genomic_features(tensor)
    except ValidationError as e:
        print(f"{e}\n")

    # Batch size mismatch
    print("Testing batch size mismatch:")
    try:
        batch = {
            "wsi_features": torch.randn(16, 100, 1024),
            "genomic": torch.randn(8, 2000),  # Wrong batch size
        }
        validate_multimodal_batch(batch)
    except ValidationError as e:
        print(f"{e}\n")


def example_7_disabling_validation():
    """Example 7: Disabling validation for production."""
    print("=" * 70)
    print("Example 7: Disabling Validation for Production")
    print("=" * 70)

    # Disable validation
    set_validation_enabled(False)
    print("Validation disabled")

    # This should not raise even though shape is wrong
    tensor = torch.randn(16, 100, 512)  # Wrong feature dim
    try:
        validate_wsi_features(tensor)
        print("✓ Validation skipped (no error raised)")
    except ValidationError as e:
        print(f"✗ Validation still active: {e}")

    # Re-enable validation
    set_validation_enabled(True)
    print("Validation re-enabled")

    # Now it should raise
    try:
        validate_wsi_features(tensor)
        print("✓ Validation skipped (unexpected)")
    except ValidationError as e:
        print("✗ Validation active (expected): Feature dimension mismatch")

    print()


def example_8_variable_length_patches():
    """Example 8: Handling variable-length patch sequences."""
    print("=" * 70)
    print("Example 8: Variable-Length Patch Sequences")
    print("=" * 70)

    # Batch with variable number of patches per sample
    batch = {
        "wsi_features": [
            torch.randn(100, 1024),  # Sample 1: 100 patches
            torch.randn(50, 1024),  # Sample 2: 50 patches
            torch.randn(75, 1024),  # Sample 3: 75 patches
        ],
        "genomic": torch.randn(3, 2000),
        "labels": torch.randint(0, 2, (3,)),
    }

    try:
        validate_multimodal_batch(batch)
        print("✓ Variable-length batch validation passed")
        print(f"  Sample 1: {batch['wsi_features'][0].shape[0]} patches")
        print(f"  Sample 2: {batch['wsi_features'][1].shape[0]} patches")
        print(f"  Sample 3: {batch['wsi_features'][2].shape[0]} patches")
    except ValidationError as e:
        print(f"✗ Batch validation failed: {e}")

    print()


def main():
    """Run all examples."""
    print("\n")
    print("=" * 70)
    print("VALIDATION MODULE USAGE EXAMPLES")
    print("=" * 70)
    print()

    example_1_basic_tensor_validation()
    example_2_modality_specific_validation()
    example_3_batch_validation()
    example_4_handling_missing_modalities()
    example_5_decorator_usage()
    example_6_error_messages()
    example_7_disabling_validation()
    example_8_variable_length_patches()

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
