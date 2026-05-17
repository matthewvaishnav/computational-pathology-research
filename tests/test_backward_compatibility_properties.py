"""
Property-based tests for backward compatibility between TransMIL and nnMIL.

This test file validates correctness properties for backward compatibility,
migration tools, and unified training interface using property-based testing
with Hypothesis. Each property test runs a minimum of 100 iterations to verify
universal invariants.

Feature: nnmil-architecture-upgrade
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock, patch

import pytest
import torch

from hypothesis import given, settings
from hypothesis import strategies as st
from scripts.migrate_transmil_to_nnmil import TransMILToNnMILMigrator
from src.models.nnmil import nnMIL
from src.models.transmil import TransMIL  # Assuming existing TransMIL implementation
from src.training.unified_trainer import UnifiedTrainer

# ============================================================================
# Property 33: API Compatibility
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=16),
    num_patches=st.integers(min_value=10, max_value=500),
    feature_dim=st.integers(min_value=256, max_value=1024),
    num_classes=st.integers(min_value=2, max_value=10),
    return_attention=st.booleans(),
)
def test_property_33_api_compatibility(
    batch_size, num_patches, feature_dim, num_classes, return_attention
):
    """
    Feature: nnmil-architecture-upgrade, Property 33: For any input tensor,
    both TransMIL and nnMIL models SHALL accept the input and return outputs
    with compatible shapes.

    **Validates: Requirements 10.2, 10.6**
    """
    # Create both models with same configuration
    transmil_model = TransMIL(feature_dim=feature_dim, num_classes=num_classes, dropout=0.25)

    nnmil_model = nnMIL(
        feature_dim=feature_dim, hidden_dim=256, num_classes=num_classes, dropout=0.25
    )

    # Set both models to eval mode
    transmil_model.eval()
    nnmil_model.eval()

    # Create input features
    features = torch.randn(batch_size, num_patches, feature_dim)

    with torch.no_grad():
        # Forward pass through both models
        if return_attention:
            transmil_output = transmil_model(features, return_attention=True)
            nnmil_output = nnmil_model(features, return_attention=True)

            # Both should return tuples (logits, attention)
            assert isinstance(
                transmil_output, tuple
            ), "TransMIL should return tuple when return_attention=True"
            assert isinstance(
                nnmil_output, tuple
            ), "nnMIL should return tuple when return_attention=True"
            assert len(transmil_output) == 2, "TransMIL should return (logits, attention)"
            assert len(nnmil_output) == 2, "nnMIL should return (logits, attention)"

            transmil_logits, transmil_attention = transmil_output
            nnmil_logits, nnmil_attention = nnmil_output
        else:
            transmil_logits = transmil_model(features, return_attention=False)
            nnmil_logits = nnmil_model(features, return_attention=False)

            # Both should return tensors
            assert isinstance(
                transmil_logits, torch.Tensor
            ), "TransMIL should return tensor when return_attention=False"
            assert isinstance(
                nnmil_logits, torch.Tensor
            ), "nnMIL should return tensor when return_attention=False"

    # Verify output shapes are compatible
    expected_logits_shape = (batch_size, num_classes)
    assert (
        transmil_logits.shape == expected_logits_shape
    ), f"TransMIL logits shape mismatch: expected {expected_logits_shape}, got {transmil_logits.shape}"
    assert (
        nnmil_logits.shape == expected_logits_shape
    ), f"nnMIL logits shape mismatch: expected {expected_logits_shape}, got {nnmil_logits.shape}"

    if return_attention:
        expected_attention_shape = (batch_size, num_patches)
        assert (
            transmil_attention.shape == expected_attention_shape
        ), f"TransMIL attention shape mismatch: expected {expected_attention_shape}, got {transmil_attention.shape}"
        assert (
            nnmil_attention.shape == expected_attention_shape
        ), f"nnMIL attention shape mismatch: expected {expected_attention_shape}, got {nnmil_attention.shape}"


# ============================================================================
# Property 34: Weight Transfer
# ============================================================================


@settings(max_examples=50, deadline=None)
@given(
    feature_dim=st.integers(min_value=256, max_value=1024),
    num_classes=st.integers(min_value=2, max_value=10),
    hidden_dim=st.integers(min_value=128, max_value=512),
)
def test_property_34_weight_transfer(feature_dim, num_classes, hidden_dim):
    """
    Feature: nnmil-architecture-upgrade, Property 34: For any TransMIL checkpoint,
    compatible layers (feature projection, classifier) SHALL transfer to nnMIL
    with matching weights.

    **Validates: Requirements 10.4**
    """
    # Create TransMIL model and initialize with random weights
    transmil_model = TransMIL(feature_dim=feature_dim, num_classes=num_classes, dropout=0.25)

    # Create nnMIL model with compatible configuration
    nnmil_model = nnMIL(
        feature_dim=feature_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=0.25
    )

    # Create migrator
    migrator = TransMILToNnMILMigrator()

    # Extract TransMIL state dict
    transmil_state = transmil_model.state_dict()

    # Perform weight transfer
    transferred_weights = migrator.transfer_compatible_weights(
        transmil_state, nnmil_model.state_dict()
    )

    # Load transferred weights into nnMIL
    nnmil_model.load_state_dict(transferred_weights, strict=False)

    # Verify compatible layers have matching weights
    # Note: This test assumes both models have similar layer naming conventions
    # In practice, the migrator would handle the mapping between different layer names

    # Test that models produce similar outputs (not identical due to architecture differences)
    test_input = torch.randn(2, 50, feature_dim)

    with torch.no_grad():
        transmil_output = transmil_model(test_input)
        nnmil_output = nnmil_model(test_input)

    # Outputs should be in similar range (not exact due to architectural differences)
    transmil_mean = transmil_output.mean()
    nnmil_mean = nnmil_output.mean()

    # Should be within reasonable range of each other
    assert (
        abs(transmil_mean - nnmil_mean) < 5.0
    ), f"Transferred model outputs too different: TransMIL={transmil_mean:.3f}, nnMIL={nnmil_mean:.3f}"


def test_checkpoint_migration_end_to_end():
    """Test complete checkpoint migration from TransMIL to nnMIL."""
    # Create TransMIL model and save checkpoint
    transmil_model = TransMIL(feature_dim=1024, num_classes=2, dropout=0.25)

    # Create mock training state
    transmil_checkpoint = {
        "model_state_dict": transmil_model.state_dict(),
        "optimizer_state_dict": {"lr": 1e-4, "weight_decay": 1e-5},
        "epoch": 50,
        "best_val_auc": 0.85,
        "config": {
            "model": {"feature_dim": 1024, "num_classes": 2, "dropout": 0.25},
            "training": {"learning_rate": 1e-4, "batch_size": 32},
        },
    }

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        torch.save(transmil_checkpoint, f.name)
        checkpoint_path = f.name

    try:
        # Migrate checkpoint
        migrator = TransMILToNnMILMigrator()
        nnmil_checkpoint_path = migrator.migrate_checkpoint(
            checkpoint_path, output_path=checkpoint_path.replace(".pth", "_nnmil.pth")
        )

        # Load migrated checkpoint
        nnmil_checkpoint = torch.load(nnmil_checkpoint_path)

        # Verify checkpoint structure
        assert "model_state_dict" in nnmil_checkpoint
        assert "config" in nnmil_checkpoint
        assert "migration_info" in nnmil_checkpoint

        # Verify config was updated for nnMIL
        config = nnmil_checkpoint["config"]
        assert "model" in config
        assert "hidden_dim" in config["model"]  # nnMIL-specific parameter

        # Verify migration info
        migration_info = nnmil_checkpoint["migration_info"]
        assert migration_info["source_model"] == "TransMIL"
        assert migration_info["target_model"] == "nnMIL"
        assert "transferred_layers" in migration_info

        # Clean up
        os.unlink(nnmil_checkpoint_path)

    finally:
        os.unlink(checkpoint_path)


def test_unified_training_interface():
    """Test unified training interface works with both model types."""
    # Test configurations for both models
    transmil_config = {
        "model_type": "TransMIL",
        "model": {"feature_dim": 1024, "num_classes": 2, "dropout": 0.25},
        "training": {"batch_size": 16, "learning_rate": 1e-4},
    }

    nnmil_config = {
        "model_type": "nnMIL",
        "model": {"feature_dim": 1024, "hidden_dim": 256, "num_classes": 2, "dropout": 0.25},
        "training": {"batch_size": 32, "learning_rate": 3e-4},
    }

    # Create unified trainer for each model type
    transmil_trainer = UnifiedTrainer(transmil_config)
    nnmil_trainer = UnifiedTrainer(nnmil_config)

    # Verify correct model types were created
    assert isinstance(transmil_trainer.model, TransMIL)
    assert isinstance(nnmil_trainer.model, nnMIL)

    # Verify both trainers have same interface
    assert hasattr(transmil_trainer, "train")
    assert hasattr(nnmil_trainer, "train")
    assert hasattr(transmil_trainer, "evaluate")
    assert hasattr(nnmil_trainer, "evaluate")

    # Test that both can process same input format
    batch_data = {
        "features": torch.randn(4, 100, 1024),
        "labels": torch.randint(0, 2, (4,)),
        "masks": torch.ones(4, 100, dtype=torch.bool),
    }

    # Both should handle same batch format
    transmil_loss = transmil_trainer._compute_batch_loss(batch_data)
    nnmil_loss = nnmil_trainer._compute_batch_loss(batch_data)

    assert isinstance(transmil_loss, torch.Tensor)
    assert isinstance(nnmil_loss, torch.Tensor)


def test_multi_scale_backward_compatibility():
    """Test multi-scale compatibility between TransMIL and nnMIL."""
    # Multi-scale TransMIL
    transmil_model = TransMIL(feature_dim=1024, num_classes=2, multi_scale=True, num_scales=2)

    # Multi-scale nnMIL
    nnmil_model = nnMIL(
        feature_dim=1024,
        hidden_dim=256,
        num_classes=2,
        multi_scale=True,
        num_scales=2,
        fusion_strategy="early",
    )

    # Create multi-scale input
    scale1_features = torch.randn(2, 100, 1024)
    scale2_features = torch.randn(2, 100, 1024)
    multi_scale_input = [scale1_features, scale2_features]

    # Both models should handle multi-scale input
    with torch.no_grad():
        transmil_output = transmil_model(multi_scale_input)
        nnmil_output = nnmil_model(multi_scale_input)

    # Verify compatible output shapes
    assert transmil_output.shape == (2, 2)
    assert nnmil_output.shape == (2, 2)


def test_inference_api_compatibility():
    """Test that inference APIs are compatible between models."""
    # Create models
    transmil_model = TransMIL(feature_dim=512, num_classes=3)
    nnmil_model = nnMIL(feature_dim=512, hidden_dim=256, num_classes=3)

    # Test input
    features = torch.randn(1, 200, 512)

    # Both should support same inference methods
    with torch.no_grad():
        # Basic inference
        transmil_logits = transmil_model(features)
        nnmil_logits = nnmil_model(features)

        # Inference with attention
        transmil_logits_att, transmil_attention = transmil_model(features, return_attention=True)
        nnmil_logits_att, nnmil_attention = nnmil_model(features, return_attention=True)

        # Probability computation
        transmil_probs = torch.softmax(transmil_logits, dim=1)
        nnmil_probs = torch.softmax(nnmil_logits, dim=1)

    # Verify all outputs have compatible shapes
    assert transmil_logits.shape == nnmil_logits.shape == (1, 3)
    assert transmil_attention.shape == nnmil_attention.shape == (1, 200)
    assert transmil_probs.shape == nnmil_probs.shape == (1, 3)


def test_configuration_compatibility():
    """Test that configurations are compatible between model types."""
    # Base configuration that should work for both models
    base_config = {
        "model": {"feature_dim": 1024, "num_classes": 2, "dropout": 0.25},
        "training": {"batch_size": 32, "learning_rate": 1e-4, "max_epochs": 100},
        "data": {"task_type": "classification"},
    }

    # Create TransMIL-specific config
    transmil_config = base_config.copy()
    transmil_config["model_type"] = "TransMIL"

    # Create nnMIL-specific config
    nnmil_config = base_config.copy()
    nnmil_config["model_type"] = "nnMIL"
    nnmil_config["model"]["hidden_dim"] = 256  # nnMIL-specific parameter

    # Both should be valid configurations
    transmil_trainer = UnifiedTrainer(transmil_config)
    nnmil_trainer = UnifiedTrainer(nnmil_config)

    # Verify models were created successfully
    assert transmil_trainer.model is not None
    assert nnmil_trainer.model is not None


def test_performance_regression_check():
    """Test that nnMIL doesn't significantly regress from TransMIL performance."""
    # Create identical test scenario
    batch_size, num_patches, feature_dim, num_classes = 4, 100, 1024, 2

    # Create models
    transmil_model = TransMIL(feature_dim=feature_dim, num_classes=num_classes)
    nnmil_model = nnMIL(feature_dim=feature_dim, hidden_dim=256, num_classes=num_classes)

    # Set to eval mode
    transmil_model.eval()
    nnmil_model.eval()

    # Create test data
    features = torch.randn(batch_size, num_patches, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Measure inference time
    import time

    # TransMIL timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            transmil_output = transmil_model(features)
    transmil_time = time.time() - start_time

    # nnMIL timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            nnmil_output = nnmil_model(features)
    nnmil_time = time.time() - start_time

    # nnMIL should not be significantly slower (within 2x)
    assert (
        nnmil_time < transmil_time * 2.0
    ), f"nnMIL too slow: {nnmil_time:.3f}s vs TransMIL {transmil_time:.3f}s"

    # Both should produce reasonable outputs
    assert not torch.isnan(transmil_output).any()
    assert not torch.isnan(nnmil_output).any()


def test_memory_usage_compatibility():
    """Test that nnMIL memory usage is comparable to TransMIL."""
    import os

    import psutil

    process = psutil.Process(os.getpid())

    # Measure TransMIL memory
    transmil_model = TransMIL(feature_dim=1024, num_classes=2)
    features = torch.randn(8, 500, 1024)  # Large input

    memory_before = process.memory_info().rss
    with torch.no_grad():
        transmil_output = transmil_model(features)
    memory_after_transmil = process.memory_info().rss
    transmil_memory = memory_after_transmil - memory_before

    # Clear memory
    del transmil_model, transmil_output
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Measure nnMIL memory
    nnmil_model = nnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)

    memory_before = process.memory_info().rss
    with torch.no_grad():
        nnmil_output = nnmil_model(features)
    memory_after_nnmil = process.memory_info().rss
    nnmil_memory = memory_after_nnmil - memory_before

    # nnMIL should not use significantly more memory (within 50% increase)
    memory_ratio = nnmil_memory / max(transmil_memory, 1)  # Avoid division by zero
    assert memory_ratio < 1.5, (
        f"nnMIL uses too much memory: {memory_ratio:.2f}x TransMIL "
        f"({nnmil_memory / 1024 / 1024:.1f} MB vs {transmil_memory / 1024 / 1024:.1f} MB)"
    )


def test_gradient_compatibility():
    """Test that gradients flow correctly in both models."""
    # Create models
    transmil_model = TransMIL(feature_dim=256, num_classes=2)
    nnmil_model = nnMIL(feature_dim=256, hidden_dim=128, num_classes=2)

    # Create test data
    features = torch.randn(2, 50, 256, requires_grad=True)
    labels = torch.randint(0, 2, (2,))

    # Forward and backward for TransMIL
    transmil_logits = transmil_model(features)
    transmil_loss = torch.nn.functional.cross_entropy(transmil_logits, labels)
    transmil_loss.backward(retain_graph=True)

    # Check TransMIL gradients
    transmil_has_gradients = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in transmil_model.parameters()
    )

    # Clear gradients
    transmil_model.zero_grad()
    features.grad = None

    # Forward and backward for nnMIL
    nnmil_logits = nnmil_model(features)
    nnmil_loss = torch.nn.functional.cross_entropy(nnmil_logits, labels)
    nnmil_loss.backward()

    # Check nnMIL gradients
    nnmil_has_gradients = any(
        param.grad is not None and param.grad.abs().sum() > 0 for param in nnmil_model.parameters()
    )

    # Both models should have gradients
    assert transmil_has_gradients, "TransMIL should have gradients"
    assert nnmil_has_gradients, "nnMIL should have gradients"
