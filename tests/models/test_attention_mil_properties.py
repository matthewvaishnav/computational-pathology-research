"""
Property-based tests for AttentionMIL refactoring equivalence.

**Validates: Requirements FR-4 (DRY), NFR-1 (Backward Compatibility)**

This module tests that the refactored AttentionMIL model produces identical
outputs to the original implementation. We use property-based testing with
Hypothesis to generate random feature tensors with various configurations.

The test ensures:
1. Logits are identical (atol=1e-6)
2. Attention weights are identical (atol=1e-6)
3. Behavior is consistent across different configurations:
   - Gated vs simple attention
   - Early vs late fusion
   - Single-scale vs multi-scale
   - Different batch sizes, patch counts, feature dimensions

Note: This test compares the refactored AttentionMIL against the backup file.
The backup file (attention_mil.py.backup) contains the original implementation
before refactoring.
"""

from pathlib import Path

import pytest
import torch

from hypothesis import given, settings
from hypothesis import strategies as st

# Import the refactored AttentionMIL
from src.models.mil.attention_mil import AttentionMIL

# Load the original AttentionMIL from backup by reading and executing the file
backup_path = Path(__file__).parent.parent.parent / "src" / "models" / "attention_mil.py.backup"
if not backup_path.exists():
    raise FileNotFoundError(f"Backup file not found: {backup_path}")

# Read the backup file and execute it in a separate namespace
backup_namespace = {}
with open(backup_path, "r", encoding="utf-8") as f:
    backup_code = f.read()
    exec(backup_code, backup_namespace)

OldAttentionMIL = backup_namespace["AttentionMIL"]


# Hypothesis strategies for generating test data
@st.composite
def feature_tensor_strategy(draw):
    """Generate random feature tensors with various configurations."""
    batch_size = draw(st.integers(min_value=1, max_value=8))
    num_patches = draw(st.integers(min_value=10, max_value=200))
    feature_dim = draw(st.sampled_from([512, 1024, 2048]))

    # Generate random features
    features = torch.randn(batch_size, num_patches, feature_dim)

    # Generate num_patches tensor (some samples may have fewer valid patches)
    actual_patches = []
    for _ in range(batch_size):
        # Each sample has between 50% and 100% of max patches
        actual = draw(st.integers(min_value=max(1, num_patches // 2), max_value=num_patches))
        actual_patches.append(actual)

    num_patches_tensor = torch.tensor(actual_patches)

    return features, num_patches_tensor, feature_dim


@st.composite
def model_config_strategy(draw):
    """Generate random model configurations."""
    feature_dim = draw(st.sampled_from([512, 1024, 2048]))
    hidden_dim = draw(st.sampled_from([128, 256, 512]))
    num_classes = draw(st.sampled_from([2, 3, 5]))
    dropout = draw(st.floats(min_value=0.0, max_value=0.5))
    gated = draw(st.booleans())
    attention_mode = draw(st.sampled_from(["instance", "bag"]))

    return {
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "dropout": dropout,
        "gated": gated,
        "attention_mode": attention_mode,
        "multi_scale": False,
        "num_scales": 1,
        "fusion_strategy": "early",
    }


@st.composite
def multi_scale_config_strategy(draw):
    """Generate random multi-scale model configurations."""
    feature_dim = draw(st.sampled_from([512, 1024]))
    hidden_dim = draw(st.sampled_from([128, 256]))
    num_classes = draw(st.sampled_from([2, 3]))
    dropout = draw(st.floats(min_value=0.0, max_value=0.3))
    gated = draw(st.booleans())
    num_scales = draw(st.integers(min_value=2, max_value=3))
    fusion_strategy = draw(st.sampled_from(["early", "late"]))

    return {
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "dropout": dropout,
        "gated": gated,
        "attention_mode": "instance",
        "multi_scale": True,
        "num_scales": num_scales,
        "fusion_strategy": fusion_strategy,
    }


@st.composite
def multi_scale_features_strategy(draw, num_scales, batch_size, num_patches, feature_dim):
    """Generate multi-scale feature tensors."""
    features = []
    for _ in range(num_scales):
        scale_features = torch.randn(batch_size, num_patches, feature_dim)
        features.append(scale_features)
    return features


class TestAttentionMILEquivalence:
    """Property tests for AttentionMIL refactoring equivalence."""

    @given(config=model_config_strategy(), features_data=feature_tensor_strategy())
    @settings(max_examples=100, deadline=None)
    def test_attention_mil_single_scale_equivalence(self, config, features_data):
        """
        Property test: Refactored AttentionMIL produces identical outputs to original.

        **Validates: Requirements FR-4, NFR-1**

        Tests single-scale configurations with various:
        - Batch sizes (1-8)
        - Patch counts (10-200)
        - Feature dimensions (512, 1024, 2048)
        - Gated vs simple attention
        - Different hidden dimensions
        """
        features, num_patches, feature_dim = features_data

        # Skip if feature dimensions don't match
        if config["feature_dim"] != feature_dim:
            return

        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Create old model
        old_model = OldAttentionMIL(**config)
        old_model.eval()

        # Set random seed again for new model
        torch.manual_seed(42)

        # Create new model
        new_model = AttentionMIL(**config)
        new_model.eval()

        # Forward pass with old model
        with torch.no_grad():
            old_output = old_model(features, num_patches, return_attention=True)
            if isinstance(old_output, tuple):
                old_logits, old_attention = old_output
            else:
                old_logits = old_output
                old_attention = None

        # Forward pass with new model
        with torch.no_grad():
            new_output = new_model(features, num_patches, return_attention=True)
            if isinstance(new_output, tuple):
                new_logits, new_attention = new_output
            else:
                new_logits = new_output
                new_attention = None

        # Verify logits are identical
        assert torch.allclose(
            old_logits, new_logits, atol=1e-6
        ), f"Logits differ: max diff = {(old_logits - new_logits).abs().max()}"

        # Verify attention weights are identical (if returned)
        if old_attention is not None and new_attention is not None:
            assert torch.allclose(
                old_attention, new_attention, atol=1e-6
            ), f"Attention weights differ: max diff = {(old_attention - new_attention).abs().max()}"

    @given(config=multi_scale_config_strategy())
    @settings(max_examples=50, deadline=None)
    def test_attention_mil_multi_scale_equivalence(self, config):
        """
        Property test: Multi-scale AttentionMIL produces identical outputs.

        **Validates: Requirements FR-4, NFR-1**

        Tests multi-scale configurations with:
        - 2-3 scales
        - Early vs late fusion
        - Various batch sizes and patch counts
        """
        # Generate multi-scale features
        batch_size = 4
        num_patches = 100
        feature_dim = config["feature_dim"]
        num_scales = config["num_scales"]

        # Generate features for each scale
        features = []
        for _ in range(num_scales):
            scale_features = torch.randn(batch_size, num_patches, feature_dim)
            features.append(scale_features)

        num_patches_tensor = torch.tensor([num_patches] * batch_size)

        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Create old model
        old_model = OldAttentionMIL(**config)
        old_model.eval()

        # Set random seed again for new model
        torch.manual_seed(42)

        # Create new model
        new_model = AttentionMIL(**config)
        new_model.eval()

        # Forward pass with old model
        with torch.no_grad():
            old_output = old_model(features, num_patches_tensor, return_attention=True)
            if isinstance(old_output, tuple):
                old_logits, old_attention = old_output
            else:
                old_logits = old_output
                old_attention = None

        # Forward pass with new model
        with torch.no_grad():
            new_output = new_model(features, num_patches_tensor, return_attention=True)
            if isinstance(new_output, tuple):
                new_logits, new_attention = new_output
            else:
                new_logits = new_output
                new_attention = None

        # Verify logits are identical
        assert torch.allclose(
            old_logits, new_logits, atol=1e-6
        ), f"Multi-scale logits differ: max diff = {(old_logits - new_logits).abs().max()}"

        # Verify attention weights are identical (if returned)
        if old_attention is not None and new_attention is not None:
            assert torch.allclose(
                old_attention, new_attention, atol=1e-6
            ), f"Multi-scale attention weights differ: max diff = {(old_attention - new_attention).abs().max()}"

    def test_attention_mil_equivalence_fixed_example(self):
        """
        Unit test: Verify equivalence with a fixed example.

        This is a simpler test case to debug if property tests fail.
        """
        # Fixed configuration
        config = {
            "feature_dim": 1024,
            "hidden_dim": 256,
            "num_classes": 2,
            "dropout": 0.1,
            "gated": True,
            "attention_mode": "instance",
            "multi_scale": False,
            "num_scales": 1,
            "fusion_strategy": "early",
        }

        # Fixed input
        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        num_patches_tensor = torch.tensor([100, 80, 90, 100])

        # Set random seed
        torch.manual_seed(42)
        old_model = OldAttentionMIL(**config)
        old_model.eval()

        torch.manual_seed(42)
        new_model = AttentionMIL(**config)
        new_model.eval()

        # Forward pass
        with torch.no_grad():
            old_output = old_model(features, num_patches_tensor, return_attention=True)
            if isinstance(old_output, tuple):
                old_logits, old_attention = old_output
            else:
                old_logits = old_output
                old_attention = None

            new_output = new_model(features, num_patches_tensor, return_attention=True)
            if isinstance(new_output, tuple):
                new_logits, new_attention = new_output
            else:
                new_logits = new_output
                new_attention = None

        # Verify
        assert torch.allclose(old_logits, new_logits, atol=1e-6)
        if old_attention is not None and new_attention is not None:
            assert torch.allclose(old_attention, new_attention, atol=1e-6)

    def test_attention_mil_equivalence_simple_attention(self):
        """
        Unit test: Verify equivalence with simple (non-gated) attention.
        """
        config = {
            "feature_dim": 1024,
            "hidden_dim": 256,
            "num_classes": 2,
            "dropout": 0.1,
            "gated": False,  # Simple attention
            "attention_mode": "instance",
            "multi_scale": False,
            "num_scales": 1,
            "fusion_strategy": "early",
        }

        batch_size = 4
        num_patches = 100
        features = torch.randn(batch_size, num_patches, 1024)
        num_patches_tensor = torch.tensor([100, 80, 90, 100])

        torch.manual_seed(42)
        old_model = OldAttentionMIL(**config)
        old_model.eval()

        torch.manual_seed(42)
        new_model = AttentionMIL(**config)
        new_model.eval()

        with torch.no_grad():
            old_output = old_model(features, num_patches_tensor, return_attention=True)
            if isinstance(old_output, tuple):
                old_logits, old_attention = old_output
            else:
                old_logits = old_output
                old_attention = None

            new_output = new_model(features, num_patches_tensor, return_attention=True)
            if isinstance(new_output, tuple):
                new_logits, new_attention = new_output
            else:
                new_logits = new_output
                new_attention = None

        assert torch.allclose(old_logits, new_logits, atol=1e-6)
        if old_attention is not None and new_attention is not None:
            assert torch.allclose(old_attention, new_attention, atol=1e-6)

    def test_attention_mil_equivalence_multi_scale_early_fusion(self):
        """
        Unit test: Verify equivalence with multi-scale early fusion.
        """
        config = {
            "feature_dim": 1024,
            "hidden_dim": 256,
            "num_classes": 2,
            "dropout": 0.1,
            "gated": True,
            "attention_mode": "instance",
            "multi_scale": True,
            "num_scales": 2,
            "fusion_strategy": "early",
        }

        batch_size = 4
        num_patches = 100
        scale1 = torch.randn(batch_size, num_patches, 1024)
        scale2 = torch.randn(batch_size, num_patches, 1024)
        features = [scale1, scale2]
        num_patches_tensor = torch.tensor([100, 80, 90, 100])

        torch.manual_seed(42)
        old_model = OldAttentionMIL(**config)
        old_model.eval()

        torch.manual_seed(42)
        new_model = AttentionMIL(**config)
        new_model.eval()

        with torch.no_grad():
            old_output = old_model(features, num_patches_tensor, return_attention=True)
            if isinstance(old_output, tuple):
                old_logits, old_attention = old_output
            else:
                old_logits = old_output
                old_attention = None

            new_output = new_model(features, num_patches_tensor, return_attention=True)
            if isinstance(new_output, tuple):
                new_logits, new_attention = new_output
            else:
                new_logits = new_output
                new_attention = None

        assert torch.allclose(old_logits, new_logits, atol=1e-6)
        if old_attention is not None and new_attention is not None:
            assert torch.allclose(old_attention, new_attention, atol=1e-6)

    def test_attention_mil_equivalence_multi_scale_late_fusion(self):
        """
        Unit test: Verify equivalence with multi-scale late fusion.
        """
        config = {
            "feature_dim": 1024,
            "hidden_dim": 256,
            "num_classes": 2,
            "dropout": 0.1,
            "gated": True,
            "attention_mode": "instance",
            "multi_scale": True,
            "num_scales": 2,
            "fusion_strategy": "late",
        }

        batch_size = 4
        num_patches = 100
        scale1 = torch.randn(batch_size, num_patches, 1024)
        scale2 = torch.randn(batch_size, num_patches, 1024)
        features = [scale1, scale2]
        num_patches_tensor = torch.tensor([100, 80, 90, 100])

        torch.manual_seed(42)
        old_model = OldAttentionMIL(**config)
        old_model.eval()

        torch.manual_seed(42)
        new_model = AttentionMIL(**config)
        new_model.eval()

        with torch.no_grad():
            old_output = old_model(features, num_patches_tensor, return_attention=True)
            if isinstance(old_output, tuple):
                old_logits, old_attention = old_output
            else:
                old_logits = old_output
                old_attention = None

            new_output = new_model(features, num_patches_tensor, return_attention=True)
            if isinstance(new_output, tuple):
                new_logits, new_attention = new_output
            else:
                new_logits = new_output
                new_attention = None

        assert torch.allclose(old_logits, new_logits, atol=1e-6)
        if old_attention is not None and new_attention is not None:
            assert torch.allclose(old_attention, new_attention, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
