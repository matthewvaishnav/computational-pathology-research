"""
Unit tests for attention mechanisms.

Tests the attention mechanisms extracted from attention_mil.py:
- GatedAttention
- SimpleAttention
- TransformerAttention
"""

import pytest
import torch

from src.models.components.attention_mechanisms import (
    GatedAttention,
    SimpleAttention,
    TransformerAttention,
)


class TestGatedAttention:
    """Tests for GatedAttention mechanism."""

    def test_initialization(self):
        """Test GatedAttention can be initialized."""
        attention = GatedAttention(feature_dim=256, hidden_dim=128)
        assert attention.feature_dim == 256
        assert attention.hidden_dim == 128

    def test_initialization_default_hidden_dim(self):
        """Test GatedAttention uses feature_dim as default hidden_dim."""
        attention = GatedAttention(feature_dim=256)
        assert attention.feature_dim == 256
        assert attention.hidden_dim == 256

    def test_forward_shape(self):
        """Test GatedAttention output shape is correct."""
        attention = GatedAttention(feature_dim=256, hidden_dim=128)
        features = torch.randn(4, 100, 256)
        attention_weights = attention(features)

        assert attention_weights.shape == (4, 100)

    def test_attention_weights_sum_to_one(self):
        """Test attention weights sum to 1 for each sample."""
        attention = GatedAttention(feature_dim=256, hidden_dim=128)
        features = torch.randn(4, 100, 256)
        attention_weights = attention(features)

        # Check each sample's weights sum to ~1.0
        for i in range(4):
            assert torch.allclose(attention_weights[i].sum(), torch.tensor(1.0), atol=1e-6)

    def test_attention_weights_non_negative(self):
        """Test attention weights are non-negative."""
        attention = GatedAttention(feature_dim=256, hidden_dim=128)
        features = torch.randn(4, 100, 256)
        attention_weights = attention(features)

        assert (attention_weights >= 0).all()

    def test_masking(self):
        """Test masking sets padded patches to zero weight."""
        attention = GatedAttention(feature_dim=256, hidden_dim=128)
        features = torch.randn(4, 100, 256)

        # Create mask: first sample has only 80 valid patches
        mask = torch.ones(4, 100, dtype=torch.bool)
        mask[0, 80:] = False

        attention_weights = attention(features, mask)

        # Masked patches should have ~0 weight
        assert torch.allclose(attention_weights[0, 80:].sum(), torch.tensor(0.0), atol=1e-6)

        # Valid patches should still sum to ~1.0
        assert torch.allclose(attention_weights[0, :80].sum(), torch.tensor(1.0), atol=1e-6)

    def test_different_batch_sizes(self):
        """Test GatedAttention works with different batch sizes."""
        attention = GatedAttention(feature_dim=256, hidden_dim=128)

        for batch_size in [1, 2, 8, 16]:
            features = torch.randn(batch_size, 100, 256)
            attention_weights = attention(features)
            assert attention_weights.shape == (batch_size, 100)

    def test_different_num_patches(self):
        """Test GatedAttention works with different numbers of patches."""
        attention = GatedAttention(feature_dim=256, hidden_dim=128)

        for num_patches in [10, 50, 100, 500]:
            features = torch.randn(4, num_patches, 256)
            attention_weights = attention(features)
            assert attention_weights.shape == (4, num_patches)


class TestSimpleAttention:
    """Tests for SimpleAttention mechanism."""

    def test_initialization(self):
        """Test SimpleAttention can be initialized."""
        attention = SimpleAttention(feature_dim=256, hidden_dim=128)
        assert attention.feature_dim == 256
        assert attention.hidden_dim == 128

    def test_initialization_default_hidden_dim(self):
        """Test SimpleAttention uses feature_dim // 2 as default hidden_dim."""
        attention = SimpleAttention(feature_dim=256)
        assert attention.feature_dim == 256
        assert attention.hidden_dim == 128

    def test_forward_shape(self):
        """Test SimpleAttention output shape is correct."""
        attention = SimpleAttention(feature_dim=256, hidden_dim=128)
        features = torch.randn(4, 100, 256)
        attention_weights = attention(features)

        assert attention_weights.shape == (4, 100)

    def test_attention_weights_sum_to_one(self):
        """Test attention weights sum to 1 for each sample."""
        attention = SimpleAttention(feature_dim=256, hidden_dim=128)
        features = torch.randn(4, 100, 256)
        attention_weights = attention(features)

        # Check each sample's weights sum to ~1.0
        for i in range(4):
            assert torch.allclose(attention_weights[i].sum(), torch.tensor(1.0), atol=1e-6)

    def test_attention_weights_non_negative(self):
        """Test attention weights are non-negative."""
        attention = SimpleAttention(feature_dim=256, hidden_dim=128)
        features = torch.randn(4, 100, 256)
        attention_weights = attention(features)

        assert (attention_weights >= 0).all()

    def test_masking(self):
        """Test masking sets padded patches to zero weight."""
        attention = SimpleAttention(feature_dim=256, hidden_dim=128)
        features = torch.randn(4, 100, 256)

        # Create mask: first sample has only 80 valid patches
        mask = torch.ones(4, 100, dtype=torch.bool)
        mask[0, 80:] = False

        attention_weights = attention(features, mask)

        # Masked patches should have ~0 weight
        assert torch.allclose(attention_weights[0, 80:].sum(), torch.tensor(0.0), atol=1e-6)

        # Valid patches should still sum to ~1.0
        assert torch.allclose(attention_weights[0, :80].sum(), torch.tensor(1.0), atol=1e-6)


class TestTransformerAttention:
    """Tests for TransformerAttention mechanism."""

    def test_initialization(self):
        """Test TransformerAttention can be initialized."""
        attention = TransformerAttention(feature_dim=256, num_heads=8, num_layers=2, dropout=0.1)
        assert attention.feature_dim == 256
        assert attention.num_heads == 8
        assert attention.num_layers == 2

    def test_initialization_invalid_heads(self):
        """Test TransformerAttention raises error if feature_dim not divisible by num_heads."""
        with pytest.raises(ValueError, match="must be divisible by num_heads"):
            TransformerAttention(feature_dim=256, num_heads=7)

    def test_forward_shape(self):
        """Test TransformerAttention output shape is correct."""
        attention = TransformerAttention(feature_dim=256, num_heads=8, num_layers=2)
        features = torch.randn(4, 100, 256)
        output, attn_weights = attention(features)

        # Output includes CLS token at position 0
        assert output.shape == (4, 101, 256)  # 100 patches + 1 CLS token

    def test_forward_without_pos_encoding(self):
        """Test TransformerAttention works without positional encoding."""
        attention = TransformerAttention(
            feature_dim=256, num_heads=8, num_layers=2, use_pos_encoding=False
        )
        features = torch.randn(4, 100, 256)
        output, attn_weights = attention(features)

        assert output.shape == (4, 101, 256)

    def test_masking(self):
        """Test masking works with TransformerAttention."""
        attention = TransformerAttention(feature_dim=256, num_heads=8, num_layers=2)
        features = torch.randn(4, 100, 256)

        # Create mask: first sample has only 80 valid patches
        mask = torch.ones(4, 100, dtype=torch.bool)
        mask[0, 80:] = False

        output, attn_weights = attention(features, mask)

        # Output should still have correct shape
        assert output.shape == (4, 101, 256)

    def test_get_cls_token(self):
        """Test extracting CLS token from output."""
        attention = TransformerAttention(feature_dim=256, num_heads=8, num_layers=2)
        features = torch.randn(4, 100, 256)
        output, _ = attention(features)

        cls_token = attention.get_cls_token(output)
        assert cls_token.shape == (4, 256)

    def test_get_patch_features(self):
        """Test extracting patch features from output."""
        attention = TransformerAttention(feature_dim=256, num_heads=8, num_layers=2)
        features = torch.randn(4, 100, 256)
        output, _ = attention(features)

        patch_features = attention.get_patch_features(output)
        assert patch_features.shape == (4, 100, 256)

    def test_different_num_layers(self):
        """Test TransformerAttention works with different numbers of layers."""
        for num_layers in [1, 2, 4, 6]:
            attention = TransformerAttention(feature_dim=256, num_heads=8, num_layers=num_layers)
            features = torch.randn(4, 100, 256)
            output, _ = attention(features)
            assert output.shape == (4, 101, 256)

    def test_different_num_heads(self):
        """Test TransformerAttention works with different numbers of heads."""
        for num_heads in [1, 2, 4, 8]:
            attention = TransformerAttention(feature_dim=256, num_heads=num_heads, num_layers=2)
            features = torch.randn(4, 100, 256)
            output, _ = attention(features)
            assert output.shape == (4, 101, 256)

    def test_deterministic_with_seed(self):
        """Test TransformerAttention produces same output with same seed."""
        attention = TransformerAttention(feature_dim=256, num_heads=8, num_layers=2)
        features = torch.randn(4, 100, 256)

        # First forward pass
        torch.manual_seed(42)
        output1, _ = attention(features)

        # Second forward pass with same seed
        torch.manual_seed(42)
        output2, _ = attention(features)

        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)


class TestAttentionMechanismComparison:
    """Tests comparing different attention mechanisms."""

    def test_all_mechanisms_produce_valid_output(self):
        """Test all attention mechanisms produce valid output."""
        features = torch.randn(4, 100, 256)

        # GatedAttention
        gated = GatedAttention(feature_dim=256, hidden_dim=256)
        gated_weights = gated(features)
        assert gated_weights.shape == (4, 100)
        assert (gated_weights >= 0).all()

        # SimpleAttention
        simple = SimpleAttention(feature_dim=256, hidden_dim=128)
        simple_weights = simple(features)
        assert simple_weights.shape == (4, 100)
        assert (simple_weights >= 0).all()

        # TransformerAttention
        transformer = TransformerAttention(feature_dim=256, num_heads=8, num_layers=2)
        transformer_output, _ = transformer(features)
        assert transformer_output.shape == (4, 101, 256)

    def test_gated_vs_simple_attention_different(self):
        """Test gated and simple attention produce different weights."""
        features = torch.randn(4, 100, 256)

        # Set same seed for both
        torch.manual_seed(42)
        gated = GatedAttention(feature_dim=256, hidden_dim=256)
        gated_weights = gated(features)

        torch.manual_seed(42)
        simple = SimpleAttention(feature_dim=256, hidden_dim=128)
        simple_weights = simple(features)

        # Weights should be different (different architectures)
        assert not torch.allclose(gated_weights, simple_weights, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
