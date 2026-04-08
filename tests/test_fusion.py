"""
Unit tests for fusion mechanisms including CrossModalAttention and MultiModalFusionLayer.
"""

import pytest
import torch

from src.models.fusion import CrossModalAttention, MultiModalFusionLayer


class TestCrossModalAttention:
    """Tests for CrossModalAttention module."""

    def test_basic_forward(self):
        """Test basic cross-modal attention forward pass."""
        fusion = CrossModalAttention(embed_dim=256, num_heads=8)

        query = torch.randn(4, 256)
        key = torch.randn(4, 256)

        output = fusion(query, key)

        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()

    def test_with_value(self):
        """Test cross-modal attention with explicit value."""
        fusion = CrossModalAttention(embed_dim=256, num_heads=8)

        query = torch.randn(4, 256)
        key = torch.randn(4, 256)
        value = torch.randn(4, 256)

        output = fusion(query, key, value)

        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()

    def test_with_key_mask_missing_modality(self):
        """Test cross-modal attention with key_mask simulating missing modality."""
        fusion = CrossModalAttention(embed_dim=256, num_heads=8)

        batch_size = 4
        query = torch.randn(batch_size, 256)
        key = torch.randn(batch_size, 256)

        # Create mask where only first 2 samples have valid key modality
        key_mask = torch.zeros(batch_size, dtype=torch.bool)
        key_mask[:2] = True

        output = fusion(query, key, key_mask=key_mask)

        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()

    def test_with_all_missing_key_modality(self):
        """Test behavior when all key modalities are missing."""
        fusion = CrossModalAttention(embed_dim=256, num_heads=8)

        batch_size = 4
        query = torch.randn(batch_size, 256)
        key = torch.randn(batch_size, 256)

        # All keys are masked out (missing modality)
        key_mask = torch.zeros(batch_size, dtype=torch.bool)

        output = fusion(query, key, key_mask=key_mask)

        # Should still produce output without NaN
        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow through cross-modal attention."""
        fusion = CrossModalAttention(embed_dim=128, num_heads=4)

        query = torch.randn(2, 128, requires_grad=True)
        key = torch.randn(2, 128, requires_grad=True)

        output = fusion(query, key)
        loss = output.sum()
        loss.backward()

        assert query.grad is not None
        assert key.grad is not None

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        fusion = CrossModalAttention(embed_dim=64, num_heads=4)

        for batch_size in [1, 2, 8, 16]:
            query = torch.randn(batch_size, 64)
            key = torch.randn(batch_size, 64)

            output = fusion(query, key)
            assert output.shape == (batch_size, 64)

    def test_different_embed_dims(self):
        """Test with various embedding dimensions."""
        for embed_dim in [64, 128, 256, 512]:
            fusion = CrossModalAttention(embed_dim=embed_dim, num_heads=8)

            query = torch.randn(4, embed_dim)
            key = torch.randn(4, embed_dim)

            output = fusion(query, key)
            assert output.shape == (4, embed_dim)


class TestMultiModalFusionLayer:
    """Tests for MultiModalFusionLayer module."""

    def test_basic_fusion(self):
        """Test fusion with all modalities present."""
        fusion = MultiModalFusionLayer(embed_dim=256, num_heads=8)

        embeddings = {
            "wsi": torch.randn(4, 256),
            "genomic": torch.randn(4, 256),
            "clinical": torch.randn(4, 256),
        }

        output = fusion(embeddings)

        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()

    def test_fusion_with_missing_modality(self):
        """Test fusion with one missing modality."""
        fusion = MultiModalFusionLayer(embed_dim=256, num_heads=8)

        embeddings = {
            "wsi": torch.randn(4, 256),
            "genomic": None,  # Missing
            "clinical": torch.randn(4, 256),
        }

        output = fusion(embeddings)

        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()

    def test_fusion_with_two_missing_modalities(self):
        """Test fusion with two missing modalities."""
        fusion = MultiModalFusionLayer(embed_dim=256, num_heads=8)

        embeddings = {
            "wsi": torch.randn(4, 256),
            "genomic": None,  # Missing
            "clinical": None,  # Missing
        }

        output = fusion(embeddings)

        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()

    def test_fusion_with_all_missing_modalities(self):
        """Test fusion with all modalities missing - should raise ValueError."""
        fusion = MultiModalFusionLayer(embed_dim=256, num_heads=8)

        embeddings = {"wsi": None, "genomic": None, "clinical": None}

        # Should raise ValueError with descriptive message
        with pytest.raises(ValueError, match="At least one modality must be present"):
            fusion(embeddings)

    def test_fusion_with_modality_masks(self):
        """Test fusion with explicit modality masks."""
        fusion = MultiModalFusionLayer(embed_dim=256, num_heads=8)

        batch_size = 4
        embeddings = {
            "wsi": torch.randn(batch_size, 256),
            "genomic": torch.randn(batch_size, 256),
            "clinical": torch.randn(batch_size, 256),
        }

        # Mask out some samples for each modality
        modality_masks = {
            "wsi": torch.tensor([True, True, True, False]),
            "genomic": torch.tensor([True, True, False, True]),
            "clinical": torch.tensor([True, False, True, True]),
        }

        output = fusion(embeddings, modality_masks)

        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow through fusion layer."""
        fusion = MultiModalFusionLayer(embed_dim=128, num_heads=4)

        embeddings = {
            "wsi": torch.randn(2, 128, requires_grad=True),
            "genomic": torch.randn(2, 128, requires_grad=True),
            "clinical": torch.randn(2, 128, requires_grad=True),
        }

        output = fusion(embeddings)
        loss = output.sum()
        loss.backward()

        assert embeddings["wsi"].grad is not None
        assert embeddings["genomic"].grad is not None
        assert embeddings["clinical"].grad is not None

    def test_custom_modalities(self):
        """Test fusion with custom modality names."""
        fusion = MultiModalFusionLayer(
            embed_dim=128, num_heads=4, modalities=["image", "text", "audio"]
        )

        embeddings = {
            "image": torch.randn(4, 128),
            "text": torch.randn(4, 128),
            "audio": torch.randn(4, 128),
        }

        output = fusion(embeddings)

        assert output.shape == (4, 128)
