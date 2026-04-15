"""
Unit tests for temporal reasoning modules.
"""

import torch

from src.models.temporal import CrossSlideTemporalReasoner, TemporalAttention


class TestTemporalAttention:
    """Tests for TemporalAttention module."""

    def test_basic_forward(self):
        """Test basic temporal attention forward pass."""
        temporal_attn = TemporalAttention(embed_dim=256, num_heads=8)

        slide_embeddings = torch.randn(4, 5, 256)  # [batch, num_slides, embed_dim]
        output = temporal_attn(slide_embeddings)

        assert output.shape == (4, 5, 256)
        assert not torch.isnan(output).any()

    def test_with_timestamps(self):
        """Test temporal attention with timestamps."""
        temporal_attn = TemporalAttention(embed_dim=256, num_heads=8)

        slide_embeddings = torch.randn(4, 5, 256)
        timestamps = torch.tensor(
            [
                [0, 30, 60, 90, 120],  # Days
                [0, 45, 90, 135, 180],
                [0, 60, 120, 180, 240],
                [0, 15, 30, 45, 60],
            ],
            dtype=torch.float32,
        )

        output = temporal_attn(slide_embeddings, timestamps)

        assert output.shape == (4, 5, 256)
        assert not torch.isnan(output).any()

    def test_single_slide(self):
        """Test temporal attention with single slide (edge case)."""
        temporal_attn = TemporalAttention(embed_dim=256, num_heads=8)

        # Single slide per batch
        slide_embeddings = torch.randn(4, 1, 256)
        timestamps = torch.tensor([[0], [0], [0], [0]], dtype=torch.float32)

        output = temporal_attn(slide_embeddings, timestamps)

        assert output.shape == (4, 1, 256)
        assert not torch.isnan(output).any()

    def test_with_mask(self):
        """Test temporal attention with padding mask."""
        temporal_attn = TemporalAttention(embed_dim=256, num_heads=8)

        slide_embeddings = torch.randn(4, 5, 256)

        # Mask: only first 3 slides valid for first 2 batches, all 5 for others
        mask = torch.zeros(4, 5, dtype=torch.bool)
        mask[0, :3] = True
        mask[1, :3] = True
        mask[2, :] = True
        mask[3, :] = True

        output = temporal_attn(slide_embeddings, mask=mask)

        assert output.shape == (4, 5, 256)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow through temporal attention."""
        temporal_attn = TemporalAttention(embed_dim=128, num_heads=4)

        slide_embeddings = torch.randn(2, 5, 128, requires_grad=True)
        output = temporal_attn(slide_embeddings)

        loss = output.sum()
        loss.backward()

        assert slide_embeddings.grad is not None

    def test_different_sequence_lengths(self):
        """Test with varying sequence lengths."""
        temporal_attn = TemporalAttention(embed_dim=128, num_heads=4)

        for num_slides in [1, 2, 5, 10]:
            slide_embeddings = torch.randn(4, num_slides, 128)
            output = temporal_attn(slide_embeddings)
            assert output.shape == (4, num_slides, 128)


class TestCrossSlideTemporalReasoner:
    """Tests for CrossSlideTemporalReasoner module."""

    def test_basic_forward(self):
        """Test basic temporal reasoning forward pass."""
        reasoner = CrossSlideTemporalReasoner(embed_dim=256, num_heads=8)

        slide_embeddings = torch.randn(4, 5, 256)
        timestamps = torch.randn(4, 5)

        sequence_emb, progression_features = reasoner(slide_embeddings, timestamps)

        assert sequence_emb.shape == (4, 256)
        assert progression_features.shape == (4, 128)  # embed_dim // 2
        assert not torch.isnan(sequence_emb).any()
        assert not torch.isnan(progression_features).any()

    def test_single_slide(self):
        """Test temporal reasoning with single slide (no progression features possible)."""
        reasoner = CrossSlideTemporalReasoner(embed_dim=256, num_heads=8)

        # Single slide per batch - edge case
        slide_embeddings = torch.randn(4, 1, 256)
        timestamps = torch.tensor([[0], [0], [0], [0]], dtype=torch.float32)

        sequence_emb, progression_features = reasoner(slide_embeddings, timestamps)

        assert sequence_emb.shape == (4, 256)
        assert progression_features.shape == (4, 128)
        assert not torch.isnan(sequence_emb).any()
        assert not torch.isnan(progression_features).any()
        # Progression features should be zeros for single slide
        assert torch.allclose(progression_features, torch.zeros_like(progression_features))

    def test_two_slides(self):
        """Test with minimum number of slides for progression features."""
        reasoner = CrossSlideTemporalReasoner(embed_dim=256, num_heads=8)

        slide_embeddings = torch.randn(4, 2, 256)
        timestamps = torch.tensor(
            [
                [0, 30],
                [0, 60],
                [0, 45],
                [0, 90],
            ],
            dtype=torch.float32,
        )

        sequence_emb, progression_features = reasoner(slide_embeddings, timestamps)

        assert sequence_emb.shape == (4, 256)
        assert progression_features.shape == (4, 128)
        assert not torch.isnan(sequence_emb).any()
        assert not torch.isnan(progression_features).any()

    def test_different_pooling_strategies(self):
        """Test different temporal pooling strategies."""
        slide_embeddings = torch.randn(4, 5, 256)
        timestamps = torch.randn(4, 5)

        for pooling in ["attention", "mean", "max", "last"]:
            reasoner = CrossSlideTemporalReasoner(embed_dim=256, num_heads=8, pooling=pooling)
            sequence_emb, progression_features = reasoner(slide_embeddings, timestamps)

            assert sequence_emb.shape == (4, 256)
            assert progression_features.shape == (4, 128)
            assert not torch.isnan(sequence_emb).any()

    def test_with_mask(self):
        """Test temporal reasoning with padding mask."""
        reasoner = CrossSlideTemporalReasoner(embed_dim=256, num_heads=8)

        slide_embeddings = torch.randn(4, 5, 256)

        # Mask with varying valid slides per batch
        mask = torch.zeros(4, 5, dtype=torch.bool)
        mask[0, :5] = True  # All 5 valid
        mask[1, :3] = True  # Only 3 valid
        mask[2, :4] = True  # Only 4 valid
        mask[3, :2] = True  # Only 2 valid

        sequence_emb, progression_features = reasoner(slide_embeddings, mask=mask)

        assert sequence_emb.shape == (4, 256)
        assert progression_features.shape == (4, 128)
        assert not torch.isnan(sequence_emb).any()
        assert not torch.isnan(progression_features).any()

    def test_gradient_flow(self):
        """Test that gradients flow through temporal reasoner."""
        reasoner = CrossSlideTemporalReasoner(embed_dim=128, num_heads=4)

        slide_embeddings = torch.randn(2, 5, 128, requires_grad=True)
        timestamps = torch.randn(2, 5)

        sequence_emb, progression_features = reasoner(slide_embeddings, timestamps)
        loss = sequence_emb.sum() + progression_features.sum()
        loss.backward()

        assert slide_embeddings.grad is not None

    def test_no_timestamps(self):
        """Test temporal reasoning without timestamps (uses positional encoding)."""
        reasoner = CrossSlideTemporalReasoner(embed_dim=256, num_heads=8)

        slide_embeddings = torch.randn(4, 5, 256)
        # No timestamps provided

        sequence_emb, progression_features = reasoner(slide_embeddings)

        assert sequence_emb.shape == (4, 256)
        assert progression_features.shape == (4, 128)
        assert not torch.isnan(sequence_emb).any()
        assert not torch.isnan(progression_features).any()

    def test_different_embed_dims(self):
        """Test with various embedding dimensions."""
        for embed_dim in [64, 128, 256, 512]:
            reasoner = CrossSlideTemporalReasoner(embed_dim=embed_dim, num_heads=8)

            slide_embeddings = torch.randn(4, 5, embed_dim)
            sequence_emb, progression_features = reasoner(slide_embeddings)

            assert sequence_emb.shape == (4, embed_dim)
            assert progression_features.shape == (4, embed_dim // 2)
