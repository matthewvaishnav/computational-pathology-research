"""
Unit tests for stain normalization transformer.
"""

import pytest
import torch

from src.models.stain_normalization import (
    ColorFeatureEncoder,
    PatchEmbedding,
    StainNormalizationTransformer,
    StyleConditioner,
    StyleTransferDecoder,
)


class TestPatchEmbedding:
    """Tests for PatchEmbedding module."""

    def test_patch_embedding_forward(self):
        """Test patch embedding with various input sizes."""
        patch_embed = PatchEmbedding(patch_size=16, in_channels=3, embed_dim=256)

        # Test with 256x256 image
        x = torch.randn(2, 3, 256, 256)
        output = patch_embed(x)

        expected_num_patches = (256 // 16) * (256 // 16)
        assert output.shape == (2, expected_num_patches, 256)

    def test_patch_embedding_different_sizes(self):
        """Test patch embedding with different image sizes."""
        patch_embed = PatchEmbedding(patch_size=16, in_channels=3, embed_dim=128)

        # Test with 128x128 image
        x = torch.randn(1, 3, 128, 128)
        output = patch_embed(x)

        expected_num_patches = (128 // 16) * (128 // 16)
        assert output.shape == (1, expected_num_patches, 128)


class TestColorFeatureEncoder:
    """Tests for ColorFeatureEncoder module."""

    def test_encoder_forward(self):
        """Test encoder forward pass."""
        encoder = ColorFeatureEncoder(embed_dim=256, num_heads=8, num_layers=4)

        x = torch.randn(2, 256, 256)  # [batch, num_patches, embed_dim]
        output = encoder(x)

        assert output.shape == x.shape

    def test_encoder_preserves_dimensions(self):
        """Test that encoder preserves input dimensions."""
        encoder = ColorFeatureEncoder(embed_dim=128, num_heads=4, num_layers=2)

        x = torch.randn(4, 64, 128)
        output = encoder(x)

        assert output.shape == (4, 64, 128)


class TestStyleConditioner:
    """Tests for StyleConditioner module."""

    def test_style_conditioner_with_style(self):
        """Test style conditioning with reference style."""
        conditioner = StyleConditioner(embed_dim=256, style_dim=128)

        content = torch.randn(2, 256, 256)
        style = torch.randn(2, 256, 256)

        output = conditioner(content, style)

        assert output.shape == content.shape

    def test_style_conditioner_without_style(self):
        """Test style conditioning without reference style."""
        conditioner = StyleConditioner(embed_dim=256, style_dim=128)

        content = torch.randn(2, 256, 256)
        output = conditioner(content, None)

        # Should return content unchanged when no style provided
        assert output.shape == content.shape
        assert torch.allclose(output, content)


class TestStyleTransferDecoder:
    """Tests for StyleTransferDecoder module."""

    def test_decoder_forward(self):
        """Test decoder forward pass."""
        decoder = StyleTransferDecoder(embed_dim=256, patch_size=16, out_channels=3)

        features = torch.randn(2, 256, 256)  # [batch, num_patches, embed_dim]
        memory = torch.randn(2, 256, 256)

        output = decoder(features, memory, height=256, width=256)

        assert output.shape == (2, 3, 256, 256)

    def test_decoder_different_sizes(self):
        """Test decoder with different image sizes."""
        decoder = StyleTransferDecoder(embed_dim=128, patch_size=16, out_channels=3)

        num_patches = (128 // 16) * (128 // 16)
        features = torch.randn(1, num_patches, 128)
        memory = torch.randn(1, num_patches, 128)

        output = decoder(features, memory, height=128, width=128)

        assert output.shape == (1, 3, 128, 128)


class TestStainNormalizationTransformer:
    """Tests for complete StainNormalizationTransformer."""

    def test_forward_without_reference(self):
        """Test forward pass without reference style."""
        model = StainNormalizationTransformer(
            patch_size=16, embed_dim=256, num_encoder_layers=2, num_decoder_layers=2
        )

        x = torch.randn(2, 3, 256, 256)
        output = model(x)

        assert output.shape == x.shape
        assert output.min() >= -1.0 and output.max() <= 1.0  # tanh bounded

    def test_forward_with_reference(self):
        """Test forward pass with reference style."""
        model = StainNormalizationTransformer(
            patch_size=16, embed_dim=256, num_encoder_layers=2, num_decoder_layers=2
        )

        x = torch.randn(2, 3, 256, 256)
        reference = torch.randn(2, 3, 256, 256)

        output = model(x, reference)

        assert output.shape == x.shape
        assert output.min() >= -1.0 and output.max() <= 1.0

    def test_different_image_sizes(self):
        """Test with different valid image sizes."""
        model = StainNormalizationTransformer(patch_size=16, embed_dim=128)

        # Test 128x128
        x1 = torch.randn(1, 3, 128, 128)
        output1 = model(x1)
        assert output1.shape == (1, 3, 128, 128)

        # Test 256x256
        x2 = torch.randn(1, 3, 256, 256)
        output2 = model(x2)
        assert output2.shape == (1, 3, 256, 256)

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise assertion error."""
        model = StainNormalizationTransformer(patch_size=16)

        # Image size not divisible by patch_size
        x = torch.randn(1, 3, 100, 100)

        with pytest.raises(AssertionError):
            model(x)

    def test_get_num_params(self):
        """Test parameter counting."""
        model = StainNormalizationTransformer(embed_dim=128)

        num_params = model.get_num_params()

        assert num_params > 0
        assert isinstance(num_params, int)

    def test_output_shape_consistency(self):
        """Test that output shape matches input shape."""
        model = StainNormalizationTransformer(patch_size=16, embed_dim=256)

        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 256, 256)
            output = model(x)
            assert output.shape == x.shape
