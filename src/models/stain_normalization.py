"""
Stain Normalization Transformer for computational pathology.

This module implements a transformer-based architecture for normalizing color variations
across different staining protocols while preserving tissue morphology.
"""

from typing import Optional

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts image patches into embeddings for transformer processing.

    Args:
        patch_size: Size of each patch (default: 16)
        in_channels: Number of input channels (default: 3 for RGB)
        embed_dim: Dimension of patch embeddings (default: 256)
    """

    def __init__(self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [batch_size, channels, height, width]

        Returns:
            Patch embeddings [batch_size, num_patches, embed_dim]
        """
        # Project patches to embeddings
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]

        # Flatten spatial dimensions
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Normalize
        x = self.norm(x)

        return x


class ColorFeatureEncoder(nn.Module):
    """
    Transformer encoder for extracting color features from image patches.

    Args:
        embed_dim: Dimension of embeddings (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 4)
        mlp_ratio: Ratio of MLP hidden dim to embedding dim (default: 4.0)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch embeddings [batch_size, num_patches, embed_dim]

        Returns:
            Encoded features [batch_size, num_patches, embed_dim]
        """
        return self.transformer(x)


class StyleConditioner(nn.Module):
    """
    Conditions the normalization on a reference style.

    Args:
        embed_dim: Dimension of embeddings (default: 256)
        style_dim: Dimension of style representation (default: 128)
    """

    def __init__(self, embed_dim: int = 256, style_dim: int = 128):
        super().__init__()

        # Style encoder: processes reference style image
        self.style_encoder = nn.Sequential(
            nn.Linear(embed_dim, style_dim), nn.GELU(), nn.LayerNorm(style_dim)
        )

        # Adaptive instance normalization parameters
        self.gamma_net = nn.Linear(style_dim, embed_dim)
        self.beta_net = nn.Linear(style_dim, embed_dim)

    def forward(
        self, content_features: torch.Tensor, style_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply style conditioning to content features.

        Args:
            content_features: Features to be styled [batch_size, num_patches, embed_dim]
            style_features: Reference style features [batch_size, num_patches, embed_dim]
                           If None, returns content_features unchanged

        Returns:
            Style-conditioned features [batch_size, num_patches, embed_dim]
        """
        if style_features is None:
            return content_features

        # Encode style: global average pooling over patches
        style_code = style_features.mean(dim=1)  # [B, embed_dim]
        style_code = self.style_encoder(style_code)  # [B, style_dim]

        # Generate adaptive parameters
        gamma = self.gamma_net(style_code).unsqueeze(1)  # [B, 1, embed_dim]
        beta = self.beta_net(style_code).unsqueeze(1)  # [B, 1, embed_dim]

        # Normalize content features over sequence dimension (AdaIN)
        content_mean = content_features.mean(dim=1, keepdim=True)
        content_std = content_features.std(dim=1, keepdim=True) + 1e-6
        normalized = (content_features - content_mean) / content_std

        # Apply style transformation
        styled = gamma * normalized + beta

        return styled


class StyleTransferDecoder(nn.Module):
    """
    Decoder that reconstructs normalized image from styled features.

    Args:
        embed_dim: Dimension of embeddings (default: 256)
        patch_size: Size of each patch (default: 16)
        out_channels: Number of output channels (default: 3 for RGB)
        num_layers: Number of decoder layers (default: 4)
        dropout: Dropout rate for transformer layers (default: 0.1)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        patch_size: int = 16,
        out_channels: int = 3,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Projection to pixel space
        self.to_pixels = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, patch_size * patch_size * out_channels),
        )

    def forward(
        self, features: torch.Tensor, memory: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        Decode features to normalized image.

        Args:
            features: Styled features [batch_size, num_patches, embed_dim]
            memory: Encoder output for cross-attention [batch_size, num_patches, embed_dim]
            height: Original image height
            width: Original image width

        Returns:
            Reconstructed image [batch_size, out_channels, height, width]
        """
        # Transformer decoding with cross-attention to encoder features
        decoded = self.transformer_decoder(features, memory)  # [B, num_patches, embed_dim]

        # Project to pixel space
        pixels = self.to_pixels(decoded)  # [B, num_patches, patch_size^2 * channels]

        # Reshape to image
        B, num_patches, _ = pixels.shape
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size
        if patch_h * self.patch_size != height or patch_w * self.patch_size != width:
            raise ValueError(
                f"Patch size mismatch: patch_h={patch_h}, patch_w={patch_w}, "
                f"height={height}, width={width}"
            )

        # Reshape: [B, num_patches, patch_size^2 * C] -> [B, C, H, W]
        pixels = pixels.reshape(B, patch_h, patch_w, self.patch_size, self.patch_size, -1)
        pixels = pixels.permute(
            0, 5, 1, 3, 2, 4
        )  # [B, C, patch_h, patch_size, patch_w, patch_size]
        pixels = pixels.reshape(B, -1, height, width)

        return pixels


class StainNormalizationTransformer(nn.Module):
    """
    Transformer-based stain normalization for whole-slide images.

    Normalizes color variations across different staining protocols while preserving
    tissue morphology. Supports optional reference style conditioning.

    Args:
        patch_size: Size of image patches (default: 16)
        in_channels: Number of input channels (default: 3 for RGB)
        embed_dim: Dimension of patch embeddings (default: 256)
        num_encoder_layers: Number of encoder transformer layers (default: 4)
        num_decoder_layers: Number of decoder transformer layers (default: 4)
        num_heads: Number of attention heads (default: 8)
        mlp_ratio: Ratio of MLP hidden dim to embedding dim (default: 4.0)
        dropout: Dropout rate (default: 0.1)
        style_dim: Dimension of style representation (default: 128)

    Example:
        >>> model = StainNormalizationTransformer()
        >>> input_image = torch.randn(2, 3, 256, 256)
        >>> reference_style = torch.randn(2, 3, 256, 256)
        >>> normalized = model(input_image, reference_style)
        >>> normalized.shape
        torch.Size([2, 3, 256, 256])
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        style_dim: int = 128,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)

        # Color feature encoder
        self.encoder = ColorFeatureEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Style conditioner
        self.style_conditioner = StyleConditioner(embed_dim, style_dim)

        # Style transfer decoder
        self.decoder = StyleTransferDecoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            out_channels=in_channels,
            num_layers=num_decoder_layers,
        )

    def forward(
        self, x: torch.Tensor, reference_style: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Normalize staining of input image.

        Args:
            x: Input image [batch_size, channels, height, width]
            reference_style: Optional reference style image [batch_size, channels, height, width]
                           If provided, normalizes to match this style

        Returns:
            Normalized image [batch_size, channels, height, width]
        """
        B, C, H, W = x.shape

        # Validate dimensions
        if not (():
            raise AssertionError("(")
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), f"Image dimensions ({H}, {W}) must be divisible by patch_size ({self.patch_size})"

        # Embed patches
        patches = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Encode color features
        encoded = self.encoder(patches)  # [B, num_patches, embed_dim]

        # Apply style conditioning if reference provided
        if reference_style is not None:
            # Encode reference style
            style_patches = self.patch_embed(reference_style)
            style_encoded = self.encoder(style_patches)

            # Condition on reference style
            styled = self.style_conditioner(encoded, style_encoded)
        else:
            # No style conditioning
            styled = self.style_conditioner(encoded, None)

        # Decode to normalized image
        normalized = self.decoder(styled, encoded, H, W)

        # Apply tanh to bound output to [-1, 1] range
        normalized = torch.tanh(normalized)

        return normalized

    def get_num_params(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
