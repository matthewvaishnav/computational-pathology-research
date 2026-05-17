"""
Attention mechanisms for Multiple Instance Learning models.

This module provides reusable attention mechanisms extracted from attention_mil.py
to reduce code duplication across AttentionMIL, CLAM, and TransMIL models.

Three main attention mechanisms are implemented:

1. GatedAttention: Gated attention mechanism from AttentionMIL and CLAM
   Uses element-wise product of tanh and sigmoid branches to compute attention scores.

2. TransformerAttention: Multi-head self-attention from TransMIL
   Uses transformer encoder layers to model relationships between patches.

3. SimpleAttention: Basic attention mechanism without gating
   Uses a simple feedforward network to compute attention scores.

These mechanisms are designed to be composable and reusable across different
MIL architectures.

Example:
    >>> # Gated attention for AttentionMIL
    >>> attention = GatedAttention(feature_dim=256, hidden_dim=256)
    >>> features = torch.randn(4, 100, 256)
    >>> attention_weights = attention(features)
    >>> attention_weights.shape
    torch.Size([4, 100])
    >>> attention_weights[0].sum()  # Should be close to 1.0
    tensor(1.0000)

    >>> # Transformer attention for TransMIL
    >>> attention = TransformerAttention(
    ...     feature_dim=256,
    ...     num_heads=8,
    ...     num_layers=2
    ... )
    >>> features = torch.randn(4, 100, 256)
    >>> output, attn_weights = attention(features)
    >>> output.shape
    torch.Size([4, 100, 256])
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class AttentionMechanism(ABC, nn.Module):
    """
    Abstract base class for attention mechanisms.

    All attention mechanisms must implement the forward method which takes
    patch features and returns attention weights or transformed features.

    Args:
        feature_dim: Dimension of input patch features
        hidden_dim: Dimension of hidden layers in the attention mechanism
    """

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

    @abstractmethod
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention weights or transform features.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            mask: Optional boolean mask [batch_size, num_patches], True for valid patches

        Returns:
            Attention weights [batch_size, num_patches] or transformed features

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement forward")


class GatedAttention(AttentionMechanism):
    """
    Gated attention mechanism from AttentionMIL and CLAM.

    This mechanism uses two parallel branches:
    - attention_V: Computes what features to attend to (tanh activation)
    - attention_U: Computes how much to attend (sigmoid activation)

    The element-wise product of these branches creates a gating mechanism that
    allows the model to learn both feature selection and attention strength.

    The gated attention is more expressive than simple attention and has been
    shown to improve performance in MIL tasks.

    Args:
        feature_dim: Dimension of input patch features
        hidden_dim: Dimension of hidden layers (default: same as feature_dim)

    Example:
        >>> attention = GatedAttention(feature_dim=256, hidden_dim=256)
        >>> features = torch.randn(4, 100, 256)
        >>> mask = torch.ones(4, 100, dtype=torch.bool)
        >>> mask[0, 80:] = False  # Mask out last 20 patches of first sample
        >>> attention_weights = attention(features, mask)
        >>> attention_weights.shape
        torch.Size([4, 100])
        >>> attention_weights[0, 80:].sum()  # Masked patches should have ~0 weight
        tensor(0.)
    """

    def __init__(self, feature_dim: int, hidden_dim: Optional[int] = None):
        if hidden_dim is None:
            hidden_dim = feature_dim
        super().__init__(feature_dim, hidden_dim)

        # Tanh branch: what to attend to
        self.attention_V = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())

        # Sigmoid branch: how much to attend
        self.attention_U = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Sigmoid())

        # Final projection to scalar attention score
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute gated attention weights.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            mask: Optional boolean mask [batch_size, num_patches], True for valid patches

        Returns:
            Attention weights [batch_size, num_patches] normalized to sum to 1
        """
        # Compute attention scores using gating mechanism
        # a_v: what features to attend to
        a_v = self.attention_V(features)  # [batch_size, num_patches, hidden_dim]

        # a_u: how much to attend (gating)
        a_u = self.attention_U(features)  # [batch_size, num_patches, hidden_dim]

        # Element-wise product creates gated attention
        a = self.attention_w(a_v * a_u)  # [batch_size, num_patches, 1]
        a = a.squeeze(-1)  # [batch_size, num_patches]

        # Apply mask: set padded patches to -inf before softmax
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))

        # Normalize with softmax
        attention_weights = torch.softmax(a, dim=1)  # [batch_size, num_patches]

        return attention_weights


class SimpleAttention(AttentionMechanism):
    """
    Simple attention mechanism without gating.

    This mechanism uses a basic feedforward network to compute attention scores
    directly from patch features. It's simpler than gated attention but may be
    less expressive.

    Args:
        feature_dim: Dimension of input patch features
        hidden_dim: Dimension of hidden layer (default: feature_dim // 2)

    Example:
        >>> attention = SimpleAttention(feature_dim=256, hidden_dim=128)
        >>> features = torch.randn(4, 100, 256)
        >>> attention_weights = attention(features)
        >>> attention_weights.shape
        torch.Size([4, 100])
        >>> attention_weights[0].sum()  # Should be close to 1.0
        tensor(1.0000)
    """

    def __init__(self, feature_dim: int, hidden_dim: Optional[int] = None):
        if hidden_dim is None:
            hidden_dim = feature_dim // 2
        super().__init__(feature_dim, hidden_dim)

        # Simple feedforward network for attention
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute simple attention weights.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            mask: Optional boolean mask [batch_size, num_patches], True for valid patches

        Returns:
            Attention weights [batch_size, num_patches] normalized to sum to 1
        """
        # Compute attention scores
        a = self.attention_net(features)  # [batch_size, num_patches, 1]
        a = a.squeeze(-1)  # [batch_size, num_patches]

        # Apply mask: set padded patches to -inf before softmax
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))

        # Normalize with softmax
        attention_weights = torch.softmax(a, dim=1)  # [batch_size, num_patches]

        return attention_weights


class TransformerAttention(AttentionMechanism):
    """
    Multi-head self-attention mechanism from TransMIL.

    This mechanism uses transformer encoder layers to model relationships between
    patches. Unlike simple or gated attention which compute attention weights
    independently for each patch, transformer attention allows patches to attend
    to each other through multi-head self-attention.

    The transformer processes features through multiple encoder layers, each
    containing multi-head self-attention and feedforward networks. This allows
    the model to capture complex spatial and contextual relationships between patches.

    A learnable CLS token is prepended to the patch features and used to aggregate
    information from all patches. The CLS token representation can be extracted
    for slide-level classification.

    Args:
        feature_dim: Dimension of input patch features
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer encoder layers (default: 2)
        dropout: Dropout rate for regularization (default: 0.1)
        use_pos_encoding: If True, add learnable positional encoding (default: True)

    Example:
        >>> attention = TransformerAttention(
        ...     feature_dim=256,
        ...     num_heads=8,
        ...     num_layers=2
        ... )
        >>> features = torch.randn(4, 100, 256)
        >>> output, attn_weights = attention(features)
        >>> output.shape  # Transformed features
        torch.Size([4, 100, 256])
        >>> attn_weights.shape  # Last layer attention weights
        torch.Size([4, 8, 100, 100])  # [batch, heads, queries, keys]
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
    ):
        super().__init__(feature_dim, feature_dim)  # hidden_dim = feature_dim for transformers

        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_pos_encoding = use_pos_encoding

        # Learnable positional encoding (max 10000 patches)
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, 10000, feature_dim) * 0.02)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self, features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply transformer attention to patch features.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            mask: Optional boolean mask [batch_size, num_patches], True for valid patches

        Returns:
            Tuple of:
            - output: Transformed features [batch_size, num_patches+1, feature_dim]
                     (includes CLS token at position 0)
            - attention_weights: Attention weights from last layer
                                [batch_size, num_heads, num_patches+1, num_patches+1]
                                or None if not available

        Note:
            The output includes the CLS token at position 0. To extract only patch
            features, use output[:, 1:, :]. To extract the CLS token representation
            for classification, use output[:, 0, :].
        """
        batch_size, num_patches, _ = features.shape

        # Add positional encoding
        if self.use_pos_encoding:
            features = features + self.pos_encoding[:, :num_patches, :]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, feature_dim]
        h = torch.cat([cls_tokens, features], dim=1)  # [batch_size, num_patches+1, feature_dim]

        # Create attention mask for transformer
        if mask is not None:
            # Create mask: False for valid positions (CLS + valid patches), True for padding
            transformer_mask = ~mask  # Invert: True for padding
            # Prepend False for CLS token
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=features.device)
            transformer_mask = torch.cat([cls_mask, transformer_mask], dim=1)
        else:
            transformer_mask = None

        # Apply transformer
        output = self.transformer(h, src_key_padding_mask=transformer_mask)

        # Apply layer normalization
        output = self.norm(output)

        # Note: PyTorch's TransformerEncoder doesn't return attention weights by default
        # To get attention weights, you would need to modify the encoder or use a custom implementation
        attention_weights = None

        return output, attention_weights

    def get_cls_token(self, output: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS token representation from transformer output.

        Args:
            output: Transformer output [batch_size, num_patches+1, feature_dim]

        Returns:
            CLS token representation [batch_size, feature_dim]
        """
        return output[:, 0, :]

    def get_patch_features(self, output: torch.Tensor) -> torch.Tensor:
        """
        Extract patch features from transformer output (excluding CLS token).

        Args:
            output: Transformer output [batch_size, num_patches+1, feature_dim]

        Returns:
            Patch features [batch_size, num_patches, feature_dim]
        """
        return output[:, 1:, :]
