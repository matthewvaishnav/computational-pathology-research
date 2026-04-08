"""
Modality-specific encoders for multimodal fusion architecture.

This module implements encoders for WSI features, genomic data, and clinical text
that transform each modality into a common embedding space for fusion.
"""

from typing import List, Optional

import torch
import torch.nn as nn


class WSIEncoder(nn.Module):
    """
    Encoder for whole-slide image (WSI) patch features using attention-based aggregation.

    Takes variable-length sequences of patch features and produces a fixed-size
    embedding using multi-head self-attention for patch aggregation.

    Args:
        input_dim: Dimension of input patch features (default: 1024)
        hidden_dim: Dimension of hidden representations (default: 512)
        output_dim: Dimension of output embedding (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        pooling: Pooling strategy for aggregation ('attention', 'mean', 'max') (default: 'attention')

    Example:
        >>> encoder = WSIEncoder(input_dim=1024, output_dim=256)
        >>> patches = torch.randn(16, 100, 1024)  # [batch, num_patches, feature_dim]
        >>> embedding = encoder(patches)
        >>> embedding.shape
        torch.Size([16, 256])
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = "attention",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pooling = pooling

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Transformer encoder for patch interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Attention-based pooling
        if pooling == "attention":
            self.attention_pool = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim)
        )

    def forward(
        self, patch_features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode WSI patch features to fixed-size embedding.

        Args:
            patch_features: Patch features [batch_size, num_patches, input_dim]
            mask: Optional padding mask [batch_size, num_patches] where True indicates valid patches

        Returns:
            WSI embedding [batch_size, output_dim]
        """
        # Project input features
        x = self.input_proj(patch_features)  # [B, num_patches, hidden_dim]

        # Apply transformer encoder
        # Create attention mask for transformer (inverted: True = ignore)
        all_masked = None
        if mask is not None:
            all_masked = ~mask.any(dim=1)
            safe_mask = mask
            if all_masked.any():
                # Keep one placeholder token visible so the transformer does not emit NaNs
                # for samples that have no valid patches in a mixed batch. We zero these rows
                # back out after pooling so they do not contribute signal.
                safe_mask = mask.clone()
                safe_mask[all_masked, 0] = True
            attn_mask = ~safe_mask  # Invert: True means ignore in transformer
        else:
            attn_mask = None

        x = self.transformer(x, src_key_padding_mask=attn_mask)  # [B, num_patches, hidden_dim]
        x = torch.nan_to_num(x, nan=0.0)

        # Aggregate patches
        if self.pooling == "attention":
            # Attention-weighted pooling
            attn_weights = self.attention_pool(x)  # [B, num_patches, 1]
            if mask is not None:
                # Mask out invalid patches
                attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), 0.0)
                # Renormalize
                attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-9)

            aggregated = (x * attn_weights).sum(dim=1)  # [B, hidden_dim]

        elif self.pooling == "mean":
            # Mean pooling
            if mask is not None:
                # Masked mean
                x_masked = x * mask.unsqueeze(-1)
                aggregated = x_masked.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-9)
            else:
                aggregated = x.mean(dim=1)

        elif self.pooling == "max":
            # Max pooling
            if mask is not None:
                x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                aggregated = x_masked.max(dim=1)[0]
            else:
                aggregated = x.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        if all_masked is not None and all_masked.any():
            aggregated = aggregated.masked_fill(all_masked.unsqueeze(-1), 0.0)

        # Project to output dimension
        embedding = self.output_proj(aggregated)  # [B, output_dim]

        return embedding


class GenomicEncoder(nn.Module):
    """
    Encoder for genomic features using MLP with batch normalization.

    Transforms genomic feature vectors (e.g., gene expression profiles) into
    a common embedding space through a multi-layer perceptron with batch
    normalization for stable training.

    Args:
        input_dim: Dimension of input genomic features (default: 2000)
        hidden_dims: List of hidden layer dimensions (default: [1024, 512])
        output_dim: Dimension of output embedding (default: 256)
        dropout: Dropout rate (default: 0.3)
        use_batch_norm: Whether to use batch normalization (default: True)

    Example:
        >>> encoder = GenomicEncoder(input_dim=2000, output_dim=256)
        >>> genomic_data = torch.randn(16, 2000)  # [batch, num_genes]
        >>> embedding = encoder(genomic_data)
        >>> embedding.shape
        torch.Size([16, 256])
    """

    def __init__(
        self,
        input_dim: int = 2000,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 256,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 512]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(nn.GELU())

            # Dropout
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        # Final normalization
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        else:
            layers.append(nn.LayerNorm(output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, genomic_features: torch.Tensor) -> torch.Tensor:
        """
        Encode genomic features to fixed-size embedding.

        Args:
            genomic_features: Genomic data [batch_size, input_dim]

        Returns:
            Genomic embedding [batch_size, output_dim]
        """
        embedding = self.mlp(genomic_features)
        return embedding


class ClinicalTextEncoder(nn.Module):
    """
    Encoder for clinical text using transformer-based architecture.

    Processes tokenized clinical text through embedding layer and transformer
    encoder to produce a fixed-size text representation.

    Args:
        vocab_size: Size of vocabulary (default: 30000)
        embed_dim: Dimension of token embeddings (default: 256)
        hidden_dim: Dimension of transformer hidden states (default: 512)
        output_dim: Dimension of output embedding (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 3)
        max_seq_length: Maximum sequence length (default: 512)
        dropout: Dropout rate (default: 0.1)
        pooling: Pooling strategy ('cls', 'mean', 'max') (default: 'mean')

    Example:
        >>> encoder = ClinicalTextEncoder(vocab_size=30000, output_dim=256)
        >>> token_ids = torch.randint(0, 30000, (16, 128))  # [batch, seq_len]
        >>> embedding = encoder(token_ids)
        >>> embedding.shape
        torch.Size([16, 256])
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length
        self.pooling = pooling

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        nn.init.normal_(self.positional_encoding, std=0.02)

        # Projection to hidden dimension
        self.embed_proj = nn.Linear(embed_dim, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # CLS token for classification pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.cls_token, std=0.02)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode clinical text to fixed-size embedding.

        Args:
            token_ids: Token IDs [batch_size, seq_len]
            attention_mask: Optional mask [batch_size, seq_len] where True indicates valid tokens

        Returns:
            Text embedding [batch_size, output_dim]
        """
        batch_size, seq_len = token_ids.shape

        # Create attention mask if not provided (non-zero tokens are valid)
        if attention_mask is None:
            attention_mask = token_ids != 0

        all_masked = ~attention_mask.any(dim=1)

        # Token embedding
        x = self.token_embedding(token_ids)  # [B, seq_len, embed_dim]

        # Add positional encoding
        if seq_len <= self.max_seq_length:
            x = x + self.positional_encoding[:, :seq_len, :]
        else:
            # Truncate if sequence is too long
            x = x[:, : self.max_seq_length, :]
            attention_mask = attention_mask[:, : self.max_seq_length]
            x = x + self.positional_encoding[:, : self.max_seq_length, :]

        x = self.dropout(x)

        # Project to hidden dimension
        x = self.embed_proj(x)  # [B, seq_len, hidden_dim]

        # Add CLS token if using CLS pooling
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, hidden_dim]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, seq_len+1, hidden_dim]

            # Extend attention mask for CLS token
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=token_ids.device)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        # Apply transformer encoder
        # Create attention mask for transformer (inverted: True = ignore)
        safe_attention_mask = attention_mask
        if all_masked.any():
            # Keep one placeholder token visible so mixed batches with empty text rows
            # stay numerically stable through the transformer.
            safe_attention_mask = attention_mask.clone()
            safe_attention_mask[all_masked, 0] = True

        attn_mask = ~safe_attention_mask  # Invert for transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)  # [B, seq_len, hidden_dim]
        x = torch.nan_to_num(x, nan=0.0)

        # Pool sequence to fixed-size representation
        if self.pooling == "cls":
            # Use CLS token representation
            pooled = x[:, 0, :]  # [B, hidden_dim]

        elif self.pooling == "mean":
            # Mean pooling over valid tokens
            x_masked = x * attention_mask.unsqueeze(-1)
            pooled = x_masked.sum(dim=1) / (attention_mask.sum(dim=1, keepdim=True) + 1e-9)

        elif self.pooling == "max":
            # Max pooling over valid tokens
            x_masked = x.masked_fill(~attention_mask.unsqueeze(-1), float("-inf"))
            pooled = x_masked.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        if all_masked.any():
            pooled = pooled.masked_fill(all_masked.unsqueeze(-1), 0.0)

        # Project to output dimension
        embedding = self.output_proj(pooled)  # [B, output_dim]

        return embedding
