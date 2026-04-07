"""
Cross-modal attention fusion for multimodal integration.

This module implements attention-based fusion mechanisms that enable
interaction between different modalities (WSI, genomic, clinical text)
through multi-head cross-attention.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for modality interaction.

    Implements multi-head attention where queries come from one modality
    and keys/values come from another, enabling cross-modal information flow.
    Supports missing modalities through masking.

    Args:
        embed_dim: Dimension of input embeddings (default: 256)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
        bias: Whether to use bias in projections (default: True)

    Example:
        >>> fusion = CrossModalAttention(embed_dim=256, num_heads=8)
        >>> wsi_emb = torch.randn(16, 256)  # [batch, embed_dim]
        >>> genomic_emb = torch.randn(16, 256)
        >>> fused = fusion(wsi_emb, genomic_emb)
        >>> fused.shape
        torch.Size([16, 256])
    """

    def __init__(
        self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1, bias: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Multi-head attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply cross-modal attention.

        Args:
            query: Query embeddings [batch_size, embed_dim]
            key: Key embeddings [batch_size, embed_dim]
            value: Value embeddings [batch_size, embed_dim] (defaults to key if None)
            key_mask: Mask for valid keys [batch_size] where True indicates valid

        Returns:
            Attended output [batch_size, embed_dim]
        """
        if value is None:
            value = key

        batch_size = query.shape[0]

        # Project to multi-head space
        q = self.q_proj(query)  # [B, embed_dim]
        k = self.k_proj(key)  # [B, embed_dim]
        v = self.v_proj(value)  # [B, embed_dim]

        # Reshape for multi-head attention: [B, num_heads, head_dim]
        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)

        # Compute attention scores: [B, num_heads, 1]
        # For single embeddings, we compute similarity between q and k
        attn_scores = (q * k).sum(dim=-1, keepdim=True) / (self.head_dim**0.5)

        # Apply key mask if provided
        if key_mask is not None:
            # Expand mask for heads: [B, num_heads, 1]
            mask_expanded = key_mask.view(batch_size, 1, 1).expand(-1, self.num_heads, 1)
            attn_scores = attn_scores.masked_fill(~mask_expanded, float("-inf"))

            # Check if all keys are masked (all False) - handle per sample
            all_masked = ~key_mask  # [B]
            if all_masked.all():
                # All samples have all keys masked - return query unchanged
                return self.norm(query)
            elif all_masked.any():
                # Some samples have all keys masked - handle them separately
                # For masked samples, we'll zero out the attention contribution
                pass  # Will be handled after softmax

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_heads, 1]

        # Handle NaN from all-masked samples (softmax of all -inf)
        if key_mask is not None:
            all_masked = ~key_mask  # [B]
            if all_masked.any():
                # Replace NaN attention weights with zeros for all-masked samples
                all_masked_expanded = all_masked.view(batch_size, 1, 1).expand(
                    -1, self.num_heads, 1
                )
                attn_weights = torch.where(
                    all_masked_expanded, torch.zeros_like(attn_weights), attn_weights
                )

        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        # [B, num_heads, 1] * [B, num_heads, head_dim] -> [B, num_heads, head_dim]
        # Transpose and reshape to concatenate heads: [B, num_heads, head_dim] -> [B, embed_dim]
        attended = (attn_weights * v).transpose(1, 2).contiguous().view(batch_size, self.embed_dim)

        # Output projection
        output = self.out_proj(attended)
        output = self.out_dropout(output)

        # Residual connection and normalization
        output = self.norm(query + output)

        return output


class MultiModalFusionLayer(nn.Module):
    """
    Complete fusion layer with pairwise cross-modal attention.

    Performs all-pairs cross-modal attention between modalities and
    combines the results through concatenation and projection.

    Args:
        embed_dim: Dimension of embeddings (default: 256)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
        modalities: List of modality names (default: ['wsi', 'genomic', 'clinical'])

    Example:
        >>> fusion = MultiModalFusionLayer(embed_dim=256)
        >>> embeddings = {
        ...     'wsi': torch.randn(16, 256),
        ...     'genomic': torch.randn(16, 256),
        ...     'clinical': torch.randn(16, 256)
        ... }
        >>> fused = fusion(embeddings)
        >>> fused.shape
        torch.Size([16, 256])
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        modalities: List[str] = None,
    ):
        super().__init__()

        if modalities is None:
            modalities = ["wsi", "genomic", "clinical"]

        self.embed_dim = embed_dim
        self.modalities = modalities
        self.num_modalities = len(modalities)

        # Create cross-attention modules for each modality pair
        self.cross_attentions = nn.ModuleDict()
        for query_mod in modalities:
            for key_mod in modalities:
                if query_mod != key_mod:
                    name = f"{query_mod}_to_{key_mod}"
                    self.cross_attentions[name] = CrossModalAttention(
                        embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
                    )

        # Modality-specific projection layers
        self.modality_projections = nn.ModuleDict(
            {
                mod: nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for mod in modalities
            }
        )

        # Fusion pooling: concatenate all modalities and project
        fusion_input_dim = embed_dim * self.num_modalities
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        embeddings: Dict[str, Optional[torch.Tensor]],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Fuse multimodal embeddings through cross-modal attention.

        Args:
            embeddings: Dict mapping modality names to embeddings [batch_size, embed_dim]
                       Can contain None for missing modalities
            modality_masks: Optional dict of masks [batch_size] indicating valid samples

        Returns:
            Fused embedding [batch_size, embed_dim]
        """
        first_emb = next(emb for emb in embeddings.values() if emb is not None)
        batch_size = first_emb.shape[0]
        device = first_emb.device

        # Initialize modality masks if not provided
        if modality_masks is None:
            modality_masks = {
                mod: (
                    torch.ones(batch_size, dtype=torch.bool, device=device)
                    if embeddings.get(mod) is not None
                    else torch.zeros(batch_size, dtype=torch.bool, device=device)
                )
                for mod in self.modalities
            }

        # Replace None embeddings with zeros
        processed_embeddings = {}
        for mod in self.modalities:
            if embeddings.get(mod) is not None:
                processed_embeddings[mod] = embeddings[mod]
            else:
                processed_embeddings[mod] = torch.zeros(batch_size, self.embed_dim, device=device)

        # Apply cross-modal attention for each modality
        attended_embeddings = {}
        for query_mod in self.modalities:
            # Start with the original embedding
            attended = processed_embeddings[query_mod].clone()

            # Apply cross-attention with each other modality
            for key_mod in self.modalities:
                if query_mod != key_mod:
                    name = f"{query_mod}_to_{key_mod}"
                    # Only attend if key modality is available
                    attended = self.cross_attentions[name](
                        query=attended,
                        key=processed_embeddings[key_mod],
                        key_mask=modality_masks[key_mod],
                    )

            # Apply modality-specific projection
            attended = self.modality_projections[query_mod](attended)

            # Mask out if modality is not available
            attended = attended * modality_masks[query_mod].unsqueeze(-1).float()

            attended_embeddings[query_mod] = attended

        # Concatenate all modality embeddings
        fused_list = [attended_embeddings[mod] for mod in self.modalities]
        concatenated = torch.cat(fused_list, dim=-1)  # [B, embed_dim * num_modalities]

        # Project to final embedding
        fused = self.fusion_proj(concatenated)  # [B, embed_dim]

        return fused
