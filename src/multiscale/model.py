"""
End-to-end multi-scale MIL model for slide-level classification/regression.

Architecture:
  1. MultiScaleFeatureExtractor: per-magnification projection
  2. CrossScaleAttention: bidirectional cross-scale information exchange
  3. HierarchicalAttentionPool: → slide embedding
  4. Task head: classification or survival prediction
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .encoder import MultiScaleFeatureExtractor
from .attention import CrossScaleAttention, HierarchicalAttentionPool

logger = logging.getLogger(__name__)


class MultiScaleMIL(nn.Module):
    """
    Multi-scale hierarchical MIL classifier.

    Input: dict of magnification → patch feature tensors
    Output: logits (classification) or risk score (survival)

    Args:
        input_dim: backbone feature dimension per patch
        embed_dim: shared latent dimension
        num_classes: classification output size (0 = survival regression)
        magnifications: list of magnification levels expected
        num_cross_attn_layers: cross-scale attention depth
        num_heads: attention heads
        dropout: dropout probability
    """

    def __init__(
        self,
        input_dim: int = 1024,
        embed_dim: int = 512,
        num_classes: int = 2,
        magnifications: Optional[List[int]] = None,
        num_cross_attn_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if magnifications is None:
            magnifications = [4, 10, 20, 40]
        self.magnifications = magnifications
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.extractor = MultiScaleFeatureExtractor(input_dim, embed_dim, magnifications, dropout)

        # Cross-scale attention pairs: each lower scale attends to next higher
        self.cross_attn_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CrossScaleAttention(embed_dim, num_heads, dropout)
                        for _ in range(len(magnifications) - 1)
                    ]
                )
                for _ in range(num_cross_attn_layers)
            ]
        )

        self.pool = HierarchicalAttentionPool(embed_dim, len(magnifications), dropout)

        self.norm = nn.LayerNorm(embed_dim)

        if num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            # Survival: hazard ratio (log-partial-hazard)
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1)
            )

    def forward(self, features: Dict[int, torch.Tensor]) -> dict:
        """
        Args:
            features: dict magnification → (B, N_patches, input_dim)

        Returns:
            dict with 'logits', 'embedding', optionally 'attention_weights'
        """
        scale_feats = self.extractor(features)  # dict mag → (B, N, D)

        mags_sorted = sorted(scale_feats.keys())

        for cross_layer in self.cross_attn_layers:
            updated = {}
            for i, mag in enumerate(mags_sorted):
                x = scale_feats[mag]
                for j, other_mag in enumerate(mags_sorted):
                    if other_mag == mag:
                        continue
                    # Attend to the other scale; only one cross-attn module per adjacent pair
                    pair_idx = min(i, len(mags_sorted) - 2)
                    x = cross_layer[pair_idx](x, scale_feats[other_mag])
                updated[mag] = x
            scale_feats = updated

        slide_emb = self.norm(self.pool(scale_feats))
        logits = self.head(slide_emb)

        return {"logits": logits, "embedding": slide_emb}

    def get_slide_embedding(self, features: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Convenience: returns just the (B, embed_dim) slide embedding."""
        return self.forward(features)["embedding"]
