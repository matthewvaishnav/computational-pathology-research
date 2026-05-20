"""
Cross-scale and hierarchical attention for multi-magnification feature fusion.

CrossScaleAttention: patches at one magnification attend to patches at another.
HierarchicalAttentionPool: aggregates across all scales → slide-level vector.
"""

import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossScaleAttention(nn.Module):
    """
    Transformer cross-attention: query from scale A, key/value from scale B.

    Allows high-magnification patches to incorporate low-magnification context
    and vice versa. Both directions are useful:
      - Low→High: global context guides local classification
      - High→Low: cellular detail enriches region-level features
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, Nq, D) — patches at query scale
            context: (B, Nk, D) — patches at context scale
            query_mask: (B, Nq) bool mask (True = valid)

        Returns:
            (B, Nq, D) updated query features
        """
        B, Nq, D = query.shape
        Nk = context.size(1)

        q = (
            self.q_proj(self.norm_q(query))
            .view(B, Nq, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(self.norm_kv(context))
            .view(B, Nk, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = self.v_proj(context).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, Nq, D)
        return query + self.out_proj(out)  # residual


class HierarchicalAttentionPool(nn.Module):
    """
    Two-stage hierarchical pooling:
      1. Within each scale: attention-weighted pool → scale token
      2. Across scales: attention-weighted pool of scale tokens → slide vector

    This mirrors the tumour hierarchy: cells → regions → tissue → slide.
    """

    def __init__(
        self,
        dim: int,
        num_scales: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Within-scale attention
        self.patch_attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )
        # Across-scale attention
        self.scale_attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )
        self.scale_pe = nn.Parameter(torch.randn(1, num_scales, dim) * 0.02)

    def forward(self, scale_features: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            scale_features: dict magnification → (B, N_mag, D)

        Returns:
            (B, D) slide-level embedding
        """
        scale_tokens = []
        for mag in sorted(scale_features.keys()):
            x = scale_features[mag]  # (B, N, D)
            w = self.patch_attn(x)  # (B, N, 1)
            w = torch.softmax(w, dim=1)
            token = (w * x).sum(dim=1)  # (B, D)
            scale_tokens.append(token)

        tokens = torch.stack(scale_tokens, dim=1)  # (B, S, D)
        tokens = tokens + self.scale_pe[:, : tokens.size(1), :]

        w = self.scale_attn(tokens)  # (B, S, 1)
        w = torch.softmax(w, dim=1)
        return (w * tokens).sum(dim=1)  # (B, D)
