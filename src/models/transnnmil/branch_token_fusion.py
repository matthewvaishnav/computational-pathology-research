"""Experimental branch-level fusion operators for TransnnMIL.

These modules accept an explicit branch-token tensor with shape ``[B, M, D]``.
They do not imitate ``nn.MultiheadAttention`` and do not label pooling weights as
self-attention weights.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class BranchAttentionFusion(nn.Module):
    """Fuse branch embeddings with branch-aware self-attention.

    Learned branch-type embeddings preserve the semantic distinction between
    TransMIL and nnMIL tokens. The module returns the fused vector and a details
    dictionary containing separately named pooling and self-attention weights.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_branches: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError(
                f"num_heads must divide embed_dim; got embed_dim={embed_dim}, num_heads={num_heads}"
            )
        if num_branches < 2:
            raise ValueError(f"num_branches must be at least 2, got {num_branches}")

        self.embed_dim = embed_dim
        self.num_branches = num_branches
        self.branch_type_embeddings = nn.Parameter(torch.zeros(num_branches, embed_dim))
        nn.init.normal_(self.branch_type_embeddings, std=0.02)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pool_score = nn.Linear(embed_dim, 1, bias=False)

    def forward(
        self,
        branch_tokens: torch.Tensor,
        branch_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if branch_tokens.ndim != 3:
            raise ValueError(
                f"branch_tokens must have shape [B, M, D], got {tuple(branch_tokens.shape)}"
            )
        batch_size, num_branches, channels = branch_tokens.shape
        if channels != self.embed_dim:
            raise ValueError(f"expected channel dimension {self.embed_dim}, got {channels}")
        if num_branches > self.num_branches:
            raise ValueError(
                f"received {num_branches} branch tokens but configured for {self.num_branches}"
            )
        if branch_mask is not None:
            if branch_mask.shape != (batch_size, num_branches):
                raise ValueError(
                    f"branch_mask must have shape {(batch_size, num_branches)}, got {tuple(branch_mask.shape)}"
                )
            branch_mask = branch_mask.bool()
            if (~branch_mask).all(dim=1).any():
                raise ValueError("each sample must retain at least one branch token")

        tokens = branch_tokens + self.branch_type_embeddings[:num_branches].unsqueeze(0)
        key_padding_mask = None if branch_mask is None else ~branch_mask
        attended, self_attention_weights = self.self_attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        attended = self.norm(tokens + attended)

        pool_logits = self.pool_score(attended).squeeze(-1)
        if branch_mask is not None:
            pool_logits = pool_logits.masked_fill(~branch_mask, torch.finfo(pool_logits.dtype).min)
        branch_pool_weights = torch.softmax(pool_logits, dim=1)
        fused = torch.sum(attended * branch_pool_weights.unsqueeze(-1), dim=1)
        return fused, {
            "branch_pool_weights": branch_pool_weights,
            "self_attention_weights": self_attention_weights,
        }


class BranchConcatFusion(nn.Module):
    """Concatenate two branch vectors and project back to the shared dimension."""

    def __init__(self, embed_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, branch_tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if branch_tokens.ndim != 3 or branch_tokens.shape[1] != 2:
            raise ValueError("BranchConcatFusion requires branch_tokens with shape [B, 2, D]")
        fused = self.projection(branch_tokens.flatten(start_dim=1))
        return fused, {}


class BranchGateFusion(nn.Module):
    """Learn sample-specific normalized weights over two branch vectors."""

    def __init__(self, embed_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2),
        )

    def forward(self, branch_tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if branch_tokens.ndim != 3 or branch_tokens.shape[1] != 2:
            raise ValueError("BranchGateFusion requires branch_tokens with shape [B, 2, D]")
        branch_pool_weights = torch.softmax(self.gate(branch_tokens.flatten(start_dim=1)), dim=1)
        fused = torch.sum(branch_tokens * branch_pool_weights.unsqueeze(-1), dim=1)
        return fused, {"branch_pool_weights": branch_pool_weights}


# Backward import alias for the initial draft only. New code should use the honest name.
BranchTokenFusion = BranchAttentionFusion
