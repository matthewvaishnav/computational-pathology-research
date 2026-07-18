"""Non-degenerate branch-token fusion for TransnnMIL.

The original TransnnMIL fusion passes one query token from TransMIL and one
key/value token from nnMIL into ``nn.MultiheadAttention``. With exactly one key,
the softmax weight is always one, so the output is independent of the query and
the TransMIL branch cannot influence the fused representation.

This module treats branch representations as a short token sequence, performs
self-attention across the branch tokens, then learns a normalized pooling weight
over the attended tokens. Both branch inputs therefore have a direct path to the
fused representation and receive task gradients.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class BranchTokenFusion(nn.Module):
    """Fuse two or more branch embeddings as tokens.

    The call signature intentionally matches ``nn.MultiheadAttention`` so this
    module can replace the existing ``fusion_attention`` member without changing
    the rest of the TransnnMIL forward path.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError(
                f"num_heads must be positive and divide embed_dim; got embed_dim={embed_dim}, "
                f"num_heads={num_heads}"
            )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pool_score = nn.Linear(embed_dim, 1, bias=False)

    @staticmethod
    def _build_tokens(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query, key, and value must have shape [batch, tokens, channels]")
        if query.shape[0] != value.shape[0] or query.shape[2] != value.shape[2]:
            raise ValueError(
                f"query/value batch and channel dimensions must match, got {query.shape} and {value.shape}"
            )

        # Existing two-branch TransnnMIL call: query=[TransMIL], value=[nnMIL].
        if query.shape[1] == 1 and value.shape[1] == 1:
            return torch.cat([query, value], dim=1)

        # Existing topology path already supplies all branch tokens as query/key/value.
        if query.shape == key.shape == value.shape:
            return query

        # General fallback: preserve all distinct query and value tokens.
        return torch.cat([query, value], dim=1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self._build_tokens(query, key, value)
        attended, _ = self.self_attention(tokens, tokens, tokens, need_weights=False)
        attended = self.norm(tokens + attended)

        pool_weights = torch.softmax(self.pool_score(attended).squeeze(-1), dim=1)
        fused = torch.sum(attended * pool_weights.unsqueeze(-1), dim=1, keepdim=True)

        # Match MultiheadAttention's averaged attention-weight shape expected by callers.
        return fused, pool_weights.unsqueeze(1)
