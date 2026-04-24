"""
Multi-omics fusion strategies for combining encoded modality embeddings.

MultiOmicsFusion: gated attention fusion of arbitrary modality embeddings.
ModalityDropoutFusion: training-time modality dropout for robustness to missing data.
"""

import logging
import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiOmicsFusion(nn.Module):
    """
    Gated multi-head attention fusion: all modality embeddings → fused vector.

    Each modality token gets a learned gate weight (attention over modalities).
    Self-attention across modality tokens captures inter-modality interactions.

    Args:
        embed_dim: shared embedding dimension (all modalities same dim after projection)
        num_heads: attention heads for cross-modality transformer
        num_layers: number of transformer layers
        output_dim: output embedding size
        dropout: dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling over modality tokens → fused vector
        self.pool_attn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.out_proj = nn.Linear(embed_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            modality_embeddings: name → (N, embed_dim)
            modality_mask: name → (N,) bool True=observed (for padding mask)

        Returns:
            (N, output_dim) fused embedding
        """
        names = sorted(modality_embeddings.keys())
        tokens = torch.stack([modality_embeddings[n] for n in names], dim=1)  # (N, M, D)

        # Build key_padding_mask: (N, M) True = IGNORE (missing modality)
        key_mask = None
        if modality_mask is not None:
            masks = [~modality_mask.get(n, torch.ones(tokens.size(0), dtype=torch.bool, device=tokens.device)) for n in names]
            key_mask = torch.stack(masks, dim=1)  # (N, M) True = padding

        tokens = self.transformer(tokens, src_key_padding_mask=key_mask)

        attn_w = self.pool_attn(tokens)  # (N, M, 1)
        if key_mask is not None:
            attn_w = attn_w.masked_fill(key_mask.unsqueeze(-1), float("-inf"))
        attn_w = torch.softmax(attn_w, dim=1)
        fused = (attn_w * tokens).sum(dim=1)  # (N, D)

        return self.norm(self.out_proj(fused))


class ModalityDropoutFusion(nn.Module):
    """
    Wraps MultiOmicsFusion with training-time modality dropout.

    Randomly drops entire modalities during training to force the model
    to learn from any subset → robust to missing data at inference.

    Args:
        fusion: MultiOmicsFusion module
        modality_dropout_rate: probability of dropping each modality per forward pass
        min_modalities: minimum number of modalities always kept (default 1)
    """

    def __init__(
        self,
        fusion: MultiOmicsFusion,
        modality_dropout_rate: float = 0.3,
        min_modalities: int = 1,
    ):
        super().__init__()
        self.fusion = fusion
        self.drop_rate = modality_dropout_rate
        self.min_modalities = min_modalities

    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.training and self.drop_rate > 0:
            names = list(modality_embeddings.keys())
            num_keep = max(self.min_modalities, int(len(names) * (1 - self.drop_rate)))
            kept = set(random.sample(names, num_keep))
            modality_embeddings = {k: v for k, v in modality_embeddings.items() if k in kept}
            if modality_mask is not None:
                modality_mask = {k: v for k, v in modality_mask.items() if k in kept}

        return self.fusion(modality_embeddings, modality_mask)
