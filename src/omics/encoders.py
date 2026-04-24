"""
Modality-specific encoders for genomics, proteomics, and imaging data.

Each encoder maps raw high-dimensional modality data into a compact embedding
suitable for fusion or factorization.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class OmicsEncoder(nn.Module):
    """
    Generic encoder for tabular omics data (RNAseq, CNV, methylation, proteomics).

    Architecture: FC → BN → ReLU stack with skip connection at each block.
    Input assumed to be pre-normalised (log1p + standard scale recommended).
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.3,
        noise_std: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.input_noise_std = noise_std
        dims = [input_dim] + hidden_dims + [embed_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)

        # Skip projection if input_dim != embed_dim
        self.skip = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, input_dim) omics features, NaN imputed to 0 upstream

        Returns:
            (N, embed_dim)
        """
        if self.training and self.input_noise_std > 0:
            x = x + torch.randn_like(x) * self.input_noise_std
        return self.norm(self.encoder(x) + self.skip(x))


class ImageOmicsEncoder(nn.Module):
    """
    Bridges image (WSI) embeddings with omics embeddings via cross-modal projection.

    Projects each modality into a shared space and optionally concatenates
    them for downstream tasks. Dropout on image branch simulates missing imaging.
    """

    def __init__(
        self,
        image_dim: int,
        omics_dims: dict,   # name → dim
        shared_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.omics_projs = nn.ModuleDict({
            name: OmicsEncoder(dim, shared_dim, dropout=dropout)
            for name, dim in omics_dims.items()
        })
        self.shared_dim = shared_dim

    def forward(
        self,
        image_emb: torch.Tensor,
        omics: dict,
    ) -> dict:
        """
        Args:
            image_emb: (N, image_dim)
            omics: name → (N, dim) tensors

        Returns:
            dict with 'image', per-omics embeddings, all (N, shared_dim)
        """
        out = {"image": self.image_proj(image_emb)}
        for name, x in omics.items():
            if name in self.omics_projs:
                out[name] = self.omics_projs[name](torch.nan_to_num(x, nan=0.0))
        return out
