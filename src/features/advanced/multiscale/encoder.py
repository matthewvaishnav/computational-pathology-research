"""
Per-magnification patch encoders for multi-scale WSI processing.

Each magnification level gets its own projection head that maps backbone
features into a shared latent space, enabling cross-scale comparison.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

MAGNIFICATION_LEVELS = [4, 10, 20, 40]


class MagnificationEncoder(nn.Module):
    """
    Projects patch embeddings from one magnification level into shared space.

    Adds learnable magnification positional embedding before projection so
    the model knows which scale each token came from.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        magnification: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.magnification = magnification
        self.mag_embedding = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, input_dim) patch features at this magnification

        Returns:
            (B, N, output_dim) projected features with mag embedding added
        """
        return self.proj(x) + self.mag_embedding


class MultiScaleFeatureExtractor(nn.Module):
    """
    Runs independent MagnificationEncoders for each scale and returns
    a dict of scale → projected tensor.

    Input dict maps magnification int → (B, N_mag, input_dim) patch features.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        magnifications: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if magnifications is None:
            magnifications = MAGNIFICATION_LEVELS
        self.magnifications = magnifications
        self.encoders = nn.ModuleDict(
            {
                str(m): MagnificationEncoder(input_dim, output_dim, m, dropout)
                for m in magnifications
            }
        )

    def forward(self, features: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Args:
            features: dict magnification → (B, N, input_dim)

        Returns:
            dict magnification → (B, N, output_dim)
        """
        out = {}
        for mag, x in features.items():
            key = str(mag)
            if key not in self.encoders:
                raise ValueError(
                    f"No encoder for magnification {mag}. Available: {self.magnifications}"
                )
            out[mag] = self.encoders[key](x)
        return out
