"""Feature projector for foundation model adaptation.

Maps variable foundation model dimensions (512-1024) to HistoCore's
standard 256-dim space, keeping downstream MIL code unchanged.
"""

import torch
import torch.nn as nn


class FeatureProjector(nn.Module):
    """Trainable adapter: foundation model dims → 256-dim HistoCore space.

    This is the only component that trains in Phase 1 (frozen encoder experiments).
    The two-layer MLP with residual-style normalization avoids representation
    collapse when training on small downstream datasets.

    Architecture: Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm

    Args:
        input_dim: Foundation model feature dimension (512, 768, or 1024)
        output_dim: Target dimension for downstream MIL (default: 256)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> projector = FeatureProjector(input_dim=768, output_dim=256)
        >>> raw_features = phikon_encoder(patches)  # [B, 768]
        >>> features = projector(raw_features)       # [B, 256]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to target dimension.

        Args:
            x: [B, input_dim] foundation model features

        Returns:
            [B, output_dim] projected features
        """
        return self.proj(x)

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
