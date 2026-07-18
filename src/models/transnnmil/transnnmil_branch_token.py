"""Corrected TransnnMIL variant with non-degenerate branch-token fusion."""

from __future__ import annotations

from src.models.transnnmil.branch_token_fusion import BranchTokenFusion
from src.models.transnnmil.transnnmil import TransnnMIL


class TransnnMILBranchToken(TransnnMIL):
    """TransnnMIL with branch-token self-attention and learned pooling.

    This preserves the existing TransnnMIL branches, projections, classifier,
    hierarchical option, topology option, and public forward API. Only the
    degenerate single-key cross-attention module is replaced.

    Existing TransnnMIL checkpoints are not numerically equivalent because the
    corrected fusion introduces trainable self-attention, normalization, and
    branch-pooling parameters. This class is therefore intentionally exposed as
    a separate experimental model rather than silently changing old results.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion_attention = BranchTokenFusion(
            embed_dim=512,
            num_heads=8,
            dropout=self.dropout,
        )
