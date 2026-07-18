"""Experimental, non-degenerate TransnnMIL fusion variants.

The historical ``TransnnMIL`` class remains untouched for checkpoint and result
reproducibility. These classes share its two trained branches and projections but
replace only the final branch-fusion operator.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

from src.models.transnnmil.branch_token_fusion import (
    BranchAttentionFusion,
    BranchConcatFusion,
    BranchGateFusion,
)
from src.models.transnnmil.transnnmil import TransnnMIL


class _TransnnMILFusionExperimentalBase(TransnnMIL):
    """Common execution path for experimental two-branch fusion variants."""

    fusion_name = "experimental"

    def __init__(self, *args, **kwargs) -> None:
        if kwargs.get("enable_topology", False):
            raise ValueError(
                "Experimental fusion variants currently disable enable_topology=True because "
                "the historical parent creates an unregistered random topology projection inside forward()."
            )
        super().__init__(*args, **kwargs)
        self.enable_topology = False
        self.branch_fusion = self._make_fusion()

    def _make_fusion(self):
        raise NotImplementedError

    def _prepare_branch_input(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor],
        coordinates: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.enable_hierarchical:
            return features, num_patches
        if coordinates is None:
            raise ValueError("coordinates required when enable_hierarchical=True")

        assignments = self.hierarchical_pooling(coordinates)
        projected = self.region_feature_proj(features)
        region_features = self.region_pooling(projected, assignments)
        region_centers = self.hierarchical_pooling.get_centers()
        region_tokens = self.region_transformer(region_features, region_centers=region_centers)
        region_counts = torch.full(
            (features.size(0),),
            self.num_regions,
            dtype=torch.long,
            device=features.device,
        )
        return region_tokens, region_counts

    def _fuse_projected_branches(
        self,
        projected_a: torch.Tensor,
        projected_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        branch_tokens = torch.stack([projected_a, projected_b], dim=1)
        return self.branch_fusion(branch_tokens)

    def forward(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        coordinates: Optional[torch.Tensor] = None,
        return_fusion_details: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        branch_input, branch_num_patches = self._prepare_branch_input(
            features, num_patches, coordinates
        )

        attention_a = None
        if return_attention:
            _, attention_a = self.branch_a(
                branch_input,
                branch_num_patches,
                return_attention=True,
            )

        features_a = self.branch_a.get_features(branch_input, branch_num_patches)
        features_b = self.branch_b.get_features(branch_input, branch_num_patches)
        projected_a = self.proj_a(features_a)
        projected_b = self.proj_b(features_b)
        fused, fusion_details = self._fuse_projected_branches(projected_a, projected_b)
        logits = self.fusion_classifier(fused)

        if return_attention and return_fusion_details:
            return logits, attention_a, fusion_details
        if return_attention:
            return logits, attention_a
        if return_fusion_details:
            return logits, fusion_details
        return logits

    def get_branch_outputs(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        coordinates: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_input, branch_num_patches = self._prepare_branch_input(
            features, num_patches, coordinates
        )
        logits_a = self.branch_a(branch_input, branch_num_patches)
        logits_b = self.branch_b(branch_input, branch_num_patches)
        logits_fused = self.forward(
            features,
            num_patches=num_patches,
            coordinates=coordinates,
        )
        return logits_a, logits_b, logits_fused


class TransnnMILBranchAttentionExperimental(_TransnnMILFusionExperimentalBase):
    """Two branch tokens, explicit branch identities, self-attention, learned pooling."""

    fusion_name = "branch_attention"

    def _make_fusion(self) -> BranchAttentionFusion:
        return BranchAttentionFusion(
            embed_dim=512,
            num_heads=8,
            num_branches=2,
            dropout=self.dropout,
        )


class TransnnMILConcatExperimental(_TransnnMILFusionExperimentalBase):
    """Simple concatenation-plus-projection control baseline."""

    fusion_name = "concat"

    def _make_fusion(self) -> BranchConcatFusion:
        return BranchConcatFusion(embed_dim=512, dropout=self.dropout)


class TransnnMILGateExperimental(_TransnnMILFusionExperimentalBase):
    """Sample-specific learned soft gate control baseline."""

    fusion_name = "gate"

    def _make_fusion(self) -> BranchGateFusion:
        return BranchGateFusion(embed_dim=512, dropout=self.dropout)


# Temporary compatibility alias for the first PR draft. Do not use in new configs.
TransnnMILBranchToken = TransnnMILBranchAttentionExperimental
