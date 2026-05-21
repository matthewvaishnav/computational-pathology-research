"""
Clustering-Constrained Attention Multiple Instance Learning (CLAM).

CLAM extends attention-based MIL by adding instance-level clustering to identify
distinct subgroups of patches within a slide using a two-stage approach:
1. Instance-level classifier predicts cluster assignments for each patch
2. Attention mechanism aggregates features within each cluster

Supports single-branch (one attention for all patches) or multi-branch (separate
attention for positive/negative clusters) modes.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.components.attention_mechanisms import GatedAttention
from src.models.components.fusion_strategies import EarlyFusion, LateFusion
from src.models.mil.mil_base import MILBase


class CLAM(MILBase):
    """
    Clustering-Constrained Attention Multiple Instance Learning (CLAM).

    Args:
        feature_dim: Dimension of input patch features
        hidden_dim: Dimension of hidden layers (default: 256)
        num_classes: Number of output classes (default: 2)
        num_clusters: Number of instance-level clusters (default: 10)
        dropout: Dropout rate (default: 0.1)
        multi_branch: If True, use separate attention for pos/neg clusters (default: True)
        instance_loss_weight: Weight for instance-level loss (default: 0.3)
        multi_scale: If True, support multi-scale features (default: False)
        num_scales: Number of scales for multi-scale features (default: 1)
        fusion_strategy: 'early' or 'late' fusion for multi-scale (default: 'early')
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_clusters: int = 10,
        dropout: float = 0.1,
        multi_branch: bool = True,
        instance_loss_weight: float = 0.3,
        multi_scale: bool = False,
        num_scales: int = 1,
        fusion_strategy: str = "early",
    ):
        # Create attention mechanism
        attention = GatedAttention(feature_dim=hidden_dim, hidden_dim=hidden_dim)

        # Create fusion strategy if multi-scale
        fusion = None
        if multi_scale and num_scales > 1:
            if fusion_strategy == "early":
                fusion = EarlyFusion(
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    num_scales=num_scales,
                    dropout=dropout,
                )
            elif fusion_strategy == "late":
                fusion = LateFusion(
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    num_scales=num_scales,
                    dropout=dropout,
                )
            else:
                raise ValueError(
                    f"fusion_strategy must be 'early' or 'late', got {fusion_strategy}"
                )

        super().__init__(feature_dim, num_classes, attention, fusion)

        if num_clusters < 2:
            raise ValueError(f"num_clusters must be >= 2, got {num_clusters}")

        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.multi_branch = multi_branch
        self.instance_loss_weight = instance_loss_weight
        self.multi_scale = multi_scale
        self.num_scales = num_scales
        self.fusion_strategy = fusion_strategy

        # Feature projection layer(s)
        if multi_scale and num_scales > 1:
            self.feature_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
                    )
                    for _ in range(num_scales)
                ]
            )
        else:
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )

        # Instance-level classifier for clustering
        if multi_scale and num_scales > 1 and fusion_strategy == "late":
            self.instance_classifier = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, num_clusters),
                    )
                    for _ in range(num_scales)
                ]
            )
        else:
            self.instance_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_clusters),
            )

        # Attention branches
        if multi_scale and num_scales > 1 and fusion_strategy == "late":
            if multi_branch:
                self.attention_pos = nn.ModuleList(
                    [GatedAttention(hidden_dim, hidden_dim) for _ in range(num_scales)]
                )
                self.attention_neg = nn.ModuleList(
                    [GatedAttention(hidden_dim, hidden_dim) for _ in range(num_scales)]
                )
            else:
                self.attention_net = nn.ModuleList(
                    [GatedAttention(hidden_dim, hidden_dim) for _ in range(num_scales)]
                )
        else:
            if multi_branch:
                self.attention_pos = GatedAttention(hidden_dim, hidden_dim)
                self.attention_neg = GatedAttention(hidden_dim, hidden_dim)
            else:
                self.attention_net = GatedAttention(hidden_dim, hidden_dim)

        # Bag-level classifier
        if multi_scale and fusion_strategy == "late":
            bag_input_dim = hidden_dim * 2 * num_scales if multi_branch else hidden_dim * num_scales
        else:
            bag_input_dim = hidden_dim * 2 if multi_branch else hidden_dim

        self.bag_classifier = nn.Sequential(
            nn.Linear(bag_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _project_features(self, features: torch.Tensor, scale_idx: int = 0) -> torch.Tensor:
        """Project features using scale-specific or shared projection."""
        if isinstance(self.feature_proj, nn.ModuleList):
            return self.feature_proj[scale_idx](features)
        else:
            return self.feature_proj(features)

    def _compute_instance_predictions(self, h: torch.Tensor, scale_idx: int = 0) -> torch.Tensor:
        """Compute instance-level cluster predictions."""
        if isinstance(self.instance_classifier, nn.ModuleList):
            return self.instance_classifier[scale_idx](h)
        else:
            return self.instance_classifier(h)

    def _compute_attention_weights(
        self, h: torch.Tensor, mask: Optional[torch.Tensor], branch: str, scale_idx: int = 0
    ) -> torch.Tensor:
        """Compute attention weights for specified branch."""
        if self.multi_branch:
            if branch == "positive":
                attention_module = (
                    self.attention_pos[scale_idx]
                    if isinstance(self.attention_pos, nn.ModuleList)
                    else self.attention_pos
                )
            else:
                attention_module = (
                    self.attention_neg[scale_idx]
                    if isinstance(self.attention_neg, nn.ModuleList)
                    else self.attention_neg
                )
        else:
            attention_module = (
                self.attention_net[scale_idx]
                if isinstance(self.attention_net, nn.ModuleList)
                else self.attention_net
            )

        return attention_module(h, mask)

    def _aggregate_with_attention(
        self, h: torch.Tensor, attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate features using attention weights."""
        return torch.bmm(attention_weights.unsqueeze(1), h).squeeze(1)

    def _process_single_scale(
        self, features: torch.Tensor, mask: Optional[torch.Tensor], scale_idx: int = 0
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        """Process single scale features."""
        h = self._project_features(features, scale_idx)
        instance_preds = self._compute_instance_predictions(h, scale_idx)

        if self.multi_branch:
            attention_pos = self._compute_attention_weights(h, mask, "positive", scale_idx)
            attention_neg = self._compute_attention_weights(h, mask, "negative", scale_idx)
            slide_repr = torch.cat(
                [
                    self._aggregate_with_attention(h, attention_pos),
                    self._aggregate_with_attention(h, attention_neg),
                ],
                dim=1,
            )
            attention_weights = {"positive": attention_pos, "negative": attention_neg}
        else:
            attention_weights = self._compute_attention_weights(h, mask, "single", scale_idx)
            slide_repr = self._aggregate_with_attention(h, attention_weights)

        return slide_repr, attention_weights, instance_preds

    def _early_fusion_forward(
        self, multi_scale_features: List[torch.Tensor], mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        """Early fusion: concatenate features from all scales before attention."""
        h = self.fusion(multi_scale_features, mask)
        instance_preds = self._compute_instance_predictions(h, 0)

        if self.multi_branch:
            attention_pos = self._compute_attention_weights(h, mask, "positive", 0)
            attention_neg = self._compute_attention_weights(h, mask, "negative", 0)
            slide_repr = torch.cat(
                [
                    self._aggregate_with_attention(h, attention_pos),
                    self._aggregate_with_attention(h, attention_neg),
                ],
                dim=1,
            )
            attention_weights = {"positive": attention_pos, "negative": attention_neg}
        else:
            attention_weights = self._compute_attention_weights(h, mask, "single", 0)
            slide_repr = self._aggregate_with_attention(h, attention_weights)

        return slide_repr, attention_weights, instance_preds

    def _late_fusion_forward(
        self, multi_scale_features: List[torch.Tensor], mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        """Late fusion: separate attention per scale, then combine outputs."""
        scale_representations, scale_attention_pos, scale_attention_neg = [], [], []
        scale_attention_single, scale_instance_preds = [], []

        for scale_idx, scale_features in enumerate(multi_scale_features):
            if scale_features is None:
                batch_size = next((f.size(0) for f in multi_scale_features if f is not None), 1)
                device = next(self.parameters()).device
                dim = self.hidden_dim * 2 if self.multi_branch else self.hidden_dim
                scale_representations.append(torch.zeros(batch_size, dim, device=device))
                continue

            slide_repr, attention_weights, instance_preds = self._process_single_scale(
                scale_features, mask, scale_idx
            )
            scale_representations.append(slide_repr)
            scale_instance_preds.append(instance_preds)

            if self.multi_branch:
                scale_attention_pos.append(attention_weights["positive"])
                scale_attention_neg.append(attention_weights["negative"])
            else:
                scale_attention_single.append(attention_weights)

        slide_repr = torch.cat(scale_representations, dim=-1)

        if self.multi_branch:
            attention_weights = {
                "positive": torch.stack(scale_attention_pos, dim=0).mean(dim=0),
                "negative": torch.stack(scale_attention_neg, dim=0).mean(dim=0),
            }
        else:
            attention_weights = torch.stack(scale_attention_single, dim=0).mean(dim=0)

        instance_preds = torch.stack(scale_instance_preds, dim=0).mean(dim=0)
        return slide_repr, attention_weights, instance_preds

    def forward(
        self,
        features: Union[torch.Tensor, List[torch.Tensor]],
        num_patches: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
    ]:
        """Forward pass through CLAM model."""
        if isinstance(features, list):
            if not self.multi_scale:
                raise ValueError(
                    "Model was not initialized with multi_scale=True but received list of features"
                )

            first_scale = next((f for f in features if f is not None), None)
            if first_scale is None:
                raise ValueError("All scales are None in multi-scale input")

            batch_size, max_patches, _ = first_scale.shape
            mask = None
            if num_patches is not None:
                mask = torch.arange(max_patches, device=first_scale.device).unsqueeze(
                    0
                ) < num_patches.unsqueeze(1)

            if self.fusion_strategy == "early":
                slide_repr, attention_weights, instance_preds = self._early_fusion_forward(
                    features, mask
                )
            elif self.fusion_strategy == "late":
                slide_repr, attention_weights, instance_preds = self._late_fusion_forward(
                    features, mask
                )
            else:
                raise ValueError(f"Unknown fusion_strategy: {self.fusion_strategy}")
        else:
            batch_size, max_patches, _ = features.shape
            mask = None
            if num_patches is not None:
                mask = torch.arange(max_patches, device=features.device).unsqueeze(
                    0
                ) < num_patches.unsqueeze(1)

            slide_repr, attention_weights, instance_preds = self._process_single_scale(
                features, mask, 0
            )

        logits = self.bag_classifier(slide_repr)
        return (logits, attention_weights, instance_preds) if return_attention else logits
