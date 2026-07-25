"""TransnnMIL: non-degenerate fusion of TransMIL and nnMIL.

The historical implementation used one TransMIL query token attending to exactly
one nnMIL key/value token. A one-element softmax is always one, so that fusion
was query-invariant and could not use the TransMIL branch as intended. This
implementation keeps the public API and registered module names while fusing
both branch tokens with self-attention.

Topology mode also keeps its projection as a registered module. No layer is
created inside ``forward``.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.mil.nnmil import nnMIL
from src.models.mil.transmil import TransMIL
from src.models.transnnmil.hierarchical_pooling import (
    HierarchicalPooling,
    RegionAttentionPooling,
    RegionMaxPooling,
    RegionMeanPooling,
    RegionTransformer,
)

try:
    from src.models.transnnmil.topology_branch import TopologyBranch

    TOPOLOGY_AVAILABLE = True
except ImportError:
    TOPOLOGY_AVAILABLE = False


class TransnnMIL(nn.Module):
    """Dual-branch multiple-instance learner with genuine branch fusion.

    Both branch representations are projected to 512 dimensions, stacked as
    branch tokens, processed with multi-head self-attention, and mean pooled.
    Consequently, changing either branch can change the fused prediction.

    Historical checkpoints remain loadable for the non-topology path because
    the original registered names are retained. Their predictions will differ
    from the historical defective execution path and must therefore be treated
    as new evaluations.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_pos_encoding: bool = False,
        enable_hierarchical: bool = False,
        num_regions: int = 16,
        region_hidden_dim: int = 512,
        clustering_method: str = "learnable",
        pooling_method: str = "attention",
        temperature: float = 1.0,
        enable_topology: bool = False,
        k_neighbors: int = 8,
        gnn_type: str = "gat",
        gnn_pooling: str = "attention",
    ) -> None:
        super().__init__()

        if feature_dim <= 0 or hidden_dim <= 0 or num_classes <= 0:
            raise ValueError("feature_dim, hidden_dim, and num_classes must be positive")
        if num_layers <= 0 or num_heads <= 0:
            raise ValueError("num_layers and num_heads must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if num_regions <= 0 or region_hidden_dim <= 0:
            raise ValueError("num_regions and region_hidden_dim must be positive")
        if clustering_method not in {"learnable", "kmeans", "grid"}:
            raise ValueError("unsupported clustering_method")
        if pooling_method not in {"attention", "mean", "max"}:
            raise ValueError("unsupported pooling_method")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if enable_topology and not TOPOLOGY_AVAILABLE:
            raise ImportError("TopologyBranch requires torch-geometric")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_pos_encoding = use_pos_encoding
        self.enable_hierarchical = enable_hierarchical
        self.num_regions = num_regions
        self.region_hidden_dim = region_hidden_dim
        self.clustering_method = clustering_method
        self.pooling_method = pooling_method
        self.temperature = temperature
        self.enable_topology = enable_topology
        self.k_neighbors = k_neighbors
        self.gnn_type = gnn_type
        self.gnn_pooling = gnn_pooling

        if enable_hierarchical:
            self.hierarchical_pooling = HierarchicalPooling(
                num_clusters=num_regions,
                temperature=temperature,
                init_method="uniform",
                clustering_method=clustering_method,
            )
            self.region_feature_proj = nn.Sequential(
                nn.Linear(feature_dim, region_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            if pooling_method == "attention":
                self.region_pooling = RegionAttentionPooling(
                    feature_dim=region_hidden_dim,
                    hidden_dim=128,
                    dropout=dropout,
                )
            elif pooling_method == "mean":
                self.region_pooling = RegionMeanPooling()
            else:
                self.region_pooling = RegionMaxPooling()
            self.region_transformer = RegionTransformer(
                feature_dim=region_hidden_dim,
                num_layers=2,
                num_heads=8,
                mlp_ratio=4.0,
                dropout=dropout,
                use_pos_encoding=False,
            )
            branch_input_dim = region_hidden_dim
        else:
            branch_input_dim = feature_dim

        self.branch_a = TransMIL(
            feature_dim=branch_input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding,
        )
        self.branch_b = nnMIL(
            feature_dim=branch_input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=0.25,
        )

        self.proj_a = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        branch_b_output_dim = region_hidden_dim if enable_hierarchical else feature_dim
        self.proj_b = nn.Sequential(
            nn.Linear(branch_b_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if enable_topology:
            self.topology_branch = TopologyBranch(
                feature_dim=feature_dim,
                hidden_dim=region_hidden_dim,
                num_layers=2,
                k_neighbors=k_neighbors,
                gnn_type=gnn_type,
                pooling=gnn_pooling,
                dropout=dropout,
            )
            self.proj_c = nn.Linear(region_hidden_dim, 512)
        else:
            self.topology_branch = None
            self.proj_c = None

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Retained only for checkpoint/API compatibility. It is not a scientific
        # estimate of branch importance; inspect branch-token ablations instead.
        self.gate_param = nn.Parameter(torch.zeros(1))

    def _prepare_branch_input(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor],
        coordinates: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.enable_hierarchical:
            return features, num_patches
        if coordinates is None:
            raise ValueError("coordinates required when enable_hierarchical=True")

        assignments = self.hierarchical_pooling(coordinates)
        projected = self.region_feature_proj(features)
        region_features = self.region_pooling(projected, assignments)
        region_centers = self.hierarchical_pooling.get_centers()
        region_tokens = self.region_transformer(
            region_features,
            region_centers=region_centers,
        )
        region_counts = torch.full(
            (features.size(0),),
            self.num_regions,
            dtype=torch.long,
            device=features.device,
        )
        return region_tokens, region_counts

    def _fused_features(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor],
        coordinates: Optional[torch.Tensor],
    ) -> torch.Tensor:
        branch_input, branch_counts = self._prepare_branch_input(
            features,
            num_patches,
            coordinates,
        )
        branch_a_features = self.branch_a.get_features(branch_input, branch_counts)
        branch_b_features = self.branch_b.get_features(branch_input, branch_counts)
        tokens = [self.proj_a(branch_a_features), self.proj_b(branch_b_features)]

        if self.enable_topology:
            if coordinates is None:
                raise ValueError("coordinates required when enable_topology=True")
            mask = None
            if num_patches is not None:
                mask = torch.arange(features.size(1), device=features.device).unsqueeze(0)
                mask = mask < num_patches.unsqueeze(1)
            topology_features = self.topology_branch(features, coordinates, mask)
            tokens.append(self.proj_c(topology_features))

        branch_tokens = torch.stack(tokens, dim=1)
        fused_tokens, _ = self.fusion_attention(
            branch_tokens,
            branch_tokens,
            branch_tokens,
            need_weights=False,
        )
        return fused_tokens.mean(dim=1)

    def forward(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        coordinates: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attention_a = None
        if return_attention:
            branch_input, branch_counts = self._prepare_branch_input(
                features,
                num_patches,
                coordinates,
            )
            _, attention_a = self.branch_a(
                branch_input,
                branch_counts,
                return_attention=True,
            )

        logits = self.fusion_classifier(self._fused_features(features, num_patches, coordinates))
        if return_attention:
            return logits, attention_a
        return logits

    def get_gate_value(self) -> float:
        """Return the retained compatibility parameter, not branch importance."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_param).item()

    def get_branch_outputs(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        coordinates: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_input, branch_counts = self._prepare_branch_input(
            features,
            num_patches,
            coordinates,
        )
        logits_a = self.branch_a(branch_input, branch_counts)
        logits_b = self.branch_b(branch_input, branch_counts)
        logits_fused = self.forward(
            features,
            num_patches=num_patches,
            coordinates=coordinates,
        )
        return logits_a, logits_b, logits_fused
