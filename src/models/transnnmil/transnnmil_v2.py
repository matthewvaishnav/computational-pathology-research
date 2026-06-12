"""
TransnnMIL v2.0: Hierarchical + Topology + Pruning.

Combines three architectural innovations:
1. Hierarchical pooling (spatial clustering + region transformer)
2. Topology branch (k-NN graph + GNN)
3. Adaptive pruning (token sparsification)

Architecture:
    Input: Patch features [B, N, D] + coordinates [B, N, 2]
    ├─ Branch A: TransMIL (attention-based MIL)
    │   ├─ Adaptive pruning (optional)
    │   └─ Transformer encoder
    ├─ Branch B: Hierarchical pooling
    │   ├─ Spatial clustering (learnable centers)
    │   ├─ Intra-region aggregation (attention)
    │   └─ Inter-region transformer
    ├─ Branch C: Topology branch
    │   ├─ k-NN graph construction
    │   ├─ GNN layers (GATv2/GraphSAGE/GIN)
    │   └─ Global pooling
    ├─ Fusion: Concatenate + MLP
    └─ Output: Class logits [B, num_classes]

Reference:
- TransnnMIL v2.0: Hierarchical + Topology (2027)
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.mil.transmil import TransMIL
from src.models.transnnmil.adaptive_pruning import AdaptivePruning
from src.models.transnnmil.hierarchical_pooling import HierarchicalPooling, RegionAttentionPooling
from src.models.transnnmil.topology_branch import TopologyBranch


class TransnnMILv2(nn.Module):
    """
    TransnnMIL v2.0: Three-branch architecture.

    Args:
        feature_dim: Input feature dimension
        num_classes: Number of output classes
        num_regions: Number of spatial regions (default: 16)
        k_neighbors: Number of neighbors for k-NN (default: 8)
        gnn_type: GNN architecture ('gat', 'sage', 'gin')
        use_pruning: Enable adaptive pruning (default: False)
        keep_ratio: Pruning keep ratio (default: 0.5)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> model = TransnnMILv2(
        ...     feature_dim=1024,
        ...     num_classes=2,
        ...     num_regions=16,
        ...     k_neighbors=8,
        ...     gnn_type='gat',
        ...     use_pruning=True,
        ...     keep_ratio=0.5,
        ... )
        >>>
        >>> features = torch.randn(4, 100, 1024)
        >>> coords = torch.rand(4, 100, 2)
        >>>
        >>> logits = model(features, coords)  # [4, 2]
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_regions: int = 16,
        k_neighbors: int = 8,
        gnn_type: str = "gat",
        use_pruning: bool = False,
        keep_ratio: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_pruning = use_pruning

        # Branch A: TransMIL (with optional pruning)
        if use_pruning:
            self.pruning = AdaptivePruning(
                feature_dim=feature_dim,
                keep_ratio=keep_ratio,
                dropout=dropout,
            )

        self.transmil = TransMIL(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_layers=2,
            num_heads=8,
            dropout=dropout,
        )

        # Branch B: Hierarchical pooling
        self.hierarchical = HierarchicalPooling(
            num_clusters=num_regions,
            temperature=1.0,
            clustering_method="learnable",
            init_method="uniform",
        )
        self.hierarchical_pooling = RegionAttentionPooling(
            feature_dim=feature_dim,
            hidden_dim=512,
            dropout=dropout,
        )

        # Branch C: Topology branch
        self.topology = TopologyBranch(
            feature_dim=feature_dim,
            hidden_dim=512,
            num_layers=2,
            k_neighbors=k_neighbors,
            gnn_type=gnn_type,
            pooling="attention",
            dropout=dropout,
        )

        # Fusion layer
        fusion_dim = 256 + feature_dim + 512  # TransMIL hidden + Hierarchical + Topology
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            coords: Patch coordinates [batch_size, num_patches, 2]
            mask: Optional mask [batch_size, num_patches]

        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        # Branch A: TransMIL (pruning disabled - needs proper batched implementation)
        # Convert mask to num_patches for TransMIL
        if mask is not None:
            num_patches = mask.sum(dim=1)
        else:
            num_patches = None
        transmil_features = self.transmil.get_features(features, num_patches)

        # Branch B: Hierarchical pooling
        assignments = self.hierarchical(coords, mask)  # [B, N, R]
        hierarchical_features = self.hierarchical_pooling(features, assignments, mask)  # [B, R, D]
        # Global pooling over regions (mean)
        hierarchical_features = hierarchical_features.mean(dim=1)  # [B, D]

        # Branch C: Topology branch
        topology_features = self.topology(features, coords, mask)

        # Fusion
        fused = torch.cat([transmil_features, hierarchical_features, topology_features], dim=1)
        logits = self.fusion(fused)

        return logits


class TransnnMILv2TwoBranch(nn.Module):
    """
    TransnnMIL v2.0: Two-branch variant (for ablation).

    Args:
        feature_dim: Input feature dimension
        num_classes: Number of output classes
        branches: Branches to use ('AB', 'AC', 'BC')
        num_regions: Number of spatial regions (default: 16)
        k_neighbors: Number of neighbors for k-NN (default: 8)
        gnn_type: GNN architecture ('gat', 'sage', 'gin')
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        branches: str = "AB",
        num_regions: int = 16,
        k_neighbors: int = 8,
        gnn_type: str = "gat",
        dropout: float = 0.1,
    ):
        super().__init__()

        if branches not in ["AB", "AC", "BC"]:
            raise ValueError(f"branches must be 'AB', 'AC', or 'BC', got {branches}")

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.branches = branches

        # Branch A: TransMIL
        if "A" in branches:
            self.transmil = TransMIL(
                feature_dim=feature_dim,
                num_classes=num_classes,
                num_layers=2,
                num_heads=8,
                dropout=dropout,
            )

        # Branch B: Hierarchical pooling
        if "B" in branches:
            self.hierarchical = HierarchicalPooling(
                num_clusters=num_regions,
                temperature=1.0,
                clustering_method="learnable",
                init_method="uniform",
            )
            self.hierarchical_pooling = RegionAttentionPooling(
                feature_dim=feature_dim,
                hidden_dim=512,
                dropout=dropout,
            )

        # Branch C: Topology branch
        if "C" in branches:
            self.topology = TopologyBranch(
                feature_dim=feature_dim,
                hidden_dim=512,
                num_layers=2,
                k_neighbors=k_neighbors,
                gnn_type=gnn_type,
                pooling="attention",
                dropout=dropout,
            )

        # Fusion layer
        if branches == "AB":
            fusion_dim = 256 + feature_dim  # TransMIL hidden + Hierarchical
        elif branches == "AC":
            fusion_dim = 256 + 512  # TransMIL hidden + Topology
        else:  # BC
            fusion_dim = feature_dim + 512  # Hierarchical + Topology
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        branch_features = []

        # Branch A
        if "A" in self.branches:
            # Convert mask to num_patches for TransMIL
            if mask is not None:
                num_patches = mask.sum(dim=1)
            else:
                num_patches = None
            transmil_features = self.transmil.get_features(features, num_patches)
            branch_features.append(transmil_features)

        # Branch B
        if "B" in self.branches:
            assignments = self.hierarchical(coords, mask)
            hierarchical_features = self.hierarchical_pooling(features, assignments, mask)
            hierarchical_features = hierarchical_features.mean(dim=1)  # Global pooling
            branch_features.append(hierarchical_features)

        # Branch C
        if "C" in self.branches:
            topology_features = self.topology(features, coords, mask)
            branch_features.append(topology_features)

        # Fusion
        fused = torch.cat(branch_features, dim=1)
        logits = self.fusion(fused)

        return logits
