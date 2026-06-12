"""
TransnnMIL: Fusion of TransMIL and nnMIL

This module implements TransnnMIL, a novel dual-branch MIL architecture that combines:
- Branch A: TransMIL (transformer-based correlator with self-attention)
- Branch B: nnMIL (lightweight gated attention aggregator)

The two branches are fused using a learnable scalar gate parameter that balances
their contributions. This design leverages both the global context modeling of
transformers and the efficiency of gated attention mechanisms.

Key features:
- Dual-branch architecture with learnable fusion gate
- TransMIL branch captures long-range patch dependencies via self-attention
- nnMIL branch provides efficient attention-weighted aggregation
- Positional encoding disabled in TransMIL for random sub-bag compatibility
- Compatible with existing training infrastructure (nnMILTrainer, samplers)
- Supports uncertainty estimation via sliding window inference

Architecture:
    Input: Bag of patch embeddings [B, K, D]
    ├─ Branch A (TransMIL): Transformer → CLS token → MLP → logits_A
    ├─ Branch B (nnMIL): Gated attention → Weighted sum → MLP → logits_B
    └─ Fusion: gate * logits_A + (1 - gate) * logits_B → final logits

The gate parameter is initialized to 0.0 (sigmoid(0) = 0.5), giving equal weight
to both branches initially. During training, the model learns the optimal balance.

Reference:
- TransMIL: Shao et al., "TransMIL: Transformer based Correlated Multiple Instance Learning"
- nnMIL: Stanford/NIH (2024), "No-New-Net Multiple Instance Learning"
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.mil.nnmil import nnMIL
from src.models.mil.transmil import TransMIL
from src.models.transnnmil.hierarchical_pooling import (
    HierarchicalPooling,
    RegionAttentionPooling,
    RegionTransformer,
)

# Import topology branch if available
try:
    from src.models.transnnmil.topology_branch import TopologyBranch

    TOPOLOGY_AVAILABLE = True
except ImportError:
    TOPOLOGY_AVAILABLE = False


class TransnnMIL(nn.Module):
    """
    TransnnMIL: Dual-branch MIL with learnable fusion.

    Combines TransMIL (transformer-based) and nnMIL (gated attention) through
    a learnable scalar gate. The gate parameter controls the contribution of
    each branch to the final prediction.

    Args:
        feature_dim: Dimension of input patch features (e.g., 1024 for UNI)
        hidden_dim: Hidden dimension for both branches (default: 256)
        num_classes: Number of output classes (default: 2)
        num_layers: Number of transformer layers in Branch A (default: 2)
        num_heads: Number of attention heads in Branch A (default: 8)
        dropout: Dropout rate (default: 0.1 for TransMIL, 0.25 for nnMIL)
        use_pos_encoding: Enable positional encoding in TransMIL (default: False)
                         Set to False for random sub-bag sampling compatibility

    Example:
        >>> # Create TransnnMIL model
        >>> model = TransnnMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        >>>
        >>> # Forward pass with fixed-length bags
        >>> features = torch.randn(4, 100, 1024)  # [batch, patches, features]
        >>> num_patches = torch.tensor([100, 80, 90, 100])  # actual patch counts
        >>>
        >>> # Get predictions
        >>> logits = model(features, num_patches)
        >>> logits.shape
        torch.Size([4, 2])
        >>>
        >>> # Get predictions with attention weights
        >>> logits, attention = model(features, num_patches, return_attention=True)
        >>> attention.shape  # From Branch A (TransMIL)
        torch.Size([4, 100])
        >>>
        >>> # Check learned gate value
        >>> gate_value = torch.sigmoid(model.gate_param)
        >>> print(f"Branch A weight: {gate_value.item():.3f}")
        >>> print(f"Branch B weight: {1 - gate_value.item():.3f}")

    Notes:
        - The gate parameter is initialized to 0.0, giving equal weight (0.5) to both branches
        - During training, the gate learns to balance the branches based on the task
        - For uncertainty estimation, run the model on multiple sliding windows and
          compute variance across predictions
        - Both branches process the same input bag in parallel for efficiency
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
    ):
        super().__init__()

        # Validate inputs
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if num_regions <= 0:
            raise ValueError(f"num_regions must be positive, got {num_regions}")
        if region_hidden_dim <= 0:
            raise ValueError(f"region_hidden_dim must be positive, got {region_hidden_dim}")
        if clustering_method not in ["learnable", "kmeans", "grid"]:
            raise ValueError(
                f"clustering_method must be 'learnable', 'kmeans', or 'grid', got {clustering_method}"
            )
        if pooling_method not in ["attention", "mean", "max"]:
            raise ValueError(
                f"pooling_method must be 'attention', 'mean', or 'max', got {pooling_method}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

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

        # Topology branch (optional)
        if enable_topology:
            if not TOPOLOGY_AVAILABLE:
                raise ImportError(
                    "TopologyBranch requires torch_geometric. Install with: pip install torch-geometric"
                )

            self.topology_branch = TopologyBranch(
                feature_dim=feature_dim,
                hidden_dim=region_hidden_dim,
                num_layers=2,
                k_neighbors=k_neighbors,
                gnn_type=gnn_type,
                pooling=gnn_pooling,
                dropout=dropout,
            )

        # Hierarchical pooling module (optional)
        if enable_hierarchical:
            # Spatial clustering
            self.hierarchical_pooling = HierarchicalPooling(
                num_clusters=num_regions,
                temperature=temperature,
                init_method="uniform",
                clustering_method=clustering_method,
            )

            # Feature projection for regions
            self.region_feature_proj = nn.Sequential(
                nn.Linear(feature_dim, region_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Intra-region pooling
            if pooling_method == "attention":
                self.region_pooling = RegionAttentionPooling(
                    feature_dim=region_hidden_dim,
                    hidden_dim=128,
                    dropout=dropout,
                )
            elif pooling_method == "mean":
                from src.models.transnnmil.hierarchical_pooling import RegionMeanPooling

                self.region_pooling = RegionMeanPooling()
            elif pooling_method == "max":
                from src.models.transnnmil.hierarchical_pooling import RegionMaxPooling

                self.region_pooling = RegionMaxPooling()

            # Inter-region transformer
            self.region_transformer = RegionTransformer(
                feature_dim=region_hidden_dim,
                num_layers=2,
                num_heads=8,
                mlp_ratio=4.0,
                dropout=dropout,
                use_pos_encoding=False,
            )

            # Branches process region tokens
            branch_input_dim = region_hidden_dim
        else:
            # Branches process raw patches
            branch_input_dim = feature_dim

        # Branch A: TransMIL (Transformer-based correlator)
        # Disable positional encoding for random sub-bag compatibility
        self.branch_a = TransMIL(
            feature_dim=branch_input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding,
        )

        # Branch B: nnMIL (Lightweight gated attention aggregator)
        # Use higher dropout (0.25) as per nnMIL paper
        self.branch_b = nnMIL(
            feature_dim=branch_input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=0.25,  # nnMIL uses higher dropout than TransMIL
        )

        # Feature-level fusion components
        # Project Branch A features (256-dim) to common embedding space (512-dim)
        self.proj_a = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Project Branch B features to common embedding space (512-dim)
        # Input dim depends on hierarchical mode:
        # - Hierarchical: region_hidden_dim (512)
        # - Non-hierarchical: feature_dim (1024)
        branch_b_output_dim = region_hidden_dim if enable_hierarchical else feature_dim
        self.proj_b = nn.Sequential(
            nn.Linear(branch_b_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Cross-attention fusion module
        # Uses multi-head attention to fuse features from both branches
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Fusion classifier: maps fused features to class predictions
        # Architecture: 512 → 256 → num_classes
        self.fusion_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Learnable fusion gate (maintained for backward compatibility)
        # Initialized to 0.0 → sigmoid(0) = 0.5 (equal weight to both branches)
        # Note: Not used in feature-level fusion, but kept for API compatibility
        self.gate_param = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        coordinates: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through TransnnMIL with optional hierarchical pooling.

        Processes the input bag through optional hierarchical pooling, then
        through both branches, extracts features before classification, projects
        them to a common dimension, fuses them via cross-attention, and classifies
        the fused features.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            num_patches: Actual patch counts [batch_size] for masking padded patches
            return_attention: If True, return attention weights from Branch A
            coordinates: Patch coordinates [batch_size, num_patches, 2] (required if enable_hierarchical=True)

        Returns:
            logits: Class predictions [batch_size, num_classes]
            attention_weights: (optional) Attention from Branch A [batch_size, num_patches or num_regions]

        Notes:
            - If enable_hierarchical=True, patches are first grouped into regions
            - Regions are processed by intra-region pooling and inter-region transformer
            - Both branches process the same input (patches or regions) in parallel
            - Features are extracted before classification using get_features()
            - Features are projected to common 512-dim space
            - Cross-attention fuses features from both branches
            - Fusion classifier produces final predictions
            - Attention weights are returned from Branch A (TransMIL) for interpretability
        """
        batch_size = features.size(0)

        # Hierarchical pooling (optional)
        if self.enable_hierarchical:
            if coordinates is None:
                raise ValueError("coordinates required when enable_hierarchical=True")

            # 1. Spatial clustering: assign patches to regions
            assignments = self.hierarchical_pooling(coordinates)  # [B, N, R]

            # 2. Project features
            h = self.region_feature_proj(features)  # [B, N, region_hidden_dim]

            # 3. Intra-region aggregation
            region_features = self.region_pooling(h, assignments)  # [B, R, region_hidden_dim]

            # 4. Inter-region transformer
            region_centers = self.hierarchical_pooling.get_centers()  # [R, 2]
            region_tokens = self.region_transformer(
                region_features,
                region_centers=region_centers,
            )  # [B, R, region_hidden_dim]

            # Use region tokens as input to branches
            branch_input = region_tokens
            branch_num_patches = torch.full(
                (batch_size,), self.num_regions, dtype=torch.long, device=features.device
            )
        else:
            # Use raw patches
            branch_input = features
            branch_num_patches = num_patches

        # Extract attention weights from Branch A if requested
        attention_a = None
        if return_attention:
            _, attention_a = self.branch_a(branch_input, branch_num_patches, return_attention=True)

        # Extract features from both branches (before classification)
        # Branch A: CLS token representation [batch_size, 256]
        features_a = self.branch_a.get_features(branch_input, branch_num_patches)

        # Branch B: Aggregated features [batch_size, 1024]
        features_b = self.branch_b.get_features(branch_input, branch_num_patches)

        # Branch C: Topology (optional)
        features_c = None
        if self.enable_topology:
            if coordinates is None:
                raise ValueError("coordinates required when enable_topology=True")

            # Create mask from num_patches
            mask = None
            if num_patches is not None:
                mask = torch.arange(features.size(1), device=features.device).unsqueeze(
                    0
                ) < num_patches.unsqueeze(1)

            # Get topology features [batch_size, region_hidden_dim]
            features_c = self.topology_branch(features, coordinates, mask)

        # Project to common dimension (512)
        proj_a = self.proj_a(features_a)  # [batch_size, 512]
        proj_b = self.proj_b(features_b)  # [batch_size, 512]

        # Three-branch fusion if topology enabled
        if self.enable_topology and features_c is not None:
            # Project topology features
            proj_c = nn.Linear(self.region_hidden_dim, 512).to(features.device)(
                features_c
            )  # [B, 512]

            # Concatenate all three branches
            combined = torch.stack([proj_a, proj_b, proj_c], dim=1)  # [B, 3, 512]

            # Multi-head attention over branches
            fused, _ = self.fusion_attention(combined, combined, combined)  # [B, 3, 512]
            fused = fused.mean(dim=1)  # [B, 512] - average over branches
        else:
            # Two-branch fusion (original)
            # Validate output shapes
            batch_size = features.size(0)
            assert proj_a.shape == (
                batch_size,
                512,
            ), f"proj_a shape mismatch: expected ({batch_size}, 512), got {proj_a.shape}"
            assert proj_b.shape == (
                batch_size,
                512,
            ), f"proj_b shape mismatch: expected ({batch_size}, 512), got {proj_b.shape}"

            # Reshape for multi-head attention: [B, 512] → [B, 1, 512]
            query = proj_a.unsqueeze(1)
            key = proj_b.unsqueeze(1)
            value = proj_b.unsqueeze(1)

            # Apply cross-attention fusion
            fused, _ = self.fusion_attention(query, key, value)  # [B, 1, 512]
            fused = fused.squeeze(1)  # [B, 512]

        # Validate fused features shape
        assert fused.shape == (
            batch_size,
            512,
        ), f"fused shape mismatch: expected ({batch_size}, 512), got {fused.shape}"

        # Classify fused features
        logits = self.fusion_classifier(fused)  # [B, num_classes]

        # Validate output logits shape
        assert logits.shape == (
            batch_size,
            self.num_classes,
        ), f"logits shape mismatch: expected ({batch_size}, {self.num_classes}), got {logits.shape}"

        if return_attention:
            # Return attention weights from Branch A (TransMIL)
            # These provide interpretability via transformer attention patterns
            return logits, attention_a
        else:
            return logits

    def get_gate_value(self) -> float:
        """
        Get the current fusion gate value.

        Returns:
            gate_value: Weight given to Branch A (TransMIL), in range (0, 1)
                       Branch B (nnMIL) receives weight (1 - gate_value)

        Example:
            >>> model = TransnnMIL(feature_dim=1024)
            >>> gate = model.get_gate_value()
            >>> print(f"TransMIL weight: {gate:.3f}, nnMIL weight: {1-gate:.3f}")
        """
        with torch.no_grad():
            return torch.sigmoid(self.gate_param).item()

    def get_branch_outputs(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        coordinates: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get outputs from both branches separately (for analysis/debugging).

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            num_patches: Actual patch counts [batch_size]
            coordinates: Patch coordinates [batch_size, num_patches, 2] (required if enable_hierarchical=True)

        Returns:
            logits_a: Predictions from Branch A (TransMIL) [batch_size, num_classes]
            logits_b: Predictions from Branch B (nnMIL) [batch_size, num_classes]
            logits_fused: Final fused predictions [batch_size, num_classes]

        Example:
            >>> model = TransnnMIL(feature_dim=1024, enable_hierarchical=True)
            >>> features = torch.randn(4, 100, 1024)
            >>> coords = torch.rand(4, 100, 2)
            >>> logits_a, logits_b, logits_fused = model.get_branch_outputs(features, coordinates=coords)
            >>>
            >>> # Analyze branch agreement
            >>> preds_a = logits_a.argmax(dim=1)
            >>> preds_b = logits_b.argmax(dim=1)
            >>> agreement = (preds_a == preds_b).float().mean()
            >>> print(f"Branch agreement: {agreement:.2%}")
        """
        with torch.no_grad():
            batch_size = features.size(0)

            # Hierarchical pooling (optional)
            if self.enable_hierarchical:
                if coordinates is None:
                    raise ValueError("coordinates required when enable_hierarchical=True")

                # Process through hierarchical pooling
                assignments = self.hierarchical_pooling(coordinates)
                h = self.region_feature_proj(features)
                region_features = self.region_pooling(h, assignments)
                region_centers = self.hierarchical_pooling.get_centers()
                region_tokens = self.region_transformer(
                    region_features, region_centers=region_centers
                )

                branch_input = region_tokens
                branch_num_patches = torch.full(
                    (batch_size,), self.num_regions, dtype=torch.long, device=features.device
                )
            else:
                branch_input = features
                branch_num_patches = num_patches

            # Get logits from original branch classifiers
            logits_a = self.branch_a(branch_input, branch_num_patches, return_attention=False)
            logits_b = self.branch_b(branch_input, branch_num_patches, return_attention=False)

            # Compute fused logits via feature-level fusion pipeline
            # Extract features from both branches (before classification)
            features_a = self.branch_a.get_features(branch_input, branch_num_patches)
            features_b = self.branch_b.get_features(branch_input, branch_num_patches)

            # Project to common dimension (512)
            proj_a = self.proj_a(features_a)
            proj_b = self.proj_b(features_b)

            # Reshape for multi-head attention: [B, 512] → [B, 1, 512]
            query = proj_a.unsqueeze(1)
            key = proj_b.unsqueeze(1)
            value = proj_b.unsqueeze(1)

            # Apply cross-attention fusion
            fused, _ = self.fusion_attention(query, key, value)
            fused = fused.squeeze(1)

            # Classify fused features
            logits_fused = self.fusion_classifier(fused)

            return logits_a, logits_b, logits_fused
