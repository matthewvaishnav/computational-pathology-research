"""
Hierarchical Pooling: Spatial clustering for MIL

Implements learnable spatial clustering to group patches into regions,
enabling hierarchical aggregation for large WSI bags.

Key components:
- Learnable cluster centers (nn.Parameter)
- Soft assignment via softmax over distances
- Supports k-means and grid-based baselines

Architecture:
    Input: Patch features [B, N, D] + coordinates [B, N, 2]
    ├─ Compute distances to cluster centers
    ├─ Soft assignment: softmax(-distances / temperature)
    └─ Output: Region assignments [B, N, K]

Reference:
- TransnnMIL v2.0: Hierarchical + Topology (2027)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class LearnableClusterCenters(nn.Module):
    """
    Learnable spatial cluster centers for patch grouping.

    Learns K cluster centers in 2D coordinate space. Patches are assigned
    to clusters via soft assignment (softmax over distances).

    Args:
        num_clusters: Number of spatial regions (K)
        temperature: Softmax temperature for soft assignment (default: 1.0)
                    Lower = harder assignment, higher = softer
        init_method: Initialization method ('uniform', 'random')
                    - 'uniform': Grid layout in [0, 1]^2
                    - 'random': Random positions in [0, 1]^2

    Example:
        >>> # Create learnable cluster centers
        >>> clusterer = LearnableClusterCenters(num_clusters=16, temperature=0.5)
        >>>
        >>> # Patch coordinates (normalized to [0, 1])
        >>> coords = torch.rand(4, 100, 2)  # [batch, patches, xy]
        >>>
        >>> # Get soft assignments
        >>> assignments = clusterer(coords)  # [4, 100, 16]
        >>> assignments.sum(dim=-1)  # Should be all 1.0
        tensor([1., 1., 1., ...])
        >>>
        >>> # Get hard assignments (argmax)
        >>> hard_assign = assignments.argmax(dim=-1)  # [4, 100]

    Notes:
        - Cluster centers are initialized in [0, 1]^2 space
        - Input coordinates should be normalized to [0, 1]
        - Soft assignment allows gradients to flow to all clusters
        - Temperature controls assignment sharpness
    """

    def __init__(
        self,
        num_clusters: int,
        temperature: float = 1.0,
        init_method: str = "uniform",
    ):
        super().__init__()

        # Validate inputs
        if num_clusters <= 0:
            raise ValueError(f"num_clusters must be positive, got {num_clusters}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if init_method not in ["uniform", "random"]:
            raise ValueError(f"init_method must be 'uniform' or 'random', got {init_method}")

        self.num_clusters = num_clusters
        self.temperature = temperature
        self.init_method = init_method

        # Initialize cluster centers [K, 2]
        centers = self._initialize_centers()
        self.centers = nn.Parameter(centers)

    def _initialize_centers(self) -> torch.Tensor:
        """
        Initialize cluster centers in [0, 1]^2 space.

        Returns:
            centers: Cluster centers [num_clusters, 2]
        """
        if self.init_method == "uniform":
            # Grid layout
            k = int(self.num_clusters**0.5)
            if k * k != self.num_clusters:
                # Not perfect square, use random
                return torch.rand(self.num_clusters, 2)

            # Create grid
            x = torch.linspace(0, 1, k + 2)[1:-1]  # Exclude boundaries
            y = torch.linspace(0, 1, k + 2)[1:-1]
            grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
            centers = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
            return centers

        elif self.init_method == "random":
            # Random positions
            return torch.rand(self.num_clusters, 2)

        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

    def forward(
        self,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute soft assignments for patches.

        Args:
            coords: Patch coordinates [batch_size, num_patches, 2]
                   Should be normalized to [0, 1]
            mask: Optional mask for valid patches [batch_size, num_patches]
                 True = valid, False = padding

        Returns:
            assignments: Soft assignments [batch_size, num_patches, num_clusters]
                        Each row sums to 1.0

        Notes:
            - Computes L2 distance to each cluster center
            - Applies softmax with temperature scaling
            - Masked patches get uniform assignment (1/K)
        """
        batch_size, num_patches, _ = coords.shape

        # Compute pairwise distances [B, N, K]
        # coords: [B, N, 2], centers: [K, 2]
        # Expand: coords [B, N, 1, 2], centers [1, 1, K, 2]
        coords_expanded = coords.unsqueeze(2)  # [B, N, 1, 2]
        centers_expanded = self.centers.unsqueeze(0).unsqueeze(0)  # [1, 1, K, 2]

        # L2 distance
        distances = torch.norm(coords_expanded - centers_expanded, dim=-1)  # [B, N, K]

        # Soft assignment: softmax(-distances / temperature)
        assignments = F.softmax(-distances / self.temperature, dim=-1)  # [B, N, K]

        # Apply mask if provided
        if mask is not None:
            # Masked patches get uniform assignment
            uniform = torch.ones_like(assignments) / self.num_clusters
            assignments = torch.where(
                mask.unsqueeze(-1),  # [B, N, 1]
                assignments,
                uniform,
            )

        return assignments

    def get_centers(self) -> torch.Tensor:
        """
        Get current cluster center positions.

        Returns:
            centers: Cluster centers [num_clusters, 2]
        """
        return self.centers.detach()


class HierarchicalPooling(nn.Module):
    """
    Hierarchical pooling module for MIL.

    Groups patches into spatial regions via learnable clustering,
    then aggregates within each region.

    Args:
        num_clusters: Number of spatial regions
        temperature: Softmax temperature for soft assignment
        init_method: Cluster center initialization ('uniform', 'random')

    Example:
        >>> # Create hierarchical pooling
        >>> pooling = HierarchicalPooling(num_clusters=16)
        >>>
        >>> # Patch features + coordinates
        >>> features = torch.randn(4, 100, 1024)
        >>> coords = torch.rand(4, 100, 2)
        >>>
        >>> # Get region assignments
        >>> assignments = pooling(coords)  # [4, 100, 16]
        >>>
        >>> # Aggregate features by region (weighted sum)
        >>> region_features = torch.bmm(
        ...     assignments.transpose(1, 2),  # [4, 16, 100]
        ...     features,  # [4, 100, 1024]
        ... )  # [4, 16, 1024]
    """

    def __init__(
        self,
        num_clusters: int,
        temperature: float = 1.0,
        init_method: str = "uniform",
        clustering_method: str = "learnable",
    ):
        super().__init__()

        self.num_clusters = num_clusters
        self.temperature = temperature
        self.init_method = init_method
        self.clustering_method = clustering_method

        # Select clustering method
        if clustering_method == "learnable":
            self.clusterer = LearnableClusterCenters(
                num_clusters=num_clusters,
                temperature=temperature,
                init_method=init_method,
            )
        elif clustering_method == "kmeans":
            self.clusterer = KMeansClusterer(
                num_clusters=num_clusters,
                temperature=temperature,
            )
        elif clustering_method == "grid":
            self.clusterer = GridClusterer(
                num_clusters=num_clusters,
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unknown clustering_method: {clustering_method}")

    def forward(
        self,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute soft region assignments.

        Args:
            coords: Patch coordinates [batch_size, num_patches, 2]
            mask: Optional mask for valid patches [batch_size, num_patches]

        Returns:
            assignments: Soft assignments [batch_size, num_patches, num_clusters]
        """
        return self.clusterer(coords, mask)

    def get_centers(self) -> torch.Tensor:
        """Get cluster center positions."""
        return self.clusterer.get_centers()


class KMeansClusterer(nn.Module):
    """
    K-means baseline for spatial clustering.

    Uses sklearn KMeans to cluster patch coordinates. Centers are fixed
    (not learnable). Useful as baseline comparison.

    Args:
        num_clusters: Number of spatial regions (K)
        temperature: Softmax temperature for soft assignment
        random_state: Random seed for reproducibility

    Example:
        >>> clusterer = KMeansClusterer(num_clusters=16)
        >>> coords = torch.rand(4, 100, 2)
        >>>
        >>> # Fit on first batch
        >>> clusterer.fit(coords[0])
        >>>
        >>> # Get assignments
        >>> assignments = clusterer(coords)  # [4, 100, 16]

    Notes:
        - Centers are fixed after fit() call
        - Must call fit() before forward()
        - Not differentiable (no gradient flow)
    """

    def __init__(
        self,
        num_clusters: int,
        temperature: float = 1.0,
        random_state: int = 42,
    ):
        super().__init__()

        if num_clusters <= 0:
            raise ValueError(f"num_clusters must be positive, got {num_clusters}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.num_clusters = num_clusters
        self.temperature = temperature
        self.random_state = random_state

        # Will be set by fit()
        self.register_buffer("centers", torch.zeros(num_clusters, 2))
        self._fitted = False

    def fit(self, coords: torch.Tensor) -> None:
        """
        Fit k-means on coordinates.

        Args:
            coords: Patch coordinates [num_patches, 2] or [batch, num_patches, 2]
                   If batched, uses first batch only
        """
        # Handle batched input
        if coords.ndim == 3:
            coords = coords[0]

        # Convert to numpy
        coords_np = coords.detach().cpu().numpy()

        # Fit k-means
        kmeans = KMeans(
            n_clusters=self.num_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        kmeans.fit(coords_np)

        # Store centers
        centers = torch.from_numpy(kmeans.cluster_centers_).float()
        self.centers.copy_(centers)
        self._fitted = True

    def forward(
        self,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute soft assignments using fixed k-means centers.

        Args:
            coords: Patch coordinates [batch_size, num_patches, 2]
            mask: Optional mask [batch_size, num_patches]

        Returns:
            assignments: Soft assignments [batch_size, num_patches, num_clusters]
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before forward()")

        batch_size, num_patches, _ = coords.shape

        # Compute distances [B, N, K]
        coords_expanded = coords.unsqueeze(2)  # [B, N, 1, 2]
        centers_expanded = self.centers.unsqueeze(0).unsqueeze(0)  # [1, 1, K, 2]
        distances = torch.norm(coords_expanded - centers_expanded, dim=-1)

        # Soft assignment
        assignments = F.softmax(-distances / self.temperature, dim=-1)

        # Apply mask
        if mask is not None:
            uniform = torch.ones_like(assignments) / self.num_clusters
            assignments = torch.where(mask.unsqueeze(-1), assignments, uniform)

        return assignments

    def get_centers(self) -> torch.Tensor:
        """Get k-means cluster centers."""
        return self.centers.detach()


class RegionAttentionPooling(nn.Module):
    """
    Attention-based pooling within spatial regions.

    Aggregates patch features within each region using learned attention.
    Each region gets independent attention weights.

    Args:
        feature_dim: Patch feature dimension
        hidden_dim: Hidden dimension for attention (default: 128)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> pooling = RegionAttentionPooling(feature_dim=1024)
        >>> features = torch.randn(4, 100, 1024)
        >>> assignments = torch.randn(4, 100, 16).softmax(dim=-1)
        >>>
        >>> region_features = pooling(features, assignments)  # [4, 16, 1024]

    Notes:
        - Uses single-layer attention (query = tanh(W * features))
        - Attention computed independently per region
        - Masked patches contribute zero to aggregation
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        assignments: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool features within regions using attention.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            assignments: Soft region assignments [batch_size, num_patches, num_regions]
            mask: Optional mask [batch_size, num_patches]

        Returns:
            region_features: Aggregated features [batch_size, num_regions, feature_dim]
        """
        batch_size, num_patches, feature_dim = features.shape
        num_regions = assignments.shape[-1]

        # Compute attention scores [B, N, 1]
        attn_scores = self.attention(features)  # [B, N, 1]

        # Apply mask to attention
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        # Compute attention weights per region
        # For each region, weight patches by (assignment * attention)
        region_features = []

        for r in range(num_regions):
            # Region assignment weights [B, N, 1]
            region_assign = assignments[:, :, r : r + 1]  # [B, N, 1]

            # Combined weights: assignment * attention
            combined_scores = attn_scores + torch.log(region_assign + 1e-8)

            # Softmax over patches (within region)
            region_attn = F.softmax(combined_scores, dim=1)  # [B, N, 1]

            # Weighted sum
            region_feat = (features * region_attn).sum(dim=1)  # [B, D]
            region_features.append(region_feat)

        # Stack regions
        region_features = torch.stack(region_features, dim=1)  # [B, R, D]

        return region_features


class RegionMeanPooling(nn.Module):
    """
    Mean pooling within spatial regions (baseline).

    Simple weighted average of patch features within each region.

    Args:
        None

    Example:
        >>> pooling = RegionMeanPooling()
        >>> features = torch.randn(4, 100, 1024)
        >>> assignments = torch.randn(4, 100, 16).softmax(dim=-1)
        >>>
        >>> region_features = pooling(features, assignments)  # [4, 16, 1024]
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        features: torch.Tensor,
        assignments: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool features within regions using mean.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            assignments: Soft region assignments [batch_size, num_patches, num_regions]
            mask: Optional mask [batch_size, num_patches]

        Returns:
            region_features: Aggregated features [batch_size, num_regions, feature_dim]
        """
        # Apply mask to assignments
        if mask is not None:
            assignments = assignments * mask.unsqueeze(-1).float()

        # Weighted sum: [B, R, N] @ [B, N, D] = [B, R, D]
        region_features = torch.bmm(
            assignments.transpose(1, 2),  # [B, R, N]
            features,  # [B, N, D]
        )

        # Normalize by sum of assignments (handle empty regions)
        region_weights = assignments.sum(dim=1, keepdim=True).transpose(1, 2)  # [B, R, 1]
        region_features = region_features / (region_weights + 1e-8)

        return region_features


class RegionMaxPooling(nn.Module):
    """
    Max pooling within spatial regions (baseline).

    Takes max feature value within each region (per dimension).

    Args:
        None

    Example:
        >>> pooling = RegionMaxPooling()
        >>> features = torch.randn(4, 100, 1024)
        >>> assignments = torch.randn(4, 100, 16).softmax(dim=-1)
        >>>
        >>> region_features = pooling(features, assignments)  # [4, 16, 1024]
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        features: torch.Tensor,
        assignments: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool features within regions using max.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            assignments: Soft region assignments [batch_size, num_patches, num_regions]
            mask: Optional mask [batch_size, num_patches]

        Returns:
            region_features: Aggregated features [batch_size, num_regions, feature_dim]
        """
        batch_size, num_patches, feature_dim = features.shape
        num_regions = assignments.shape[-1]

        # Hard assignment (argmax)
        hard_assign = assignments.argmax(dim=-1)  # [B, N]

        # Apply mask
        if mask is not None:
            hard_assign = hard_assign.masked_fill(~mask, -1)

        # Max pool per region
        region_features = []

        for r in range(num_regions):
            # Patches in region r
            region_mask = hard_assign == r  # [B, N]

            # Get features for region
            region_feat = features.clone()
            region_feat[~region_mask] = float("-inf")

            # Max over patches
            region_max = region_feat.max(dim=1)[0]  # [B, D]

            # Handle empty regions (all -inf)
            region_max = torch.where(
                torch.isinf(region_max),
                torch.zeros_like(region_max),
                region_max,
            )

            region_features.append(region_max)

        # Stack regions
        region_features = torch.stack(region_features, dim=1)  # [B, R, D]

        return region_features


class RegionTransformer(nn.Module):
    """
    Transformer for inter-region communication.

    Processes region features with multi-head self-attention to capture
    spatial relationships between regions.

    Args:
        feature_dim: Region feature dimension
        num_layers: Number of transformer layers (default: 2)
        num_heads: Number of attention heads (default: 8)
        mlp_ratio: MLP hidden dim ratio (default: 4.0)
        dropout: Dropout rate (default: 0.1)
        use_pos_encoding: Add positional encoding for regions (default: False)

    Example:
        >>> transformer = RegionTransformer(feature_dim=1024, num_layers=2)
        >>> region_features = torch.randn(4, 16, 1024)  # [B, R, D]
        >>>
        >>> # Process regions
        >>> output = transformer(region_features)  # [4, 16, 1024]

    Notes:
        - Standard transformer encoder architecture
        - Optional positional encoding based on region centers
        - LayerNorm + residual connections
    """

    def __init__(
        self,
        feature_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_pos_encoding: bool = False,
    ):
        super().__init__()

        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})"
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.use_pos_encoding = use_pos_encoding

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=int(feature_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Optional positional encoding
        if use_pos_encoding:
            self.pos_encoder = nn.Linear(2, feature_dim)
        else:
            self.pos_encoder = None

    def forward(
        self,
        region_features: torch.Tensor,
        region_centers: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process region features with transformer.

        Args:
            region_features: Region features [batch_size, num_regions, feature_dim]
            region_centers: Optional region center coords [num_regions, 2]
                          Used for positional encoding if use_pos_encoding=True
            mask: Optional mask for valid regions [batch_size, num_regions]
                 True = valid, False = padding

        Returns:
            output: Processed features [batch_size, num_regions, feature_dim]
        """
        batch_size, num_regions, feature_dim = region_features.shape

        # Add positional encoding
        if self.use_pos_encoding:
            if region_centers is None:
                raise ValueError("region_centers required when use_pos_encoding=True")

            # Encode positions [R, 2] -> [R, D]
            pos_encoding = self.pos_encoder(region_centers)  # [R, D]

            # Add to features
            region_features = region_features + pos_encoding.unsqueeze(0)  # [B, R, D]

        # Create attention mask (True = ignore)
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask  # Invert: True = padding

        # Apply transformer
        output = self.transformer(
            region_features,
            src_key_padding_mask=attn_mask,
        )

        return output


class GridClusterer(nn.Module):
    """
    Grid-based baseline for spatial clustering.

    Divides coordinate space into uniform grid. Simple, deterministic,
    no learning required. Useful as baseline.

    Args:
        num_clusters: Number of grid cells (must be perfect square)
        temperature: Softmax temperature for soft assignment

    Example:
        >>> clusterer = GridClusterer(num_clusters=16)  # 4x4 grid
        >>> coords = torch.rand(4, 100, 2)
        >>> assignments = clusterer(coords)  # [4, 100, 16]

    Notes:
        - Grid centers are fixed (not learnable)
        - num_clusters must be perfect square (4, 9, 16, 25, ...)
        - Assumes coords normalized to [0, 1]
    """

    def __init__(
        self,
        num_clusters: int,
        temperature: float = 1.0,
    ):
        super().__init__()

        if num_clusters <= 0:
            raise ValueError(f"num_clusters must be positive, got {num_clusters}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        # Check perfect square
        k = int(num_clusters**0.5)
        if k * k != num_clusters:
            raise ValueError(
                f"num_clusters must be perfect square, got {num_clusters}. "
                f"Try: {k*k} or {(k+1)*(k+1)}"
            )

        self.num_clusters = num_clusters
        self.temperature = temperature
        self.grid_size = k

        # Create fixed grid centers
        centers = self._create_grid()
        self.register_buffer("centers", centers)

    def _create_grid(self) -> torch.Tensor:
        """
        Create uniform grid centers in [0, 1]^2.

        Returns:
            centers: Grid centers [num_clusters, 2]
        """
        k = self.grid_size

        # Grid points (exclude boundaries)
        x = torch.linspace(0, 1, k + 2)[1:-1]
        y = torch.linspace(0, 1, k + 2)[1:-1]

        # Meshgrid
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        centers = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

        return centers

    def forward(
        self,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute soft assignments using fixed grid.

        Args:
            coords: Patch coordinates [batch_size, num_patches, 2]
            mask: Optional mask [batch_size, num_patches]

        Returns:
            assignments: Soft assignments [batch_size, num_patches, num_clusters]
        """
        batch_size, num_patches, _ = coords.shape

        # Compute distances [B, N, K]
        coords_expanded = coords.unsqueeze(2)  # [B, N, 1, 2]
        centers_expanded = self.centers.unsqueeze(0).unsqueeze(0)  # [1, 1, K, 2]
        distances = torch.norm(coords_expanded - centers_expanded, dim=-1)

        # Soft assignment
        assignments = F.softmax(-distances / self.temperature, dim=-1)

        # Apply mask
        if mask is not None:
            uniform = torch.ones_like(assignments) / self.num_clusters
            assignments = torch.where(mask.unsqueeze(-1), assignments, uniform)

        return assignments

    def get_centers(self) -> torch.Tensor:
        """Get grid centers."""
        return self.centers.detach()
