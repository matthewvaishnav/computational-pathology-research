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

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np


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
        init_method: str = 'uniform',
    ):
        super().__init__()
        
        # Validate inputs
        if num_clusters <= 0:
            raise ValueError(f"num_clusters must be positive, got {num_clusters}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if init_method not in ['uniform', 'random']:
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
        if self.init_method == 'uniform':
            # Grid layout
            k = int(self.num_clusters ** 0.5)
            if k * k != self.num_clusters:
                # Not perfect square, use random
                return torch.rand(self.num_clusters, 2)
            
            # Create grid
            x = torch.linspace(0, 1, k + 2)[1:-1]  # Exclude boundaries
            y = torch.linspace(0, 1, k + 2)[1:-1]
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            centers = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
            return centers
        
        elif self.init_method == 'random':
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
        init_method: str = 'uniform',
    ):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.init_method = init_method
        
        # Learnable cluster centers
        self.clusterer = LearnableClusterCenters(
            num_clusters=num_clusters,
            temperature=temperature,
            init_method=init_method,
        )
    
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
        self.register_buffer('centers', torch.zeros(num_clusters, 2))
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
        k = int(num_clusters ** 0.5)
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
        self.register_buffer('centers', centers)
    
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
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
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

