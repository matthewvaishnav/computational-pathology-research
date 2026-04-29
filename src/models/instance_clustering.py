"""
Instance-Level Clustering for MIL

CLAM-style instance clustering to refine feature space and identify
high-diagnostic-value subregions.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available")


class InstanceClusteringModule(nn.Module):
    """
    Instance-level clustering for MIL.
    
    Clusters patch features to identify distinct tissue regions
    and refine the feature space for better classification.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_clusters: int = 10,
        cluster_method: str = "kmeans"
    ):
        """
        Initialize clustering module.
        
        Args:
            feature_dim: Dimension of patch features
            num_clusters: Number of clusters
            cluster_method: Clustering method ("kmeans", "learnable")
        """
        super().__init__()
        
        if not SKLEARN_AVAILABLE and cluster_method == "kmeans":
            raise ImportError("sklearn required for kmeans")
        
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.cluster_method = cluster_method
        
        if cluster_method == "learnable":
            # Learnable cluster centers
            self.cluster_centers = nn.Parameter(
                torch.randn(num_clusters, feature_dim)
            )
        else:
            self.register_buffer("cluster_centers", torch.zeros(num_clusters, feature_dim))
    
    def fit_clusters(self, features: torch.Tensor):
        """
        Fit cluster centers to features.
        
        Args:
            features: Patch features (N, feature_dim)
        """
        if self.cluster_method == "kmeans":
            # Use sklearn KMeans
            features_np = features.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
            kmeans.fit(features_np)
            
            # Update cluster centers
            self.cluster_centers.copy_(
                torch.from_numpy(kmeans.cluster_centers_).to(features.device)
            )
        elif self.cluster_method == "learnable":
            # Learnable centers updated via backprop
            pass
    
    def assign_clusters(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign patches to clusters.
        
        Args:
            features: Patch features (N, feature_dim)
        
        Returns:
            cluster_ids: Cluster assignment (N,)
            distances: Distance to assigned cluster (N,)
        """
        # Compute distances to all clusters
        # (N, 1, D) - (1, K, D) = (N, K, D)
        dists = torch.cdist(
            features.unsqueeze(0),
            self.cluster_centers.unsqueeze(0)
        ).squeeze(0)  # (N, K)
        
        # Assign to nearest cluster
        cluster_ids = dists.argmin(dim=1)  # (N,)
        distances = dists.gather(1, cluster_ids.unsqueeze(1)).squeeze(1)  # (N,)
        
        return cluster_ids, distances
    
    def get_cluster_features(
        self,
        features: torch.Tensor,
        cluster_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Get aggregated features per cluster.
        
        Args:
            features: Patch features (N, feature_dim)
            cluster_ids: Cluster assignments (N,)
        
        Returns:
            cluster_features: Aggregated features (num_clusters, feature_dim)
        """
        cluster_features = []
        
        for k in range(self.num_clusters):
            mask = cluster_ids == k
            if mask.sum() > 0:
                # Mean pooling within cluster
                cluster_feat = features[mask].mean(dim=0)
            else:
                # Empty cluster - use cluster center
                cluster_feat = self.cluster_centers[k]
            
            cluster_features.append(cluster_feat)
        
        return torch.stack(cluster_features)  # (K, D)
    
    def forward(
        self,
        features: torch.Tensor,
        return_assignments: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with clustering.
        
        Args:
            features: Patch features (N, feature_dim)
            return_assignments: Whether to return cluster assignments
        
        Returns:
            cluster_features: Aggregated cluster features (num_clusters, feature_dim)
            cluster_ids: (optional) Cluster assignments (N,)
        """
        # Assign to clusters
        cluster_ids, _ = self.assign_clusters(features)
        
        # Aggregate per cluster
        cluster_features = self.get_cluster_features(features, cluster_ids)
        
        if return_assignments:
            return cluster_features, cluster_ids
        return cluster_features, None


class CLAMInstanceBranch(nn.Module):
    """
    CLAM instance-level branch for identifying high-value regions.
    
    Predicts instance-level scores to identify patches with
    high diagnostic value.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize instance branch.
        
        Args:
            feature_dim: Dimension of patch features
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.instance_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict instance-level scores.
        
        Args:
            features: Patch features (N, feature_dim)
        
        Returns:
            scores: Instance scores (N,)
        """
        scores = self.instance_classifier(features).squeeze(-1)
        return scores
    
    def select_top_instances(
        self,
        features: torch.Tensor,
        scores: torch.Tensor,
        top_k: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k instances by score.
        
        Args:
            features: Patch features (N, feature_dim)
            scores: Instance scores (N,)
            top_k: Number of top instances to select
        
        Returns:
            top_features: Selected features (top_k, feature_dim)
            top_indices: Selected indices (top_k,)
        """
        top_k = min(top_k, len(scores))
        top_scores, top_indices = torch.topk(scores, top_k)
        top_features = features[top_indices]
        
        return top_features, top_indices


def cluster_instances(
    features: torch.Tensor,
    num_clusters: int = 10,
    method: str = "kmeans"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function for instance clustering.
    
    Args:
        features: Patch features (N, feature_dim)
        num_clusters: Number of clusters
        method: Clustering method
    
    Returns:
        cluster_features: Aggregated cluster features (num_clusters, feature_dim)
        cluster_ids: Cluster assignments (N,)
    """
    module = InstanceClusteringModule(
        feature_dim=features.shape[1],
        num_clusters=num_clusters,
        cluster_method=method
    )
    
    # Fit clusters
    module.fit_clusters(features)
    
    # Get cluster features and assignments
    cluster_features, cluster_ids = module(features, return_assignments=True)
    
    return cluster_features, cluster_ids
