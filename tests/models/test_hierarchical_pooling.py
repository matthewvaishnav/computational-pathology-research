"""
Unit tests for hierarchical pooling clustering methods.

Tests:
- LearnableClusterCenters
- KMeansClusterer
- GridClusterer
- HierarchicalPooling
"""

import pytest
import torch
import numpy as np

from src.models.hierarchical_pooling import (
    LearnableClusterCenters,
    KMeansClusterer,
    GridClusterer,
    HierarchicalPooling,
)


class TestLearnableClusterCenters:
    """Test learnable cluster centers."""
    
    def test_init_uniform(self):
        """Test uniform grid initialization."""
        clusterer = LearnableClusterCenters(
            num_clusters=16,
            temperature=1.0,
            init_method='uniform',
        )
        
        assert clusterer.num_clusters == 16
        assert clusterer.temperature == 1.0
        assert clusterer.centers.shape == (16, 2)
        
        # Check centers in [0, 1]
        assert (clusterer.centers >= 0).all()
        assert (clusterer.centers <= 1).all()
    
    def test_init_random(self):
        """Test random initialization."""
        clusterer = LearnableClusterCenters(
            num_clusters=10,
            temperature=0.5,
            init_method='random',
        )
        
        assert clusterer.num_clusters == 10
        assert clusterer.centers.shape == (10, 2)
        assert (clusterer.centers >= 0).all()
        assert (clusterer.centers <= 1).all()
    
    def test_invalid_inputs(self):
        """Test input validation."""
        with pytest.raises(ValueError, match="num_clusters must be positive"):
            LearnableClusterCenters(num_clusters=0)
        
        with pytest.raises(ValueError, match="temperature must be positive"):
            LearnableClusterCenters(num_clusters=16, temperature=-1.0)
        
        with pytest.raises(ValueError, match="init_method must be"):
            LearnableClusterCenters(num_clusters=16, init_method='invalid')
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        clusterer = LearnableClusterCenters(num_clusters=16)
        coords = torch.rand(4, 100, 2)
        
        assignments = clusterer(coords)
        
        assert assignments.shape == (4, 100, 16)
        assert torch.allclose(assignments.sum(dim=-1), torch.ones(4, 100))
    
    def test_forward_with_mask(self):
        """Test forward with mask."""
        clusterer = LearnableClusterCenters(num_clusters=16)
        coords = torch.rand(4, 100, 2)
        mask = torch.ones(4, 100, dtype=torch.bool)
        mask[:, 50:] = False  # Mask half
        
        assignments = clusterer(coords, mask)
        
        assert assignments.shape == (4, 100, 16)
        
        # Valid patches: soft assignment
        assert not torch.allclose(
            assignments[:, :50],
            torch.ones_like(assignments[:, :50]) / 16,
        )
        
        # Masked patches: uniform
        assert torch.allclose(
            assignments[:, 50:],
            torch.ones_like(assignments[:, 50:]) / 16,
        )
    
    def test_temperature_effect(self):
        """Test temperature controls sharpness."""
        coords = torch.rand(4, 100, 2)
        
        # Low temp = sharper
        clusterer_low = LearnableClusterCenters(num_clusters=16, temperature=0.1)
        assign_low = clusterer_low(coords)
        
        # High temp = softer
        clusterer_high = LearnableClusterCenters(num_clusters=16, temperature=10.0)
        assign_high = clusterer_high(coords)
        
        # Low temp has higher max prob (sharper)
        assert assign_low.max(dim=-1)[0].mean() > assign_high.max(dim=-1)[0].mean()
    
    def test_get_centers(self):
        """Test get_centers method."""
        clusterer = LearnableClusterCenters(num_clusters=16)
        centers = clusterer.get_centers()
        
        assert centers.shape == (16, 2)
        assert not centers.requires_grad


class TestKMeansClusterer:
    """Test k-means baseline."""
    
    def test_init(self):
        """Test initialization."""
        clusterer = KMeansClusterer(num_clusters=16, temperature=1.0)
        
        assert clusterer.num_clusters == 16
        assert clusterer.temperature == 1.0
        assert not clusterer._fitted
    
    def test_fit(self):
        """Test fitting k-means."""
        clusterer = KMeansClusterer(num_clusters=16)
        coords = torch.rand(100, 2)
        
        clusterer.fit(coords)
        
        assert clusterer._fitted
        assert clusterer.centers.shape == (16, 2)
    
    def test_fit_batched(self):
        """Test fitting with batched input."""
        clusterer = KMeansClusterer(num_clusters=16)
        coords = torch.rand(4, 100, 2)
        
        clusterer.fit(coords)  # Uses first batch
        
        assert clusterer._fitted
        assert clusterer.centers.shape == (16, 2)
    
    def test_forward_without_fit(self):
        """Test forward fails without fit."""
        clusterer = KMeansClusterer(num_clusters=16)
        coords = torch.rand(4, 100, 2)
        
        with pytest.raises(RuntimeError, match="Must call fit"):
            clusterer(coords)
    
    def test_forward_after_fit(self):
        """Test forward after fitting."""
        clusterer = KMeansClusterer(num_clusters=16)
        coords = torch.rand(4, 100, 2)
        
        clusterer.fit(coords)
        assignments = clusterer(coords)
        
        assert assignments.shape == (4, 100, 16)
        assert torch.allclose(assignments.sum(dim=-1), torch.ones(4, 100))
    
    def test_get_centers(self):
        """Test get_centers."""
        clusterer = KMeansClusterer(num_clusters=16)
        coords = torch.rand(100, 2)
        
        clusterer.fit(coords)
        centers = clusterer.get_centers()
        
        assert centers.shape == (16, 2)
        assert not centers.requires_grad


class TestGridClusterer:
    """Test grid-based baseline."""
    
    def test_init_perfect_square(self):
        """Test initialization with perfect square."""
        clusterer = GridClusterer(num_clusters=16)
        
        assert clusterer.num_clusters == 16
        assert clusterer.grid_size == 4
        assert clusterer.centers.shape == (16, 2)
    
    def test_init_not_perfect_square(self):
        """Test initialization fails for non-perfect square."""
        with pytest.raises(ValueError, match="must be perfect square"):
            GridClusterer(num_clusters=15)
    
    def test_grid_layout(self):
        """Test grid centers are uniformly spaced."""
        clusterer = GridClusterer(num_clusters=16)
        centers = clusterer.get_centers()
        
        # Check 4x4 grid
        x_coords = centers[:, 0].unique().sort()[0]
        y_coords = centers[:, 1].unique().sort()[0]
        
        assert len(x_coords) == 4
        assert len(y_coords) == 4
        
        # Check uniform spacing
        x_diff = x_coords[1:] - x_coords[:-1]
        y_diff = y_coords[1:] - y_coords[:-1]
        
        assert torch.allclose(x_diff, x_diff[0])
        assert torch.allclose(y_diff, y_diff[0])
    
    def test_forward_shape(self):
        """Test forward pass."""
        clusterer = GridClusterer(num_clusters=16)
        coords = torch.rand(4, 100, 2)
        
        assignments = clusterer(coords)
        
        assert assignments.shape == (4, 100, 16)
        assert torch.allclose(assignments.sum(dim=-1), torch.ones(4, 100))
    
    def test_forward_with_mask(self):
        """Test forward with mask."""
        clusterer = GridClusterer(num_clusters=16)
        coords = torch.rand(4, 100, 2)
        mask = torch.ones(4, 100, dtype=torch.bool)
        mask[:, 50:] = False
        
        assignments = clusterer(coords, mask)
        
        # Masked patches get uniform
        assert torch.allclose(
            assignments[:, 50:],
            torch.ones_like(assignments[:, 50:]) / 16,
        )
    
    def test_get_centers(self):
        """Test get_centers."""
        clusterer = GridClusterer(num_clusters=16)
        centers = clusterer.get_centers()
        
        assert centers.shape == (16, 2)
        assert not centers.requires_grad


class TestHierarchicalPooling:
    """Test hierarchical pooling module."""
    
    def test_init(self):
        """Test initialization."""
        pooling = HierarchicalPooling(num_clusters=16)
        
        assert pooling.num_clusters == 16
        assert isinstance(pooling.clusterer, LearnableClusterCenters)
    
    def test_forward(self):
        """Test forward pass."""
        pooling = HierarchicalPooling(num_clusters=16)
        coords = torch.rand(4, 100, 2)
        
        assignments = pooling(coords)
        
        assert assignments.shape == (4, 100, 16)
        assert torch.allclose(assignments.sum(dim=-1), torch.ones(4, 100))
    
    def test_aggregate_features(self):
        """Test feature aggregation by region."""
        pooling = HierarchicalPooling(num_clusters=16)
        
        features = torch.randn(4, 100, 1024)
        coords = torch.rand(4, 100, 2)
        
        # Get assignments
        assignments = pooling(coords)  # [4, 100, 16]
        
        # Aggregate features
        region_features = torch.bmm(
            assignments.transpose(1, 2),  # [4, 16, 100]
            features,  # [4, 100, 1024]
        )  # [4, 16, 1024]
        
        assert region_features.shape == (4, 16, 1024)
    
    def test_get_centers(self):
        """Test get_centers."""
        pooling = HierarchicalPooling(num_clusters=16)
        centers = pooling.get_centers()
        
        assert centers.shape == (16, 2)


class TestClustererComparison:
    """Compare different clustering methods."""
    
    def test_all_produce_valid_assignments(self):
        """Test all methods produce valid soft assignments."""
        coords = torch.rand(4, 100, 2)
        
        # Learnable
        learnable = LearnableClusterCenters(num_clusters=16)
        assign_learnable = learnable(coords)
        
        # K-means
        kmeans = KMeansClusterer(num_clusters=16)
        kmeans.fit(coords)
        assign_kmeans = kmeans(coords)
        
        # Grid
        grid = GridClusterer(num_clusters=16)
        assign_grid = grid(coords)
        
        # All valid
        for assign in [assign_learnable, assign_kmeans, assign_grid]:
            assert assign.shape == (4, 100, 16)
            assert torch.allclose(assign.sum(dim=-1), torch.ones(4, 100))
            assert (assign >= 0).all()
            assert (assign <= 1).all()
    
    def test_learnable_has_gradients(self):
        """Test only learnable method has gradients."""
        coords = torch.rand(4, 100, 2)
        
        # Learnable
        learnable = LearnableClusterCenters(num_clusters=16)
        assign_learnable = learnable(coords)
        loss = assign_learnable.sum()
        loss.backward()
        
        assert learnable.centers.grad is not None
        
        # K-means (no gradients)
        kmeans = KMeansClusterer(num_clusters=16)
        kmeans.fit(coords)
        assign_kmeans = kmeans(coords)
        
        assert not assign_kmeans.requires_grad
        
        # Grid (no gradients)
        grid = GridClusterer(num_clusters=16)
        assign_grid = grid(coords)
        
        assert not assign_grid.requires_grad


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
