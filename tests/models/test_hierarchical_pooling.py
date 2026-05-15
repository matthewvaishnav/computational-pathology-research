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
    RegionAttentionPooling,
    RegionMeanPooling,
    RegionMaxPooling,
)


class TestLearnableClusterCenters:
    """Test learnable cluster centers."""

    def test_init_uniform(self):
        """Test uniform grid initialization."""
        clusterer = LearnableClusterCenters(
            num_clusters=16,
            temperature=1.0,
            init_method="uniform",
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
            init_method="random",
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
            LearnableClusterCenters(num_clusters=16, init_method="invalid")

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


class TestRegionAttentionPooling:
    """Test attention-based region pooling."""

    def test_init(self):
        """Test initialization."""
        pooling = RegionAttentionPooling(feature_dim=1024, hidden_dim=128)

        assert pooling.feature_dim == 1024
        assert pooling.hidden_dim == 128

    def test_invalid_inputs(self):
        """Test input validation."""
        with pytest.raises(ValueError, match="feature_dim must be positive"):
            RegionAttentionPooling(feature_dim=0)

        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            RegionAttentionPooling(feature_dim=1024, hidden_dim=0)

        with pytest.raises(ValueError, match="dropout must be in"):
            RegionAttentionPooling(feature_dim=1024, dropout=1.5)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        pooling = RegionAttentionPooling(feature_dim=1024)

        features = torch.randn(4, 100, 1024)
        assignments = torch.randn(4, 100, 16).softmax(dim=-1)

        region_features = pooling(features, assignments)

        assert region_features.shape == (4, 16, 1024)

    def test_forward_with_mask(self):
        """Test forward with mask."""
        pooling = RegionAttentionPooling(feature_dim=1024)

        features = torch.randn(4, 100, 1024)
        assignments = torch.randn(4, 100, 16).softmax(dim=-1)
        mask = torch.ones(4, 100, dtype=torch.bool)
        mask[:, 50:] = False

        region_features = pooling(features, assignments, mask)

        assert region_features.shape == (4, 16, 1024)
        assert not torch.isnan(region_features).any()

    def test_gradients_flow(self):
        """Test gradients flow through attention."""
        pooling = RegionAttentionPooling(feature_dim=1024)

        features = torch.randn(4, 100, 1024, requires_grad=True)
        assignments = torch.randn(4, 100, 16).softmax(dim=-1)

        region_features = pooling(features, assignments)
        loss = region_features.sum()
        loss.backward()

        assert features.grad is not None


class TestRegionMeanPooling:
    """Test mean pooling baseline."""

    def test_init(self):
        """Test initialization."""
        pooling = RegionMeanPooling()
        assert pooling is not None

    def test_forward_shape(self):
        """Test forward pass output shape."""
        pooling = RegionMeanPooling()

        features = torch.randn(4, 100, 1024)
        assignments = torch.randn(4, 100, 16).softmax(dim=-1)

        region_features = pooling(features, assignments)

        assert region_features.shape == (4, 16, 1024)

    def test_forward_with_mask(self):
        """Test forward with mask."""
        pooling = RegionMeanPooling()

        features = torch.randn(4, 100, 1024)
        assignments = torch.randn(4, 100, 16).softmax(dim=-1)
        mask = torch.ones(4, 100, dtype=torch.bool)
        mask[:, 50:] = False

        region_features = pooling(features, assignments, mask)

        assert region_features.shape == (4, 16, 1024)
        assert not torch.isnan(region_features).any()

    def test_weighted_average(self):
        """Test mean pooling computes weighted average."""
        pooling = RegionMeanPooling()

        # Simple case: 2 patches, 2 regions
        features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [1, 2, 2]
        assignments = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # [1, 2, 2]

        region_features = pooling(features, assignments)

        # Region 0: patch 0 = [1, 2]
        # Region 1: patch 1 = [3, 4]
        expected = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

        assert torch.allclose(region_features, expected)

    def test_soft_assignment(self):
        """Test mean pooling with soft assignments."""
        pooling = RegionMeanPooling()

        features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [1, 2, 2]
        # Both patches contribute to both regions
        assignments = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])  # [1, 2, 2]

        region_features = pooling(features, assignments)

        # Both regions get average of both patches
        expected = torch.tensor([[[2.0, 3.0], [2.0, 3.0]]])

        assert torch.allclose(region_features, expected)

    def test_gradients_flow(self):
        """Test gradients flow through mean pooling."""
        pooling = RegionMeanPooling()

        features = torch.randn(4, 100, 1024, requires_grad=True)
        assignments = torch.randn(4, 100, 16).softmax(dim=-1)

        region_features = pooling(features, assignments)
        loss = region_features.sum()
        loss.backward()

        assert features.grad is not None


class TestRegionMaxPooling:
    """Test max pooling baseline."""

    def test_init(self):
        """Test initialization."""
        pooling = RegionMaxPooling()
        assert pooling is not None

    def test_forward_shape(self):
        """Test forward pass output shape."""
        pooling = RegionMaxPooling()

        features = torch.randn(4, 100, 1024)
        assignments = torch.randn(4, 100, 16).softmax(dim=-1)

        region_features = pooling(features, assignments)

        assert region_features.shape == (4, 16, 1024)

    def test_forward_with_mask(self):
        """Test forward with mask."""
        pooling = RegionMaxPooling()

        features = torch.randn(4, 100, 1024)
        assignments = torch.randn(4, 100, 16).softmax(dim=-1)
        mask = torch.ones(4, 100, dtype=torch.bool)
        mask[:, 50:] = False

        region_features = pooling(features, assignments, mask)

        assert region_features.shape == (4, 16, 1024)
        assert not torch.isnan(region_features).any()

    def test_max_selection(self):
        """Test max pooling selects maximum values."""
        pooling = RegionMaxPooling()

        # Simple case: 3 patches, 2 regions
        features = torch.tensor([[[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]]])  # [1, 3, 2]
        # Hard assignment: patch 0,1 -> region 0, patch 2 -> region 1
        assignments = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])  # [1, 3, 2]

        region_features = pooling(features, assignments)

        # Region 0: max([1,5], [3,2]) = [3, 5]
        # Region 1: max([2,4]) = [2, 4]
        expected = torch.tensor([[[3.0, 5.0], [2.0, 4.0]]])

        assert torch.allclose(region_features, expected)

    def test_empty_region(self):
        """Test max pooling handles empty regions."""
        pooling = RegionMaxPooling()

        features = torch.randn(4, 100, 1024)
        assignments = torch.zeros(4, 100, 16)
        assignments[:, :, 0] = 1.0  # All patches in region 0

        region_features = pooling(features, assignments)

        # Region 0 has features, others are zeros
        assert not torch.allclose(region_features[:, 0], torch.zeros_like(region_features[:, 0]))
        assert torch.allclose(region_features[:, 1:], torch.zeros_like(region_features[:, 1:]))

    def test_gradients_flow(self):
        """Test gradients flow through max pooling."""
        pooling = RegionMaxPooling()

        features = torch.randn(4, 100, 1024, requires_grad=True)
        assignments = torch.randn(4, 100, 16).softmax(dim=-1)

        region_features = pooling(features, assignments)
        loss = region_features.sum()
        loss.backward()

        assert features.grad is not None


class TestPoolingComparison:
    """Compare different pooling methods."""

    def test_all_produce_valid_output(self):
        """Test all pooling methods produce valid output."""
        features = torch.randn(4, 100, 1024)
        assignments = torch.randn(4, 100, 16).softmax(dim=-1)

        # Attention
        attn_pooling = RegionAttentionPooling(feature_dim=1024)
        attn_out = attn_pooling(features, assignments)

        # Mean
        mean_pooling = RegionMeanPooling()
        mean_out = mean_pooling(features, assignments)

        # Max
        max_pooling = RegionMaxPooling()
        max_out = max_pooling(features, assignments)

        # All valid
        for out in [attn_out, mean_out, max_out]:
            assert out.shape == (4, 16, 1024)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_different_aggregations(self):
        """Test pooling methods produce different results."""
        features = torch.randn(4, 100, 1024)
        assignments = torch.randn(4, 100, 16).softmax(dim=-1)

        attn_pooling = RegionAttentionPooling(feature_dim=1024)
        mean_pooling = RegionMeanPooling()
        max_pooling = RegionMaxPooling()

        attn_out = attn_pooling(features, assignments)
        mean_out = mean_pooling(features, assignments)
        max_out = max_pooling(features, assignments)

        # Should be different
        assert not torch.allclose(attn_out, mean_out)
        assert not torch.allclose(mean_out, max_out)
        assert not torch.allclose(attn_out, max_out)


class TestRegionTransformer:
    """Test region transformer."""

    def test_init(self):
        """Test initialization."""
        from src.models.hierarchical_pooling import RegionTransformer

        transformer = RegionTransformer(
            feature_dim=1024,
            num_layers=2,
            num_heads=8,
        )

        assert transformer.feature_dim == 1024
        assert transformer.num_layers == 2
        assert transformer.num_heads == 8

    def test_invalid_inputs(self):
        """Test input validation."""
        from src.models.hierarchical_pooling import RegionTransformer

        with pytest.raises(ValueError, match="feature_dim must be positive"):
            RegionTransformer(feature_dim=0)

        with pytest.raises(ValueError, match="num_layers must be positive"):
            RegionTransformer(feature_dim=1024, num_layers=0)

        with pytest.raises(ValueError, match="num_heads must be positive"):
            RegionTransformer(feature_dim=1024, num_heads=0)

        with pytest.raises(ValueError, match="must be divisible by num_heads"):
            RegionTransformer(feature_dim=1024, num_heads=7)

        with pytest.raises(ValueError, match="mlp_ratio must be positive"):
            RegionTransformer(feature_dim=1024, mlp_ratio=0)

        with pytest.raises(ValueError, match="dropout must be in"):
            RegionTransformer(feature_dim=1024, dropout=1.5)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        from src.models.hierarchical_pooling import RegionTransformer

        transformer = RegionTransformer(feature_dim=1024, num_layers=2)
        region_features = torch.randn(4, 16, 1024)

        output = transformer(region_features)

        assert output.shape == (4, 16, 1024)

    def test_forward_with_mask(self):
        """Test forward with mask."""
        from src.models.hierarchical_pooling import RegionTransformer

        transformer = RegionTransformer(feature_dim=1024, num_layers=2)
        region_features = torch.randn(4, 16, 1024)
        mask = torch.ones(4, 16, dtype=torch.bool)
        mask[:, 8:] = False  # Mask half

        output = transformer(region_features, mask=mask)

        assert output.shape == (4, 16, 1024)
        assert not torch.isnan(output).any()

    def test_positional_encoding(self):
        """Test positional encoding."""
        from src.models.hierarchical_pooling import RegionTransformer

        transformer = RegionTransformer(
            feature_dim=1024,
            num_layers=2,
            use_pos_encoding=True,
        )

        region_features = torch.randn(4, 16, 1024)
        region_centers = torch.rand(16, 2)

        output = transformer(region_features, region_centers=region_centers)

        assert output.shape == (4, 16, 1024)

    def test_positional_encoding_required(self):
        """Test positional encoding requires centers."""
        from src.models.hierarchical_pooling import RegionTransformer

        transformer = RegionTransformer(
            feature_dim=1024,
            use_pos_encoding=True,
        )

        region_features = torch.randn(4, 16, 1024)

        with pytest.raises(ValueError, match="region_centers required"):
            transformer(region_features)

    def test_gradients_flow(self):
        """Test gradients flow through transformer."""
        from src.models.hierarchical_pooling import RegionTransformer

        transformer = RegionTransformer(feature_dim=1024, num_layers=2)
        region_features = torch.randn(4, 16, 1024, requires_grad=True)

        output = transformer(region_features)
        loss = output.sum()
        loss.backward()

        assert region_features.grad is not None

    def test_multiple_layers(self):
        """Test different number of layers."""
        from src.models.hierarchical_pooling import RegionTransformer

        region_features = torch.randn(4, 16, 1024)

        for num_layers in [1, 2, 4]:
            transformer = RegionTransformer(
                feature_dim=1024,
                num_layers=num_layers,
            )
            output = transformer(region_features)
            assert output.shape == (4, 16, 1024)

    def test_different_num_heads(self):
        """Test different number of attention heads."""
        from src.models.hierarchical_pooling import RegionTransformer

        region_features = torch.randn(4, 16, 1024)

        for num_heads in [4, 8, 16]:
            transformer = RegionTransformer(
                feature_dim=1024,
                num_heads=num_heads,
            )
            output = transformer(region_features)
            assert output.shape == (4, 16, 1024)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
