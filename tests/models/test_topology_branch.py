"""
Unit tests for topology branch (k-NN graph + GNN).

Tests:
- KNNGraphBuilder: graph construction, edge features
- GNN layers: GATv2, GraphSAGE, GIN
- TopologyBranch: end-to-end forward pass
"""

import pytest
import torch
import torch.nn as nn

# Check if torch_geometric available
try:
    import torch_geometric

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

from src.models.topology_branch import (
    KNNGraphBuilder,
    GATv2Layer,
    GraphSAGELayer,
    GINLayer,
    TopologyBranch,
)


@pytest.mark.skipif(not TORCH_GEOMETRIC_AVAILABLE, reason="torch_geometric not installed")
class TestKNNGraphBuilder:
    """Test k-NN graph construction."""

    def test_basic_graph_construction(self):
        """Test basic k-NN graph building."""
        builder = KNNGraphBuilder(k=4, self_loops=False)

        # Simple 2D grid
        coords = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )

        edge_index, edge_attr = builder(coords)

        # Check shapes
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0

        # Check undirected (each edge appears twice)
        num_edges = edge_index.shape[1]
        assert num_edges % 2 == 0

    def test_self_loops(self):
        """Test self-loop inclusion."""
        coords = torch.rand(10, 2)

        # With self-loops
        builder_with = KNNGraphBuilder(k=4, self_loops=True)
        edge_index_with, _ = builder_with(coords)

        # Check self-loops exist
        self_loops = (edge_index_with[0] == edge_index_with[1]).sum()
        assert self_loops > 0

        # Without self-loops
        builder_without = KNNGraphBuilder(k=4, self_loops=False)
        edge_index_without, _ = builder_without(coords)

        # Check no self-loops
        self_loops = (edge_index_without[0] == edge_index_without[1]).sum()
        assert self_loops == 0

    def test_edge_features(self):
        """Test edge feature computation."""
        builder = KNNGraphBuilder(k=4)

        coords = torch.rand(20, 2)
        features = torch.randn(20, 128)

        edge_index, edge_attr = builder(coords, features)

        # Check edge_attr shape
        assert edge_attr is not None
        assert edge_attr.shape[0] == edge_index.shape[1]
        assert edge_attr.shape[1] == 2  # distance + similarity

        # Check distance >= 0
        distances = edge_attr[:, 0]
        assert (distances >= 0).all()

        # Check similarity in [-1, 1]
        similarities = edge_attr[:, 1]
        assert (similarities >= -1).all()
        assert (similarities <= 1).all()

    def test_batch_construction(self):
        """Test batched graph construction."""
        builder = KNNGraphBuilder(k=8)

        batch_size = 4
        num_patches = 50
        feature_dim = 256

        coords_batch = torch.rand(batch_size, num_patches, 2)
        features_batch = torch.randn(batch_size, num_patches, feature_dim)

        batch = builder.build_batch(coords_batch, features_batch)

        # Check batch
        assert batch.num_graphs == batch_size
        assert batch.x.shape[1] == feature_dim
        assert batch.edge_index.shape[0] == 2

    def test_masked_batch(self):
        """Test batched construction with mask."""
        builder = KNNGraphBuilder(k=8)

        batch_size = 4
        num_patches = 50
        feature_dim = 256

        coords_batch = torch.rand(batch_size, num_patches, 2)
        features_batch = torch.randn(batch_size, num_patches, feature_dim)

        # Create mask (some patches invalid)
        mask = torch.rand(batch_size, num_patches) > 0.2

        batch = builder.build_batch(coords_batch, features_batch, mask)

        # Check batch
        assert batch.num_graphs == batch_size
        # Total nodes should be less than batch_size * num_patches
        assert batch.x.shape[0] < batch_size * num_patches


@pytest.mark.skipif(not TORCH_GEOMETRIC_AVAILABLE, reason="torch_geometric not installed")
class TestGNNLayers:
    """Test GNN layer implementations."""

    def test_gatv2_layer(self):
        """Test GATv2 layer forward pass."""
        layer = GATv2Layer(
            in_channels=128,
            out_channels=128,
            heads=4,
            dropout=0.1,
        )

        # Simple graph
        x = torch.randn(10, 128)
        edge_index = torch.randint(0, 10, (2, 40))

        out = layer(x, edge_index)

        # Check shape
        assert out.shape == (10, 128)

        # Check finite
        assert torch.isfinite(out).all()

    def test_gatv2_with_edge_features(self):
        """Test GATv2 with edge features."""
        layer = GATv2Layer(
            in_channels=128,
            out_channels=128,
            heads=4,
            edge_dim=2,
        )

        x = torch.randn(10, 128)
        edge_index = torch.randint(0, 10, (2, 40))
        edge_attr = torch.randn(40, 2)

        out = layer(x, edge_index, edge_attr)

        assert out.shape == (10, 128)
        assert torch.isfinite(out).all()

    def test_graphsage_layer(self):
        """Test GraphSAGE layer forward pass."""
        layer = GraphSAGELayer(
            in_channels=128,
            out_channels=128,
            aggr="mean",
        )

        x = torch.randn(10, 128)
        edge_index = torch.randint(0, 10, (2, 40))

        out = layer(x, edge_index)

        assert out.shape == (10, 128)
        assert torch.isfinite(out).all()

    def test_gin_layer(self):
        """Test GIN layer forward pass."""
        layer = GINLayer(
            in_channels=128,
            out_channels=128,
        )

        x = torch.randn(10, 128)
        edge_index = torch.randint(0, 10, (2, 40))

        out = layer(x, edge_index)

        assert out.shape == (10, 128)
        assert torch.isfinite(out).all()


@pytest.mark.skipif(not TORCH_GEOMETRIC_AVAILABLE, reason="torch_geometric not installed")
class TestTopologyBranch:
    """Test end-to-end topology branch."""

    @pytest.mark.parametrize("gnn_type", ["gat", "sage", "gin"])
    def test_forward_pass(self, gnn_type):
        """Test forward pass with different GNN types."""
        branch = TopologyBranch(
            feature_dim=512,
            hidden_dim=256,
            num_layers=2,
            k_neighbors=8,
            gnn_type=gnn_type,
            pooling="mean",
        )

        batch_size = 4
        num_patches = 50

        features = torch.randn(batch_size, num_patches, 512)
        coords = torch.rand(batch_size, num_patches, 2)

        bag_features = branch(features, coords)

        # Check shape
        assert bag_features.shape == (batch_size, 256)

        # Check finite
        assert torch.isfinite(bag_features).all()

    @pytest.mark.parametrize("pooling", ["attention", "mean", "max"])
    def test_pooling_methods(self, pooling):
        """Test different pooling methods."""
        branch = TopologyBranch(
            feature_dim=512,
            hidden_dim=256,
            num_layers=2,
            k_neighbors=8,
            gnn_type="gat",
            pooling=pooling,
        )

        batch_size = 4
        num_patches = 50

        features = torch.randn(batch_size, num_patches, 512)
        coords = torch.rand(batch_size, num_patches, 2)

        bag_features = branch(features, coords)

        assert bag_features.shape == (batch_size, 256)
        assert torch.isfinite(bag_features).all()

    def test_with_mask(self):
        """Test forward pass with mask."""
        branch = TopologyBranch(
            feature_dim=512,
            hidden_dim=256,
            num_layers=2,
            k_neighbors=8,
            gnn_type="gat",
            pooling="attention",
        )

        batch_size = 4
        num_patches = 50

        features = torch.randn(batch_size, num_patches, 512)
        coords = torch.rand(batch_size, num_patches, 2)
        mask = torch.rand(batch_size, num_patches) > 0.2

        bag_features = branch(features, coords, mask)

        assert bag_features.shape == (batch_size, 256)
        assert torch.isfinite(bag_features).all()

    def test_gradient_flow(self):
        """Test gradient flow through topology branch."""
        branch = TopologyBranch(
            feature_dim=512,
            hidden_dim=256,
            num_layers=2,
            k_neighbors=8,
            gnn_type="gat",
            pooling="attention",
        )

        features = torch.randn(2, 30, 512, requires_grad=True)
        coords = torch.rand(2, 30, 2)

        bag_features = branch(features, coords)
        loss = bag_features.sum()
        loss.backward()

        # Check gradients exist
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    def test_different_k_neighbors(self):
        """Test different k values."""
        for k in [4, 8, 16]:
            branch = TopologyBranch(
                feature_dim=512,
                hidden_dim=256,
                num_layers=2,
                k_neighbors=k,
                gnn_type="gat",
                pooling="mean",
            )

            features = torch.randn(2, 50, 512)
            coords = torch.rand(2, 50, 2)

            bag_features = branch(features, coords)

            assert bag_features.shape == (2, 256)
            assert torch.isfinite(bag_features).all()

    def test_invalid_gnn_type(self):
        """Test invalid GNN type raises error."""
        with pytest.raises(ValueError, match="gnn_type must be"):
            TopologyBranch(
                feature_dim=512,
                hidden_dim=256,
                gnn_type="invalid",
            )

    def test_invalid_pooling(self):
        """Test invalid pooling raises error."""
        with pytest.raises(ValueError, match="pooling must be"):
            TopologyBranch(
                feature_dim=512,
                hidden_dim=256,
                pooling="invalid",
            )


@pytest.mark.skipif(not TORCH_GEOMETRIC_AVAILABLE, reason="torch_geometric not installed")
class TestTopologyBranchIntegration:
    """Integration tests for topology branch."""

    def test_variable_bag_sizes(self):
        """Test with variable bag sizes (via masking)."""
        branch = TopologyBranch(
            feature_dim=512,
            hidden_dim=256,
            num_layers=2,
            k_neighbors=8,
            gnn_type="gat",
            pooling="attention",
        )

        batch_size = 4
        max_patches = 100

        features = torch.randn(batch_size, max_patches, 512)
        coords = torch.rand(batch_size, max_patches, 2)

        # Variable sizes: 50, 75, 100, 60
        mask = torch.zeros(batch_size, max_patches, dtype=torch.bool)
        mask[0, :50] = True
        mask[1, :75] = True
        mask[2, :100] = True
        mask[3, :60] = True

        bag_features = branch(features, coords, mask)

        assert bag_features.shape == (batch_size, 256)
        assert torch.isfinite(bag_features).all()

    def test_deterministic_output(self):
        """Test deterministic output with same input."""
        torch.manual_seed(42)

        branch = TopologyBranch(
            feature_dim=512,
            hidden_dim=256,
            num_layers=2,
            k_neighbors=8,
            gnn_type="gat",
            pooling="mean",
        )
        branch.eval()

        features = torch.randn(2, 50, 512)
        coords = torch.rand(2, 50, 2)

        # Two forward passes
        with torch.no_grad():
            out1 = branch(features, coords)
            out2 = branch(features, coords)

        # Should be identical
        assert torch.allclose(out1, out2, atol=1e-6)
