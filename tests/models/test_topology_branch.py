"""
Unit tests for topology branch (k-NN graph + GNN).

Tests:
- KNNGraphBuilder: graph construction, edge features, FAISS
- GraphCache: precompute, load, invalidation
- GNN layers: GATv2, GraphSAGE, GIN
- TopologyBranch: end-to-end forward pass
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Check if torch_geometric available
try:
    import torch_geometric

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

# Check if FAISS available
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from src.models.graph_cache import GraphCache
from src.models.topology_branch import (
    GATv2Layer,
    GINLayer,
    GraphSAGELayer,
    KNNGraphBuilder,
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

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
    def test_faiss_approximate_knn(self):
        """Test FAISS approximate k-NN."""
        builder = KNNGraphBuilder(k=8, use_faiss=True, faiss_threshold=50)

        # Large graph (triggers FAISS)
        coords = torch.rand(100, 2)
        features = torch.randn(100, 256)

        edge_index, edge_attr = builder(coords, features)

        # Check shapes
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0
        assert edge_attr.shape[0] == edge_index.shape[1]
        assert edge_attr.shape[1] == 2

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
    def test_faiss_vs_exact(self):
        """Test FAISS vs exact k-NN (should be similar)."""
        coords = torch.rand(100, 2)
        features = torch.randn(100, 256)

        # Exact
        builder_exact = KNNGraphBuilder(k=8, use_faiss=False)
        edge_index_exact, _ = builder_exact(coords, features)

        # FAISS
        builder_faiss = KNNGraphBuilder(k=8, use_faiss=True, faiss_threshold=50)
        edge_index_faiss, _ = builder_faiss(coords, features)

        # Should have similar number of edges
        num_edges_exact = edge_index_exact.shape[1]
        num_edges_faiss = edge_index_faiss.shape[1]
        assert abs(num_edges_exact - num_edges_faiss) / num_edges_exact < 0.1  # <10% diff

    def test_small_graph_uses_exact(self):
        """Test small graphs use exact k-NN even with use_faiss=True."""
        builder = KNNGraphBuilder(k=8, use_faiss=True, faiss_threshold=1000)

        # Small graph (below threshold)
        coords = torch.rand(50, 2)
        features = torch.randn(50, 256)

        edge_index, edge_attr = builder(coords, features)

        # Should work (uses exact internally)
        assert edge_index.shape[0] == 2
        assert edge_attr is not None


@pytest.mark.skipif(not TORCH_GEOMETRIC_AVAILABLE, reason="torch_geometric not installed")
class TestGraphCache:
    """Test graph cache for precomputed k-NN graphs."""

    def test_build_and_load_cache(self):
        """Test building and loading graph cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = GraphCache(cache_dir=tmpdir, k=8)

            # Mock dataset
            class MockDataset:
                def __len__(self):
                    return 5

                def __getitem__(self, idx):
                    return {
                        "slide_id": f"slide_{idx:03d}",
                        "coords": torch.rand(50, 2),
                        "features": torch.randn(50, 256),
                    }

            dataset = MockDataset()

            # Build cache
            cache.build_cache(dataset)

            # Check cache exists
            assert cache.cache_file.exists()
            assert cache.metadata_file.exists()

            # Load graph
            edge_index, edge_attr = cache.load_graph("slide_000")

            # Check shapes
            assert edge_index.shape[0] == 2
            assert edge_index.shape[1] > 0
            assert edge_attr.shape[0] == edge_index.shape[1]
            assert edge_attr.shape[1] == 2

    def test_cache_metadata(self):
        """Test cache metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = GraphCache(cache_dir=tmpdir, k=8)

            class MockDataset:
                def __len__(self):
                    return 3

                def __getitem__(self, idx):
                    return {
                        "slide_id": f"slide_{idx:03d}",
                        "coords": torch.rand(50, 2),
                        "features": torch.randn(50, 256),
                    }

            dataset = MockDataset()
            cache.build_cache(dataset)

            # Get metadata
            metadata = cache.get_metadata()
            assert len(metadata) == 3
            assert "slide_000" in metadata
            assert metadata["slide_000"]["num_nodes"] == 50
            assert metadata["slide_000"]["has_edge_attr"] is True

            # Get specific metadata
            slide_meta = cache.get_metadata("slide_001")
            assert slide_meta["num_nodes"] == 50

    def test_cache_invalidation(self):
        """Test cache invalidation on config change."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build cache with k=8
            cache1 = GraphCache(cache_dir=tmpdir, k=8)

            class MockDataset:
                def __len__(self):
                    return 2

                def __getitem__(self, idx):
                    return {
                        "slide_id": f"slide_{idx:03d}",
                        "coords": torch.rand(50, 2),
                        "features": torch.randn(50, 256),
                    }

            dataset = MockDataset()
            cache1.build_cache(dataset)

            cache1_file = cache1.cache_file

            # Build cache with k=16 (different config)
            cache2 = GraphCache(cache_dir=tmpdir, k=16)
            cache2.build_cache(dataset)

            cache2_file = cache2.cache_file

            # Should have different cache files
            assert cache1_file != cache2_file
            assert cache1_file.exists()
            assert cache2_file.exists()

    def test_load_nonexistent_slide(self):
        """Test loading nonexistent slide raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = GraphCache(cache_dir=tmpdir, k=8)

            class MockDataset:
                def __len__(self):
                    return 2

                def __getitem__(self, idx):
                    return {
                        "slide_id": f"slide_{idx:03d}",
                        "coords": torch.rand(50, 2),
                        "features": torch.randn(50, 256),
                    }

            dataset = MockDataset()
            cache.build_cache(dataset)

            # Try to load nonexistent slide
            with pytest.raises(KeyError, match="Slide not found"):
                cache.load_graph("slide_999")

    def test_cache_without_features(self):
        """Test cache with coords only (no features)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = GraphCache(cache_dir=tmpdir, k=8)

            class MockDataset:
                def __len__(self):
                    return 2

                def __getitem__(self, idx):
                    return {
                        "slide_id": f"slide_{idx:03d}",
                        "coords": torch.rand(50, 2),
                    }

            dataset = MockDataset()
            cache.build_cache(dataset)

            # Load graph
            edge_index, edge_attr = cache.load_graph("slide_000")

            # edge_attr should be None (no features)
            assert edge_index.shape[0] == 2
            assert edge_attr is None

    def test_clear_cache(self):
        """Test clearing cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = GraphCache(cache_dir=tmpdir, k=8)

            class MockDataset:
                def __len__(self):
                    return 2

                def __getitem__(self, idx):
                    return {
                        "slide_id": f"slide_{idx:03d}",
                        "coords": torch.rand(50, 2),
                        "features": torch.randn(50, 256),
                    }

            dataset = MockDataset()
            cache.build_cache(dataset)

            # Clear cache
            cache.clear_cache()

            # Check files deleted
            assert not cache.cache_file.exists()
            assert not cache.metadata_file.exists()

    def test_force_rebuild(self):
        """Test force rebuild overwrites existing cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = GraphCache(cache_dir=tmpdir, k=8)

            class MockDataset:
                def __len__(self):
                    return 2

                def __getitem__(self, idx):
                    return {
                        "slide_id": f"slide_{idx:03d}",
                        "coords": torch.rand(50, 2),
                        "features": torch.randn(50, 256),
                    }

            dataset = MockDataset()

            # Build cache
            cache.build_cache(dataset)
            mtime1 = cache.cache_file.stat().st_mtime

            # Build again without force (should skip)
            cache.build_cache(dataset, force_rebuild=False)
            mtime2 = cache.cache_file.stat().st_mtime
            assert mtime1 == mtime2  # Not rebuilt

            # Build with force
            cache.build_cache(dataset, force_rebuild=True)
            mtime3 = cache.cache_file.stat().st_mtime
            assert mtime3 > mtime2  # Rebuilt


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
