"""
Topology Branch: Graph Neural Networks for MIL

Implements k-NN graph construction + GNN processing for spatial topology.

Key components:
- k-NN graph builder (PyTorch Geometric)
- GATv2, GraphSAGE, GIN implementations
- Global graph pooling (attention/mean/top-k)

Architecture:
    Input: Patch features [B, N, D] + coordinates [B, N, 2]
    ├─ Build k-NN graph (edges based on spatial proximity)
    ├─ GNN layers (message passing)
    ├─ Global pooling (graph → bag embedding)
    └─ Output: Bag features [B, D]

Reference:
- TransnnMIL v2.0: Hierarchical + Topology (2027)
"""

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss
    import numpy as np

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import torch_geometric
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import (
        GATv2Conv,
        GINConv,
        SAGEConv,
        global_max_pool,
        global_mean_pool,
    )

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    # Dummy types for type hints
    Batch = None
    Data = None
    print("Warning: torch_geometric not available. Topology branch disabled.")
    if TYPE_CHECKING:
        from torch_geometric.data import Batch, Data
    else:
        Data = None
        Batch = None


class KNNGraphBuilder(nn.Module):
    """
    k-NN graph builder for spatial topology.

    Constructs k-nearest neighbor graph based on patch coordinates.
    Each patch connected to k closest neighbors.

    Args:
        k: Number of neighbors per node
        self_loops: Include self-loops (default: True)
        directed: Directed edges (default: False, undirected)

    Example:
        >>> builder = KNNGraphBuilder(k=8)
        >>> coords = torch.rand(100, 2)  # [N, 2]
        >>> features = torch.randn(100, 512)  # [N, D]
        >>>
        >>> # Build graph
        >>> edge_index, edge_attr = builder(coords, features)
        >>> # edge_index: [2, E] (source, target)
        >>> # edge_attr: [E, 2] (distance, similarity)

    Notes:
        - Uses Euclidean distance in coordinate space
        - Edge features: [distance, cosine similarity]
        - Undirected: each edge appears twice (i→j, j→i)
    """

    def __init__(
        self,
        k: int = 8,
        self_loops: bool = True,
        directed: bool = False,
        use_faiss: bool = False,
        faiss_threshold: int = 1000,
    ):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for KNNGraphBuilder")

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        self.k = k
        self.self_loops = self_loops
        self.directed = directed
        self.use_faiss = use_faiss
        self.faiss_threshold = faiss_threshold

        if use_faiss and not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu or faiss-gpu required for approximate k-NN")

    def forward(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Build k-NN graph from coordinates.

        Args:
            coords: Patch coordinates [num_patches, 2]
            features: Optional patch features [num_patches, feature_dim]
                     Used for edge similarity features

        Returns:
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, 2] (distance, similarity)
                      None if features not provided
        """
        num_patches = coords.shape[0]

        # Use FAISS for large N
        if self.use_faiss and num_patches >= self.faiss_threshold:
            return self._build_graph_faiss(coords, features)
        else:
            return self._build_graph_exact(coords, features)

    def _build_graph_exact(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Exact k-NN using PyTorch (O(N²))."""
        num_patches = coords.shape[0]
        device = coords.device

        # Compute pairwise distances [N, N]
        coords_expanded = coords.unsqueeze(0)  # [1, N, 2]
        coords_t = coords.unsqueeze(1)  # [N, 1, 2]
        distances = torch.norm(coords_expanded - coords_t, dim=-1)  # [N, N]

        # Find k nearest neighbors (exclude self if no self_loops)
        if self.self_loops:
            k_actual = self.k
        else:
            k_actual = self.k + 1  # +1 to exclude self

        # Get top-k nearest (smallest distances)
        _, indices = torch.topk(distances, k=min(k_actual, num_patches), largest=False, dim=1)

        # Remove self-loops if needed
        if not self.self_loops:
            # Filter out diagonal (self-connections)
            mask = indices != torch.arange(num_patches, device=device).unsqueeze(1)
            indices = indices[mask].view(num_patches, -1)[:, : self.k]

        # Build edge_index [2, E]
        source = torch.arange(num_patches, device=device).unsqueeze(1).expand(-1, self.k)
        target = indices

        edge_index = torch.stack([source.flatten(), target.flatten()], dim=0)

        # Make undirected (add reverse edges)
        if not self.directed:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # Compute edge features
        edge_attr = None
        if features is not None:
            # Distance feature
            src_coords = coords[edge_index[0]]  # [E, 2]
            tgt_coords = coords[edge_index[1]]  # [E, 2]
            edge_distances = torch.norm(src_coords - tgt_coords, dim=-1, keepdim=True)  # [E, 1]

            # Cosine similarity feature
            src_features = features[edge_index[0]]  # [E, D]
            tgt_features = features[edge_index[1]]  # [E, D]
            edge_similarity = F.cosine_similarity(src_features, tgt_features, dim=-1).unsqueeze(
                -1
            )  # [E, 1]

            # Concatenate
            edge_attr = torch.cat([edge_distances, edge_similarity], dim=-1)  # [E, 2]

        return edge_index, edge_attr

    def _build_graph_faiss(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Approximate k-NN using FAISS (O(N log N))."""
        num_patches = coords.shape[0]
        device = coords.device

        # Convert to numpy for FAISS
        coords_np = coords.cpu().numpy().astype(np.float32)

        # Build FAISS index
        index = faiss.IndexFlatL2(2)  # L2 distance, 2D coords
        index.add(coords_np)

        # Search k+1 neighbors (includes self)
        k_search = self.k + 1 if not self.self_loops else self.k
        distances_np, indices_np = index.search(coords_np, k_search)

        # Convert back to torch
        indices = torch.from_numpy(indices_np).to(device)

        # Remove self-loops if needed
        if not self.self_loops:
            # First column is self, remove it
            indices = indices[:, 1:]

        # Build edge_index [2, E]
        source = torch.arange(num_patches, device=device).unsqueeze(1).expand(-1, self.k)
        target = indices[:, : self.k]

        edge_index = torch.stack([source.flatten(), target.flatten()], dim=0)

        # Make undirected (add reverse edges)
        if not self.directed:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # Compute edge features
        edge_attr = None
        if features is not None:
            # Distance feature
            src_coords = coords[edge_index[0]]  # [E, 2]
            tgt_coords = coords[edge_index[1]]  # [E, 2]
            edge_distances = torch.norm(src_coords - tgt_coords, dim=-1, keepdim=True)  # [E, 1]

            # Cosine similarity feature
            src_features = features[edge_index[0]]  # [E, D]
            tgt_features = features[edge_index[1]]  # [E, D]
            edge_similarity = F.cosine_similarity(src_features, tgt_features, dim=-1).unsqueeze(
                -1
            )  # [E, 1]

            # Concatenate
            edge_attr = torch.cat([edge_distances, edge_similarity], dim=-1)  # [E, 2]

        return edge_index, edge_attr

    def build_batch(
        self,
        coords_batch: torch.Tensor,
        features_batch: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Build batched PyG graphs.

        Args:
            coords_batch: Coordinates [batch_size, num_patches, 2]
            features_batch: Features [batch_size, num_patches, feature_dim]
            mask: Optional mask [batch_size, num_patches]

        Returns:
            batch: PyG Batch object
        """
        batch_size = coords_batch.shape[0]

        graphs = []
        for i in range(batch_size):
            coords = coords_batch[i]
            features = features_batch[i]

            # Apply mask
            if mask is not None:
                valid = mask[i]
                coords = coords[valid]
                features = features[valid]

            # Build graph
            edge_index, edge_attr = self(coords, features)

            # Create PyG Data
            data = Data(
                x=features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=coords,
            )
            graphs.append(data)

        # Batch graphs
        batch = Batch.from_data_list(graphs)
        return batch


class GATv2Layer(nn.Module):
    """
    Graph Attention Network v2 layer.

    Improved attention mechanism with better expressiveness.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
        edge_dim: Edge feature dimension (default: None)

    Example:
        >>> gat = GATv2Layer(in_channels=512, out_channels=512, heads=8)
        >>> x = torch.randn(100, 512)  # [N, D]
        >>> edge_index = torch.randint(0, 100, (2, 400))  # [2, E]
        >>>
        >>> out = gat(x, edge_index)  # [N, D]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for GATv2Layer")

        self.conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels // heads,  # Per-head dimension
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=True,  # Concatenate heads
        )

        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim]

        Returns:
            out: Updated node features [num_nodes, out_channels]
        """
        # GAT conv
        out = self.conv(x, edge_index, edge_attr=edge_attr)

        # Residual + norm
        if x.shape[-1] == out.shape[-1]:
            out = out + x
        out = self.norm(out)
        out = self.dropout(out)

        return out


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE layer (sample + aggregate).

    Efficient neighborhood aggregation via sampling.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        aggr: Aggregation method ('mean', 'max', 'add')
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for GraphSAGELayer")

        self.conv = SAGEConv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr=aggr,
        )

        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        out = self.conv(x, edge_index)

        # Residual + norm
        if x.shape[-1] == out.shape[-1]:
            out = out + x
        out = self.norm(out)
        out = self.dropout(out)

        return out


class GINLayer(nn.Module):
    """
    Graph Isomorphism Network layer.

    Maximally expressive GNN for graph classification.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for GINLayer")

        # MLP for GIN
        mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )

        self.conv = GINConv(mlp, train_eps=True)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        out = self.conv(x, edge_index)

        # Residual + norm
        if x.shape[-1] == out.shape[-1]:
            out = out + x
        out = self.norm(out)
        out = self.dropout(out)

        return out


class TopologyBranch(nn.Module):
    """
    Topology branch: k-NN graph + GNN.

    Processes spatial topology via graph neural networks.

    Args:
        feature_dim: Patch feature dimension
        hidden_dim: GNN hidden dimension
        num_layers: Number of GNN layers (default: 2)
        k_neighbors: Number of neighbors for k-NN (default: 8)
        gnn_type: GNN architecture ('gat', 'sage', 'gin')
        pooling: Global pooling ('attention', 'mean', 'max')
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> branch = TopologyBranch(
        ...     feature_dim=1024,
        ...     hidden_dim=512,
        ...     num_layers=2,
        ...     k_neighbors=8,
        ...     gnn_type='gat',
        ...     pooling='attention',
        ... )
        >>>
        >>> features = torch.randn(4, 100, 1024)  # [B, N, D]
        >>> coords = torch.rand(4, 100, 2)  # [B, N, 2]
        >>>
        >>> bag_features = branch(features, coords)  # [B, 512]
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        k_neighbors: int = 8,
        gnn_type: str = "gat",
        pooling: str = "attention",
        dropout: float = 0.1,
        use_faiss: bool = False,
    ):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for TopologyBranch")

        if gnn_type not in ["gat", "sage", "gin"]:
            raise ValueError(f"gnn_type must be 'gat', 'sage', or 'gin', got {gnn_type}")
        if pooling not in ["attention", "mean", "max"]:
            raise ValueError(f"pooling must be 'attention', 'mean', or 'max', got {pooling}")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors
        self.gnn_type = gnn_type
        self.pooling = pooling

        # k-NN graph builder
        self.graph_builder = KNNGraphBuilder(k=k_neighbors, use_faiss=use_faiss)

        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == "gat":
                layer = GATv2Layer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=8,
                    dropout=dropout,
                    edge_dim=2,  # distance + similarity
                )
            elif gnn_type == "sage":
                layer = GraphSAGELayer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggr="mean",
                    dropout=dropout,
                )
            elif gnn_type == "gin":
                layer = GINLayer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    dropout=dropout,
                )
            self.gnn_layers.append(layer)

        # Global pooling
        if pooling == "attention":
            self.pool_attn = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
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
            bag_features: Bag-level features [batch_size, hidden_dim]
        """
        features.shape[0]

        # Project features
        features = self.input_proj(features)  # [B, N, H]

        # Build batched graphs
        batch = self.graph_builder.build_batch(coords, features, mask)

        # GNN layers
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr if hasattr(batch, "edge_attr") else None

        for layer in self.gnn_layers:
            if self.gnn_type == "gat" and edge_attr is not None:
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x, edge_index)

        # Global pooling
        if self.pooling == "attention":
            # Attention pooling
            attn_scores = self.pool_attn(x)  # [total_nodes, 1]
            attn_weights = torch_geometric.utils.softmax(attn_scores, batch.batch, dim=0)
            bag_features = torch_geometric.nn.global_add_pool(x * attn_weights, batch.batch)
        elif self.pooling == "mean":
            bag_features = global_mean_pool(x, batch.batch)
        elif self.pooling == "max":
            bag_features = global_max_pool(x, batch.batch)

        return bag_features
