"""
Spatial Tissue Graph Construction

Build spatial graphs from segmented cells for GNN-based analysis.
Models cell-cell interactions and tissue architecture.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.spatial import Delaunay, distance_matrix
    from sklearn.neighbors import NearestNeighbors

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy/sklearn not available")

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("networkx not available. Install: pip install networkx")


@dataclass
class CellNode:
    """Cell node in tissue graph."""

    id: int
    centroid: Tuple[float, float]
    area: float
    features: Dict[str, float]


class TissueGraph:
    """
    Spatial graph representation of tissue.

    Nodes = cells, Edges = spatial relationships
    """

    def __init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx required. Install: pip install networkx")
        self.graph = nx.Graph()
        self.cells = {}

    def add_cell(self, cell: CellNode):
        """Add cell node to graph."""
        self.graph.add_node(cell.id, **cell.features)
        self.cells[cell.id] = cell

    def add_edge(self, cell1_id: int, cell2_id: int, distance: float):
        """Add spatial edge between cells."""
        self.graph.add_edge(cell1_id, cell2_id, distance=distance)

    def get_adjacency_matrix(self) -> np.ndarray:
        """Get graph adjacency matrix."""
        return nx.to_numpy_array(self.graph)

    def get_node_features(self) -> np.ndarray:
        """Get node feature matrix."""
        features = []
        for node_id in sorted(self.graph.nodes()):
            node_data = self.graph.nodes[node_id]
            features.append([node_data.get(k, 0) for k in sorted(node_data.keys())])
        return np.array(features)

    def compute_graph_metrics(self) -> Dict[str, float]:
        """Compute graph-level metrics."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_degree": np.mean([d for n, d in self.graph.degree()]),
            "avg_clustering": nx.average_clustering(self.graph),
        }


class TissueGraphBuilder:
    """
    Build tissue graphs from segmented cells.

    Supports multiple edge construction methods:
    - KNN: K nearest neighbors
    - Radius: All cells within radius
    - Delaunay: Delaunay triangulation
    """

    def __init__(
        self,
        method: str = "knn",
        k: int = 5,
        radius: float = 100.0,
        max_distance: Optional[float] = None,
    ):
        """
        Initialize graph builder.

        Args:
            method: Edge construction method ("knn", "radius", "delaunay")
            k: Number of neighbors for KNN
            radius: Radius for radius-based edges
            max_distance: Maximum edge distance (optional filter)
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy/sklearn required")

        self.method = method
        self.k = k
        self.radius = radius
        self.max_distance = max_distance

    def build_from_segmentation(
        self, labels: np.ndarray, features: Optional[List[Dict]] = None
    ) -> TissueGraph:
        """
        Build graph from segmentation mask.

        Args:
            labels: Segmentation mask (H, W) with cell IDs
            features: Optional list of feature dicts per cell

        Returns:
            TissueGraph object
        """
        from skimage.measure import regionprops

        # Extract cell properties
        props = regionprops(labels)

        # Create graph
        graph = TissueGraph()

        # Add nodes
        centroids = []
        for i, prop in enumerate(props):
            cell_id = prop.label

            # Get features
            cell_features = {
                "area": prop.area,
                "perimeter": prop.perimeter,
                "eccentricity": prop.eccentricity,
                "solidity": prop.solidity,
            }

            # Add custom features if provided
            if features and i < len(features):
                cell_features.update(features[i])

            # Create cell node
            cell = CellNode(
                id=cell_id, centroid=prop.centroid, area=prop.area, features=cell_features
            )

            graph.add_cell(cell)
            centroids.append(prop.centroid)

        # Build edges
        centroids = np.array(centroids)
        edges = self._build_edges(centroids)

        # Add edges to graph
        for i, j, dist in edges:
            cell1_id = props[i].label
            cell2_id = props[j].label
            graph.add_edge(cell1_id, cell2_id, dist)

        return graph

    def _build_edges(self, centroids: np.ndarray) -> List[Tuple[int, int, float]]:
        """Build edges based on method."""
        if self.method == "knn":
            return self._build_knn_edges(centroids)
        elif self.method == "radius":
            return self._build_radius_edges(centroids)
        elif self.method == "delaunay":
            return self._build_delaunay_edges(centroids)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _build_knn_edges(self, centroids: np.ndarray) -> List[Tuple[int, int, float]]:
        """Build K-nearest neighbor edges."""
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(centroids)
        distances, indices = nbrs.kneighbors(centroids)

        edges = []
        for i in range(len(centroids)):
            for j, dist in zip(indices[i][1:], distances[i][1:]):  # Skip self
                if self.max_distance is None or dist <= self.max_distance:
                    edges.append((i, j, dist))

        return edges

    def _build_radius_edges(self, centroids: np.ndarray) -> List[Tuple[int, int, float]]:
        """Build radius-based edges."""
        nbrs = NearestNeighbors(radius=self.radius).fit(centroids)
        distances, indices = nbrs.radius_neighbors(centroids)

        edges = []
        for i in range(len(centroids)):
            for j, dist in zip(indices[i], distances[i]):
                if i < j:  # Avoid duplicates
                    if self.max_distance is None or dist <= self.max_distance:
                        edges.append((i, j, dist))

        return edges

    def _build_delaunay_edges(self, centroids: np.ndarray) -> List[Tuple[int, int, float]]:
        """Build Delaunay triangulation edges."""
        tri = Delaunay(centroids)

        edges = set()
        for simplex in tri.simplices:
            # Add all edges of triangle
            for i in range(3):
                for j in range(i + 1, 3):
                    idx1, idx2 = simplex[i], simplex[j]
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                    edges.add((idx1, idx2))

        # Compute distances
        edge_list = []
        for i, j in edges:
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if self.max_distance is None or dist <= self.max_distance:
                edge_list.append((i, j, dist))

        return edge_list


def build_tissue_graph(
    labels: np.ndarray, method: str = "knn", k: int = 5, features: Optional[List[Dict]] = None
) -> TissueGraph:
    """
    Convenience function to build tissue graph.

    Args:
        labels: Segmentation mask
        method: Edge construction method
        k: Number of neighbors for KNN
        features: Optional cell features

    Returns:
        TissueGraph object
    """
    builder = TissueGraphBuilder(method=method, k=k)
    return builder.build_from_segmentation(labels, features)
