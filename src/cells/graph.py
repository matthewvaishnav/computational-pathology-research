"""
Cell graph construction from detected nuclei.

Spatial kNN + optional Delaunay triangulation → edges with morphological features.
Graph structure captures local tumour microenvironment architecture.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class CellGraph:
    """Graph representation of cells in a tissue patch."""
    node_features: torch.Tensor    # (N, node_dim) per-cell features
    edge_index: torch.Tensor       # (2, E) COO format
    edge_features: torch.Tensor    # (E, edge_dim) per-edge features
    centroids: np.ndarray          # (N, 2) spatial coords for visualisation
    cell_types: Optional[torch.Tensor] = None  # (N,) class labels

    @property
    def num_nodes(self) -> int:
        return self.node_features.size(0)

    @property
    def num_edges(self) -> int:
        return self.edge_index.size(1)


class CellGraphBuilder:
    """
    Constructs a CellGraph from DetectionResult.

    Node features: [mean_r, mean_g, mean_b, area, eccentricity, solidity, cell_type_onehot...]
    Edge features: [distance, delta_row, delta_col, angle]

    Usage:
        builder = CellGraphBuilder(k=8, use_delaunay=True)
        graph = builder.build(detection_result, patch_rgb)
    """

    def __init__(
        self,
        k: int = 8,
        max_distance: float = 200.0,
        use_delaunay: bool = True,
        num_cell_types: int = 0,
    ):
        self.k = k
        self.max_distance = max_distance
        self.use_delaunay = use_delaunay
        self.num_cell_types = num_cell_types

    def build(self, detection_result, patch: np.ndarray) -> Optional[CellGraph]:
        """
        Args:
            detection_result: DetectionResult from NucleusDetector
            patch: (H, W, 3) uint8 RGB — used for colour node features

        Returns:
            CellGraph or None if < 2 nuclei detected
        """
        from .detector import DetectionResult
        result: DetectionResult = detection_result

        N = result.count
        if N < 2:
            return None

        centroids = result.centroids  # (N, 2)

        node_feats = self._node_features(result, patch)
        src, dst = self._build_edges(centroids)

        if len(src) == 0:
            return None

        edge_feats = self._edge_features(centroids, src, dst)

        cell_types = None
        if result.cell_types is not None:
            cell_types = torch.from_numpy(result.cell_types).long()

        return CellGraph(
            node_features=torch.tensor(node_feats, dtype=torch.float32),
            edge_index=torch.tensor(np.array([src, dst]), dtype=torch.long),
            edge_features=torch.tensor(edge_feats, dtype=torch.float32),
            centroids=centroids,
            cell_types=cell_types,
        )

    def _node_features(self, result, patch: np.ndarray) -> np.ndarray:
        N = result.count
        feats = []
        for i in range(N):
            m = result.masks[i]
            # Colour stats
            pixels = patch[m]  # (n_pixels, 3)
            colour = pixels.mean(axis=0) / 255.0 if len(pixels) else np.zeros(3)

            # Shape stats
            area = m.sum()
            try:
                from skimage.measure import regionprops, label as sk_label
                lbl = sk_label(m.astype(np.uint8))
                props = regionprops(lbl)
                if props:
                    ecc = props[0].eccentricity
                    sol = props[0].solidity
                else:
                    ecc, sol = 0.0, 1.0
            except ImportError:
                ecc, sol = 0.0, 1.0

            row = [*colour, np.log1p(area), ecc, sol]

            # One-hot cell type
            if self.num_cell_types > 0:
                oh = np.zeros(self.num_cell_types)
                if result.cell_types is not None:
                    ct = int(result.cell_types[i])
                    if 0 <= ct < self.num_cell_types:
                        oh[ct] = 1.0
                row.extend(oh.tolist())

            feats.append(row)

        return np.array(feats, dtype=np.float32)

    def _build_edges(self, centroids: np.ndarray):
        N = len(centroids)
        src_list, dst_list = [], []

        # kNN edges
        from scipy.spatial import KDTree
        tree = KDTree(centroids)
        k_query = min(self.k + 1, N)
        dists, idxs = tree.query(centroids, k=k_query)

        for i in range(N):
            for j_idx in range(1, k_query):
                j = idxs[i, j_idx]
                d = dists[i, j_idx]
                if d <= self.max_distance:
                    src_list.append(i)
                    dst_list.append(j)

        # Delaunay triangulation edges (denser, still spatially meaningful)
        if self.use_delaunay and N >= 4:
            try:
                from scipy.spatial import Delaunay
                tri = Delaunay(centroids)
                seen = set(zip(src_list, dst_list))
                for simplex in tri.simplices:
                    for a, b in [(0, 1), (1, 2), (0, 2)]:
                        i, j = int(simplex[a]), int(simplex[b])
                        d = np.linalg.norm(centroids[i] - centroids[j])
                        if d <= self.max_distance:
                            for u, v in [(i, j), (j, i)]:
                                if (u, v) not in seen:
                                    src_list.append(u)
                                    dst_list.append(v)
                                    seen.add((u, v))
            except Exception as e:
                logger.debug("Delaunay failed: %s", e)

        return np.array(src_list, dtype=np.int64), np.array(dst_list, dtype=np.int64)

    def _edge_features(self, centroids: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        delta = centroids[dst] - centroids[src]  # (E, 2)
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        angle = np.arctan2(delta[:, 0:1], delta[:, 1:2])  # row/col → angle
        # Normalise distance
        dist_norm = dist / (self.max_distance + 1e-8)
        return np.concatenate([dist_norm, delta / (dist + 1e-8), angle], axis=1).astype(np.float32)
