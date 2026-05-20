"""
Graph Cache: Precompute and cache k-NN graphs in HDF5.

For large datasets, building k-NN graphs on-the-fly is expensive.
This module precomputes graphs offline and caches them in HDF5 for fast loading.

Key features:
- Precompute graphs for entire dataset
- Store edge_index + edge_attr in HDF5
- Fast random access during training
- Automatic cache invalidation on config change

Usage:
    # Precompute graphs
    cache = GraphCache(cache_dir='data/graph_cache', k=8)
    cache.build_cache(dataset, coords_key='coords', features_key='features')

    # Load during training
    edge_index, edge_attr = cache.load_graph(slide_id)

Reference:
- TransnnMIL v2.0: Hierarchical + Topology (2027)
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.models.transnnmil.topology_branch import KNNGraphBuilder


class GraphCache:
    """
    Graph cache for precomputed k-NN graphs.

    Args:
        cache_dir: Directory to store cache files
        k: Number of neighbors for k-NN
        use_faiss: Use FAISS for approximate k-NN
        self_loops: Include self-loops
        directed: Directed edges

    Example:
        >>> cache = GraphCache(cache_dir='data/graph_cache', k=8)
        >>>
        >>> # Build cache
        >>> cache.build_cache(dataset)
        >>>
        >>> # Load graph
        >>> edge_index, edge_attr = cache.load_graph('slide_001')
    """

    def __init__(
        self,
        cache_dir: str,
        k: int = 8,
        use_faiss: bool = False,
        self_loops: bool = True,
        directed: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.k = k
        self.use_faiss = use_faiss
        self.self_loops = self_loops
        self.directed = directed

        # Graph builder
        self.graph_builder = KNNGraphBuilder(
            k=k,
            self_loops=self_loops,
            directed=directed,
            use_faiss=use_faiss,
        )

        # Config hash for cache invalidation
        self.config_hash = self._compute_config_hash()
        self.cache_file = self.cache_dir / f"graphs_{self.config_hash}.h5"
        self.metadata_file = self.cache_dir / f"metadata_{self.config_hash}.json"

    def _compute_config_hash(self) -> str:
        """Compute hash of graph config for cache invalidation."""
        config = {
            "k": self.k,
            "use_faiss": self.use_faiss,
            "self_loops": self.self_loops,
            "directed": self.directed,
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode(), usedforsecurity=False).hexdigest()[:8]

    def build_cache(
        self,
        dataset,
        coords_key: str = "coords",
        features_key: str = "features",
        id_key: str = "slide_id",
        force_rebuild: bool = False,
    ):
        """
        Build graph cache for entire dataset.

        Args:
            dataset: Dataset with __len__ and __getitem__
            coords_key: Key for coordinates in dataset items
            features_key: Key for features in dataset items
            id_key: Key for slide ID in dataset items
            force_rebuild: Force rebuild even if cache exists
        """
        # Check if cache exists
        if self.cache_file.exists() and not force_rebuild:
            print(f"Cache already exists: {self.cache_file}")
            return

        print(f"Building graph cache: {self.cache_file}")
        print(f"Config: k={self.k}, use_faiss={self.use_faiss}")

        # Create HDF5 file
        with h5py.File(self.cache_file, "w") as f:
            metadata = {}

            # Process each sample
            for idx in tqdm(range(len(dataset)), desc="Building graphs"):
                sample = dataset[idx]

                # Extract data
                slide_id = sample.get(id_key, f"sample_{idx}")
                coords = sample[coords_key]  # [N, 2]
                features = sample.get(features_key, None)  # [N, D] or None

                # Convert to torch if needed
                if isinstance(coords, np.ndarray):
                    coords = torch.from_numpy(coords).float()
                if features is not None and isinstance(features, np.ndarray):
                    features = torch.from_numpy(features).float()

                # Build graph
                edge_index, edge_attr = self.graph_builder(coords, features)

                # Store in HDF5
                grp = f.create_group(slide_id)
                grp.create_dataset("edge_index", data=edge_index.cpu().numpy(), compression="gzip")

                if edge_attr is not None:
                    grp.create_dataset(
                        "edge_attr", data=edge_attr.cpu().numpy(), compression="gzip"
                    )

                # Metadata
                metadata[slide_id] = {
                    "num_nodes": coords.shape[0],
                    "num_edges": edge_index.shape[1],
                    "has_edge_attr": edge_attr is not None,
                }

        # Save metadata
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Cache built: {len(metadata)} graphs")

    def load_graph(
        self,
        slide_id: str,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Load precomputed graph from cache.

        Args:
            slide_id: Slide identifier
            device: Device to load tensors to

        Returns:
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, 2] or None
        """
        if not self.cache_file.exists():
            raise FileNotFoundError(f"Cache not found: {self.cache_file}")

        with h5py.File(self.cache_file, "r") as f:
            if slide_id not in f:
                raise KeyError(f"Slide not found in cache: {slide_id}")

            grp = f[slide_id]

            # Load edge_index
            edge_index = torch.from_numpy(grp["edge_index"][:])

            # Load edge_attr if exists
            edge_attr = None
            if "edge_attr" in grp:
                edge_attr = torch.from_numpy(grp["edge_attr"][:])

        # Move to device
        if device is not None:
            edge_index = edge_index.to(device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device)

        return edge_index, edge_attr

    def get_metadata(self, slide_id: Optional[str] = None) -> Dict:
        """
        Get cache metadata.

        Args:
            slide_id: Optional slide ID to get specific metadata

        Returns:
            metadata: Metadata dict
        """
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_file}")

        with open(self.metadata_file) as f:
            metadata = json.load(f)

        if slide_id is not None:
            if slide_id not in metadata:
                raise KeyError(f"Slide not found in metadata: {slide_id}")
            return metadata[slide_id]

        return metadata

    def clear_cache(self):
        """Clear all cache files."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            print(f"✓ Deleted: {self.cache_file}")

        if self.metadata_file.exists():
            self.metadata_file.unlink()
            print(f"✓ Deleted: {self.metadata_file}")

    def __repr__(self) -> str:
        exists = "✓" if self.cache_file.exists() else "✗"
        return (
            f"GraphCache(cache_dir={self.cache_dir}, k={self.k}, "
            f"use_faiss={self.use_faiss}, exists={exists})"
        )
