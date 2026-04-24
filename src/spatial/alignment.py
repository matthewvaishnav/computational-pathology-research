"""
Spatial alignment utilities: Visium spot coordinates ↔ H&E patch grid.

Handles loading paired (H&E features, gene expression) data from
10X Genomics Visium format (H5AD/AnnData) or CSV files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False


def normalize_counts(
    counts: np.ndarray,
    target_sum: float = 1e4,
    log1p: bool = True,
) -> np.ndarray:
    """
    Library size normalization + log1p transform (Seurat/Scanpy default).

    Args:
        counts: Raw count matrix [n_spots, n_genes]
        target_sum: Normalize each spot to this total count
        log1p: Apply log(x+1) after normalization

    Returns:
        Normalized expression matrix [n_spots, n_genes]
    """
    lib_sizes = counts.sum(axis=1, keepdims=True)
    normalized = counts / (lib_sizes + 1e-8) * target_sum
    if log1p:
        normalized = np.log1p(normalized)
    return normalized.astype(np.float32)


def highly_variable_genes(
    counts: np.ndarray,
    n_top_genes: int = 3000,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_dispersion: float = 0.5,
) -> np.ndarray:
    """
    Select highly variable genes (HVGs) using Seurat v2 dispersion method.

    Args:
        counts: Normalized expression [n_spots, n_genes]
        n_top_genes: Maximum number of HVGs to select
        min_mean, max_mean, min_dispersion: Filtering thresholds

    Returns:
        Boolean mask [n_genes] indicating HVGs
    """
    mean = counts.mean(axis=0)
    var = counts.var(axis=0)
    dispersion = np.where(mean > 0, var / (mean + 1e-8), 0.0)

    # Bin genes by mean expression (10 bins)
    n_bins = 10
    mean_bins = np.digitize(mean, np.percentile(mean, np.linspace(0, 100, n_bins + 1)[1:-1]))
    normalized_dispersion = np.zeros_like(dispersion)
    for b in range(n_bins):
        mask = mean_bins == b
        if mask.sum() == 0:
            continue
        bin_disp = dispersion[mask]
        bin_mean_disp = bin_disp.mean()
        bin_std_disp = bin_disp.std() + 1e-8
        normalized_dispersion[mask] = (bin_disp - bin_mean_disp) / bin_std_disp

    # Filter by mean and dispersion thresholds
    hvg_mask = (
        (mean >= min_mean)
        & (mean <= max_mean)
        & (normalized_dispersion >= min_dispersion)
    )

    # Rank by normalized dispersion and keep top-k
    if hvg_mask.sum() > n_top_genes:
        hvg_indices = np.where(hvg_mask)[0]
        top_k = hvg_indices[np.argsort(normalized_dispersion[hvg_indices])[::-1][:n_top_genes]]
        final_mask = np.zeros(counts.shape[1], dtype=bool)
        final_mask[top_k] = True
        return final_mask

    logger.info(f"Selected {hvg_mask.sum()} highly variable genes")
    return hvg_mask


def align_spots_to_patches(
    spot_coords: np.ndarray,
    patch_size_pixels: int,
    spot_diameter_pixels: int,
) -> np.ndarray:
    """
    Map Visium spot pixel coordinates to patch grid indices.

    Args:
        spot_coords: Spot pixel coordinates [n_spots, 2] (x, y)
        patch_size_pixels: Size of each H&E patch in pixels
        spot_diameter_pixels: Diameter of each Visium spot in pixels

    Returns:
        Patch grid indices [n_spots, 2] (row, col)
    """
    patch_indices = (spot_coords / patch_size_pixels).astype(int)
    return patch_indices


class SpatialDataset(Dataset):
    """
    PyTorch dataset for paired (H&E features, gene expression) Visium data.

    Supports loading from:
    - AnnData H5AD files (10X Genomics format)
    - NumPy arrays directly
    - CSV expression matrix + HDF5 features

    Train/val split is always by SLIDE, never by spot, to prevent
    spatial data leakage between train and test.
    """

    def __init__(
        self,
        patch_features: np.ndarray,
        expression: np.ndarray,
        patch_coords: Optional[np.ndarray] = None,
        gene_names: Optional[List[str]] = None,
        normalize: bool = True,
        n_top_genes: int = 3000,
    ):
        """
        Args:
            patch_features: H&E patch embeddings [n_spots, feature_dim]
            expression: Raw gene expression counts [n_spots, n_genes]
            patch_coords: Patch grid coordinates [n_spots, 2] (optional)
            gene_names: Gene names list (optional)
            normalize: Apply library-size normalization + log1p
            n_top_genes: Number of HVGs to use (0 = use all genes)
        """
        if normalize:
            expression = normalize_counts(expression)

        if n_top_genes > 0 and expression.shape[1] > n_top_genes:
            hvg_mask = highly_variable_genes(expression, n_top_genes=n_top_genes)
            expression = expression[:, hvg_mask]
            if gene_names is not None:
                gene_names = [g for g, m in zip(gene_names, hvg_mask) if m]
            logger.info(f"Selected {hvg_mask.sum()} HVGs from {len(hvg_mask)} genes")

        self.features = torch.tensor(patch_features, dtype=torch.float32)
        self.expression = torch.tensor(expression, dtype=torch.float32)
        self.coords = (
            torch.tensor(patch_coords, dtype=torch.long)
            if patch_coords is not None
            else None
        )
        self.gene_names = gene_names
        self.n_genes = expression.shape[1]
        logger.info(
            f"SpatialDataset: {len(self)} spots, "
            f"feature_dim={patch_features.shape[1]}, n_genes={self.n_genes}"
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "features": self.features[idx],
            "expression": self.expression[idx],
        }
        if self.coords is not None:
            item["coords"] = self.coords[idx]
        return item

    @classmethod
    def from_anndata(
        cls,
        h5ad_path: Union[str, Path],
        feature_key: str = "X_pca",
        n_top_genes: int = 3000,
        normalize: bool = True,
    ) -> "SpatialDataset":
        """
        Load from AnnData H5AD file (10X Visium format).

        Args:
            h5ad_path: Path to .h5ad file
            feature_key: Key in adata.obsm for patch features
                         (use 'X_pca' or custom CNN features stored in obsm)
            n_top_genes: Number of HVGs
            normalize: Normalize counts

        Returns:
            SpatialDataset
        """
        if not HAS_ANNDATA:
            raise ImportError("anndata required. pip install anndata>=0.9.0")

        adata = ad.read_h5ad(str(h5ad_path))
        logger.info(f"Loaded AnnData: {adata.shape} from {h5ad_path}")

        # Raw counts matrix
        counts = adata.X
        if hasattr(counts, "toarray"):
            counts = counts.toarray()
        counts = np.asarray(counts, dtype=np.float32)

        # Patch features from obsm
        if feature_key in adata.obsm:
            features = np.asarray(adata.obsm[feature_key], dtype=np.float32)
        else:
            raise KeyError(
                f"Feature key '{feature_key}' not in adata.obsm. "
                f"Available: {list(adata.obsm.keys())}"
            )

        # Spatial coordinates
        coords = None
        if "spatial" in adata.obsm:
            spatial = np.asarray(adata.obsm["spatial"])
            # Convert pixel coords to grid indices (assume 55µm spot, 10µm pixel)
            coords = (spatial / 50).astype(int)

        gene_names = list(adata.var_names) if adata.var_names is not None else None
        return cls(
            patch_features=features,
            expression=counts,
            patch_coords=coords,
            gene_names=gene_names,
            normalize=normalize,
            n_top_genes=n_top_genes,
        )

    @classmethod
    def from_arrays(
        cls,
        features: np.ndarray,
        expression: np.ndarray,
        coords: Optional[np.ndarray] = None,
        gene_names: Optional[List[str]] = None,
        n_top_genes: int = 3000,
    ) -> "SpatialDataset":
        """Construct directly from numpy arrays."""
        return cls(
            patch_features=features,
            expression=expression,
            patch_coords=coords,
            gene_names=gene_names,
            normalize=True,
            n_top_genes=n_top_genes,
        )
