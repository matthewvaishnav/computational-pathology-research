"""
Spatial transcriptomics decoder: H&E → gene expression prediction.

Predicts spatial gene expression profiles directly from H&E patch features,
enabling single-cell molecular profiling from standard $2 staining.
"""

from .alignment import (
    SpatialDataset,
    align_spots_to_patches,
    highly_variable_genes,
    normalize_counts,
)
from .decoder import SpatialDecoderLoss, SpatialTranscriptomicsDecoder
from .evaluation import gene_ranking_auroc, mean_pearson, per_gene_pearson, spatial_autocorrelation
from .pretrain import SpatialPretrainer

__all__ = [
    "SpatialTranscriptomicsDecoder",
    "SpatialDecoderLoss",
    "SpatialDataset",
    "align_spots_to_patches",
    "normalize_counts",
    "highly_variable_genes",
    "per_gene_pearson",
    "mean_pearson",
    "spatial_autocorrelation",
    "gene_ranking_auroc",
    "SpatialPretrainer",
]
