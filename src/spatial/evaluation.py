"""
Evaluation metrics for spatial transcriptomics prediction.

Primary metric: mean Pearson r across genes (field standard).
Secondary: Moran's I for spatial coherence, gene-level AUROC.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def per_gene_pearson(
    pred: np.ndarray,
    target: np.ndarray,
    gene_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute Pearson r for each gene between predicted and ground-truth expression.

    Args:
        pred: Predicted expression [n_spots, n_genes]
        target: True expression [n_spots, n_genes]
        gene_names: Optional gene name list for keyed output

    Returns:
        Dict mapping gene name (or index) → Pearson r
    """
    n_genes = pred.shape[1]
    results: Dict[str, float] = {}
    for g in range(n_genes):
        p = pred[:, g]
        t = target[:, g]
        if p.std() < 1e-8 or t.std() < 1e-8:
            r = 0.0
        else:
            r, _ = scipy_stats.pearsonr(p, t)
            if np.isnan(r):
                r = 0.0
        key = gene_names[g] if gene_names else str(g)
        results[key] = float(r)
    return results


def mean_pearson(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Mean Pearson r across all genes — primary benchmark metric.

    Args:
        pred: Predicted expression [n_spots, n_genes]
        target: True expression [n_spots, n_genes]

    Returns:
        Mean Pearson r (higher = better; 1.0 = perfect)
    """
    gene_r = per_gene_pearson(pred, target)
    mp = float(np.mean(list(gene_r.values())))
    logger.info(f"Mean Pearson r across {len(gene_r)} genes: {mp:.4f}")
    return mp


def spatial_autocorrelation(
    expression: np.ndarray,
    coords: np.ndarray,
    gene_idx: int = 0,
    k_neighbors: int = 6,
) -> float:
    """
    Moran's I spatial autocorrelation for a single gene.

    Measures whether nearby spots have similar expression — a positive Moran's I
    indicates spatially structured expression patterns (expected for real data).

    Args:
        expression: Expression values [n_spots, n_genes]
        coords: Spot coordinates [n_spots, 2]
        gene_idx: Which gene to compute autocorrelation for
        k_neighbors: Number of spatial neighbors to consider

    Returns:
        Moran's I statistic ([-1, 1]; positive = spatially clustered)
    """
    from sklearn.neighbors import NearestNeighbors

    y = expression[:, gene_idx]
    n = len(y)

    # Build spatial weight matrix (k-NN)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    # Exclude self (index 0)
    neighbor_indices = indices[:, 1:]

    # Row-normalize weight matrix
    W = np.zeros((n, n))
    for i in range(n):
        for j in neighbor_indices[i]:
            W[i, j] = 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    W = W / row_sums

    # Moran's I
    y_mean = y.mean()
    y_dev = y - y_mean
    W_sum = W.sum()
    numerator = n * (y_dev[:, None] * W * y_dev[None, :]).sum()
    denominator = W_sum * (y_dev**2).sum()
    morans_i = float(numerator / (denominator + 1e-8))
    return morans_i


def gene_ranking_auroc(
    pred: np.ndarray,
    target: np.ndarray,
    high_quantile: float = 0.75,
) -> float:
    """
    Mean AUROC for predicting high-expression spots per gene.

    Binarizes true expression at high_quantile threshold per gene
    and computes AUROC of predicted scores. Measures rank-order accuracy.

    Args:
        pred: Predicted expression [n_spots, n_genes]
        target: True expression [n_spots, n_genes]
        high_quantile: Quantile threshold for "high expression" binary label

    Returns:
        Mean AUROC across genes
    """
    n_genes = pred.shape[1]
    aurocs = []
    for g in range(n_genes):
        threshold = np.quantile(target[:, g], high_quantile)
        y_true = (target[:, g] >= threshold).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        try:
            auc = roc_auc_score(y_true, pred[:, g])
            aurocs.append(auc)
        except Exception:
            continue
    mean_auroc = float(np.mean(aurocs)) if aurocs else 0.5
    logger.info(f"Mean gene-ranking AUROC: {mean_auroc:.4f} (over {len(aurocs)} genes)")
    return mean_auroc


def evaluate_decoder(
    pred: np.ndarray,
    target: np.ndarray,
    coords: Optional[np.ndarray] = None,
    gene_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Full evaluation suite for spatial transcriptomics decoder.

    Returns:
        Dict with mean_pearson, median_pearson, top100_pearson,
        gene_auroc, and optionally moran_i_mean
    """
    gene_r = per_gene_pearson(pred, target, gene_names)
    r_values = np.array(list(gene_r.values()))

    metrics = {
        "mean_pearson": float(r_values.mean()),
        "median_pearson": float(np.median(r_values)),
        "top100_pearson": float(np.sort(r_values)[::-1][:100].mean()),
        "bottom100_pearson": float(np.sort(r_values)[:100].mean()),
        "gene_auroc": gene_ranking_auroc(pred, target),
        "n_genes": len(r_values),
    }

    if coords is not None and pred.shape[1] > 0:
        morans = []
        for g in range(min(50, pred.shape[1])):
            try:
                mi = spatial_autocorrelation(pred, coords, g)
                morans.append(mi)
            except Exception:
                continue
        if morans:
            metrics["moran_i_mean"] = float(np.mean(morans))

    logger.info(
        f"Decoder evaluation — mean_pearson={metrics['mean_pearson']:.4f}, "
        f"top100={metrics['top100_pearson']:.4f}, auroc={metrics['gene_auroc']:.4f}"
    )
    return metrics
