"""MAGNET v4: paired-scanner empirical *displacement* energy (development).
NOT a proof of biological purity, global invertibility, or physical scanner laws.

Complete canine replay and controls: PA_NF_MAGNET_v4_Canine_20261103.zip
This core implementation makes source-only predictions from scanner identity and
matched TRAINING region differences; no target observation is used at inference.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import pairwise_distances


@dataclass
class EmpiricalScannerResponse:
    training: np.ndarray   # n_regions x n_scanners x n_features
    sample_ids: np.ndarray
    k: int = 32
    gamma: float = 1.0

    def __post_init__(self):
        if self.training.ndim != 3 or len(self.training) != len(self.sample_ids):
            raise ValueError("Training must be matched regions x scanners x features")
        if not np.isfinite(self.training).all() or not 0 < self.k <= len(self.training):
            raise ValueError("Invalid training values/neighbors")
        if not 0 <= self.gamma <= 1:
            raise ValueError("Invalid shrinkage")
        self.mean = self.training.mean(axis=0)
        self.n_scanners = self.training.shape[1]

    def _weights(self, source, scanner, sample_ids=None, exclude_sample=False):
        if not isinstance(scanner, int) or not 0 <= scanner < self.n_scanners:
            raise ValueError("Invalid scanner")
        source = np.asarray(source, dtype=np.float64)
        if source.ndim != 2 or source.shape[1] != self.training.shape[2]:
            raise ValueError("Invalid source features")
        if not np.isfinite(source).all():
            raise ValueError("Source features not finite")
        distances = pairwise_distances(
            source, self.training[:, scanner], metric="sqeuclidean"
        )
        if exclude_sample:
            if sample_ids is None or len(sample_ids) != len(source):
                raise ValueError("Must provide a sample ID for each training query")
            distances[np.asarray(sample_ids)[:, None] == self.sample_ids[None, :]] = np.inf
        if (np.isfinite(distances).sum(axis=1) < self.k).any():
            raise ValueError("Too few independent reference regions")
        neighbors = np.argpartition(distances, self.k - 1, axis=1)[:, :self.k]
        d = np.take_along_axis(distances, neighbors, axis=1)
        weights = 1.0 / (d + 1e-6)
        weights /= weights.sum(axis=1, keepdims=True)
        return neighbors, weights

    def predict_all(self, source, scanner, sample_ids=None, exclude_sample=False):
        indices, weights = self._weights(source, scanner, sample_ids, exclude_sample)
        source = np.asarray(source, dtype=np.float64)
        all_deltas = self.training - self.training[:, scanner:scanner+1, :]
        local_deltas = np.einsum(
            "nk,nksd->nsd", weights, all_deltas[indices], optimize=True
        )
        average_delta = self.mean - self.mean[scanner]
        delta = (1.0 - self.gamma) * average_delta[None] + self.gamma * local_deltas
        result = source[:, None, :] + delta
        result[:, scanner, :] = source
        return result

    def transport(self, source, scanner, target, sample_ids=None, exclude_sample=False):
        if not isinstance(target, int) or not 0 <= target < self.n_scanners:
            raise ValueError("Invalid target scanner")
        return self.predict_all(source, scanner, sample_ids, exclude_sample)[:, target]

    def canonical(self, source, scanner, sample_ids=None, exclude_sample=False):
        # Scanner-shared consensus is a statistical coordinate, not pure biology.
        return (self.predict_all(source, scanner, sample_ids, exclude_sample)
                .mean(axis=1) - self.mean.mean(axis=0))

    def energy_gradient(self, displacement, source, scanner, target):
        if self.gamma <= 0:
            raise ValueError("Zero gamma is a limit, not a finite energy")
        idx, weight = self._weights(source, scanner)
        all_delta = self.training[:, target] - self.training[:, scanner]
        local = np.einsum("nk,nkd->nd", weight, all_delta[idx], optimize=True)
        prior = self.mean[target] - self.mean[scanner]
        lam = (1.0 - self.gamma) / self.gamma
        return lam * (displacement - prior) + (displacement - local)


def calibrate_training_gauge(model: EmpiricalScannerResponse):
    """Calibrate common B origins from training samples only.

    Training-region neighborhood queries exclude their entire biological sample,
    though global scanner centroids still use the complete training split.
    """
    encoded = np.stack([
        model.canonical(model.training[:, s], s, model.sample_ids, True)
        for s in range(model.n_scanners)
    ], axis=1)
    return encoded.mean(axis=0) - encoded.mean(axis=(0, 1))[None, :]
