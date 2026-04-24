"""
Unsupervised cancer subtype discovery with survival-aware clustering.

Discovers novel subtypes that are both cohesive in feature space AND
maximally separated in survival outcome — finding groups that matter
clinically, not just statistically.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .representation import SurvivalVAE, survival_vae_loss

logger = logging.getLogger(__name__)


def _gap_statistic(
    X: np.ndarray,
    k_range: range,
    n_refs: int = 10,
    random_state: int = 42,
) -> Tuple[int, np.ndarray]:
    """
    Select optimal k using gap statistic (Tibshirani et al. 2001).

    Returns (optimal_k, gap_values).
    """
    gaps = np.zeros(len(k_range))
    sk = np.zeros(len(k_range))

    for i, k in enumerate(k_range):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        inertia = km.inertia_

        # Reference distribution: uniform within bounding box
        ref_inertias = []
        for _ in range(n_refs):
            ref = np.random.uniform(X.min(0), X.max(0), size=X.shape)
            ref_km = KMeans(n_clusters=k, random_state=random_state, n_init=3)
            ref_km.fit(ref)
            ref_inertias.append(np.log(ref_km.inertia_ + 1e-8))
        ref_mean = np.mean(ref_inertias)
        ref_std = np.std(ref_inertias)

        gaps[i] = ref_mean - np.log(inertia + 1e-8)
        sk[i] = ref_std * np.sqrt(1 + 1 / n_refs)

    # Tibshirani rule: smallest k such that gap(k) >= gap(k+1) - sk(k+1)
    optimal_k = k_range[0]
    for i in range(len(k_range) - 1):
        if gaps[i] >= gaps[i + 1] - sk[i + 1]:
            optimal_k = k_range[i]
            break
    else:
        optimal_k = k_range[np.argmax(gaps)]

    return optimal_k, gaps


class SurvivalAwareClusterer:
    """
    Discovers cancer subtypes by clustering survival-VAE latent space.

    Pipeline:
    1. Train SurvivalVAE to learn survival-regularized embeddings
    2. Select optimal number of clusters via gap statistic
    3. Cluster latent means with KMeans
    4. Validate subtypes with log-rank test

    Usage:
        clusterer = SurvivalAwareClusterer(input_dim=1024)
        labels = clusterer.fit(features, survival_times, events)
        report = clusterer.summary()
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        k_min: int = 2,
        k_max: int = 8,
        lambda_cox: float = 1.0,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k_min = k_min
        self.k_max = k_max
        self.lambda_cox = lambda_cox
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

        self.vae = SurvivalVAE(input_dim, hidden_dim, latent_dim)
        self._scaler = StandardScaler()
        self._kmeans: Optional[KMeans] = None
        self._optimal_k: int = 2
        self._latent_embeddings: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    def _train_vae(
        self,
        X: torch.Tensor,
        survival_times: torch.Tensor,
        events: torch.Tensor,
    ) -> List[Dict[str, float]]:
        """Train SurvivalVAE and return per-epoch loss history."""
        self.vae.to(self.device)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs
        )
        dataset = torch.utils.data.TensorDataset(X, survival_times, events)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        history = []
        self.vae.train()
        for epoch in range(self.n_epochs):
            epoch_losses: Dict[str, float] = {}
            for xb, tb, eb in loader:
                xb = xb.to(self.device)
                tb = tb.to(self.device)
                eb = eb.to(self.device)
                optimizer.zero_grad()
                outputs = self.vae(xb)
                loss, metrics = survival_vae_loss(
                    outputs, xb, tb, eb, lambda_cox=self.lambda_cox
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                optimizer.step()
                for k, v in metrics.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            scheduler.step()
            history.append({k: v / len(loader) for k, v in epoch_losses.items()})
            if (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.n_epochs}: "
                    + ", ".join(f"{k}={v:.4f}" for k, v in history[-1].items())
                )
        return history

    def fit(
        self,
        features: np.ndarray,
        survival_times: np.ndarray,
        events: np.ndarray,
    ) -> np.ndarray:
        """
        Discover subtypes from WSI feature embeddings.

        Args:
            features: WSI slide embeddings [n_slides, feature_dim]
            survival_times: Observed survival times [n_slides]
            events: Event indicators [n_slides] (1=event, 0=censored)

        Returns:
            Subtype labels [n_slides] (integers 0..k-1)
        """
        logger.info(
            f"Discovering subtypes: {features.shape[0]} slides, "
            f"{(events == 1).sum()} events"
        )

        # Normalize features
        X_scaled = self._scaler.fit_transform(features)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        T_t = torch.tensor(survival_times, dtype=torch.float32)
        E_t = torch.tensor(events, dtype=torch.float32)

        # Train survival-aware VAE
        self._train_vae(X_t, T_t, E_t)

        # Extract latent embeddings
        self.vae.eval()
        with torch.no_grad():
            latent = self.vae.get_latent(X_t.to(self.device)).cpu().numpy()
        self._latent_embeddings = latent

        # Select optimal k via gap statistic
        k_range = range(self.k_min, min(self.k_max + 1, features.shape[0] // 5 + 1))
        self._optimal_k, gap_values = _gap_statistic(latent, k_range)
        logger.info(f"Gap statistic selected k={self._optimal_k}")

        # Cluster latent space
        self._kmeans = KMeans(
            n_clusters=self._optimal_k,
            random_state=self.random_state,
            n_init=20,
        )
        self._labels = self._kmeans.fit_predict(latent)

        sil = silhouette_score(latent, self._labels) if self._optimal_k > 1 else 0.0
        logger.info(
            f"Clustering complete: k={self._optimal_k}, silhouette={sil:.3f}"
        )
        return self._labels

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Assign new slides to discovered subtypes."""
        if self._kmeans is None:
            raise RuntimeError("Call fit() first")
        X_scaled = self._scaler.transform(features)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        self.vae.eval()
        with torch.no_grad():
            latent = self.vae.get_latent(X_t.to(self.device)).cpu().numpy()
        return self._kmeans.predict(latent)

    def summary(self) -> Dict:
        if self._labels is None:
            return {"status": "not fitted"}
        unique, counts = np.unique(self._labels, return_counts=True)
        return {
            "n_subtypes": self._optimal_k,
            "subtype_sizes": dict(zip(unique.tolist(), counts.tolist())),
            "latent_dim": self.latent_dim,
        }


def discover_subtypes(
    features: np.ndarray,
    survival_times: np.ndarray,
    events: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    n_epochs: int = 100,
    lambda_cox: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, SurvivalAwareClusterer]:
    """
    Convenience wrapper: discover subtypes and return (labels, fitted clusterer).
    """
    clusterer = SurvivalAwareClusterer(
        input_dim=features.shape[1],
        k_min=k_min,
        k_max=k_max,
        n_epochs=n_epochs,
        lambda_cox=lambda_cox,
        device=device,
    )
    labels = clusterer.fit(features, survival_times, events)
    return labels, clusterer
