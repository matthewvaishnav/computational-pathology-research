"""
Out-of-distribution (OOD) detection for clinical predictions.

This module provides multiple OOD detection methods to identify cases that are
significantly different from training data, flagging them for expert review.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class OODDetector(nn.Module):
    """
    Out-of-distribution detector using multiple detection methods.

    Implements multiple OOD detection approaches:
    - Mahalanobis distance: Statistical distance from training distribution
    - Reconstruction error: Autoencoder-based anomaly detection
    - Ensemble disagreement: Variance across multiple model predictions

    The detector combines multiple methods for robust OOD detection and provides
    explanations for why cases are flagged (novel tissue patterns, unusual staining,
    rare disease presentation).

    Args:
        feature_dim: Dimension of input features (default: 256)
        hidden_dim: Dimension of hidden layers (default: 128)
        num_ensemble_models: Number of models for ensemble disagreement (default: 5)
        detection_methods: List of methods to use (default: all methods)

    Example:
        >>> detector = OODDetector(feature_dim=256)
        >>> features = torch.randn(16, 256)
        >>> output = detector(features)
        >>> output['ood_scores'].shape  # [16]
        >>> output['is_ood'].shape  # [16] - boolean mask
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        num_ensemble_models: int = 5,
        detection_methods: Optional[List[str]] = None,
    ):
        super().__init__()

        if detection_methods is None:
            detection_methods = ["mahalanobis", "reconstruction", "ensemble"]

        valid_methods = ["mahalanobis", "reconstruction", "ensemble"]
        for method in detection_methods:
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid detection method '{method}'. Valid methods: {valid_methods}"
                )

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_ensemble_models = num_ensemble_models
        self.detection_methods = detection_methods

        # Mahalanobis distance: requires training distribution statistics
        if "mahalanobis" in detection_methods:
            # These will be set during fit_training_distribution()
            self.register_buffer("train_mean", torch.zeros(feature_dim))
            self.register_buffer("train_cov_inv", torch.eye(feature_dim))
            self.mahalanobis_fitted = False

        # Reconstruction error: autoencoder for anomaly detection
        if "reconstruction" in detection_methods:
            self.autoencoder = Autoencoder(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
            )

        # Ensemble disagreement: multiple prediction heads
        if "ensemble" in detection_methods:
            self.ensemble_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(feature_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                    )
                    for _ in range(num_ensemble_models)
                ]
            )

        logger.info(
            f"Initialized OODDetector with methods: {detection_methods}, "
            f"feature_dim={feature_dim}"
        )

    def forward(
        self,
        features: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Detect out-of-distribution cases.

        Args:
            features: Input features [batch_size, feature_dim]
            threshold: OOD detection threshold (default: 0.5)

        Returns:
            Dictionary containing:
                - 'ood_scores': Overall OOD scores [batch_size] in [0, 1]
                - 'is_ood': Boolean mask indicating OOD cases [batch_size]
                - 'method_scores': Individual method scores [batch_size, num_methods]
                - 'explanations': List of explanation strings for each sample

        Raises:
            ValueError: If features have incorrect shape
        """
        if features.dim() != 2:
            raise ValueError(
                f"Expected 2D features [batch_size, feature_dim], got shape {features.shape}"
            )

        if features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {features.shape[1]}")

        batch_size = features.shape[0]
        method_scores_list = []

        # Compute scores for each detection method
        if "mahalanobis" in self.detection_methods:
            mahal_scores = self._compute_mahalanobis_scores(features)
            method_scores_list.append(mahal_scores)

        if "reconstruction" in self.detection_methods:
            recon_scores = self._compute_reconstruction_scores(features)
            method_scores_list.append(recon_scores)

        if "ensemble" in self.detection_methods:
            ensemble_scores = self._compute_ensemble_disagreement_scores(features)
            method_scores_list.append(ensemble_scores)

        # Stack method scores
        method_scores = torch.stack(method_scores_list, dim=1)  # [batch_size, num_methods]

        # Combine scores (average across methods)
        ood_scores = torch.mean(method_scores, dim=1)

        # Flag OOD cases based on threshold
        is_ood = ood_scores > threshold

        # Generate explanations
        explanations = self._generate_explanations(method_scores, is_ood)

        return {
            "ood_scores": ood_scores,
            "is_ood": is_ood,
            "method_scores": method_scores,
            "explanations": explanations,
        }

    def _compute_mahalanobis_scores(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis distance-based OOD scores.

        Args:
            features: Input features [batch_size, feature_dim]

        Returns:
            Normalized OOD scores [batch_size] in [0, 1]
        """
        if not self.mahalanobis_fitted:
            logger.warning(
                "Mahalanobis detector not fitted. Call fit_training_distribution() first. "
                "Returning zero scores."
            )
            return torch.zeros(features.shape[0], device=features.device)

        # Compute Mahalanobis distance: sqrt((x - mu)^T * Sigma^-1 * (x - mu))
        centered = features - self.train_mean
        mahal_dist = torch.sqrt(torch.sum(centered @ self.train_cov_inv * centered, dim=1))

        # Normalize to [0, 1] using sigmoid
        # Scale factor chosen empirically (can be tuned)
        # Expected Mahalanobis distance for in-distribution is around sqrt(feature_dim)
        expected_dist = torch.sqrt(torch.tensor(self.feature_dim, dtype=torch.float32))
        normalized_scores = torch.sigmoid((mahal_dist - expected_dist) / (expected_dist * 0.5))

        return normalized_scores

    def _compute_reconstruction_scores(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction error-based OOD scores.

        Args:
            features: Input features [batch_size, feature_dim]

        Returns:
            Normalized OOD scores [batch_size] in [0, 1]
        """
        # Reconstruct features
        reconstructed = self.autoencoder(features)

        # Compute reconstruction error (MSE per sample)
        recon_error = torch.mean((features - reconstructed) ** 2, dim=1)

        # Normalize to [0, 1] using sigmoid
        # Use median reconstruction error as reference point
        # For well-trained autoencoder, in-distribution should have low error
        normalized_scores = torch.sigmoid((recon_error - 0.5) / 0.3)

        return normalized_scores

    def _compute_ensemble_disagreement_scores(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute ensemble disagreement-based OOD scores.

        Args:
            features: Input features [batch_size, feature_dim]

        Returns:
            Normalized OOD scores [batch_size] in [0, 1]
        """
        # Get predictions from all ensemble members
        ensemble_outputs = []
        for head in self.ensemble_heads:
            output = head(features)
            ensemble_outputs.append(output)

        # Stack outputs: [num_models, batch_size, hidden_dim]
        ensemble_outputs = torch.stack(ensemble_outputs, dim=0)

        # Compute variance across ensemble (disagreement)
        # High variance indicates uncertainty/OOD
        variance = torch.var(ensemble_outputs, dim=0)  # [batch_size, hidden_dim]
        disagreement = torch.mean(variance, dim=1)  # [batch_size]

        # Normalize to [0, 1] using sigmoid
        # Scale factor chosen empirically (can be tuned)
        normalized_scores = torch.sigmoid((disagreement - 0.05) / 0.02)

        return normalized_scores

    def _generate_explanations(
        self,
        method_scores: torch.Tensor,
        is_ood: torch.Tensor,
    ) -> List[str]:
        """
        Generate human-readable OOD explanations.

        Args:
            method_scores: Individual method scores [batch_size, num_methods]
            is_ood: Boolean mask indicating OOD cases [batch_size]

        Returns:
            List of explanation strings for each sample
        """
        explanations = []
        batch_size = method_scores.shape[0]

        # Method names for explanation
        method_names = []
        method_explanations = []
        if "mahalanobis" in self.detection_methods:
            method_names.append("mahalanobis")
            method_explanations.append("novel tissue patterns")
        if "reconstruction" in self.detection_methods:
            method_names.append("reconstruction")
            method_explanations.append("unusual staining or artifacts")
        if "ensemble" in self.detection_methods:
            method_names.append("ensemble")
            method_explanations.append("rare disease presentation")

        for i in range(batch_size):
            if not is_ood[i]:
                explanations.append("In-distribution - normal case")
                continue

            # Identify which methods flagged this case
            flagged_methods = []
            for j, method_name in enumerate(method_names):
                if method_scores[i, j] > 0.5:
                    flagged_methods.append(method_explanations[j])

            if not flagged_methods:
                explanation = "Out-of-distribution - uncertain cause"
            elif len(flagged_methods) == 1:
                explanation = f"Out-of-distribution - {flagged_methods[0]}"
            else:
                explanation = (
                    f"Out-of-distribution - {', '.join(flagged_methods[:-1])} "
                    f"and {flagged_methods[-1]}"
                )

            explanation += " - seek expert review"
            explanations.append(explanation)

        return explanations

    def fit_training_distribution(
        self,
        train_features: torch.Tensor,
        regularization: float = 1e-3,
    ) -> None:
        """
        Fit Mahalanobis detector to training distribution.

        Args:
            train_features: Training features [num_samples, feature_dim]
            regularization: Regularization for covariance matrix inversion
        """
        if "mahalanobis" not in self.detection_methods:
            logger.warning("Mahalanobis method not enabled, skipping fit")
            return

        if train_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {train_features.shape[1]}"
            )

        # Compute mean
        train_mean = torch.mean(train_features, dim=0)

        # Compute covariance matrix
        centered = train_features - train_mean
        cov = (centered.T @ centered) / (train_features.shape[0] - 1)

        # Add regularization for numerical stability
        cov = cov + regularization * torch.eye(self.feature_dim, device=cov.device)

        # Compute inverse covariance
        try:
            cov_inv = torch.linalg.inv(cov)
        except RuntimeError:
            logger.warning("Covariance matrix inversion failed, using identity")
            cov_inv = torch.eye(self.feature_dim, device=cov.device)

        # Store statistics
        self.train_mean.copy_(train_mean)
        self.train_cov_inv.copy_(cov_inv)
        self.mahalanobis_fitted = True

        logger.info(f"Fitted Mahalanobis detector on {train_features.shape[0]} training samples")

    def train_autoencoder(
        self,
        train_features: torch.Tensor,
        learning_rate: float = 0.001,
        num_epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """
        Train autoencoder for reconstruction-based OOD detection.

        Args:
            train_features: Training features [num_samples, feature_dim]
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Dictionary with training history (losses)
        """
        if "reconstruction" not in self.detection_methods:
            logger.warning("Reconstruction method not enabled, skipping training")
            return {"losses": []}

        if train_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {train_features.shape[1]}"
            )

        # Set autoencoder to training mode
        self.autoencoder.train()

        # Optimizer
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=learning_rate)

        # Training loop
        losses = []
        num_samples = train_features.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # Shuffle data
            perm = torch.randperm(num_samples)
            shuffled_features = train_features[perm]

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                batch = shuffled_features[start_idx:end_idx]

                # Forward pass
                reconstructed = self.autoencoder(batch)
                loss = F.mse_loss(reconstructed, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

        # Set back to eval mode
        self.autoencoder.eval()

        logger.info(f"Trained autoencoder for {num_epochs} epochs")

        return {"losses": losses}

    def __repr__(self) -> str:
        """String representation of OOD detector."""
        return (
            f"OODDetector(\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  detection_methods={self.detection_methods},\n"
            f"  num_ensemble_models={self.num_ensemble_models}\n"
            f")"
        )


class Autoencoder(nn.Module):
    """
    Autoencoder for reconstruction-based anomaly detection.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input features.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Reconstructed features [batch_size, input_dim]
        """
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed
