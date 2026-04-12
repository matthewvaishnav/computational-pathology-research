"""
Uncertainty quantification for clinical predictions.

This module provides calibrated confidence intervals, uncertainty explanations,
and integration with out-of-distribution detection for physician-friendly
uncertainty quantification.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class UncertaintyQuantifier(nn.Module):
    """
    Uncertainty quantifier providing calibrated confidence intervals and explanations.

    Provides calibrated confidence estimates using temperature scaling or Platt scaling,
    generates uncertainty explanations (data quality, model confidence, OOD detection),
    and calculates separate uncertainty estimates for primary and top-3 alternative diagnoses.

    The quantifier can be applied to any classifier that outputs logits or probabilities,
    and learns calibration parameters from validation data.

    Args:
        num_classes: Number of disease classes
        calibration_method: Calibration method ('temperature' or 'platt', default: 'temperature')
        initial_temperature: Initial temperature for temperature scaling (default: 1.0)

    Example:
        >>> quantifier = UncertaintyQuantifier(num_classes=5)
        >>> logits = torch.randn(16, 5)
        >>> output = quantifier(logits)
        >>> output['calibrated_probabilities'].shape  # [16, 5]
        >>> output['uncertainty_scores'].shape  # [16]
    """

    def __init__(
        self,
        num_classes: int,
        calibration_method: str = "temperature",
        initial_temperature: float = 1.0,
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        if calibration_method not in ["temperature", "platt"]:
            raise ValueError(
                f"calibration_method must be 'temperature' or 'platt', got {calibration_method}"
            )

        self.num_classes = num_classes
        self.calibration_method = calibration_method

        # Temperature scaling: single learnable parameter
        if calibration_method == "temperature":
            self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        # Platt scaling: per-class affine transformation
        elif calibration_method == "platt":
            self.platt_scale = nn.Parameter(torch.ones(num_classes))
            self.platt_bias = nn.Parameter(torch.zeros(num_classes))

        logger.info(
            f"Initialized UncertaintyQuantifier with {calibration_method} calibration "
            f"for {num_classes} classes"
        )

    def forward(
        self,
        logits: torch.Tensor,
        ood_scores: Optional[torch.Tensor] = None,
        data_quality_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute calibrated confidence intervals and uncertainty estimates.

        Args:
            logits: Raw model logits [batch_size, num_classes]
            ood_scores: Optional OOD detection scores [batch_size] in [0, 1]
                       Higher values indicate more out-of-distribution
            data_quality_scores: Optional data quality scores [batch_size] in [0, 1]
                                Higher values indicate better quality

        Returns:
            Dictionary containing:
                - 'calibrated_probabilities': Calibrated probabilities [batch_size, num_classes]
                - 'uncertainty_scores': Overall uncertainty [batch_size] in [0, 1]
                - 'primary_uncertainty': Uncertainty for primary diagnosis [batch_size]
                - 'top3_uncertainties': Uncertainties for top-3 diagnoses [batch_size, 3]
                - 'uncertainty_explanation': Uncertainty source breakdown [batch_size, 3]
                                            (model_confidence, data_quality, ood_detection)

        Raises:
            ValueError: If logits have incorrect shape
        """
        if logits.dim() != 2:
            raise ValueError(
                f"Expected 2D logits [batch_size, num_classes], got shape {logits.shape}"
            )

        if logits.shape[1] != self.num_classes:
            raise ValueError(f"Expected num_classes={self.num_classes}, got {logits.shape[1]}")

        batch_size = logits.shape[0]
        device = logits.device

        # Apply calibration
        calibrated_probs = self._calibrate(logits)

        # Compute model confidence uncertainty (entropy-based)
        model_uncertainty = self._compute_entropy_uncertainty(calibrated_probs)

        # Incorporate OOD scores if provided
        if ood_scores is not None:
            if ood_scores.shape != (batch_size,):
                raise ValueError(
                    f"ood_scores shape {ood_scores.shape} doesn't match batch_size {batch_size}"
                )
            ood_uncertainty = ood_scores
        else:
            ood_uncertainty = torch.zeros(batch_size, device=device)

        # Incorporate data quality scores if provided
        if data_quality_scores is not None:
            if data_quality_scores.shape != (batch_size,):
                raise ValueError(
                    f"data_quality_scores shape {data_quality_scores.shape} doesn't match batch_size {batch_size}"
                )
            # Convert quality scores to uncertainty (1 - quality)
            quality_uncertainty = 1.0 - data_quality_scores
        else:
            quality_uncertainty = torch.zeros(batch_size, device=device)

        # Combine uncertainty sources (weighted average)
        overall_uncertainty = (
            0.5 * model_uncertainty + 0.3 * ood_uncertainty + 0.2 * quality_uncertainty
        )

        # Get top-3 predictions
        top3_probs, top3_indices = torch.topk(calibrated_probs, k=min(3, self.num_classes), dim=1)

        # Compute uncertainties for top-3 predictions
        # Uncertainty for each prediction is 1 - probability
        top3_uncertainties = 1.0 - top3_probs

        # Primary diagnosis uncertainty (uncertainty of highest probability class)
        primary_uncertainty = top3_uncertainties[:, 0]

        # Stack uncertainty explanations
        uncertainty_explanation = torch.stack(
            [model_uncertainty, quality_uncertainty, ood_uncertainty], dim=1
        )

        return {
            "calibrated_probabilities": calibrated_probs,
            "uncertainty_scores": overall_uncertainty,
            "primary_uncertainty": primary_uncertainty,
            "top3_uncertainties": top3_uncertainties,
            "uncertainty_explanation": uncertainty_explanation,
        }

    def _calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply calibration to logits.

        Args:
            logits: Raw model logits [batch_size, num_classes]

        Returns:
            Calibrated probabilities [batch_size, num_classes]
        """
        if self.calibration_method == "temperature":
            # Temperature scaling: divide logits by temperature before softmax
            calibrated_logits = logits / self.temperature
            return F.softmax(calibrated_logits, dim=1)
        elif self.calibration_method == "platt":
            # Platt scaling: affine transformation of logits
            calibrated_logits = logits * self.platt_scale + self.platt_bias
            return F.softmax(calibrated_logits, dim=1)

    def _compute_entropy_uncertainty(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy-based uncertainty from probability distribution.

        Args:
            probabilities: Probability distribution [batch_size, num_classes]

        Returns:
            Normalized entropy uncertainty [batch_size] in [0, 1]
        """
        # Compute entropy: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -torch.sum(probabilities * torch.log(probabilities + epsilon), dim=1)

        # Normalize by maximum entropy (log(num_classes))
        max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32))
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def calibrate_on_validation(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        learning_rate: float = 0.01,
        max_iterations: int = 100,
    ) -> Dict[str, float]:
        """
        Calibrate the quantifier on validation data.

        Args:
            val_logits: Validation logits [num_samples, num_classes]
            val_labels: Validation labels [num_samples]
            learning_rate: Learning rate for calibration optimization
            max_iterations: Maximum optimization iterations

        Returns:
            Dictionary with calibration metrics (loss, ECE, MCE, Brier score)
        """
        if val_logits.shape[0] != val_labels.shape[0]:
            raise ValueError(
                f"Logits batch size {val_logits.shape[0]} doesn't match labels {val_labels.shape[0]}"
            )

        # Set to training mode for calibration
        self.train()

        # Optimizer for calibration parameters
        optimizer = torch.optim.LBFGS(self.parameters(), lr=learning_rate, max_iter=max_iterations)

        # Negative log-likelihood loss
        def closure():
            optimizer.zero_grad()
            calibrated_probs = self._calibrate(val_logits)
            loss = F.nll_loss(torch.log(calibrated_probs + 1e-10), val_labels)
            loss.backward()
            return loss

        # Optimize calibration parameters
        optimizer.step(closure)

        # Set back to eval mode
        self.eval()

        # Compute calibration metrics
        with torch.no_grad():
            calibrated_probs = self._calibrate(val_logits)
            metrics = self.compute_calibration_metrics(calibrated_probs, val_labels)

        logger.info(
            f"Calibration complete: ECE={metrics['ece']:.4f}, "
            f"MCE={metrics['mce']:.4f}, Brier={metrics['brier_score']:.4f}"
        )

        return metrics

    def compute_calibration_metrics(
        self,
        probabilities: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 15,
    ) -> Dict[str, float]:
        """
        Compute calibration metrics (ECE, MCE, Brier Score).

        Args:
            probabilities: Predicted probabilities [num_samples, num_classes]
            labels: True labels [num_samples]
            num_bins: Number of bins for calibration curve (default: 15)

        Returns:
            Dictionary with calibration metrics:
                - 'ece': Expected Calibration Error
                - 'mce': Maximum Calibration Error
                - 'brier_score': Brier Score
        """
        # Get confidence (max probability) and predictions
        confidences, predictions = torch.max(probabilities, dim=1)
        accuracies = predictions.eq(labels)

        # Compute ECE and MCE
        ece = 0.0
        mce = 0.0
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)

        for i in range(num_bins):
            # Find samples in this bin
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            if in_bin.sum() > 0:
                # Compute average confidence and accuracy in bin
                bin_confidence = confidences[in_bin].mean().item()
                bin_accuracy = accuracies[in_bin].float().mean().item()
                bin_weight = in_bin.float().mean().item()

                # Update ECE (weighted average of calibration errors)
                ece += bin_weight * abs(bin_confidence - bin_accuracy)

                # Update MCE (maximum calibration error)
                mce = max(mce, abs(bin_confidence - bin_accuracy))

        # Compute Brier Score
        # Convert labels to one-hot encoding
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()
        brier_score = torch.mean(torch.sum((probabilities - one_hot_labels) ** 2, dim=1)).item()

        return {
            "ece": ece,
            "mce": mce,
            "brier_score": brier_score,
        }

    def get_calibration_curve(
        self,
        probabilities: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 15,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute calibration curve data for visualization.

        Args:
            probabilities: Predicted probabilities [num_samples, num_classes]
            labels: True labels [num_samples]
            num_bins: Number of bins for calibration curve

        Returns:
            Tuple of:
                - bin_confidences: Average confidence per bin [num_bins]
                - bin_accuracies: Average accuracy per bin [num_bins]
                - bin_counts: Number of samples per bin [num_bins]
        """
        confidences, predictions = torch.max(probabilities, dim=1)
        accuracies = predictions.eq(labels).float()

        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_confidences = torch.zeros(num_bins)
        bin_accuracies = torch.zeros(num_bins)
        bin_counts = torch.zeros(num_bins)

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            if in_bin.sum() > 0:
                bin_confidences[i] = confidences[in_bin].mean()
                bin_accuracies[i] = accuracies[in_bin].mean()
                bin_counts[i] = in_bin.sum()

        return bin_confidences, bin_accuracies, bin_counts

    def generate_uncertainty_explanation(
        self,
        uncertainty_breakdown: torch.Tensor,
    ) -> List[str]:
        """
        Generate human-readable uncertainty explanations.

        Args:
            uncertainty_breakdown: Uncertainty source breakdown [batch_size, 3]
                                  (model_confidence, data_quality, ood_detection)

        Returns:
            List of explanation strings for each sample
        """
        explanations = []
        batch_size = uncertainty_breakdown.shape[0]

        for i in range(batch_size):
            model_unc = uncertainty_breakdown[i, 0].item()
            quality_unc = uncertainty_breakdown[i, 1].item()
            ood_unc = uncertainty_breakdown[i, 2].item()

            # Identify dominant uncertainty source
            sources = []
            if model_unc > 0.3:
                sources.append("low model confidence")
            if quality_unc > 0.3:
                sources.append("data quality concerns")
            if ood_unc > 0.3:
                sources.append("out-of-distribution detection")

            if not sources:
                explanation = "High confidence prediction"
            elif len(sources) == 1:
                explanation = f"Uncertainty due to {sources[0]}"
            else:
                explanation = f"Uncertainty due to {', '.join(sources[:-1])} and {sources[-1]}"

            # Add recommendation for high uncertainty
            total_unc = model_unc + quality_unc + ood_unc
            if total_unc > 0.6:
                explanation += " - seek expert review"

            explanations.append(explanation)

        return explanations

    def __repr__(self) -> str:
        """String representation of uncertainty quantifier."""
        return (
            f"UncertaintyQuantifier(\n"
            f"  num_classes={self.num_classes},\n"
            f"  calibration_method='{self.calibration_method}'\n"
            f")"
        )
