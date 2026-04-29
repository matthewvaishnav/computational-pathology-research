"""Streaming Attention Aggregator for progressive confidence building."""

import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceUpdate:
    """Confidence update from attention aggregation."""

    current_confidence: float
    confidence_delta: float
    patches_processed: int
    estimated_remaining: int
    attention_weights: torch.Tensor
    early_stop_recommended: bool
    prediction_logits: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Validate confidence update parameters."""
        if not (0.0 <= self.current_confidence <= 1.0):
            raise ValueError("Current confidence must be between 0.0 and 1.0")
        if self.patches_processed < 0:
            raise ValueError("Patches processed must be >= 0")

        # Validate attention weights sum to ~1.0
        if self.attention_weights is not None:
            weight_sum = torch.sum(self.attention_weights).item()
            if abs(weight_sum - 1.0) > 1e-4:  # Tolerance for numerical precision
                logger.warning(f"Attention weights sum to {weight_sum:.6f}, not 1.0")


@dataclass
class PredictionResult:
    """Prediction result with confidence and attention."""

    prediction: int
    confidence: float
    probabilities: torch.Tensor
    attention_weights: torch.Tensor
    num_patches: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": self.probabilities.cpu().numpy().tolist(),
            "num_patches": self.num_patches,
        }


class AttentionMIL(nn.Module):
    """Attention-based Multiple Instance Learning model."""

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor, return_attention: bool = False):
        """Forward pass with optional attention weights.

        Args:
            features: [batch_size, num_patches, feature_dim] or [num_patches, feature_dim]
            return_attention: Whether to return attention weights

        Returns:
            logits: [batch_size, num_classes]
            attention_weights: [batch_size, num_patches] (if return_attention=True)
        """
        # Handle single sample
        if len(features.shape) == 2:
            features = features.unsqueeze(0)  # [1, num_patches, feature_dim]

        # Compute attention weights
        attention_scores = self.attention(features)  # [batch_size, num_patches, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # Normalize across patches

        # Weighted aggregation
        weighted_features = torch.sum(
            features * attention_weights, dim=1
        )  # [batch_size, feature_dim]

        # Classification
        logits = self.classifier(weighted_features)  # [batch_size, num_classes]

        if return_attention:
            return logits, attention_weights.squeeze(-1)  # [batch_size, num_patches]
        return logits


class StreamingAttentionAggregator:
    """Progressive attention-based feature aggregation with real-time confidence updates."""

    def __init__(
        self,
        attention_model: AttentionMIL,
        confidence_threshold: float = 0.95,
        max_features: int = 10000,
        min_patches_for_confidence: int = 100,
    ):
        """Initialize streaming attention aggregator.

        Args:
            attention_model: Attention MIL model for aggregation
            confidence_threshold: Threshold for early stopping
            max_features: Maximum number of features to keep in memory
            min_patches_for_confidence: Minimum patches before trusting confidence
        """
        self.attention_model = attention_model
        self.attention_model.eval()
        self.confidence_threshold = confidence_threshold
        self.max_features = max_features
        self.min_patches_for_confidence = min_patches_for_confidence

        # Feature accumulation
        self.accumulated_features: Optional[torch.Tensor] = None
        self.accumulated_coordinates: Optional[np.ndarray] = None
        self.num_patches = 0

        # Confidence tracking
        self.confidence_history: List[float] = []
        self.prediction_history: List[int] = []
        self.attention_cache: Optional[torch.Tensor] = None

        # Early stopping
        self.stable_confidence_count = 0
        self.required_stable_updates = 3  # Require 3 stable updates before early stop

        logger.info(
            f"StreamingAttentionAggregator initialized with threshold={confidence_threshold}"
        )

    def update_features(
        self, new_features: torch.Tensor, coordinates: np.ndarray
    ) -> ConfidenceUpdate:
        """Add new features and update attention weights.

        Args:
            new_features: [batch_size, feature_dim]
            coordinates: [batch_size, 2] - (x, y) coordinates

        Returns:
            ConfidenceUpdate with current state
        """
        # Validate inputs
        if len(new_features.shape) != 2:
            raise ValueError(f"Expected 2D features, got shape {new_features.shape}")
        if coordinates.shape[0] != new_features.shape[0]:
            raise ValueError("Coordinates must match feature batch size")

        # Accumulate features
        if self.accumulated_features is None:
            self.accumulated_features = new_features.cpu()
            self.accumulated_coordinates = coordinates
        else:
            self.accumulated_features = torch.cat(
                [self.accumulated_features, new_features.cpu()], dim=0
            )
            self.accumulated_coordinates = np.vstack([self.accumulated_coordinates, coordinates])

        self.num_patches = self.accumulated_features.shape[0]

        # Memory management: keep only most recent features if exceeding limit
        if self.num_patches > self.max_features:
            self.accumulated_features = self.accumulated_features[-self.max_features :]
            self.accumulated_coordinates = self.accumulated_coordinates[-self.max_features :]
            self.num_patches = self.max_features
            logger.info(f"Trimmed features to {self.max_features} most recent patches")

        # Compute attention and confidence
        with torch.no_grad():
            # Reshape for attention model: [1, num_patches, feature_dim]
            features_batch = self.accumulated_features.unsqueeze(0)

            # Get attention weights and prediction
            logits, attention_weights = self.attention_model(features_batch, return_attention=True)

            # Compute confidence
            probabilities = F.softmax(logits, dim=1)
            confidence = torch.max(probabilities).item()
            prediction = torch.argmax(probabilities, dim=1).item()

            # Cache attention weights
            self.attention_cache = attention_weights.squeeze(0)  # [num_patches]

        # Track confidence history
        prev_confidence = self.confidence_history[-1] if self.confidence_history else 0.0
        confidence_delta = confidence - prev_confidence

        self.confidence_history.append(confidence)
        self.prediction_history.append(prediction)

        # Check for early stopping
        early_stop = self._check_early_stop(confidence, prediction)

        # Estimate remaining patches (rough estimate)
        estimated_remaining = max(0, self.min_patches_for_confidence - self.num_patches)

        return ConfidenceUpdate(
            current_confidence=confidence,
            confidence_delta=confidence_delta,
            patches_processed=self.num_patches,
            estimated_remaining=estimated_remaining,
            attention_weights=self.attention_cache,
            early_stop_recommended=early_stop,
            prediction_logits=logits,
        )

    def _check_early_stop(self, confidence: float, prediction: int) -> bool:
        """Check if early stopping criteria are met."""
        # Need minimum patches
        if self.num_patches < self.min_patches_for_confidence:
            return False

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            self.stable_confidence_count = 0
            return False

        # Check prediction stability
        if len(self.prediction_history) >= 2:
            if self.prediction_history[-1] == self.prediction_history[-2]:
                self.stable_confidence_count += 1
            else:
                self.stable_confidence_count = 0

        # Require stable high confidence
        return self.stable_confidence_count >= self.required_stable_updates

    def get_current_prediction(self) -> PredictionResult:
        """Get current prediction with confidence."""
        if self.accumulated_features is None:
            raise RuntimeError("No features accumulated yet")

        with torch.no_grad():
            features_batch = self.accumulated_features.unsqueeze(0)
            logits, attention_weights = self.attention_model(features_batch, return_attention=True)

            probabilities = F.softmax(logits, dim=1)
            confidence = torch.max(probabilities).item()
            prediction = torch.argmax(probabilities, dim=1).item()

        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities.squeeze(0),
            attention_weights=attention_weights.squeeze(0),
            num_patches=self.num_patches,
        )

    def is_confident_enough(self) -> bool:
        """Check if current confidence meets threshold."""
        if not self.confidence_history:
            return False

        current_confidence = self.confidence_history[-1]
        return (
            current_confidence >= self.confidence_threshold
            and self.num_patches >= self.min_patches_for_confidence
        )

    def finalize_prediction(self) -> PredictionResult:
        """Generate final prediction result."""
        if self.accumulated_features is None:
            raise RuntimeError("No features to finalize")

        logger.info(f"Finalizing prediction with {self.num_patches} patches")
        return self.get_current_prediction()

    def get_attention_heatmap(self, slide_dimensions: Tuple[int, int]) -> np.ndarray:
        """Generate attention heatmap for visualization.

        Args:
            slide_dimensions: (width, height) of slide

        Returns:
            Attention heatmap as numpy array
        """
        if self.attention_cache is None or self.accumulated_coordinates is None:
            raise RuntimeError("No attention weights available")

        # Create empty heatmap
        heatmap = np.zeros(slide_dimensions[::-1])  # (height, width)

        # Fill in attention weights at patch coordinates
        attention_np = self.attention_cache.cpu().numpy()

        for i, (x, y) in enumerate(self.accumulated_coordinates):
            if i < len(attention_np):
                # Simple assignment (could be improved with interpolation)
                heatmap[int(y), int(x)] = attention_np[i]

        return heatmap

    def get_confidence_progression(self) -> List[float]:
        """Get confidence progression over time."""
        return self.confidence_history.copy()

    def reset(self):
        """Reset aggregator state."""
        self.accumulated_features = None
        self.accumulated_coordinates = None
        self.num_patches = 0
        self.confidence_history.clear()
        self.prediction_history.clear()
        self.attention_cache = None
        self.stable_confidence_count = 0
        logger.info("Aggregator state reset")

    def get_statistics(self) -> dict:
        """Get aggregator statistics."""
        if not self.confidence_history:
            return {
                "num_patches": 0,
                "current_confidence": 0.0,
                "max_confidence": 0.0,
                "min_confidence": 0.0,
                "avg_confidence": 0.0,
                "confidence_std": 0.0,
                "num_updates": 0,
            }

        return {
            "num_patches": self.num_patches,
            "current_confidence": self.confidence_history[-1],
            "max_confidence": max(self.confidence_history),
            "min_confidence": min(self.confidence_history),
            "avg_confidence": np.mean(self.confidence_history),
            "confidence_std": np.std(self.confidence_history),
            "num_updates": len(self.confidence_history),
        }


class ConfidenceCalibrator:
    """Calibrates confidence estimates for better uncertainty quantification."""

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins
        self.bin_boundaries = np.linspace(0, 1, num_bins + 1)
        self.bin_counts = np.zeros(num_bins)
        self.bin_accuracies = np.zeros(num_bins)

    def update(self, confidences: np.ndarray, correctness: np.ndarray):
        """Update calibration statistics.

        Args:
            confidences: Array of confidence values [0, 1]
            correctness: Array of binary correctness (1=correct, 0=incorrect)
        """
        for i in range(self.num_bins):
            lower = self.bin_boundaries[i]
            upper = self.bin_boundaries[i + 1]

            # Find samples in this bin
            in_bin = (confidences >= lower) & (confidences < upper)

            if np.any(in_bin):
                self.bin_counts[i] += np.sum(in_bin)
                self.bin_accuracies[i] += np.sum(correctness[in_bin])

    def get_calibrated_confidence(self, confidence: float) -> float:
        """Get calibrated confidence value."""
        # Find bin
        bin_idx = np.digitize(confidence, self.bin_boundaries) - 1
        bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)

        # Return calibrated value
        if self.bin_counts[bin_idx] > 0:
            return self.bin_accuracies[bin_idx] / self.bin_counts[bin_idx]
        return confidence  # Fallback to original

    def get_expected_calibration_error(self) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        ece = 0.0
        total_samples = np.sum(self.bin_counts)

        if total_samples == 0:
            return 0.0

        for i in range(self.num_bins):
            if self.bin_counts[i] > 0:
                bin_confidence = (self.bin_boundaries[i] + self.bin_boundaries[i + 1]) / 2
                bin_accuracy = self.bin_accuracies[i] / self.bin_counts[i]
                weight = self.bin_counts[i] / total_samples

                ece += weight * abs(bin_confidence - bin_accuracy)

        return ece
