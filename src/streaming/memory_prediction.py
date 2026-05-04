"""Memory usage prediction for WSI streaming."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryPrediction:
    """Memory usage prediction."""

    predicted_peak_gb: float
    predicted_avg_gb: float
    confidence: float
    based_on_samples: int
    slide_characteristics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predicted_peak_gb": self.predicted_peak_gb,
            "predicted_avg_gb": self.predicted_avg_gb,
            "confidence": self.confidence,
            "based_on_samples": self.based_on_samples,
            "slide_characteristics": self.slide_characteristics,
        }


class MemoryUsagePredictor:
    """Predicts memory usage based on slide characteristics.

    Features:
    - Historical usage pattern analysis
    - Slide characteristic-based prediction
    - Confidence estimation
    - Preallocation recommendations
    """

    def __init__(self, enable_learning: bool = True):
        """Initialize memory usage predictor.

        Args:
            enable_learning: Enable learning from historical data
        """
        self.enable_learning = enable_learning

        # Historical data
        self.usage_history: List[Tuple[Dict[str, Any], float, float]] = []
        # Format: (slide_characteristics, peak_memory_gb, avg_memory_gb)

        # Prediction models (simple heuristics for now)
        self.base_memory_gb = 0.5  # Base memory overhead
        self.memory_per_patch_mb = 2.0  # Memory per patch in MB
        self.memory_per_feature_mb = 0.5  # Memory per feature in MB

        logger.info("MemoryUsagePredictor initialized")

    def predict(self, slide_characteristics: Dict[str, Any]) -> MemoryPrediction:
        """Predict memory usage for a slide.

        Args:
            slide_characteristics: Dictionary with slide properties
                - dimensions: (width, height)
                - estimated_patches: int
                - tile_size: int
                - batch_size: int
                - feature_dim: int

        Returns:
            Memory prediction with confidence
        """
        # Extract characteristics
        dimensions = slide_characteristics.get("dimensions", (10000, 10000))
        estimated_patches = slide_characteristics.get("estimated_patches", 1000)
        tile_size = slide_characteristics.get("tile_size", 224)
        batch_size = slide_characteristics.get("batch_size", 32)
        feature_dim = slide_characteristics.get("feature_dim", 512)

        # Base prediction using heuristics
        # Memory = base + (batch_size * tile_memory) + (patches * feature_memory)

        # Tile memory: batch_size * channels * tile_size^2 * bytes_per_element
        tile_memory_gb = (batch_size * 3 * tile_size * tile_size * 4) / (1024**3)

        # Feature memory: estimated_patches * feature_dim * bytes_per_element
        feature_memory_gb = (estimated_patches * feature_dim * 4) / (1024**3)

        # Peak memory (during processing)
        predicted_peak_gb = self.base_memory_gb + tile_memory_gb + feature_memory_gb

        # Average memory (steady state)
        predicted_avg_gb = self.base_memory_gb + (feature_memory_gb * 0.7)

        # Adjust based on historical data if available
        if self.enable_learning and self.usage_history:
            predicted_peak_gb, predicted_avg_gb = self._adjust_with_history(
                slide_characteristics, predicted_peak_gb, predicted_avg_gb
            )

        # Calculate confidence based on historical data
        confidence = self._calculate_confidence(slide_characteristics)

        return MemoryPrediction(
            predicted_peak_gb=predicted_peak_gb,
            predicted_avg_gb=predicted_avg_gb,
            confidence=confidence,
            based_on_samples=len(self.usage_history),
            slide_characteristics=slide_characteristics,
        )

    def _adjust_with_history(
        self, slide_characteristics: Dict[str, Any], predicted_peak: float, predicted_avg: float
    ) -> Tuple[float, float]:
        """Adjust prediction using historical data.

        Args:
            slide_characteristics: Current slide characteristics
            predicted_peak: Initial peak prediction
            predicted_avg: Initial average prediction

        Returns:
            Adjusted (peak, avg) predictions
        """
        # Find similar slides in history
        similar_samples = []

        current_patches = slide_characteristics.get("estimated_patches", 0)
        current_tile_size = slide_characteristics.get("tile_size", 224)

        for hist_chars, hist_peak, hist_avg in self.usage_history:
            hist_patches = hist_chars.get("estimated_patches", 0)
            hist_tile_size = hist_chars.get("tile_size", 224)

            # Simple similarity: within 20% of patches and same tile size
            if (
                abs(hist_patches - current_patches) / max(1, current_patches) < 0.2
                and hist_tile_size == current_tile_size
            ):
                similar_samples.append((hist_peak, hist_avg))

        # If we have similar samples, blend predictions
        if similar_samples:
            hist_peak_avg = np.mean([s[0] for s in similar_samples])
            hist_avg_avg = np.mean([s[1] for s in similar_samples])

            # Blend: 70% historical, 30% heuristic
            adjusted_peak = 0.7 * hist_peak_avg + 0.3 * predicted_peak
            adjusted_avg = 0.7 * hist_avg_avg + 0.3 * predicted_avg

            return adjusted_peak, adjusted_avg

        return predicted_peak, predicted_avg

    def _calculate_confidence(self, slide_characteristics: Dict[str, Any]) -> float:
        """Calculate prediction confidence.

        Args:
            slide_characteristics: Slide characteristics

        Returns:
            Confidence score [0, 1]
        """
        if not self.usage_history:
            return 0.5  # Medium confidence with no history

        # Find similar samples
        current_patches = slide_characteristics.get("estimated_patches", 0)
        similar_count = 0

        for hist_chars, _, _ in self.usage_history:
            hist_patches = hist_chars.get("estimated_patches", 0)
            if abs(hist_patches - current_patches) / max(1, current_patches) < 0.2:
                similar_count += 1

        # Confidence based on number of similar samples
        # 0 similar: 0.5, 5+ similar: 0.9
        confidence = 0.5 + min(0.4, similar_count * 0.08)

        return confidence

    def record_usage(
        self, slide_characteristics: Dict[str, Any], peak_memory_gb: float, avg_memory_gb: float
    ):
        """Record actual memory usage for learning.

        Args:
            slide_characteristics: Slide characteristics
            peak_memory_gb: Actual peak memory usage
            avg_memory_gb: Actual average memory usage
        """
        if not self.enable_learning:
            return

        self.usage_history.append((slide_characteristics.copy(), peak_memory_gb, avg_memory_gb))

        # Keep only recent history (last 100 slides)
        if len(self.usage_history) > 100:
            self.usage_history = self.usage_history[-100:]

        logger.debug(f"Recorded usage: peak={peak_memory_gb:.2f}GB, avg={avg_memory_gb:.2f}GB")

    def get_preallocation_recommendation(self, slide_characteristics: Dict[str, Any]) -> float:
        """Get recommended preallocation size.

        Args:
            slide_characteristics: Slide characteristics

        Returns:
            Recommended preallocation size in GB
        """
        prediction = self.predict(slide_characteristics)

        # Preallocate based on predicted peak with safety margin
        # Higher confidence → smaller margin
        safety_margin = 1.0 + (0.5 * (1.0 - prediction.confidence))

        recommended_gb = prediction.predicted_peak_gb * safety_margin

        return recommended_gb
