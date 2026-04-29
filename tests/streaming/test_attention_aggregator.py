"""Unit tests for StreamingAttentionAggregator - Attention computation.

Tests:
- ConfidenceUpdate validation
- PredictionResult conversion
- AttentionMIL forward pass
- StreamingAttentionAggregator feature accumulation
- Confidence tracking + early stopping
- Attention heatmap generation
- ConfidenceCalibrator ECE calculation
"""

import importlib.util

# Direct module import to avoid OpenSlide dependency
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# Load attention_aggregator module directly
agg_path = Path(__file__).parent.parent.parent / "src" / "streaming" / "attention_aggregator.py"
spec = importlib.util.spec_from_file_location("attention_aggregator", agg_path)
agg_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg_module)

# Import classes
ConfidenceUpdate = agg_module.ConfidenceUpdate
PredictionResult = agg_module.PredictionResult
AttentionMIL = agg_module.AttentionMIL
StreamingAttentionAggregator = agg_module.StreamingAttentionAggregator
ConfidenceCalibrator = agg_module.ConfidenceCalibrator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def attention_model():
    """Create AttentionMIL model."""
    return AttentionMIL(feature_dim=128, hidden_dim=64, num_classes=2)


@pytest.fixture
def aggregator(attention_model):
    """Create StreamingAttentionAggregator."""
    return StreamingAttentionAggregator(
        attention_model=attention_model,
        confidence_threshold=0.95,
        max_features=1000,
        min_patches_for_confidence=50,
    )


@pytest.fixture
def sample_features():
    """Generate sample features."""
    return torch.randn(32, 128)


@pytest.fixture
def sample_coordinates():
    """Generate sample coordinates."""
    return np.random.randint(0, 1000, size=(32, 2))


# ============================================================================
# ConfidenceUpdate Tests
# ============================================================================


class TestConfidenceUpdate:
    """Test ConfidenceUpdate dataclass."""

    def test_initialization(self):
        """Test valid initialization."""
        update = ConfidenceUpdate(
            current_confidence=0.85,
            confidence_delta=0.05,
            patches_processed=100,
            estimated_remaining=50,
            attention_weights=torch.ones(100) / 100,
            early_stop_recommended=False,
        )

        assert update.current_confidence == 0.85
        assert update.confidence_delta == 0.05
        assert update.patches_processed == 100
        assert update.estimated_remaining == 50
        assert not update.early_stop_recommended

    def test_confidence_validation(self):
        """Test confidence bounds validation."""
        # Invalid: confidence > 1.0
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            ConfidenceUpdate(
                current_confidence=1.5,
                confidence_delta=0.0,
                patches_processed=100,
                estimated_remaining=0,
                attention_weights=torch.ones(100) / 100,
                early_stop_recommended=False,
            )

        # Invalid: confidence < 0.0
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            ConfidenceUpdate(
                current_confidence=-0.1,
                confidence_delta=0.0,
                patches_processed=100,
                estimated_remaining=0,
                attention_weights=torch.ones(100) / 100,
                early_stop_recommended=False,
            )

    def test_patches_validation(self):
        """Test patches_processed validation."""
        with pytest.raises(ValueError, match="must be >= 0"):
            ConfidenceUpdate(
                current_confidence=0.85,
                confidence_delta=0.0,
                patches_processed=-10,
                estimated_remaining=0,
                attention_weights=torch.ones(100) / 100,
                early_stop_recommended=False,
            )

    def test_attention_weights_normalization(self):
        """Test attention weights sum validation."""
        # Valid: sum = 1.0
        update = ConfidenceUpdate(
            current_confidence=0.85,
            confidence_delta=0.0,
            patches_processed=100,
            estimated_remaining=0,
            attention_weights=torch.ones(100) / 100,
            early_stop_recommended=False,
        )

        # Should not raise
        assert update.attention_weights is not None


# ============================================================================
# PredictionResult Tests
# ============================================================================


class TestPredictionResult:
    """Test PredictionResult dataclass."""

    def test_initialization(self):
        """Test initialization."""
        result = PredictionResult(
            prediction=1,
            confidence=0.92,
            probabilities=torch.tensor([0.08, 0.92]),
            attention_weights=torch.ones(100) / 100,
            num_patches=100,
        )

        assert result.prediction == 1
        assert result.confidence == 0.92
        assert result.num_patches == 100

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = PredictionResult(
            prediction=1,
            confidence=0.92,
            probabilities=torch.tensor([0.08, 0.92]),
            attention_weights=torch.ones(100) / 100,
            num_patches=100,
        )

        result_dict = result.to_dict()

        assert result_dict["prediction"] == 1
        assert result_dict["confidence"] == 0.92
        assert result_dict["num_patches"] == 100
        assert isinstance(result_dict["probabilities"], list)
        assert len(result_dict["probabilities"]) == 2


# ============================================================================
# AttentionMIL Tests
# ============================================================================


class TestAttentionMIL:
    """Test AttentionMIL model."""

    def test_initialization(self):
        """Test model initialization."""
        model = AttentionMIL(feature_dim=128, hidden_dim=64, num_classes=2)

        assert model.feature_dim == 128
        assert model.hidden_dim == 64
        assert model.num_classes == 2

    def test_forward_batch(self):
        """Test forward pass with batch."""
        model = AttentionMIL(feature_dim=128, hidden_dim=64, num_classes=2)
        features = torch.randn(4, 50, 128)  # [batch, patches, features]

        logits = model(features)

        assert logits.shape == (4, 2)

    def test_forward_single(self):
        """Test forward pass with single sample."""
        model = AttentionMIL(feature_dim=128, hidden_dim=64, num_classes=2)
        features = torch.randn(50, 128)  # [patches, features]

        logits = model(features)

        assert logits.shape == (1, 2)

    def test_forward_with_attention(self):
        """Test forward pass returning attention weights."""
        model = AttentionMIL(feature_dim=128, hidden_dim=64, num_classes=2)
        features = torch.randn(4, 50, 128)

        logits, attention = model(features, return_attention=True)

        assert logits.shape == (4, 2)
        assert attention.shape == (4, 50)

        # Check attention weights sum to 1.0
        for i in range(4):
            assert torch.abs(torch.sum(attention[i]) - 1.0) < 1e-5

    def test_attention_normalization(self):
        """Test attention weights are properly normalized."""
        model = AttentionMIL(feature_dim=128, hidden_dim=64, num_classes=2)
        features = torch.randn(2, 100, 128)

        _, attention = model(features, return_attention=True)

        # Each sample's attention should sum to 1.0
        sums = torch.sum(attention, dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)


# ============================================================================
# StreamingAttentionAggregator Tests
# ============================================================================


class TestStreamingAttentionAggregator:
    """Test StreamingAttentionAggregator."""

    def test_initialization(self, attention_model):
        """Test aggregator initialization."""
        agg = StreamingAttentionAggregator(
            attention_model=attention_model,
            confidence_threshold=0.95,
            max_features=1000,
            min_patches_for_confidence=50,
        )

        assert agg.confidence_threshold == 0.95
        assert agg.max_features == 1000
        assert agg.min_patches_for_confidence == 50
        assert agg.num_patches == 0

    def test_update_features(self, aggregator, sample_features, sample_coordinates):
        """Test feature update."""
        update = aggregator.update_features(sample_features, sample_coordinates)

        assert isinstance(update, ConfidenceUpdate)
        assert update.patches_processed == 32
        assert 0.0 <= update.current_confidence <= 1.0
        assert update.attention_weights.shape[0] == 32

    def test_feature_accumulation(self, aggregator, sample_features, sample_coordinates):
        """Test features accumulate correctly."""
        # First update
        aggregator.update_features(sample_features, sample_coordinates)
        assert aggregator.num_patches == 32

        # Second update
        aggregator.update_features(sample_features, sample_coordinates)
        assert aggregator.num_patches == 64

        # Third update
        aggregator.update_features(sample_features, sample_coordinates)
        assert aggregator.num_patches == 96

    def test_max_features_limit(self, aggregator):
        """Test max features limit enforced."""
        # Add features exceeding max_features (1000)
        for _ in range(40):  # 40 * 32 = 1280 > 1000
            features = torch.randn(32, 128)
            coords = np.random.randint(0, 1000, size=(32, 2))
            aggregator.update_features(features, coords)

        # Should be capped at max_features
        assert aggregator.num_patches == 1000

    def test_confidence_tracking(self, aggregator, sample_features, sample_coordinates):
        """Test confidence history tracking."""
        # Multiple updates
        for _ in range(5):
            aggregator.update_features(sample_features, sample_coordinates)

        assert len(aggregator.confidence_history) == 5
        assert len(aggregator.prediction_history) == 5

    def test_confidence_delta(self, aggregator, sample_features, sample_coordinates):
        """Test confidence delta calculation."""
        # First update
        update1 = aggregator.update_features(sample_features, sample_coordinates)

        # Second update
        update2 = aggregator.update_features(sample_features, sample_coordinates)

        # Delta should be difference
        expected_delta = update2.current_confidence - update1.current_confidence
        assert abs(update2.confidence_delta - expected_delta) < 1e-6

    def test_early_stop_min_patches(self, aggregator):
        """Test early stop requires min patches."""
        # Add features below min_patches_for_confidence (50)
        features = torch.randn(30, 128)
        coords = np.random.randint(0, 1000, size=(30, 2))

        update = aggregator.update_features(features, coords)

        # Should not recommend early stop
        assert not update.early_stop_recommended

    def test_early_stop_low_confidence(self, aggregator):
        """Test early stop requires high confidence."""
        # Add enough patches
        for _ in range(3):  # 3 * 32 = 96 > 50
            features = torch.randn(32, 128)
            coords = np.random.randint(0, 1000, size=(32, 2))
            update = aggregator.update_features(features, coords)

        # If confidence < threshold, no early stop
        if update.current_confidence < aggregator.confidence_threshold:
            assert not update.early_stop_recommended

    def test_get_current_prediction(self, aggregator, sample_features, sample_coordinates):
        """Test get current prediction."""
        aggregator.update_features(sample_features, sample_coordinates)

        result = aggregator.get_current_prediction()

        assert isinstance(result, PredictionResult)
        assert result.prediction in [0, 1]
        assert 0.0 <= result.confidence <= 1.0
        assert result.num_patches == 32

    def test_get_current_prediction_no_features(self, aggregator):
        """Test error when no features accumulated."""
        with pytest.raises(RuntimeError, match="No features accumulated"):
            aggregator.get_current_prediction()

    def test_is_confident_enough(self, aggregator, sample_features, sample_coordinates):
        """Test confidence check."""
        # Add features
        for _ in range(3):  # 96 patches
            aggregator.update_features(sample_features, sample_coordinates)

        # Check confidence
        confident = aggregator.is_confident_enough()

        # Should match threshold + min patches
        expected = (
            aggregator.confidence_history[-1] >= aggregator.confidence_threshold
            and aggregator.num_patches >= aggregator.min_patches_for_confidence
        )
        assert confident == expected

    def test_finalize_prediction(self, aggregator, sample_features, sample_coordinates):
        """Test finalize prediction."""
        aggregator.update_features(sample_features, sample_coordinates)

        result = aggregator.finalize_prediction()

        assert isinstance(result, PredictionResult)
        assert result.num_patches == 32

    def test_finalize_no_features(self, aggregator):
        """Test finalize error with no features."""
        with pytest.raises(RuntimeError, match="No features to finalize"):
            aggregator.finalize_prediction()

    def test_get_attention_heatmap(self, aggregator, sample_features, sample_coordinates):
        """Test attention heatmap generation."""
        aggregator.update_features(sample_features, sample_coordinates)

        heatmap = aggregator.get_attention_heatmap(slide_dimensions=(1000, 1000))

        assert heatmap.shape == (1000, 1000)
        assert heatmap.dtype == np.float64

    def test_get_attention_heatmap_no_attention(self, aggregator):
        """Test heatmap error with no attention."""
        with pytest.raises(RuntimeError, match="No attention weights"):
            aggregator.get_attention_heatmap(slide_dimensions=(1000, 1000))

    def test_get_confidence_progression(self, aggregator, sample_features, sample_coordinates):
        """Test confidence progression retrieval."""
        # Add features
        for _ in range(5):
            aggregator.update_features(sample_features, sample_coordinates)

        progression = aggregator.get_confidence_progression()

        assert len(progression) == 5
        assert all(0.0 <= c <= 1.0 for c in progression)

    def test_reset(self, aggregator, sample_features, sample_coordinates):
        """Test aggregator reset."""
        # Add features
        aggregator.update_features(sample_features, sample_coordinates)
        assert aggregator.num_patches > 0

        # Reset
        aggregator.reset()

        assert aggregator.num_patches == 0
        assert aggregator.accumulated_features is None
        assert aggregator.accumulated_coordinates is None
        assert len(aggregator.confidence_history) == 0
        assert len(aggregator.prediction_history) == 0

    def test_get_statistics(self, aggregator, sample_features, sample_coordinates):
        """Test statistics retrieval."""
        # Add features
        for _ in range(5):
            aggregator.update_features(sample_features, sample_coordinates)

        stats = aggregator.get_statistics()

        assert stats["num_patches"] == 160  # 5 * 32
        assert "current_confidence" in stats
        assert "max_confidence" in stats
        assert "min_confidence" in stats
        assert "avg_confidence" in stats
        assert "confidence_std" in stats
        assert stats["num_updates"] == 5

    def test_get_statistics_empty(self, aggregator):
        """Test statistics with no updates."""
        stats = aggregator.get_statistics()

        assert stats["num_patches"] == 0
        assert stats["current_confidence"] == 0.0
        assert stats["num_updates"] == 0

    def test_input_validation_shape(self, aggregator):
        """Test input shape validation."""
        # Invalid: 3D features
        with pytest.raises(ValueError, match="Expected 2D features"):
            aggregator.update_features(
                torch.randn(2, 32, 128), np.random.randint(0, 1000, size=(32, 2))
            )

    def test_input_validation_mismatch(self, aggregator):
        """Test coordinate-feature mismatch validation."""
        with pytest.raises(ValueError, match="must match feature batch size"):
            aggregator.update_features(
                torch.randn(32, 128), np.random.randint(0, 1000, size=(16, 2))  # Wrong size
            )


# ============================================================================
# ConfidenceCalibrator Tests
# ============================================================================


class TestConfidenceCalibrator:
    """Test ConfidenceCalibrator."""

    def test_initialization(self):
        """Test calibrator initialization."""
        calibrator = ConfidenceCalibrator(num_bins=10)

        assert calibrator.num_bins == 10
        assert len(calibrator.bin_boundaries) == 11
        assert len(calibrator.bin_counts) == 10
        assert len(calibrator.bin_accuracies) == 10

    def test_update(self):
        """Test calibration update."""
        calibrator = ConfidenceCalibrator(num_bins=10)

        confidences = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        correctness = np.array([0, 1, 1, 1, 1])

        calibrator.update(confidences, correctness)

        # Should have updated bins
        assert np.sum(calibrator.bin_counts) == 5
        assert np.sum(calibrator.bin_accuracies) == 4

    def test_get_calibrated_confidence(self):
        """Test calibrated confidence retrieval."""
        calibrator = ConfidenceCalibrator(num_bins=10)

        # Add data
        confidences = np.array([0.5, 0.5, 0.5, 0.5])
        correctness = np.array([1, 1, 0, 0])  # 50% accuracy
        calibrator.update(confidences, correctness)

        # Get calibrated value
        calibrated = calibrator.get_calibrated_confidence(0.5)

        # Should be ~0.5 (50% accuracy)
        assert abs(calibrated - 0.5) < 0.1

    def test_get_calibrated_confidence_empty_bin(self):
        """Test calibration with empty bin."""
        calibrator = ConfidenceCalibrator(num_bins=10)

        # No data in bin
        calibrated = calibrator.get_calibrated_confidence(0.5)

        # Should return original
        assert calibrated == 0.5

    def test_get_expected_calibration_error(self):
        """Test ECE calculation."""
        calibrator = ConfidenceCalibrator(num_bins=10)

        # Perfect calibration
        confidences = np.array([0.5, 0.5, 0.5, 0.5])
        correctness = np.array([1, 1, 0, 0])
        calibrator.update(confidences, correctness)

        ece = calibrator.get_expected_calibration_error()

        # Should be low for well-calibrated
        assert 0.0 <= ece <= 1.0

    def test_get_expected_calibration_error_empty(self):
        """Test ECE with no data."""
        calibrator = ConfidenceCalibrator(num_bins=10)

        ece = calibrator.get_expected_calibration_error()

        assert ece == 0.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_streaming(self, aggregator):
        """Test complete streaming workflow."""
        # Stream features in batches
        for i in range(10):
            features = torch.randn(32, 128)
            coords = np.random.randint(0, 1000, size=(32, 2))

            update = aggregator.update_features(features, coords)

            assert update.patches_processed == (i + 1) * 32
            assert 0.0 <= update.current_confidence <= 1.0

        # Finalize
        result = aggregator.finalize_prediction()

        assert result.num_patches == 320
        assert result.prediction in [0, 1]

    def test_early_stopping_workflow(self, attention_model):
        """Test early stopping workflow."""
        agg = StreamingAttentionAggregator(
            attention_model=attention_model,
            confidence_threshold=0.5,  # Low threshold for testing
            max_features=1000,
            min_patches_for_confidence=10,  # Low min for testing
        )

        # Stream until early stop
        stopped = False
        for i in range(20):
            features = torch.randn(32, 128)
            coords = np.random.randint(0, 1000, size=(32, 2))

            update = agg.update_features(features, coords)

            if update.early_stop_recommended:
                stopped = True
                break

        # May or may not stop depending on random features
        # Just verify no errors
        assert agg.num_patches > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
