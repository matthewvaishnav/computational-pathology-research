"""
Unit tests for out-of-distribution detection.

Tests OOD detection with synthetic out-of-distribution samples using multiple
detection methods (Mahalanobis distance, reconstruction error, ensemble disagreement).
"""

import pytest
import torch

from src.clinical.ood_detection import Autoencoder, OODDetector


class TestOODDetector:
    """Test cases for OODDetector class."""

    @pytest.fixture
    def detector_all_methods(self):
        """OOD detector with all methods enabled."""
        return OODDetector(
            feature_dim=256,
            hidden_dim=128,
            num_ensemble_models=5,
            detection_methods=["mahalanobis", "reconstruction", "ensemble"],
        )

    @pytest.fixture
    def detector_mahalanobis_only(self):
        """OOD detector with only Mahalanobis method."""
        return OODDetector(
            feature_dim=256,
            hidden_dim=128,
            detection_methods=["mahalanobis"],
        )

    @pytest.fixture
    def detector_reconstruction_only(self):
        """OOD detector with only reconstruction method."""
        return OODDetector(
            feature_dim=256,
            hidden_dim=128,
            detection_methods=["reconstruction"],
        )

    @pytest.fixture
    def detector_ensemble_only(self):
        """OOD detector with only ensemble method."""
        return OODDetector(
            feature_dim=256,
            hidden_dim=128,
            num_ensemble_models=5,
            detection_methods=["ensemble"],
        )

    def test_initialization_all_methods(self):
        """Test initialization with all detection methods."""
        detector = OODDetector(
            feature_dim=256,
            hidden_dim=128,
            detection_methods=["mahalanobis", "reconstruction", "ensemble"],
        )

        assert detector.feature_dim == 256
        assert detector.hidden_dim == 128
        assert len(detector.detection_methods) == 3
        assert hasattr(detector, "train_mean")
        assert hasattr(detector, "train_cov_inv")
        assert hasattr(detector, "autoencoder")
        assert hasattr(detector, "ensemble_heads")

    def test_initialization_single_method(self):
        """Test initialization with single detection method."""
        detector = OODDetector(
            feature_dim=256,
            detection_methods=["mahalanobis"],
        )

        assert len(detector.detection_methods) == 1
        assert "mahalanobis" in detector.detection_methods

    def test_initialization_invalid_method(self):
        """Test initialization with invalid detection method."""
        with pytest.raises(ValueError, match="Invalid detection method"):
            OODDetector(feature_dim=256, detection_methods=["invalid_method"])

    def test_forward_output_structure(self, detector_all_methods):
        """Test forward pass output structure."""
        batch_size = 16
        features = torch.randn(batch_size, 256)

        output = detector_all_methods(features)

        # Check output keys
        assert "ood_scores" in output
        assert "is_ood" in output
        assert "method_scores" in output
        assert "explanations" in output

        # Check shapes
        assert output["ood_scores"].shape == (batch_size,)
        assert output["is_ood"].shape == (batch_size,)
        assert output["method_scores"].shape == (batch_size, 3)
        assert len(output["explanations"]) == batch_size

    def test_forward_invalid_features_shape(self, detector_all_methods):
        """Test forward pass with invalid features shape."""
        # 1D features
        with pytest.raises(ValueError, match="Expected 2D features"):
            detector_all_methods(torch.randn(16))

        # Wrong feature dimension
        with pytest.raises(ValueError, match="Expected feature_dim=256"):
            detector_all_methods(torch.randn(16, 128))

    def test_ood_scores_in_range(self, detector_all_methods):
        """Test that OOD scores are in [0, 1]."""
        batch_size = 16
        features = torch.randn(batch_size, 256)

        output = detector_all_methods(features)
        ood_scores = output["ood_scores"]

        assert torch.all(ood_scores >= 0.0)
        assert torch.all(ood_scores <= 1.0)

    def test_method_scores_in_range(self, detector_all_methods):
        """Test that individual method scores are in [0, 1]."""
        batch_size = 16
        features = torch.randn(batch_size, 256)

        output = detector_all_methods(features)
        method_scores = output["method_scores"]

        assert torch.all(method_scores >= 0.0)
        assert torch.all(method_scores <= 1.0)

    def test_is_ood_threshold(self, detector_all_methods):
        """Test OOD flagging with different thresholds."""
        batch_size = 16
        features = torch.randn(batch_size, 256)

        # Low threshold - more cases flagged
        output_low = detector_all_methods(features, threshold=0.3)
        num_ood_low = output_low["is_ood"].sum().item()

        # High threshold - fewer cases flagged
        output_high = detector_all_methods(features, threshold=0.7)
        num_ood_high = output_high["is_ood"].sum().item()

        # Lower threshold should flag more or equal cases
        assert num_ood_low >= num_ood_high

    def test_mahalanobis_not_fitted_warning(self, detector_mahalanobis_only):
        """Test that Mahalanobis detector warns when not fitted."""
        batch_size = 16
        features = torch.randn(batch_size, 256)

        # Should not raise error, but return zero scores
        output = detector_mahalanobis_only(features)

        # Scores should be zero when not fitted
        assert torch.all(output["method_scores"] == 0.0)

    def test_fit_training_distribution(self, detector_mahalanobis_only):
        """Test fitting Mahalanobis detector to training distribution."""
        num_samples = 100
        train_features = torch.randn(num_samples, 256)

        detector_mahalanobis_only.fit_training_distribution(train_features)

        # Check that statistics are updated
        assert detector_mahalanobis_only.mahalanobis_fitted
        assert not torch.all(detector_mahalanobis_only.train_mean == 0.0)

    def test_fit_training_distribution_invalid_shape(self, detector_mahalanobis_only):
        """Test fitting with invalid feature shape."""
        train_features = torch.randn(100, 128)  # Wrong feature dimension

        with pytest.raises(ValueError, match="Expected feature_dim=256"):
            detector_mahalanobis_only.fit_training_distribution(train_features)

    def test_mahalanobis_detects_ood(self, detector_mahalanobis_only):
        """Test that Mahalanobis method detects out-of-distribution samples."""
        # Fit on in-distribution data (centered around 0)
        train_features = torch.randn(100, 256)
        detector_mahalanobis_only.fit_training_distribution(train_features)

        # In-distribution samples (similar to training)
        in_dist_features = torch.randn(16, 256)
        output_in = detector_mahalanobis_only(in_dist_features, threshold=0.5)

        # Out-of-distribution samples (shifted far from training)
        ood_features = torch.randn(16, 256) + 10.0  # Large shift
        output_ood = detector_mahalanobis_only(ood_features, threshold=0.5)

        # OOD samples should have higher or equal scores
        # (may saturate to 1.0 for very far OOD samples)
        assert output_ood["ood_scores"].mean() >= output_in["ood_scores"].mean()

    def test_train_autoencoder(self, detector_reconstruction_only):
        """Test training autoencoder for reconstruction-based detection."""
        num_samples = 100
        train_features = torch.randn(num_samples, 256)

        history = detector_reconstruction_only.train_autoencoder(
            train_features,
            learning_rate=0.001,
            num_epochs=10,
            batch_size=32,
        )

        # Check training history
        assert "losses" in history
        assert len(history["losses"]) == 10

        # Loss should generally decrease
        assert history["losses"][-1] < history["losses"][0]

    def test_train_autoencoder_invalid_shape(self, detector_reconstruction_only):
        """Test training autoencoder with invalid feature shape."""
        train_features = torch.randn(100, 128)  # Wrong feature dimension

        with pytest.raises(ValueError, match="Expected feature_dim=256"):
            detector_reconstruction_only.train_autoencoder(train_features)

    def test_reconstruction_detects_ood(self, detector_reconstruction_only):
        """Test that reconstruction method detects out-of-distribution samples."""
        # Train on in-distribution data
        train_features = torch.randn(100, 256)
        detector_reconstruction_only.train_autoencoder(train_features, num_epochs=20, batch_size=32)

        # In-distribution samples (similar to training)
        in_dist_features = torch.randn(16, 256)
        output_in = detector_reconstruction_only(in_dist_features, threshold=0.5)

        # Out-of-distribution samples (very different from training)
        ood_features = torch.randn(16, 256) * 5.0 + 10.0  # Large scale and shift
        output_ood = detector_reconstruction_only(ood_features, threshold=0.5)

        # OOD samples should have higher scores
        assert output_ood["ood_scores"].mean() > output_in["ood_scores"].mean()

    def test_ensemble_disagreement_scores(self, detector_ensemble_only):
        """Test ensemble disagreement scoring."""
        batch_size = 16
        features = torch.randn(batch_size, 256)

        output = detector_ensemble_only(features)

        # Check that scores are computed
        assert output["ood_scores"].shape == (batch_size,)
        assert torch.all(output["ood_scores"] >= 0.0)
        assert torch.all(output["ood_scores"] <= 1.0)

    def test_explanations_format(self, detector_all_methods):
        """Test that explanations are properly formatted."""
        batch_size = 16
        features = torch.randn(batch_size, 256)

        output = detector_all_methods(features, threshold=0.5)
        explanations = output["explanations"]

        # Check we get explanations for all samples
        assert len(explanations) == batch_size

        # Check explanations are strings
        for explanation in explanations:
            assert isinstance(explanation, str)

    def test_explanations_ood_cases(self, detector_all_methods):
        """Test that OOD cases get appropriate explanations."""
        # Fit Mahalanobis detector
        train_features = torch.randn(100, 256)
        detector_all_methods.fit_training_distribution(train_features)

        # Create clear OOD samples
        ood_features = torch.randn(5, 256) + 10.0

        output = detector_all_methods(ood_features, threshold=0.3)
        explanations = output["explanations"]

        # OOD cases should have "seek expert review" in explanation
        for i, is_ood in enumerate(output["is_ood"]):
            if is_ood:
                assert "seek expert review" in explanations[i]

    def test_explanations_in_distribution_cases(self, detector_all_methods):
        """Test that in-distribution cases get appropriate explanations."""
        # Fit Mahalanobis detector
        train_features = torch.randn(100, 256)
        detector_all_methods.fit_training_distribution(train_features)

        # Create in-distribution samples
        in_dist_features = torch.randn(5, 256)

        output = detector_all_methods(in_dist_features, threshold=0.8)
        explanations = output["explanations"]

        # In-distribution cases should have "In-distribution" in explanation
        for i, is_ood in enumerate(output["is_ood"]):
            if not is_ood:
                assert "In-distribution" in explanations[i]

    def test_combined_methods_average_scores(self, detector_all_methods):
        """Test that combined methods average individual scores."""
        batch_size = 16
        features = torch.randn(batch_size, 256)

        output = detector_all_methods(features)
        ood_scores = output["ood_scores"]
        method_scores = output["method_scores"]

        # Overall scores should be average of method scores
        expected_scores = method_scores.mean(dim=1)
        assert torch.allclose(ood_scores, expected_scores)

    def test_repr(self, detector_all_methods):
        """Test string representation."""
        repr_str = repr(detector_all_methods)

        assert "OODDetector" in repr_str
        assert "feature_dim=256" in repr_str
        assert "detection_methods" in repr_str


class TestAutoencoder:
    """Test cases for Autoencoder class."""

    @pytest.fixture
    def autoencoder(self):
        """Basic autoencoder instance."""
        return Autoencoder(input_dim=256, hidden_dim=128)

    def test_initialization(self):
        """Test autoencoder initialization."""
        autoencoder = Autoencoder(input_dim=256, hidden_dim=128)

        assert hasattr(autoencoder, "encoder")
        assert hasattr(autoencoder, "decoder")

    def test_forward_output_shape(self, autoencoder):
        """Test forward pass output shape."""
        batch_size = 16
        features = torch.randn(batch_size, 256)

        reconstructed = autoencoder(features)

        # Output should have same shape as input
        assert reconstructed.shape == features.shape

    def test_reconstruction_quality(self, autoencoder):
        """Test that autoencoder can reconstruct simple patterns."""
        # Create simple pattern (all ones)
        features = torch.ones(16, 256)

        # Train for a few steps
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
        for _ in range(50):
            reconstructed = autoencoder(features)
            loss = torch.nn.functional.mse_loss(reconstructed, features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # After training, reconstruction should be close to input
        with torch.no_grad():
            reconstructed = autoencoder(features)
            mse = torch.nn.functional.mse_loss(reconstructed, features)

        # MSE should be small after training
        assert mse < 0.1
