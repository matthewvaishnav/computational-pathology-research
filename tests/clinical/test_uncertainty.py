"""
Unit tests for uncertainty quantification.

Tests calibration methods (temperature scaling, Platt scaling), calibration metrics
(ECE, MCE, Brier Score), and uncertainty explanation generation.
"""

import pytest
import torch

from src.clinical.uncertainty import UncertaintyQuantifier


class TestUncertaintyQuantifier:
    """Test cases for UncertaintyQuantifier class."""

    @pytest.fixture
    def quantifier_temperature(self):
        """Temperature scaling quantifier."""
        return UncertaintyQuantifier(
            num_classes=5,
            calibration_method="temperature",
            initial_temperature=1.0,
        )

    @pytest.fixture
    def quantifier_platt(self):
        """Platt scaling quantifier."""
        return UncertaintyQuantifier(
            num_classes=5,
            calibration_method="platt",
        )

    def test_initialization_temperature(self):
        """Test initialization with temperature scaling."""
        quantifier = UncertaintyQuantifier(num_classes=5, calibration_method="temperature")

        assert quantifier.num_classes == 5
        assert quantifier.calibration_method == "temperature"
        assert hasattr(quantifier, "temperature")
        assert quantifier.temperature.item() == 1.0

    def test_initialization_platt(self):
        """Test initialization with Platt scaling."""
        quantifier = UncertaintyQuantifier(num_classes=5, calibration_method="platt")

        assert quantifier.num_classes == 5
        assert quantifier.calibration_method == "platt"
        assert hasattr(quantifier, "platt_scale")
        assert hasattr(quantifier, "platt_bias")

    def test_initialization_invalid_num_classes(self):
        """Test initialization with invalid num_classes."""
        with pytest.raises(ValueError, match="num_classes must be >= 2"):
            UncertaintyQuantifier(num_classes=1)

    def test_initialization_invalid_calibration_method(self):
        """Test initialization with invalid calibration method."""
        with pytest.raises(ValueError, match="calibration_method must be"):
            UncertaintyQuantifier(num_classes=5, calibration_method="invalid")

    def test_forward_output_structure(self, quantifier_temperature):
        """Test forward pass output structure."""
        batch_size = 16
        logits = torch.randn(batch_size, 5)

        output = quantifier_temperature(logits)

        # Check output keys
        assert "calibrated_probabilities" in output
        assert "uncertainty_scores" in output
        assert "primary_uncertainty" in output
        assert "top3_uncertainties" in output
        assert "uncertainty_explanation" in output

        # Check shapes
        assert output["calibrated_probabilities"].shape == (batch_size, 5)
        assert output["uncertainty_scores"].shape == (batch_size,)
        assert output["primary_uncertainty"].shape == (batch_size,)
        assert output["top3_uncertainties"].shape == (batch_size, 3)
        assert output["uncertainty_explanation"].shape == (batch_size, 3)

    def test_forward_with_ood_scores(self, quantifier_temperature):
        """Test forward pass with OOD scores."""
        batch_size = 16
        logits = torch.randn(batch_size, 5)
        ood_scores = torch.rand(batch_size)

        output = quantifier_temperature(logits, ood_scores=ood_scores)

        assert "uncertainty_scores" in output
        assert output["uncertainty_scores"].shape == (batch_size,)

    def test_forward_with_data_quality_scores(self, quantifier_temperature):
        """Test forward pass with data quality scores."""
        batch_size = 16
        logits = torch.randn(batch_size, 5)
        data_quality_scores = torch.rand(batch_size)

        output = quantifier_temperature(logits, data_quality_scores=data_quality_scores)

        assert "uncertainty_scores" in output
        assert output["uncertainty_scores"].shape == (batch_size,)

    def test_forward_invalid_logits_shape(self, quantifier_temperature):
        """Test forward pass with invalid logits shape."""
        # 1D logits
        with pytest.raises(ValueError, match="Expected 2D logits"):
            quantifier_temperature(torch.randn(16))

        # Wrong number of classes
        with pytest.raises(ValueError, match="Expected num_classes=5"):
            quantifier_temperature(torch.randn(16, 3))

    def test_forward_invalid_ood_scores_shape(self, quantifier_temperature):
        """Test forward pass with invalid OOD scores shape."""
        logits = torch.randn(16, 5)
        ood_scores = torch.rand(8)  # Wrong batch size

        with pytest.raises(ValueError, match="ood_scores shape"):
            quantifier_temperature(logits, ood_scores=ood_scores)

    def test_forward_invalid_data_quality_scores_shape(self, quantifier_temperature):
        """Test forward pass with invalid data quality scores shape."""
        logits = torch.randn(16, 5)
        data_quality_scores = torch.rand(8)  # Wrong batch size

        with pytest.raises(ValueError, match="data_quality_scores shape"):
            quantifier_temperature(logits, data_quality_scores=data_quality_scores)

    def test_calibrated_probabilities_sum_to_one(self, quantifier_temperature):
        """Test that calibrated probabilities sum to 1.0."""
        batch_size = 16
        logits = torch.randn(batch_size, 5)

        output = quantifier_temperature(logits)
        prob_sums = output["calibrated_probabilities"].sum(dim=1)

        # Check all sums are close to 1.0
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)

    def test_calibrated_probabilities_in_range(self, quantifier_temperature):
        """Test that calibrated probabilities are in [0, 1]."""
        batch_size = 16
        logits = torch.randn(batch_size, 5)

        output = quantifier_temperature(logits)
        probs = output["calibrated_probabilities"]

        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)

    def test_uncertainty_scores_in_range(self, quantifier_temperature):
        """Test that uncertainty scores are in [0, 1]."""
        batch_size = 16
        logits = torch.randn(batch_size, 5)

        output = quantifier_temperature(logits)
        uncertainty = output["uncertainty_scores"]

        assert torch.all(uncertainty >= 0.0)
        assert torch.all(uncertainty <= 1.0)

    def test_temperature_scaling_effect(self):
        """Test that temperature scaling affects probabilities."""
        quantifier = UncertaintyQuantifier(num_classes=5, calibration_method="temperature")
        logits = torch.randn(16, 5)

        # Get probabilities with default temperature (1.0)
        output1 = quantifier(logits)
        probs1 = output1["calibrated_probabilities"]

        # Change temperature
        quantifier.temperature.data.fill_(2.0)
        output2 = quantifier(logits)
        probs2 = output2["calibrated_probabilities"]

        # Probabilities should be different
        assert not torch.allclose(probs1, probs2)

    def test_platt_scaling_effect(self):
        """Test that Platt scaling affects probabilities."""
        quantifier = UncertaintyQuantifier(num_classes=5, calibration_method="platt")
        logits = torch.randn(16, 5)

        # Get probabilities with default parameters
        output1 = quantifier(logits)
        probs1 = output1["calibrated_probabilities"]

        # Change Platt parameters
        quantifier.platt_scale.data.fill_(2.0)
        quantifier.platt_bias.data.fill_(0.5)
        output2 = quantifier(logits)
        probs2 = output2["calibrated_probabilities"]

        # Probabilities should be different
        assert not torch.allclose(probs1, probs2)

    def test_calibrate_on_validation(self, quantifier_temperature):
        """Test calibration on validation data."""
        num_samples = 100
        val_logits = torch.randn(num_samples, 5)
        val_labels = torch.randint(0, 5, (num_samples,))

        metrics = quantifier_temperature.calibrate_on_validation(
            val_logits, val_labels, learning_rate=0.01, max_iterations=50
        )

        # Check metrics are returned
        assert "ece" in metrics
        assert "mce" in metrics
        assert "brier_score" in metrics

        # Check metrics are valid
        assert 0.0 <= metrics["ece"] <= 1.0
        assert 0.0 <= metrics["mce"] <= 1.0
        assert 0.0 <= metrics["brier_score"] <= 2.0

    def test_calibrate_on_validation_invalid_shapes(self, quantifier_temperature):
        """Test calibration with mismatched shapes."""
        val_logits = torch.randn(100, 5)
        val_labels = torch.randint(0, 5, (50,))  # Wrong size

        with pytest.raises(ValueError, match="doesn't match labels"):
            quantifier_temperature.calibrate_on_validation(val_logits, val_labels)

    def test_compute_calibration_metrics(self, quantifier_temperature):
        """Test calibration metrics computation."""
        num_samples = 100
        probabilities = torch.softmax(torch.randn(num_samples, 5), dim=1)
        labels = torch.randint(0, 5, (num_samples,))

        metrics = quantifier_temperature.compute_calibration_metrics(probabilities, labels)

        # Check metrics are returned
        assert "ece" in metrics
        assert "mce" in metrics
        assert "brier_score" in metrics

        # Check metrics are valid
        assert 0.0 <= metrics["ece"] <= 1.0
        assert 0.0 <= metrics["mce"] <= 1.0
        assert 0.0 <= metrics["brier_score"] <= 2.0

    def test_get_calibration_curve(self, quantifier_temperature):
        """Test calibration curve computation."""
        num_samples = 100
        probabilities = torch.softmax(torch.randn(num_samples, 5), dim=1)
        labels = torch.randint(0, 5, (num_samples,))

        bin_confidences, bin_accuracies, bin_counts = quantifier_temperature.get_calibration_curve(
            probabilities, labels, num_bins=10
        )

        # Check shapes
        assert bin_confidences.shape == (10,)
        assert bin_accuracies.shape == (10,)
        assert bin_counts.shape == (10,)

        # Check values are valid
        assert torch.all(bin_confidences >= 0.0) and torch.all(bin_confidences <= 1.0)
        assert torch.all(bin_accuracies >= 0.0) and torch.all(bin_accuracies <= 1.0)
        assert torch.all(bin_counts >= 0)

    def test_generate_uncertainty_explanation(self, quantifier_temperature):
        """Test uncertainty explanation generation."""
        batch_size = 5
        uncertainty_breakdown = torch.tensor(
            [
                [0.1, 0.1, 0.1],  # Low uncertainty
                [0.5, 0.1, 0.1],  # Model uncertainty
                [0.1, 0.5, 0.1],  # Data quality uncertainty
                [0.1, 0.1, 0.5],  # OOD uncertainty
                [0.4, 0.4, 0.4],  # High overall uncertainty
            ]
        )

        explanations = quantifier_temperature.generate_uncertainty_explanation(
            uncertainty_breakdown
        )

        # Check we get explanations for all samples
        assert len(explanations) == batch_size

        # Check explanations are strings
        for explanation in explanations:
            assert isinstance(explanation, str)

        # Check high uncertainty case includes "seek expert review"
        assert "seek expert review" in explanations[4]

    def test_top3_uncertainties_ordered(self, quantifier_temperature):
        """Test that top-3 uncertainties are ordered by probability."""
        batch_size = 16
        logits = torch.randn(batch_size, 5)

        output = quantifier_temperature(logits)
        top3_uncertainties = output["top3_uncertainties"]

        # Top-3 uncertainties should be ordered (lowest to highest)
        # because they're 1 - probability, and probabilities are ordered high to low
        for i in range(batch_size):
            assert top3_uncertainties[i, 0] <= top3_uncertainties[i, 1]
            assert top3_uncertainties[i, 1] <= top3_uncertainties[i, 2]

    def test_primary_uncertainty_matches_top3(self, quantifier_temperature):
        """Test that primary uncertainty matches first top-3 uncertainty."""
        batch_size = 16
        logits = torch.randn(batch_size, 5)

        output = quantifier_temperature(logits)
        primary_uncertainty = output["primary_uncertainty"]
        top3_uncertainties = output["top3_uncertainties"]

        # Primary uncertainty should match first top-3 uncertainty
        assert torch.allclose(primary_uncertainty, top3_uncertainties[:, 0])

    def test_entropy_uncertainty_uniform_distribution(self, quantifier_temperature):
        """Test entropy uncertainty for uniform distribution (maximum entropy)."""
        batch_size = 16
        # Create uniform probabilities (maximum entropy)
        uniform_probs = torch.ones(batch_size, 5) / 5.0

        entropy_uncertainty = quantifier_temperature._compute_entropy_uncertainty(uniform_probs)

        # Uniform distribution should have maximum normalized entropy (close to 1.0)
        assert torch.all(entropy_uncertainty > 0.9)

    def test_entropy_uncertainty_certain_distribution(self, quantifier_temperature):
        """Test entropy uncertainty for certain distribution (minimum entropy)."""
        batch_size = 16
        # Create certain probabilities (minimum entropy)
        certain_probs = torch.zeros(batch_size, 5)
        certain_probs[:, 0] = 1.0

        entropy_uncertainty = quantifier_temperature._compute_entropy_uncertainty(certain_probs)

        # Certain distribution should have minimum normalized entropy (close to 0.0)
        assert torch.all(entropy_uncertainty < 0.1)

    def test_repr(self, quantifier_temperature):
        """Test string representation."""
        repr_str = repr(quantifier_temperature)

        assert "UncertaintyQuantifier" in repr_str
        assert "num_classes=5" in repr_str
        assert "temperature" in repr_str
