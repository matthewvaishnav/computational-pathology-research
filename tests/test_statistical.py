"""
Unit tests for statistical utilities (bootstrap confidence intervals).
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from src.utils.statistical import compute_all_metrics_with_ci, compute_bootstrap_ci


class TestComputeBootstrapCI:
    """Test compute_bootstrap_ci function."""

    def test_perfect_predictions(self):
        """Test with perfect predictions (all correct)."""
        n_samples = 100
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = y_true.copy()
        y_prob = y_pred.astype(float)

        metric_fn = lambda yt, yp, yprob: accuracy_score(yt, yp)

        value, ci_lower, ci_upper = compute_bootstrap_ci(
            y_true, y_pred, y_prob, metric_fn, n_bootstrap=100, random_state=42
        )

        # Perfect accuracy
        assert value == 1.0
        # CI should be tight around 1.0
        assert ci_lower >= 0.95
        assert ci_upper == 1.0

    def test_random_predictions(self):
        """Test with random predictions (50% accuracy)."""
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = np.random.randint(0, 2, size=n_samples)
        y_prob = np.random.rand(n_samples)

        metric_fn = lambda yt, yp, yprob: accuracy_score(yt, yp)

        value, ci_lower, ci_upper = compute_bootstrap_ci(
            y_true, y_pred, y_prob, metric_fn, n_bootstrap=100, random_state=42
        )

        # Should be around 50% accuracy
        assert 0.4 < value < 0.6
        # CI should be reasonable
        assert ci_lower < value < ci_upper
        assert ci_upper - ci_lower < 0.2  # CI width should be reasonable

    def test_known_distribution(self):
        """Test with known distribution (80% accuracy)."""
        np.random.seed(42)
        n_samples = 500
        y_true = np.array([0] * 250 + [1] * 250)

        # Create predictions with 80% accuracy
        y_pred = y_true.copy()
        # Flip 20% of predictions
        flip_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
        y_pred[flip_indices] = 1 - y_pred[flip_indices]

        y_prob = y_pred.astype(float)

        metric_fn = lambda yt, yp, yprob: accuracy_score(yt, yp)

        value, ci_lower, ci_upper = compute_bootstrap_ci(
            y_true, y_pred, y_prob, metric_fn, n_bootstrap=200, random_state=42
        )

        # Should be around 80% accuracy
        assert 0.75 < value < 0.85
        # CI should contain the true value
        assert ci_lower <= value <= ci_upper
        # CI should be reasonable width
        assert 0.02 < ci_upper - ci_lower < 0.15

    def test_single_class_edge_case(self):
        """Test with single class (edge case)."""
        n_samples = 100
        y_true = np.ones(n_samples)  # All class 1
        y_pred = np.ones(n_samples)
        y_prob = np.ones(n_samples)

        metric_fn = lambda yt, yp, yprob: accuracy_score(yt, yp)

        value, ci_lower, ci_upper = compute_bootstrap_ci(
            y_true, y_pred, y_prob, metric_fn, n_bootstrap=100, random_state=42
        )

        # Should handle gracefully
        assert value == 1.0
        # CI might be degenerate but should not crash
        assert ci_lower <= ci_upper

    def test_all_wrong_predictions(self):
        """Test with all wrong predictions."""
        n_samples = 100
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = 1 - y_true  # All predictions wrong
        y_prob = y_pred.astype(float)

        metric_fn = lambda yt, yp, yprob: accuracy_score(yt, yp)

        value, ci_lower, ci_upper = compute_bootstrap_ci(
            y_true, y_pred, y_prob, metric_fn, n_bootstrap=100, random_state=42
        )

        # Should be 0% accuracy
        assert value == 0.0
        # CI should be tight around 0.0
        assert ci_lower == 0.0
        assert ci_upper <= 0.05

    def test_confidence_level_95(self):
        """Test that 95% CI is wider than 90% CI."""
        np.random.seed(42)
        n_samples = 500
        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = np.random.randint(0, 2, size=n_samples)
        y_prob = np.random.rand(n_samples)

        metric_fn = lambda yt, yp, yprob: accuracy_score(yt, yp)

        # 90% CI
        _, ci_lower_90, ci_upper_90 = compute_bootstrap_ci(
            y_true, y_pred, y_prob, metric_fn, n_bootstrap=100, confidence_level=0.90, random_state=42
        )

        # 95% CI
        _, ci_lower_95, ci_upper_95 = compute_bootstrap_ci(
            y_true, y_pred, y_prob, metric_fn, n_bootstrap=100, confidence_level=0.95, random_state=42
        )

        # 95% CI should be wider than 90% CI
        width_90 = ci_upper_90 - ci_lower_90
        width_95 = ci_upper_95 - ci_lower_95
        assert width_95 >= width_90

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        np.random.seed(42)
        n_samples = 200
        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = np.random.randint(0, 2, size=n_samples)
        y_prob = np.random.rand(n_samples)

        metric_fn = lambda yt, yp, yprob: accuracy_score(yt, yp)

        # First run
        value1, ci_lower1, ci_upper1 = compute_bootstrap_ci(
            y_true, y_pred, y_prob, metric_fn, n_bootstrap=100, random_state=42
        )

        # Second run with same seed
        value2, ci_lower2, ci_upper2 = compute_bootstrap_ci(
            y_true, y_pred, y_prob, metric_fn, n_bootstrap=100, random_state=42
        )

        # Should be identical
        assert value1 == value2
        assert ci_lower1 == ci_lower2
        assert ci_upper1 == ci_upper2

    def test_2d_probability_array(self):
        """Test with 2D probability array (multiclass format)."""
        n_samples = 100
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = y_true.copy()

        # 2D probability array [N, 2]
        y_prob = np.zeros((n_samples, 2))
        y_prob[y_true == 0, 0] = 1.0
        y_prob[y_true == 1, 1] = 1.0

        metric_fn = lambda yt, yp, yprob: accuracy_score(yt, yp)

        value, ci_lower, ci_upper = compute_bootstrap_ci(
            y_true, y_pred, y_prob, metric_fn, n_bootstrap=100, random_state=42
        )

        # Should handle 2D arrays correctly
        assert value == 1.0
        assert ci_lower <= ci_upper

    def test_small_sample_size(self):
        """Test with small sample size."""
        n_samples = 20
        y_true = np.array([0] * 10 + [1] * 10)
        y_pred = y_true.copy()
        y_prob = y_pred.astype(float)

        metric_fn = lambda yt, yp, yprob: accuracy_score(yt, yp)

        value, ci_lower, ci_upper = compute_bootstrap_ci(
            y_true, y_pred, y_prob, metric_fn, n_bootstrap=50, random_state=42
        )

        # Should still work with small samples
        assert value == 1.0
        assert ci_lower <= ci_upper

    def test_ci_bounds_are_reasonable(self):
        """Test that CI bounds are within [0, 1] for accuracy."""
        np.random.seed(42)
        n_samples = 300
        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = np.random.randint(0, 2, size=n_samples)
        y_prob = np.random.rand(n_samples)

        metric_fn = lambda yt, yp, yprob: accuracy_score(yt, yp)

        value, ci_lower, ci_upper = compute_bootstrap_ci(
            y_true, y_pred, y_prob, metric_fn, n_bootstrap=100, random_state=42
        )

        # Bounds should be in valid range
        assert 0.0 <= ci_lower <= 1.0
        assert 0.0 <= ci_upper <= 1.0
        assert 0.0 <= value <= 1.0


class TestComputeAllMetricsWithCI:
    """Test compute_all_metrics_with_ci function."""

    def test_all_metrics_computed(self):
        """Test that all metrics are computed."""
        np.random.seed(42)
        n_samples = 200
        y_true = np.array([0] * 100 + [1] * 100)
        y_pred = y_true.copy()
        y_prob = y_pred.astype(float)

        results = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob, n_bootstrap=50, random_state=42
        )

        # Check all metrics are present
        assert "accuracy" in results
        assert "auc" in results
        assert "f1" in results
        assert "precision" in results
        assert "recall" in results

        # Check structure
        for metric_name, metric_dict in results.items():
            if "error" not in metric_dict:
                assert "value" in metric_dict
                assert "ci_lower" in metric_dict
                assert "ci_upper" in metric_dict

    def test_perfect_predictions_all_metrics(self):
        """Test all metrics with perfect predictions."""
        n_samples = 100
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = y_true.copy()
        y_prob = y_pred.astype(float)

        results = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob, n_bootstrap=50, random_state=42
        )

        # All metrics should be 1.0 for perfect predictions
        assert results["accuracy"]["value"] == 1.0
        assert results["f1"]["value"] == 1.0
        assert results["precision"]["value"] == 1.0
        assert results["recall"]["value"] == 1.0

        # AUC should also be 1.0
        if "error" not in results["auc"]:
            assert results["auc"]["value"] == 1.0

    def test_binary_classification_1d_probs(self):
        """Test with 1D probability array (binary classification)."""
        np.random.seed(42)
        n_samples = 200
        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = np.random.randint(0, 2, size=n_samples)
        y_prob = np.random.rand(n_samples)

        results = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob, n_bootstrap=50, random_state=42
        )

        # Should handle 1D probabilities
        assert "accuracy" in results
        assert "auc" in results

    def test_binary_classification_2d_probs(self):
        """Test with 2D probability array (binary classification)."""
        np.random.seed(42)
        n_samples = 200
        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = np.random.randint(0, 2, size=n_samples)

        # 2D probabilities [N, 2]
        y_prob = np.random.rand(n_samples, 2)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize

        results = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob, n_bootstrap=50, random_state=42
        )

        # Should handle 2D probabilities
        assert "accuracy" in results
        assert "auc" in results

    def test_edge_case_single_class(self):
        """Test edge case with single class."""
        n_samples = 100
        y_true = np.ones(n_samples)  # All class 1
        y_pred = np.ones(n_samples)
        y_prob = np.ones(n_samples)

        results = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob, n_bootstrap=50, random_state=42
        )

        # Should handle gracefully
        assert results["accuracy"]["value"] == 1.0
        # AUC might fail for single class
        # Other metrics should handle with zero_division=0

    def test_ci_bounds_ordering(self):
        """Test that CI bounds are properly ordered."""
        np.random.seed(42)
        n_samples = 200
        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = np.random.randint(0, 2, size=n_samples)
        y_prob = np.random.rand(n_samples)

        results = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob, n_bootstrap=50, random_state=42
        )

        # Check CI bounds ordering for all metrics
        for metric_name, metric_dict in results.items():
            if "error" not in metric_dict:
                assert metric_dict["ci_lower"] <= metric_dict["value"]
                assert metric_dict["value"] <= metric_dict["ci_upper"]

    def test_reproducibility(self):
        """Test reproducibility with same random seed."""
        np.random.seed(42)
        n_samples = 150
        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = np.random.randint(0, 2, size=n_samples)
        y_prob = np.random.rand(n_samples)

        # First run
        results1 = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob, n_bootstrap=50, random_state=42
        )

        # Second run with same seed
        results2 = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob, n_bootstrap=50, random_state=42
        )

        # Should be identical
        for metric_name in results1.keys():
            if "error" not in results1[metric_name]:
                assert results1[metric_name]["value"] == results2[metric_name]["value"]
                assert results1[metric_name]["ci_lower"] == results2[metric_name]["ci_lower"]
                assert results1[metric_name]["ci_upper"] == results2[metric_name]["ci_upper"]

    def test_different_confidence_levels(self):
        """Test with different confidence levels."""
        np.random.seed(42)
        n_samples = 200
        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = np.random.randint(0, 2, size=n_samples)
        y_prob = np.random.rand(n_samples)

        # 90% CI
        results_90 = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob, n_bootstrap=50, confidence_level=0.90, random_state=42
        )

        # 95% CI
        results_95 = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob, n_bootstrap=50, confidence_level=0.95, random_state=42
        )

        # 95% CI should be wider than 90% CI
        for metric_name in ["accuracy", "f1", "precision", "recall"]:
            width_90 = (
                results_90[metric_name]["ci_upper"] - results_90[metric_name]["ci_lower"]
            )
            width_95 = (
                results_95[metric_name]["ci_upper"] - results_95[metric_name]["ci_lower"]
            )
            assert width_95 >= width_90 - 1e-6  # Allow small numerical differences


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
