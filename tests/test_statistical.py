"""
Tests for statistical utilities.

Tests bootstrap confidence intervals and metric computation.
"""

import pytest
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.utils.statistical import (
    compute_bootstrap_ci,
    compute_all_metrics_with_ci,
)
from src.exceptions import ValidationError


class TestBootstrapCI:
    """Test bootstrap confidence interval computation."""

    def test_bootstrap_ci_basic(self):
        """Test basic bootstrap CI computation."""
        np.random.seed(42)
        
        # Create simple binary classification data
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 10)
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 0] * 10)  # 87.5% accuracy
        y_prob = np.random.rand(80)
        
        # Compute bootstrap CI for accuracy
        point_est, ci_lower, ci_upper = compute_bootstrap_ci(
            y_true, y_pred, y_prob,
            lambda yt, yp, yprob: accuracy_score(yt, yp),
            n_bootstrap=100,
            random_state=42
        )
        
        # Check point estimate
        assert point_est == pytest.approx(0.875, abs=0.01)
        
        # Check CI bounds
        assert ci_lower <= point_est <= ci_upper
        assert 0.0 <= ci_lower <= 1.0
        assert 0.0 <= ci_upper <= 1.0

    def test_bootstrap_ci_perfect_classifier(self):
        """Test bootstrap CI with perfect classifier."""
        y_true = np.array([0, 1, 0, 1, 0, 1] * 10)
        y_pred = np.array([0, 1, 0, 1, 0, 1] * 10)  # Perfect
        y_prob = np.random.rand(60)
        
        point_est, ci_lower, ci_upper = compute_bootstrap_ci(
            y_true, y_pred, y_prob,
            lambda yt, yp, yprob: accuracy_score(yt, yp),
            n_bootstrap=100,
            random_state=42
        )
        
        assert point_est == 1.0
        assert ci_lower >= 0.9  # Should be high
        assert ci_upper == 1.0

    def test_bootstrap_ci_with_2d_probabilities(self):
        """Test bootstrap CI with 2D probability array."""
        np.random.seed(42)
        
        y_true = np.array([0, 1, 0, 1, 0, 1] * 10)
        y_pred = np.array([0, 1, 0, 1, 0, 1] * 10)
        y_prob = np.random.rand(60, 2)  # 2D array
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
        
        point_est, ci_lower, ci_upper = compute_bootstrap_ci(
            y_true, y_pred, y_prob,
            lambda yt, yp, yprob: accuracy_score(yt, yp),
            n_bootstrap=100,
            random_state=42
        )
        
        assert point_est == 1.0
        assert ci_lower <= point_est <= ci_upper

    def test_bootstrap_ci_confidence_levels(self):
        """Test different confidence levels."""
        np.random.seed(42)
        
        y_true = np.array([0, 1, 0, 1, 0, 1] * 20)
        y_pred = np.array([0, 1, 0, 1, 0, 0] * 20)
        y_prob = np.random.rand(120)
        
        # 95% CI
        _, ci_lower_95, ci_upper_95 = compute_bootstrap_ci(
            y_true, y_pred, y_prob,
            lambda yt, yp, yprob: accuracy_score(yt, yp),
            n_bootstrap=100,
            confidence_level=0.95,
            random_state=42
        )
        
        # 99% CI (should be wider)
        _, ci_lower_99, ci_upper_99 = compute_bootstrap_ci(
            y_true, y_pred, y_prob,
            lambda yt, yp, yprob: accuracy_score(yt, yp),
            n_bootstrap=100,
            confidence_level=0.99,
            random_state=42
        )
        
        # 99% CI should be wider than 95% CI
        ci_width_95 = ci_upper_95 - ci_lower_95
        ci_width_99 = ci_upper_99 - ci_lower_99
        assert ci_width_99 >= ci_width_95

    def test_bootstrap_ci_reproducibility(self):
        """Test bootstrap CI is reproducible with same random state."""
        y_true = np.array([0, 1, 0, 1, 0, 1] * 10)
        y_pred = np.array([0, 1, 0, 1, 0, 0] * 10)
        y_prob = np.random.rand(60)
        
        result1 = compute_bootstrap_ci(
            y_true, y_pred, y_prob,
            lambda yt, yp, yprob: accuracy_score(yt, yp),
            n_bootstrap=100,
            random_state=42
        )
        
        result2 = compute_bootstrap_ci(
            y_true, y_pred, y_prob,
            lambda yt, yp, yprob: accuracy_score(yt, yp),
            n_bootstrap=100,
            random_state=42
        )
        
        assert result1 == result2


class TestComputeAllMetrics:
    """Test compute_all_metrics_with_ci function."""

    def test_all_metrics_binary_classification(self):
        """Test all metrics for binary classification."""
        np.random.seed(42)
        
        # Binary classification data
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 10)
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 0] * 10)
        y_prob = np.random.rand(80)
        
        metrics = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob,
            n_bootstrap=50,
            random_state=42
        )
        
        # Check all metrics are present
        assert 'accuracy' in metrics
        assert 'auc' in metrics
        assert 'f1' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        
        # Check structure
        for metric_name, metric_dict in metrics.items():
            assert 'value' in metric_dict
            assert 'ci_lower' in metric_dict
            assert 'ci_upper' in metric_dict
            
            # Check bounds
            assert metric_dict['ci_lower'] <= metric_dict['value'] <= metric_dict['ci_upper']

    def test_all_metrics_with_2d_probabilities(self):
        """Test all metrics with 2D probability array."""
        np.random.seed(42)
        
        y_true = np.array([0, 1, 0, 1, 0, 1] * 10)
        y_pred = np.array([0, 1, 0, 1, 0, 1] * 10)
        y_prob = np.random.rand(60, 2)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        metrics = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob,
            n_bootstrap=50,
            random_state=42
        )
        
        # Should handle 2D probabilities
        assert 'accuracy' in metrics
        assert 'auc' in metrics
        assert metrics['accuracy']['value'] == 1.0

    def test_all_metrics_multiclass(self):
        """Test all metrics for multiclass classification."""
        np.random.seed(42)
        
        # 3-class classification
        y_true = np.array([0, 1, 2, 0, 1, 2] * 10)
        y_pred = np.array([0, 1, 2, 0, 1, 1] * 10)  # Some errors
        y_prob = np.random.rand(60, 3)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        metrics = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob,
            n_bootstrap=50,
            random_state=42
        )
        
        # Check all metrics computed
        assert 'accuracy' in metrics
        assert 'auc' in metrics
        assert 'f1' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics

    def test_all_metrics_perfect_classifier(self):
        """Test all metrics with perfect classifier."""
        y_true = np.array([0, 1, 0, 1, 0, 1] * 10)
        y_pred = np.array([0, 1, 0, 1, 0, 1] * 10)
        y_prob = np.concatenate([
            np.zeros(30),
            np.ones(30)
        ])
        
        metrics = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob,
            n_bootstrap=50,
            random_state=42
        )
        
        # Perfect classifier should have high metrics
        assert metrics['accuracy']['value'] == 1.0
        assert metrics['f1']['value'] == pytest.approx(1.0, abs=0.01)
        assert metrics['precision']['value'] == pytest.approx(1.0, abs=0.01)
        assert metrics['recall']['value'] == pytest.approx(1.0, abs=0.01)

    def test_all_metrics_handles_auc_failure(self):
        """Test all metrics handles AUC computation failure gracefully."""
        # Create data that might cause AUC issues
        y_true = np.array([0, 0, 0, 0])  # Only one class
        y_pred = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])
        
        metrics = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob,
            n_bootstrap=10,
            random_state=42
        )
        
        # Should still compute other metrics
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        
        # AUC might have error field
        if 'error' in metrics.get('auc', {}):
            assert metrics['auc']['value'] == 0.0

    def test_all_metrics_mismatched_shapes_raises_error(self):
        """Test mismatched input shapes raise appropriate error."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1])  # Wrong shape
        y_prob = np.array([[0.1, 0.9], [0.8, 0.2]])  # Wrong shape
        
        # Should raise ValidationError or ValueError or IndexError
        with pytest.raises((ValidationError, ValueError, IndexError)):
            compute_all_metrics_with_ci(y_true, y_pred, y_prob)

    def test_all_metrics_custom_bootstrap_params(self):
        """Test custom bootstrap parameters."""
        np.random.seed(42)
        
        y_true = np.array([0, 1, 0, 1, 0, 1] * 10)
        y_pred = np.array([0, 1, 0, 1, 0, 0] * 10)
        y_prob = np.random.rand(60)
        
        metrics = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob,
            n_bootstrap=200,  # More bootstrap samples
            confidence_level=0.99,  # Higher confidence
            random_state=123
        )
        
        # Should still compute all metrics
        assert len(metrics) == 5
        assert all(k in metrics for k in ['accuracy', 'auc', 'f1', 'precision', 'recall'])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_bootstrap_samples_fallback(self):
        """Test fallback when no valid bootstrap samples."""
        # Create data where bootstrap might fail
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        y_prob = np.array([0.1, 0.9])
        
        # With very few samples, some bootstrap iterations might fail
        point_est, ci_lower, ci_upper = compute_bootstrap_ci(
            y_true, y_pred, y_prob,
            lambda yt, yp, yprob: accuracy_score(yt, yp),
            n_bootstrap=10,
            random_state=42
        )
        
        # Should return point estimate for all values if bootstrap fails
        assert point_est == 1.0
        # CI might equal point estimate if bootstrap failed
        assert ci_lower <= point_est <= ci_upper

    def test_single_class_in_bootstrap_sample(self):
        """Test handling of single class in bootstrap sample."""
        # Create imbalanced data
        y_true = np.array([0, 0, 0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.9])
        
        # Should handle bootstrap samples with only one class
        point_est, ci_lower, ci_upper = compute_bootstrap_ci(
            y_true, y_pred, y_prob,
            lambda yt, yp, yprob: accuracy_score(yt, yp),
            n_bootstrap=50,
            random_state=42
        )
        
        assert point_est == 1.0
        assert ci_lower <= point_est <= ci_upper


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
