"""
Statistical utilities for computing bootstrap confidence intervals.

This module provides functions for computing bootstrap confidence intervals
for classification metrics, enabling statistically rigorous performance reporting.
"""

from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth labels [N]
        y_pred: Predicted labels [N]
        y_prob: Predicted probabilities [N] or [N, num_classes]
        metric_fn: Function that computes metric from (y_true, y_pred, y_prob)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)

    Example:
        >>> accuracy_ci = compute_bootstrap_ci(
        ...     y_true, y_pred, y_prob,
        ...     lambda yt, yp, yprob: accuracy_score(yt, yp),
        ...     n_bootstrap=1000
        ... )
        >>> print(f"Accuracy: {accuracy_ci[0]:.4f} [{accuracy_ci[1]:.4f}, {accuracy_ci[2]:.4f}]")
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)

    # Compute point estimate
    point_estimate = metric_fn(y_true, y_pred, y_prob)

    # Bootstrap resampling
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Handle both 1D and 2D probability arrays
        if y_prob.ndim == 1:
            y_prob_boot = y_prob[indices]
        else:
            y_prob_boot = y_prob[indices, :]

        # Skip if bootstrap sample has only one class
        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            score = metric_fn(y_true_boot, y_pred_boot, y_prob_boot)
            bootstrap_scores.append(score)
        except Exception:
            # Skip bootstrap samples that cause errors
            continue

    if len(bootstrap_scores) == 0:
        # Fallback if no valid bootstrap samples
        return point_estimate, point_estimate, point_estimate

    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)

    return point_estimate, ci_lower, ci_upper


def compute_all_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute all classification metrics with bootstrap CIs.

    Args:
        y_true: Ground truth labels [N]
        y_pred: Predicted labels [N]
        y_prob: Predicted probabilities [N] or [N, num_classes]
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with structure:
        {
            'accuracy': {'value': 0.95, 'ci_lower': 0.94, 'ci_upper': 0.96},
            'auc': {'value': 0.97, 'ci_lower': 0.96, 'ci_upper': 0.98},
            'f1': {'value': 0.94, 'ci_lower': 0.93, 'ci_upper': 0.95},
            'precision': {'value': 0.93, 'ci_lower': 0.92, 'ci_upper': 0.94},
            'recall': {'value': 0.95, 'ci_lower': 0.94, 'ci_upper': 0.96}
        }

    Example:
        >>> metrics = compute_all_metrics_with_ci(y_true, y_pred, y_prob)
        >>> print(f"Accuracy: {metrics['accuracy']['value']:.4f} "
        ...       f"[{metrics['accuracy']['ci_lower']:.4f}, "
        ...       f"{metrics['accuracy']['ci_upper']:.4f}]")
    """
    results = {}

    # Accuracy
    acc_value, acc_lower, acc_upper = compute_bootstrap_ci(
        y_true,
        y_pred,
        y_prob,
        lambda yt, yp, yprob: accuracy_score(yt, yp),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )
    results["accuracy"] = {"value": acc_value, "ci_lower": acc_lower, "ci_upper": acc_upper}

    # AUC (handle binary and multiclass)
    try:
        # For binary classification, y_prob can be 1D or 2D
        if y_prob.ndim == 1 or (y_prob.ndim == 2 and y_prob.shape[1] == 2):
            # Binary classification
            if y_prob.ndim == 2:
                y_prob_for_auc = y_prob[:, 1]  # Use probability of positive class
            else:
                y_prob_for_auc = y_prob

            auc_value, auc_lower, auc_upper = compute_bootstrap_ci(
                y_true,
                y_pred,
                y_prob_for_auc,
                lambda yt, yp, yprob: roc_auc_score(yt, yprob),
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_state=random_state,
            )
        else:
            # Multiclass classification
            auc_value, auc_lower, auc_upper = compute_bootstrap_ci(
                y_true,
                y_pred,
                y_prob,
                lambda yt, yp, yprob: roc_auc_score(yt, yprob, multi_class="ovr", average="macro"),
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_state=random_state,
            )

        results["auc"] = {"value": auc_value, "ci_lower": auc_lower, "ci_upper": auc_upper}
    except Exception as e:
        # AUC computation can fail for edge cases
        results["auc"] = {"value": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "error": str(e)}

    # F1 Score
    f1_value, f1_lower, f1_upper = compute_bootstrap_ci(
        y_true,
        y_pred,
        y_prob,
        lambda yt, yp, yprob: f1_score(yt, yp, average="macro", zero_division=0),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )
    results["f1"] = {"value": f1_value, "ci_lower": f1_lower, "ci_upper": f1_upper}

    # Precision
    prec_value, prec_lower, prec_upper = compute_bootstrap_ci(
        y_true,
        y_pred,
        y_prob,
        lambda yt, yp, yprob: precision_score(yt, yp, average="macro", zero_division=0),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )
    results["precision"] = {"value": prec_value, "ci_lower": prec_lower, "ci_upper": prec_upper}

    # Recall
    rec_value, rec_lower, rec_upper = compute_bootstrap_ci(
        y_true,
        y_pred,
        y_prob,
        lambda yt, yp, yprob: recall_score(yt, yp, average="macro", zero_division=0),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )
    results["recall"] = {"value": rec_value, "ci_lower": rec_lower, "ci_upper": rec_upper}

    return results
