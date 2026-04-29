"""
Comprehensive Performance Metrics for Clinical AI Validation

Implements clinical-grade performance metrics including sensitivity, specificity,
calibration, and inter-rater agreement for medical AI systems.
"""

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from statsmodels.stats.inter_rater import cohens_kappa, fleiss_kappa

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""

    # Basic metrics
    accuracy: float
    sensitivity: float  # True positive rate / Recall
    specificity: float  # True negative rate
    precision: float  # Positive predictive value
    npv: float  # Negative predictive value
    f1_score: float

    # Advanced metrics
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None

    # Confidence intervals
    sensitivity_ci: Optional[Tuple[float, float]] = None
    specificity_ci: Optional[Tuple[float, float]] = None
    precision_ci: Optional[Tuple[float, float]] = None
    auc_roc_ci: Optional[Tuple[float, float]] = None

    # Calibration metrics
    calibration_error: Optional[float] = None
    brier_score: Optional[float] = None

    # Sample size
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0

    # Thresholds
    optimal_threshold: Optional[float] = None
    youden_index: Optional[float] = None


class ClinicalPerformanceAnalyzer:
    """Comprehensive performance analysis for clinical AI"""

    def __init__(self, confidence_level: float = 0.95):
        """Initialize performance analyzer"""
        self.confidence_level = confidence_level
        self.z_score = stats.norm.ppf((1 + confidence_level) / 2)

    def calculate_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        predictions_proba: Optional[np.ndarray] = None,
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""

        # Ensure binary predictions
        predictions = np.asarray(predictions, dtype=int)
        true_labels = np.asarray(true_labels, dtype=int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = (
            2 * (precision * sensitivity) / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else 0
        )

        # Confidence intervals
        sensitivity_ci = self._wilson_ci(sensitivity, tp + fn)
        specificity_ci = self._wilson_ci(specificity, tn + fp)
        precision_ci = self._wilson_ci(precision, tp + fp)

        # AUC metrics (if probabilistic predictions available)
        auc_roc = None
        auc_pr = None
        auc_roc_ci = None

        if predictions_proba is not None:
            predictions_proba = np.asarray(predictions_proba)

            # Handle multi-class case
            if predictions_proba.ndim > 1:
                proba = (
                    predictions_proba[:, 1]
                    if predictions_proba.shape[1] == 2
                    else predictions_proba
                )
            else:
                proba = predictions_proba

            if len(np.unique(true_labels)) > 1:
                auc_roc = roc_auc_score(true_labels, proba)
                auc_pr = average_precision_score(true_labels, proba)
                auc_roc_ci = self._bootstrap_ci(
                    true_labels, proba, roc_auc_score, n_iterations=1000
                )

        # Calibration metrics
        calibration_error = None
        brier_score = None

        if predictions_proba is not None:
            calibration_error = self._calculate_calibration_error(predictions_proba, true_labels)
            brier_score = np.mean((predictions_proba - true_labels) ** 2)

        # Find optimal threshold
        optimal_threshold = None
        youden_index = None

        if predictions_proba is not None:
            optimal_threshold, youden_index = self._find_optimal_threshold(
                true_labels, predictions_proba
            )

        return PerformanceMetrics(
            accuracy=accuracy,
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
            npv=npv,
            f1_score=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            sensitivity_ci=sensitivity_ci,
            specificity_ci=specificity_ci,
            precision_ci=precision_ci,
            auc_roc_ci=auc_roc_ci,
            calibration_error=calibration_error,
            brier_score=brier_score,
            n_samples=len(true_labels),
            n_positive=np.sum(true_labels),
            n_negative=len(true_labels) - np.sum(true_labels),
            optimal_threshold=optimal_threshold,
            youden_index=youden_index,
        )

    def _wilson_ci(self, p: float, n: int) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval for proportion"""
        if n == 0:
            return (0.0, 0.0)

        p_adj = (p + self.z_score**2 / (2 * n)) / (1 + self.z_score**2 / n)
        margin = (
            self.z_score
            * np.sqrt(p * (1 - p) / n + self.z_score**2 / (4 * n**2))
            / (1 + self.z_score**2 / n)
        )

        return (max(0, p_adj - margin), min(1, p_adj + margin))

    def _bootstrap_ci(
        self,
        true_labels: np.ndarray,
        predictions_proba: np.ndarray,
        metric_func,
        n_iterations: int = 1000,
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        bootstrap_scores = []
        n = len(true_labels)

        for _ in range(n_iterations):
            indices = np.random.choice(n, size=n, replace=True)
            score = metric_func(true_labels[indices], predictions_proba[indices])
            bootstrap_scores.append(score)

        alpha = (1 - self.confidence_level) / 2
        lower = np.percentile(bootstrap_scores, alpha * 100)
        upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)

        return (lower, upper)

    def _calculate_calibration_error(
        self, predictions_proba: np.ndarray, true_labels: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        bins = np.linspace(0, 1, n_bins + 1)
        calibration_error = 0

        for i in range(len(bins) - 1):
            mask = (predictions_proba >= bins[i]) & (predictions_proba < bins[i + 1])

            if np.sum(mask) > 0:
                bin_accuracy = np.mean(true_labels[mask])
                bin_confidence = np.mean(predictions_proba[mask])
                calibration_error += (
                    np.sum(mask) / len(predictions_proba) * abs(bin_accuracy - bin_confidence)
                )

        return calibration_error

    def _find_optimal_threshold(
        self, true_labels: np.ndarray, predictions_proba: np.ndarray
    ) -> Tuple[float, float]:
        """Find optimal classification threshold using Youden's index"""
        fpr, tpr, thresholds = roc_curve(true_labels, predictions_proba)

        # Youden's index = TPR - FPR
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)

        optimal_threshold = thresholds[optimal_idx]
        optimal_youden = youden_index[optimal_idx]

        return optimal_threshold, optimal_youden

    def calculate_disease_specific_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        disease_labels: np.ndarray,
        predictions_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, PerformanceMetrics]:
        """Calculate metrics for each disease type"""

        disease_metrics = {}
        unique_diseases = np.unique(disease_labels)

        for disease in unique_diseases:
            mask = disease_labels == disease

            disease_preds = predictions[mask]
            disease_labels_subset = true_labels[mask]
            disease_proba = predictions_proba[mask] if predictions_proba is not None else None

            metrics = self.calculate_metrics(disease_preds, disease_labels_subset, disease_proba)
            disease_metrics[str(disease)] = metrics

        return disease_metrics

    def calculate_inter_rater_agreement(
        self, rater1: np.ndarray, rater2: np.ndarray, rater_type: str = "binary"
    ) -> Dict[str, float]:
        """Calculate inter-rater agreement metrics"""

        agreement_metrics = {}

        # Simple agreement
        agreement_metrics["simple_agreement"] = np.mean(rater1 == rater2)

        if rater_type == "binary":
            # Cohen's kappa
            agreement_metrics["cohens_kappa"] = cohens_kappa(np.column_stack([rater1, rater2]))[0]

            # Sensitivity agreement (both positive)
            both_positive = np.sum((rater1 == 1) & (rater2 == 1))
            total_positive = np.sum((rater1 == 1) | (rater2 == 1))
            agreement_metrics["sensitivity_agreement"] = (
                both_positive / total_positive if total_positive > 0 else 0
            )

            # Specificity agreement (both negative)
            both_negative = np.sum((rater1 == 0) & (rater2 == 0))
            total_negative = np.sum((rater1 == 0) | (rater2 == 0))
            agreement_metrics["specificity_agreement"] = (
                both_negative / total_negative if total_negative > 0 else 0
            )

        return agreement_metrics

    def generate_roc_curve(
        self,
        true_labels: np.ndarray,
        predictions_proba: np.ndarray,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Generate ROC curve"""

        fpr, tpr, thresholds = roc_curve(true_labels, predictions_proba)
        auc_score = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        return fig

    def generate_precision_recall_curve(
        self,
        true_labels: np.ndarray,
        predictions_proba: np.ndarray,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Generate precision-recall curve"""

        precision, recall, thresholds = precision_recall_curve(true_labels, predictions_proba)
        ap_score = average_precision_score(true_labels, predictions_proba)

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            recall, precision, color="darkorange", lw=2, label=f"PR curve (AP = {ap_score:.3f})"
        )
        ax.axhline(y=np.mean(true_labels), color="navy", linestyle="--", lw=2, label="Baseline")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)

        return fig

    def generate_calibration_plot(
        self,
        predictions_proba: np.ndarray,
        true_labels: np.ndarray,
        n_bins: int = 10,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Generate calibration plot"""

        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(len(bins) - 1):
            mask = (predictions_proba >= bins[i]) & (predictions_proba < bins[i + 1])

            if np.sum(mask) > 0:
                bin_accuracies.append(np.mean(true_labels[mask]))
                bin_confidences.append(np.mean(predictions_proba[mask]))
                bin_counts.append(np.sum(mask))

        fig, ax = plt.subplots(figsize=figsize)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", lw=2, label="Perfect calibration")

        # Calibration curve
        ax.plot(bin_confidences, bin_accuracies, "o-", lw=2, markersize=8, label="Model")

        # Histogram of predictions
        ax2 = ax.twinx()
        ax2.hist(
            predictions_proba, bins=bins, alpha=0.3, color="gray", label="Prediction distribution"
        )
        ax2.set_ylabel("Frequency")

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Plot")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        return fig

    def generate_metrics_report(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""

        report = {
            "summary": {
                "accuracy": f"{metrics.accuracy:.4f}",
                "sensitivity": f"{metrics.sensitivity:.4f}",
                "specificity": f"{metrics.specificity:.4f}",
                "precision": f"{metrics.precision:.4f}",
                "npv": f"{metrics.npv:.4f}",
                "f1_score": f"{metrics.f1_score:.4f}",
            },
            "advanced_metrics": {
                "auc_roc": f"{metrics.auc_roc:.4f}" if metrics.auc_roc else "N/A",
                "auc_pr": f"{metrics.auc_pr:.4f}" if metrics.auc_pr else "N/A",
                "calibration_error": (
                    f"{metrics.calibration_error:.4f}" if metrics.calibration_error else "N/A"
                ),
                "brier_score": f"{metrics.brier_score:.4f}" if metrics.brier_score else "N/A",
            },
            "confidence_intervals": {
                "sensitivity": (
                    f"({metrics.sensitivity_ci[0]:.4f}, {metrics.sensitivity_ci[1]:.4f})"
                    if metrics.sensitivity_ci
                    else "N/A"
                ),
                "specificity": (
                    f"({metrics.specificity_ci[0]:.4f}, {metrics.specificity_ci[1]:.4f})"
                    if metrics.specificity_ci
                    else "N/A"
                ),
                "precision": (
                    f"({metrics.precision_ci[0]:.4f}, {metrics.precision_ci[1]:.4f})"
                    if metrics.precision_ci
                    else "N/A"
                ),
                "auc_roc": (
                    f"({metrics.auc_roc_ci[0]:.4f}, {metrics.auc_roc_ci[1]:.4f})"
                    if metrics.auc_roc_ci
                    else "N/A"
                ),
            },
            "sample_info": {
                "total_samples": metrics.n_samples,
                "positive_samples": metrics.n_positive,
                "negative_samples": metrics.n_negative,
                "prevalence": (
                    f"{metrics.n_positive / metrics.n_samples:.4f}" if metrics.n_samples > 0 else 0
                ),
            },
            "optimal_threshold": {
                "threshold": (
                    f"{metrics.optimal_threshold:.4f}" if metrics.optimal_threshold else "N/A"
                ),
                "youden_index": f"{metrics.youden_index:.4f}" if metrics.youden_index else "N/A",
            },
        }

        return report


# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ClinicalPerformanceAnalyzer(confidence_level=0.95)

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    true_labels = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    predictions_proba = np.random.random(n_samples)

    # Make predictions correlated with true labels
    predictions_proba[true_labels == 1] += 0.3
    predictions_proba = np.clip(predictions_proba, 0, 1)

    predictions = (predictions_proba > 0.5).astype(int)

    print("Clinical Performance Metrics Analysis")
    print("=" * 50)

    # Calculate metrics
    metrics = analyzer.calculate_metrics(predictions, true_labels, predictions_proba)

    # Generate report
    report = analyzer.generate_metrics_report(metrics)

    print("\nPerformance Summary:")
    for key, value in report["summary"].items():
        print(f"  {key.title()}: {value}")

    print("\nAdvanced Metrics:")
    for key, value in report["advanced_metrics"].items():
        print(f"  {key.title()}: {value}")

    print("\nConfidence Intervals (95%):")
    for key, value in report["confidence_intervals"].items():
        print(f"  {key.title()}: {value}")

    print("\nSample Information:")
    for key, value in report["sample_info"].items():
        print(f"  {key.title()}: {value}")

    print("\nOptimal Threshold:")
    for key, value in report["optimal_threshold"].items():
        print(f"  {key.title()}: {value}")
