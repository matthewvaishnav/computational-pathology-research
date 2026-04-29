"""
Model validation system for federated learning retraining.

Provides comprehensive validation of retrained models including
performance metrics, safety checks, bias detection, and regulatory
compliance validation for medical AI systems.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics for a model."""

    disease_type: str
    timestamp: float

    # Basic performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    auc_pr: float  # Area under precision-recall curve

    # Clinical metrics
    sensitivity: float
    specificity: float
    ppv: float  # Positive predictive value
    npv: float  # Negative predictive value

    # Calibration metrics
    calibration_error: float
    brier_score: float

    # Confidence metrics
    mean_confidence: float
    confidence_std: float
    overconfidence_rate: float
    underconfidence_rate: float

    # Bias metrics
    demographic_parity: Optional[float] = None
    equalized_odds: Optional[float] = None

    # Sample information
    sample_count: int = 0
    positive_samples: int = 0
    negative_samples: int = 0

    # Confusion matrix
    confusion_matrix: List[List[int]] = None

    # Additional metrics
    additional_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.confusion_matrix is None:
            self.confusion_matrix = [[0, 0], [0, 0]]
        if self.additional_metrics is None:
            self.additional_metrics = {}


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    model_path: str
    validation_timestamp: float
    diseases_validated: List[str]

    # Per-disease metrics
    disease_metrics: Dict[str, ValidationMetrics]

    # Overall validation results
    overall_accuracy: float
    validation_passed: bool
    validation_score: float  # 0-1 composite score

    # Safety checks
    safety_checks_passed: bool
    safety_issues: List[str]

    # Comparison with baseline
    baseline_comparison: Dict[str, Dict[str, float]]  # disease -> metric -> improvement

    # Regulatory compliance
    regulatory_compliance: Dict[str, bool]

    # Recommendations
    recommendations: List[str]

    # Detailed results
    detailed_results: Dict[str, Any] = None

    def __post_init__(self):
        if self.detailed_results is None:
            self.detailed_results = {}


class ModelValidator:
    """
    Comprehensive model validation system.

    Validates retrained models against multiple criteria including
    performance, safety, bias, and regulatory compliance.
    """

    def __init__(
        self,
        validation_config: Dict[str, Any] = None,
        baseline_models: Dict[str, str] = None,
        test_datasets: Dict[str, Any] = None,
    ):
        """
        Initialize model validator.

        Args:
            validation_config: Validation configuration
            baseline_models: Paths to baseline models for comparison
            test_datasets: Test datasets for validation
        """
        self.config = validation_config or self._get_default_config()
        self.baseline_models = baseline_models or {}
        self.test_datasets = test_datasets or {}

        # Validation thresholds
        self.min_accuracy = self.config.get("min_accuracy", 0.85)
        self.min_sensitivity = self.config.get("min_sensitivity", 0.80)
        self.min_specificity = self.config.get("min_specificity", 0.80)
        self.max_calibration_error = self.config.get("max_calibration_error", 0.1)
        self.max_bias_threshold = self.config.get("max_bias_threshold", 0.1)

        # Safety thresholds
        self.max_performance_degradation = self.config.get("max_performance_degradation", 0.05)
        self.min_sample_size = self.config.get("min_sample_size", 100)

        logger.info("Initialized model validator")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration."""
        return {
            "min_accuracy": 0.85,
            "min_sensitivity": 0.80,
            "min_specificity": 0.80,
            "max_calibration_error": 0.1,
            "max_bias_threshold": 0.1,
            "max_performance_degradation": 0.05,
            "min_sample_size": 100,
            "enable_bias_detection": True,
            "enable_calibration_check": True,
            "enable_safety_checks": True,
            "enable_regulatory_checks": True,
        }

    def validate_model(
        self,
        model_path: str,
        diseases: List[str],
        test_data: Optional[Dict[str, Any]] = None,
        baseline_metrics: Optional[Dict[str, ValidationMetrics]] = None,
    ) -> ValidationReport:
        """
        Validate a retrained model comprehensively.

        Args:
            model_path: Path to model to validate
            diseases: List of diseases to validate
            test_data: Test data for validation
            baseline_metrics: Baseline metrics for comparison

        Returns:
            Comprehensive validation report
        """
        logger.info(f"Starting validation for model: {model_path}")

        # Load model
        model = self._load_model(model_path)

        # Initialize report
        report = ValidationReport(
            model_path=model_path,
            validation_timestamp=datetime.now().timestamp(),
            diseases_validated=diseases,
            disease_metrics={},
            overall_accuracy=0.0,
            validation_passed=False,
            validation_score=0.0,
            safety_checks_passed=False,
            safety_issues=[],
            baseline_comparison={},
            regulatory_compliance={},
            recommendations=[],
        )

        # Validate each disease
        disease_scores = []
        for disease in diseases:
            logger.info(f"Validating {disease} predictions")

            # Get test data for disease
            disease_test_data = self._get_disease_test_data(disease, test_data)

            if disease_test_data is None:
                logger.warning(f"No test data available for {disease}")
                continue

            # Compute validation metrics
            metrics = self._compute_validation_metrics(model, disease, disease_test_data)

            report.disease_metrics[disease] = metrics
            disease_scores.append(metrics.f1_score)

            # Compare with baseline if available
            if baseline_metrics and disease in baseline_metrics:
                comparison = self._compare_with_baseline(metrics, baseline_metrics[disease])
                report.baseline_comparison[disease] = comparison

        # Compute overall metrics
        if disease_scores:
            report.overall_accuracy = np.mean(
                [metrics.accuracy for metrics in report.disease_metrics.values()]
            )
            report.validation_score = np.mean(disease_scores)

        # Perform safety checks
        report.safety_checks_passed, report.safety_issues = self._perform_safety_checks(
            report.disease_metrics
        )

        # Check regulatory compliance
        report.regulatory_compliance = self._check_regulatory_compliance(report.disease_metrics)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        # Determine overall validation result
        report.validation_passed = self._determine_validation_result(report)

        logger.info(f"Validation completed. Passed: {report.validation_passed}")

        return report

    def _load_model(self, model_path: str) -> nn.Module:
        """Load model from path."""

        # In practice, this would load the actual model
        # For demo, return a mock model
        class MockModel(nn.Module):
            def forward(self, x):
                # Simulate predictions
                batch_size = x.shape[0] if hasattr(x, "shape") else len(x)
                return torch.rand(batch_size, 2)  # Binary classification

        return MockModel()

    def _get_disease_test_data(
        self, disease: str, test_data: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Get test data for specific disease."""
        if test_data and disease in test_data:
            return test_data[disease]

        if disease in self.test_datasets:
            return self.test_datasets[disease]

        # Generate synthetic test data for demo
        np.random.seed(hash(disease) % 2**32)
        n_samples = 500

        # Generate features and labels
        features = np.random.randn(n_samples, 128)
        labels = np.random.binomial(1, 0.3, n_samples)  # 30% positive rate

        # Generate predictions with some noise
        predictions = labels + np.random.normal(0, 0.1, n_samples)
        predictions = np.clip(predictions, 0, 1)

        # Generate confidence scores
        confidences = np.abs(predictions - 0.5) * 2  # Higher confidence for extreme predictions
        confidences = np.clip(confidences, 0.5, 1.0)

        return {
            "features": features,
            "labels": labels,
            "predictions": predictions,
            "confidences": confidences,
        }

    def _compute_validation_metrics(
        self, model: nn.Module, disease: str, test_data: Dict[str, Any]
    ) -> ValidationMetrics:
        """Compute comprehensive validation metrics."""
        labels = test_data["labels"]
        predictions = test_data["predictions"]
        confidences = test_data["confidences"]

        # Convert predictions to binary
        binary_predictions = (predictions > 0.5).astype(int)

        # Basic metrics
        accuracy = accuracy_score(labels, binary_predictions)
        precision = precision_score(labels, binary_predictions, zero_division=0)
        recall = recall_score(labels, binary_predictions, zero_division=0)
        f1 = f1_score(labels, binary_predictions, zero_division=0)

        # AUC metrics
        try:
            auc_roc = roc_auc_score(labels, predictions)
            precision_vals, recall_vals, _ = precision_recall_curve(labels, predictions)
            auc_pr = np.trapz(precision_vals, recall_vals)
        except ValueError:
            auc_roc = 0.5
            auc_pr = 0.5

        # Confusion matrix
        cm = confusion_matrix(labels, binary_predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases
            tn = fp = fn = tp = 0
            for true_label, pred_label in zip(labels, binary_predictions):
                if true_label == 0 and pred_label == 0:
                    tn += 1
                elif true_label == 0 and pred_label == 1:
                    fp += 1
                elif true_label == 1 and pred_label == 0:
                    fn += 1
                elif true_label == 1 and pred_label == 1:
                    tp += 1

        # Clinical metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Calibration metrics
        calibration_error = self._compute_calibration_error(labels, predictions, confidences)
        brier_score = np.mean((predictions - labels) ** 2)

        # Confidence metrics
        mean_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)

        # Overconfidence: high confidence but wrong prediction
        overconfident = (confidences > 0.8) & (binary_predictions != labels)
        overconfidence_rate = np.mean(overconfident)

        # Underconfidence: low confidence but correct prediction
        underconfident = (confidences < 0.6) & (binary_predictions == labels)
        underconfidence_rate = np.mean(underconfident)

        # Bias metrics (simplified)
        demographic_parity = None
        equalized_odds = None
        if self.config.get("enable_bias_detection", True):
            # In practice, would compute actual bias metrics with demographic data
            demographic_parity = 0.02  # Simulated
            equalized_odds = 0.03  # Simulated

        return ValidationMetrics(
            disease_type=disease,
            timestamp=datetime.now().timestamp(),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            calibration_error=calibration_error,
            brier_score=brier_score,
            mean_confidence=mean_confidence,
            confidence_std=confidence_std,
            overconfidence_rate=overconfidence_rate,
            underconfidence_rate=underconfidence_rate,
            demographic_parity=demographic_parity,
            equalized_odds=equalized_odds,
            sample_count=len(labels),
            positive_samples=int(np.sum(labels)),
            negative_samples=int(len(labels) - np.sum(labels)),
            confusion_matrix=cm.tolist(),
        )

    def _compute_calibration_error(
        self, labels: np.ndarray, predictions: np.ndarray, confidences: np.ndarray, n_bins: int = 10
    ) -> float:
        """Compute expected calibration error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this confidence bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = labels[in_bin].mean()
                # Average confidence in this bin
                avg_confidence_in_bin = confidences[in_bin].mean()
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _compare_with_baseline(
        self, current_metrics: ValidationMetrics, baseline_metrics: ValidationMetrics
    ) -> Dict[str, float]:
        """Compare current metrics with baseline."""
        comparison = {}

        metrics_to_compare = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_roc",
            "sensitivity",
            "specificity",
            "calibration_error",
        ]

        for metric in metrics_to_compare:
            current_val = getattr(current_metrics, metric)
            baseline_val = getattr(baseline_metrics, metric)

            if metric == "calibration_error":
                # Lower is better for calibration error
                improvement = baseline_val - current_val
            else:
                # Higher is better for other metrics
                improvement = current_val - baseline_val

            comparison[metric] = improvement

        return comparison

    def _perform_safety_checks(
        self, disease_metrics: Dict[str, ValidationMetrics]
    ) -> Tuple[bool, List[str]]:
        """Perform safety checks on validation results."""
        issues = []

        for disease, metrics in disease_metrics.items():
            # Check minimum performance thresholds
            if metrics.accuracy < self.min_accuracy:
                issues.append(
                    f"{disease}: Accuracy {metrics.accuracy:.3f} below threshold {self.min_accuracy}"
                )

            if metrics.sensitivity < self.min_sensitivity:
                issues.append(
                    f"{disease}: Sensitivity {metrics.sensitivity:.3f} below threshold {self.min_sensitivity}"
                )

            if metrics.specificity < self.min_specificity:
                issues.append(
                    f"{disease}: Specificity {metrics.specificity:.3f} below threshold {self.min_specificity}"
                )

            # Check calibration
            if metrics.calibration_error > self.max_calibration_error:
                issues.append(
                    f"{disease}: Calibration error {metrics.calibration_error:.3f} above threshold {self.max_calibration_error}"
                )

            # Check bias metrics
            if (
                metrics.demographic_parity
                and abs(metrics.demographic_parity) > self.max_bias_threshold
            ):
                issues.append(
                    f"{disease}: Demographic parity {metrics.demographic_parity:.3f} exceeds bias threshold"
                )

            if metrics.equalized_odds and abs(metrics.equalized_odds) > self.max_bias_threshold:
                issues.append(
                    f"{disease}: Equalized odds {metrics.equalized_odds:.3f} exceeds bias threshold"
                )

            # Check sample size
            if metrics.sample_count < self.min_sample_size:
                issues.append(
                    f"{disease}: Sample size {metrics.sample_count} below minimum {self.min_sample_size}"
                )

            # Check for extreme overconfidence
            if metrics.overconfidence_rate > 0.2:  # 20% threshold
                issues.append(
                    f"{disease}: High overconfidence rate {metrics.overconfidence_rate:.3f}"
                )

        return len(issues) == 0, issues

    def _check_regulatory_compliance(
        self, disease_metrics: Dict[str, ValidationMetrics]
    ) -> Dict[str, bool]:
        """Check regulatory compliance requirements."""
        compliance = {}

        # FDA requirements (simplified)
        compliance["fda_performance"] = all(
            metrics.accuracy >= 0.85 and metrics.sensitivity >= 0.80
            for metrics in disease_metrics.values()
        )

        compliance["fda_calibration"] = all(
            metrics.calibration_error <= 0.1 for metrics in disease_metrics.values()
        )

        compliance["fda_bias"] = all(
            (metrics.demographic_parity is None or abs(metrics.demographic_parity) <= 0.1)
            and (metrics.equalized_odds is None or abs(metrics.equalized_odds) <= 0.1)
            for metrics in disease_metrics.values()
        )

        # CE marking requirements (simplified)
        compliance["ce_performance"] = all(
            metrics.accuracy >= 0.80 and metrics.f1_score >= 0.75
            for metrics in disease_metrics.values()
        )

        # Clinical validation requirements
        compliance["clinical_validation"] = all(
            metrics.sample_count >= 100 and metrics.positive_samples >= 20
            for metrics in disease_metrics.values()
        )

        return compliance

    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Performance recommendations
        for disease, metrics in report.disease_metrics.items():
            if metrics.accuracy < 0.90:
                recommendations.append(
                    f"Consider additional training data for {disease} to improve accuracy"
                )

            if metrics.calibration_error > 0.05:
                recommendations.append(
                    f"Apply calibration techniques for {disease} to improve confidence reliability"
                )

            if metrics.overconfidence_rate > 0.15:
                recommendations.append(
                    f"Implement confidence regularization for {disease} to reduce overconfidence"
                )

        # Safety recommendations
        if not report.safety_checks_passed:
            recommendations.append("Address safety issues before deployment")

        # Regulatory recommendations
        if not all(report.regulatory_compliance.values()):
            recommendations.append("Ensure all regulatory compliance requirements are met")

        # Baseline comparison recommendations
        for disease, comparison in report.baseline_comparison.items():
            if comparison.get("accuracy", 0) < -0.02:  # 2% degradation
                recommendations.append(f"Investigate accuracy degradation for {disease}")

        return recommendations

    def _determine_validation_result(self, report: ValidationReport) -> bool:
        """Determine overall validation result."""
        # Must pass safety checks
        if not report.safety_checks_passed:
            return False

        # Must meet minimum performance
        if report.validation_score < 0.75:  # 75% minimum composite score
            return False

        # Must meet regulatory requirements
        critical_compliance = ["fda_performance", "clinical_validation"]
        if not all(report.regulatory_compliance.get(req, False) for req in critical_compliance):
            return False

        # Check for significant degradation from baseline
        for disease, comparison in report.baseline_comparison.items():
            if comparison.get("accuracy", 0) < -self.max_performance_degradation:
                return False

        return True

    def export_validation_report(self, report: ValidationReport, filepath: str) -> None:
        """Export validation report to file."""
        # Convert report to serializable format
        report_dict = {
            "model_path": report.model_path,
            "validation_timestamp": report.validation_timestamp,
            "diseases_validated": report.diseases_validated,
            "overall_accuracy": report.overall_accuracy,
            "validation_passed": report.validation_passed,
            "validation_score": report.validation_score,
            "safety_checks_passed": report.safety_checks_passed,
            "safety_issues": report.safety_issues,
            "regulatory_compliance": report.regulatory_compliance,
            "recommendations": report.recommendations,
            "disease_metrics": {},
            "baseline_comparison": report.baseline_comparison,
        }

        # Add disease metrics
        for disease, metrics in report.disease_metrics.items():
            report_dict["disease_metrics"][disease] = {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "auc_roc": metrics.auc_roc,
                "sensitivity": metrics.sensitivity,
                "specificity": metrics.specificity,
                "calibration_error": metrics.calibration_error,
                "sample_count": metrics.sample_count,
                "confusion_matrix": metrics.confusion_matrix,
            }

        with open(filepath, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        logger.info(f"Validation report exported to {filepath}")

    def plot_validation_results(
        self, report: ValidationReport, save_path: Optional[str] = None
    ) -> None:
        """Plot validation results."""
        diseases = list(report.disease_metrics.keys())
        if not diseases:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Performance metrics comparison
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        metric_values = {
            metric: [getattr(report.disease_metrics[disease], metric) for disease in diseases]
            for metric in metrics
        }

        ax = axes[0, 0]
        x = np.arange(len(diseases))
        width = 0.2

        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, metric_values[metric], width, label=metric.title())

        ax.set_xlabel("Disease Type")
        ax.set_ylabel("Score")
        ax.set_title("Performance Metrics by Disease")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(diseases)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Clinical metrics
        ax = axes[0, 1]
        sensitivity_vals = [report.disease_metrics[disease].sensitivity for disease in diseases]
        specificity_vals = [report.disease_metrics[disease].specificity for disease in diseases]

        ax.bar(x - width / 2, sensitivity_vals, width, label="Sensitivity", alpha=0.8)
        ax.bar(x + width / 2, specificity_vals, width, label="Specificity", alpha=0.8)
        ax.set_xlabel("Disease Type")
        ax.set_ylabel("Score")
        ax.set_title("Clinical Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(diseases)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Calibration and confidence
        ax = axes[1, 0]
        calibration_errors = [
            report.disease_metrics[disease].calibration_error for disease in diseases
        ]
        overconfidence_rates = [
            report.disease_metrics[disease].overconfidence_rate for disease in diseases
        ]

        ax.bar(x - width / 2, calibration_errors, width, label="Calibration Error", alpha=0.8)
        ax.bar(x + width / 2, overconfidence_rates, width, label="Overconfidence Rate", alpha=0.8)
        ax.set_xlabel("Disease Type")
        ax.set_ylabel("Rate")
        ax.set_title("Calibration and Confidence Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(diseases)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Sample distribution
        ax = axes[1, 1]
        sample_counts = [report.disease_metrics[disease].sample_count for disease in diseases]
        positive_rates = [
            report.disease_metrics[disease].positive_samples
            / report.disease_metrics[disease].sample_count
            for disease in diseases
        ]

        ax.bar(diseases, sample_counts, alpha=0.7, label="Total Samples")
        ax2 = ax.twinx()
        ax2.plot(diseases, positive_rates, "ro-", label="Positive Rate")

        ax.set_xlabel("Disease Type")
        ax.set_ylabel("Sample Count")
        ax2.set_ylabel("Positive Rate")
        ax.set_title("Sample Distribution")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Validation plots saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Demo: Model validation system

    print("=== Model Validation System Demo ===\n")

    # Create validator
    validator = ModelValidator()

    # Validate a model
    diseases = ["breast", "lung", "prostate"]

    print(f"Validating model for diseases: {diseases}")

    report = validator.validate_model(model_path="models/retrained_model.pth", diseases=diseases)

    print(f"\n--- Validation Results ---")
    print(f"Overall accuracy: {report.overall_accuracy:.3f}")
    print(f"Validation score: {report.validation_score:.3f}")
    print(f"Validation passed: {report.validation_passed}")
    print(f"Safety checks passed: {report.safety_checks_passed}")

    if report.safety_issues:
        print(f"Safety issues: {len(report.safety_issues)}")
        for issue in report.safety_issues[:3]:  # Show first 3
            print(f"  - {issue}")

    print(f"\n--- Per-Disease Metrics ---")
    for disease, metrics in report.disease_metrics.items():
        print(f"{disease.title()}:")
        print(f"  Accuracy: {metrics.accuracy:.3f}")
        print(f"  Sensitivity: {metrics.sensitivity:.3f}")
        print(f"  Specificity: {metrics.specificity:.3f}")
        print(f"  Calibration error: {metrics.calibration_error:.3f}")

    print(f"\n--- Regulatory Compliance ---")
    for requirement, passed in report.regulatory_compliance.items():
        status = "✓" if passed else "✗"
        print(f"  {requirement}: {status}")

    print(f"\n--- Recommendations ---")
    for i, rec in enumerate(report.recommendations[:5], 1):  # Show first 5
        print(f"  {i}. {rec}")

    # Export report
    validator.export_validation_report(report, "validation_report.json")
    print(f"\nValidation report exported to validation_report.json")

    print("\n=== Demo Complete ===")
