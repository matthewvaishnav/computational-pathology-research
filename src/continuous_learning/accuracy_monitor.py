"""
Accuracy monitoring for federated learning drift detection.

Monitors accuracy metrics across disease types, hospitals, and time periods
to detect performance degradation in medical AI systems.
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy metrics for a time period."""

    timestamp: float
    round_number: int
    disease_type: str
    hospital_id: Optional[str]
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: Optional[float]
    sample_count: int
    confusion_matrix: Optional[np.ndarray] = None


@dataclass
class AccuracyAlert:
    """Alert for accuracy degradation."""

    timestamp: float
    alert_type: str  # "accuracy_drop", "precision_drop", "recall_drop", "f1_drop"
    severity: str  # "warning", "critical"
    disease_type: str
    hospital_id: Optional[str]
    current_value: float
    baseline_value: float
    degradation_percent: float
    description: str


class AccuracyMonitor:
    """
    Monitors accuracy metrics for drift detection.

    Tracks performance across:
    - Disease types
    - Hospital sites
    - Time periods
    - Individual metrics (accuracy, precision, recall, F1, AUC)
    """

    def __init__(
        self,
        window_size: int = 50,
        accuracy_threshold: float = 0.05,
        precision_threshold: float = 0.05,
        recall_threshold: float = 0.05,
        f1_threshold: float = 0.05,
        min_samples: int = 10,
    ):
        """Initialize accuracy monitor."""
        self.window_size = window_size
        self.accuracy_threshold = accuracy_threshold
        self.precision_threshold = precision_threshold
        self.recall_threshold = recall_threshold
        self.f1_threshold = f1_threshold
        self.min_samples = min_samples

        # Tracking structures
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_metrics: Dict[str, AccuracyMetrics] = {}
        self.alerts: List[AccuracyAlert] = []
        self.alert_callbacks = []

        logger.info("Initialized accuracy monitor")

    def update_metrics(
        self,
        round_number: int,
        predictions: Dict[str, List],
        ground_truth: Dict[str, List],
        disease_type: str,
        hospital_id: Optional[str] = None,
    ) -> AccuracyMetrics:
        """Update accuracy metrics with new predictions."""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        if len(predictions) < self.min_samples:
            logger.warning(f"Insufficient samples ({len(predictions)}) for reliable metrics")

        timestamp = datetime.now().timestamp()

        # Compute metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision = precision_score(ground_truth, predictions, average="weighted", zero_division=0)
        recall = recall_score(ground_truth, predictions, average="weighted", zero_division=0)
        f1 = f1_score(ground_truth, predictions, average="weighted", zero_division=0)

        # AUC score (if binary classification)
        auc = None
        try:
            if len(set(ground_truth)) == 2:  # Binary classification
                auc = roc_auc_score(ground_truth, predictions)
        except Exception:
            pass

        # Create metrics object
        metrics = AccuracyMetrics(
            timestamp=timestamp,
            round_number=round_number,
            disease_type=disease_type,
            hospital_id=hospital_id,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            sample_count=len(predictions),
        )

        # Store in history
        key = self._get_metrics_key(disease_type, hospital_id)
        self.metrics_history[key].append(metrics)

        # Establish baseline if needed
        if key not in self.baseline_metrics:
            self._establish_baseline(key)

        # Check for degradation
        self._check_accuracy_degradation(key, metrics)

        logger.debug(f"Updated metrics for {key}: acc={accuracy:.4f}, f1={f1:.4f}")

        return metrics

    def _get_metrics_key(self, disease_type: str, hospital_id: Optional[str]) -> str:
        """Generate key for metrics tracking."""
        if hospital_id:
            return f"{disease_type}_{hospital_id}"
        return disease_type

    def _establish_baseline(self, key: str) -> None:
        """Establish baseline metrics."""
        if len(self.metrics_history[key]) < 5:
            return

        # Use first 5 metrics as baseline
        baseline_metrics = list(self.metrics_history[key])[:5]

        # Average the baseline metrics
        avg_accuracy = np.mean([m.accuracy for m in baseline_metrics])
        avg_precision = np.mean([m.precision for m in baseline_metrics])
        avg_recall = np.mean([m.recall for m in baseline_metrics])
        avg_f1 = np.mean([m.f1_score for m in baseline_metrics])
        avg_auc = np.mean([m.auc_score for m in baseline_metrics if m.auc_score is not None])

        # Create baseline
        latest = baseline_metrics[-1]
        self.baseline_metrics[key] = AccuracyMetrics(
            timestamp=latest.timestamp,
            round_number=latest.round_number,
            disease_type=latest.disease_type,
            hospital_id=latest.hospital_id,
            accuracy=avg_accuracy,
            precision=avg_precision,
            recall=avg_recall,
            f1_score=avg_f1,
            auc_score=avg_auc if not np.isnan(avg_auc) else None,
            sample_count=sum(m.sample_count for m in baseline_metrics),
        )

        logger.info(f"Established baseline for {key}: acc={avg_accuracy:.4f}")

    def _check_accuracy_degradation(self, key: str, current_metrics: AccuracyMetrics) -> None:
        """Check for accuracy degradation."""
        if key not in self.baseline_metrics:
            return

        baseline = self.baseline_metrics[key]

        # Check each metric
        checks = [
            ("accuracy", current_metrics.accuracy, baseline.accuracy, self.accuracy_threshold),
            ("precision", current_metrics.precision, baseline.precision, self.precision_threshold),
            ("recall", current_metrics.recall, baseline.recall, self.recall_threshold),
            ("f1_score", current_metrics.f1_score, baseline.f1_score, self.f1_threshold),
        ]

        for metric_name, current_val, baseline_val, threshold in checks:
            degradation = baseline_val - current_val
            if degradation > threshold:
                degradation_percent = (degradation / baseline_val) * 100
                severity = "critical" if degradation > threshold * 2 else "warning"

                self._trigger_accuracy_alert(
                    alert_type=f"{metric_name}_drop",
                    severity=severity,
                    disease_type=current_metrics.disease_type,
                    hospital_id=current_metrics.hospital_id,
                    current_value=current_val,
                    baseline_value=baseline_val,
                    degradation_percent=degradation_percent,
                )

    def _trigger_accuracy_alert(
        self,
        alert_type: str,
        severity: str,
        disease_type: str,
        hospital_id: Optional[str],
        current_value: float,
        baseline_value: float,
        degradation_percent: float,
    ) -> None:
        """Trigger accuracy degradation alert."""
        timestamp = datetime.now().timestamp()

        description = (
            f"{alert_type.replace('_', ' ').title()} detected for {disease_type}"
            f"{f' at {hospital_id}' if hospital_id else ''}: "
            f"{current_value:.4f} vs baseline {baseline_value:.4f} "
            f"({degradation_percent:.1f}% degradation)"
        )

        alert = AccuracyAlert(
            timestamp=timestamp,
            alert_type=alert_type,
            severity=severity,
            disease_type=disease_type,
            hospital_id=hospital_id,
            current_value=current_value,
            baseline_value=baseline_value,
            degradation_percent=degradation_percent,
            description=description,
        )

        self.alerts.append(alert)

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in accuracy alert callback: {e}")

        logger.warning(f"Accuracy alert: {description}")

    def add_alert_callback(self, callback) -> None:
        """Add callback for accuracy alerts."""
        self.alert_callbacks.append(callback)

    def get_current_metrics(
        self, disease_type: str, hospital_id: Optional[str] = None
    ) -> Optional[AccuracyMetrics]:
        """Get most recent metrics."""
        key = self._get_metrics_key(disease_type, hospital_id)
        if key in self.metrics_history and self.metrics_history[key]:
            return self.metrics_history[key][-1]
        return None

    def get_metrics_trend(
        self, disease_type: str, hospital_id: Optional[str] = None, hours: int = 24
    ) -> Dict:
        """Get metrics trend analysis."""
        key = self._get_metrics_key(disease_type, hospital_id)

        if key not in self.metrics_history:
            return {"error": f"No metrics for {key}"}

        # Filter recent metrics
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        recent_metrics = [m for m in self.metrics_history[key] if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {"error": "No recent metrics"}

        # Compute trends
        accuracies = [m.accuracy for m in recent_metrics]
        f1_scores = [m.f1_score for m in recent_metrics]

        return {
            "disease_type": disease_type,
            "hospital_id": hospital_id,
            "time_range_hours": hours,
            "sample_count": len(recent_metrics),
            "current_accuracy": accuracies[-1],
            "baseline_accuracy": (
                self.baseline_metrics[key].accuracy if key in self.baseline_metrics else None
            ),
            "accuracy_trend": (
                np.polyfit(range(len(accuracies)), accuracies, 1)[0] if len(accuracies) > 1 else 0
            ),
            "f1_trend": (
                np.polyfit(range(len(f1_scores)), f1_scores, 1)[0] if len(f1_scores) > 1 else 0
            ),
            "recent_metrics": recent_metrics[-10:],  # Last 10 for details
        }

    def get_alert_summary(self, hours: int = 24) -> Dict:
        """Get summary of recent accuracy alerts."""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        recent_alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]

        # Group by type and severity
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[f"{alert.alert_type}_{alert.severity}"] += 1

        return {
            "total_alerts": len(recent_alerts),
            "time_range_hours": hours,
            "alert_breakdown": dict(alert_counts),
            "affected_diseases": list(set(alert.disease_type for alert in recent_alerts)),
            "affected_hospitals": list(
                set(alert.hospital_id for alert in recent_alerts if alert.hospital_id)
            ),
            "most_recent_alert": recent_alerts[-1] if recent_alerts else None,
        }

    def plot_accuracy_trends(
        self, disease_type: str, hospital_id: Optional[str] = None, save_path: Optional[str] = None
    ) -> None:
        """Plot accuracy trends."""
        key = self._get_metrics_key(disease_type, hospital_id)

        if key not in self.metrics_history or not self.metrics_history[key]:
            logger.warning(f"No metrics to plot for {key}")
            return

        metrics = list(self.metrics_history[key])

        # Extract data
        timestamps = [m.timestamp for m in metrics]
        accuracies = [m.accuracy for m in metrics]
        precisions = [m.precision for m in metrics]
        recalls = [m.recall for m in metrics]
        f1_scores = [m.f1_score for m in metrics]

        # Convert to relative hours
        start_time = timestamps[0]
        time_hours = [(t - start_time) / 3600 for t in timestamps]

        # Create plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        ax1.plot(time_hours, accuracies, "b-", linewidth=2, label="Accuracy")
        if key in self.baseline_metrics:
            baseline_acc = self.baseline_metrics[key].accuracy
            ax1.axhline(y=baseline_acc, color="g", linestyle="--", alpha=0.7, label="Baseline")
        ax1.set_ylabel("Accuracy")
        ax1.set_title(f"Accuracy Trends - {disease_type}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Precision
        ax2.plot(time_hours, precisions, "r-", linewidth=2, label="Precision")
        if key in self.baseline_metrics:
            baseline_prec = self.baseline_metrics[key].precision
            ax2.axhline(y=baseline_prec, color="g", linestyle="--", alpha=0.7, label="Baseline")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision Trends")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Recall
        ax3.plot(time_hours, recalls, "orange", linewidth=2, label="Recall")
        if key in self.baseline_metrics:
            baseline_rec = self.baseline_metrics[key].recall
            ax3.axhline(y=baseline_rec, color="g", linestyle="--", alpha=0.7, label="Baseline")
        ax3.set_xlabel("Time (hours)")
        ax3.set_ylabel("Recall")
        ax3.set_title("Recall Trends")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # F1 Score
        ax4.plot(time_hours, f1_scores, "purple", linewidth=2, label="F1 Score")
        if key in self.baseline_metrics:
            baseline_f1 = self.baseline_metrics[key].f1_score
            ax4.axhline(y=baseline_f1, color="g", linestyle="--", alpha=0.7, label="Baseline")
        ax4.set_xlabel("Time (hours)")
        ax4.set_ylabel("F1 Score")
        ax4.set_title("F1 Score Trends")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved accuracy trends plot to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Demo
    print("=== Accuracy Monitor Demo ===")

    monitor = AccuracyMonitor()

    def alert_handler(alert):
        print(f"🚨 ACCURACY ALERT: {alert.description}")

    monitor.add_alert_callback(alert_handler)

    # Simulate rounds with degrading accuracy
    np.random.seed(42)

    for round_num in range(1, 51):
        # Simulate predictions for breast cancer
        base_accuracy = 0.90

        # Introduce gradual degradation after round 25
        if round_num > 25:
            degradation = (round_num - 25) * 0.01
            base_accuracy -= degradation

        # Generate synthetic predictions and ground truth
        n_samples = 100
        ground_truth = np.random.randint(0, 2, n_samples)

        # Create predictions with target accuracy
        predictions = ground_truth.copy()
        n_errors = int(n_samples * (1 - base_accuracy))
        error_indices = np.random.choice(n_samples, n_errors, replace=False)
        predictions[error_indices] = 1 - predictions[error_indices]

        # Update monitor
        metrics = monitor.update_metrics(
            round_number=round_num,
            predictions=predictions.tolist(),
            ground_truth=ground_truth.tolist(),
            disease_type="breast",
        )

        if round_num % 10 == 0:
            print(f"Round {round_num}: accuracy={metrics.accuracy:.4f}, f1={metrics.f1_score:.4f}")

    # Final analysis
    print("\nFinal Analysis:")
    trend = monitor.get_metrics_trend("breast", hours=24)
    print(f"Accuracy trend slope: {trend['accuracy_trend']:.6f}")

    alert_summary = monitor.get_alert_summary()
    print(f"Total alerts: {alert_summary['total_alerts']}")

    print("Demo complete")
