"""
Performance monitoring infrastructure for clinical deployment.

Provides real-time monitoring of model performance including concept drift
detection, distribution shift detection, and automated alerting.
"""

import logging
import smtplib
import time
from collections import defaultdict, deque
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import requests
import torch
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Real-time performance monitoring for deployed models.

    Monitors model performance over time, detects concept drift and
    distribution shift, and provides automated alerting capabilities.
    """

    def __init__(
        self, window_size: int = 1000, drift_threshold: float = 0.05, alert_threshold: float = 0.1
    ):
        """
        Initialize performance monitor.

        Args:
            window_size: Size of sliding window for metrics
            drift_threshold: Threshold for concept drift detection
            alert_threshold: Threshold for performance alerts
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.alert_threshold = alert_threshold

        # Performance tracking
        self.performance_history = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=window_size)
        self.label_history = deque(maxlen=window_size)

        # Baseline performance
        self.baseline_accuracy = None
        self.baseline_distribution = None

        # Alert callbacks
        self.alert_callbacks = []

        # Monitoring state
        self.is_monitoring = False
        self.last_alert_time = 0
        self.alert_cooldown = 3600  # 1 hour

    def start_monitoring(self):
        """Start performance monitoring."""
        self.is_monitoring = True
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        logger.info("Performance monitoring stopped")

    def track_performance(
        self,
        predictions: np.ndarray,
        labels: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
    ):
        """
        Track model performance for a batch of predictions.

        Args:
            predictions: Model predictions
            labels: Ground truth labels (if available)
            probabilities: Prediction probabilities
        """
        if not self.is_monitoring:
            return

        timestamp = time.time()

        # Store predictions and labels
        self.prediction_history.extend(predictions)
        if labels is not None:
            self.label_history.extend(labels)

        # Calculate performance metrics if labels available
        if labels is not None and len(labels) > 0:
            accuracy = accuracy_score(labels, predictions)

            performance_record = {
                "timestamp": timestamp,
                "accuracy": accuracy,
                "sample_count": len(predictions),
            }

            self.performance_history.append(performance_record)

            # Check for performance degradation
            if self.detect_performance_degradation():
                self._trigger_alerts(
                    "performance_degradation",
                    {
                        "current_accuracy": accuracy,
                        "baseline_accuracy": self.baseline_accuracy,
                        "degradation": self.baseline_accuracy - accuracy,
                    },
                )

        # Check for concept drift
        if self.detect_concept_drift():
            self._trigger_alerts("concept_drift", {"drift_detected": True, "timestamp": timestamp})

    def detect_performance_degradation(self) -> bool:
        """
        Detect if model performance has degraded significantly.

        Returns:
            True if performance degradation detected
        """
        if len(self.performance_history) < 10:
            return False

        if self.baseline_accuracy is None:
            # Set baseline from first 100 samples
            if len(self.performance_history) >= 100:
                baseline_records = list(self.performance_history)[:100]
                self.baseline_accuracy = np.mean([r["accuracy"] for r in baseline_records])
            return False

        # Calculate recent performance
        recent_records = list(self.performance_history)[-10:]
        recent_accuracy = np.mean([r["accuracy"] for r in recent_records])

        # Check for significant degradation
        degradation = self.baseline_accuracy - recent_accuracy
        return degradation > self.alert_threshold

    def detect_concept_drift(self) -> bool:
        """
        Detect concept drift using prediction distribution changes.

        Returns:
            True if concept drift detected
        """
        if len(self.prediction_history) < self.window_size:
            return False

        # Get recent and historical predictions
        all_predictions = list(self.prediction_history)
        recent_predictions = all_predictions[-self.window_size // 4 :]
        historical_predictions = all_predictions[: self.window_size // 4]

        if len(recent_predictions) < 50 or len(historical_predictions) < 50:
            return False

        # Calculate distribution shift using simple statistical test
        recent_mean = np.mean(recent_predictions)
        historical_mean = np.mean(historical_predictions)

        # Simple threshold-based drift detection
        drift_magnitude = abs(recent_mean - historical_mean)
        return drift_magnitude > self.drift_threshold

    def recommend_retraining(self) -> Dict[str, Any]:
        """
        Analyze performance and recommend if retraining is needed.

        Returns:
            Dictionary with retraining recommendation
        """
        recommendation = {"should_retrain": False, "confidence": 0.0, "reasons": []}

        # Check performance degradation
        if self.detect_performance_degradation():
            recommendation["should_retrain"] = True
            recommendation["reasons"].append("Performance degradation detected")
            recommendation["confidence"] += 0.4

        # Check concept drift
        if self.detect_concept_drift():
            recommendation["should_retrain"] = True
            recommendation["reasons"].append("Concept drift detected")
            recommendation["confidence"] += 0.3

        # Check data volume
        if len(self.performance_history) > self.window_size * 0.8:
            recommendation["reasons"].append("Sufficient new data available")
            recommendation["confidence"] += 0.2

        # Check time since last training
        # This would require tracking last training time
        recommendation["confidence"] = min(recommendation["confidence"], 1.0)

        return recommendation

    def add_alert_callback(self, callback: Callable[[str, Dict], None]):
        """
        Add callback function for alerts.

        Args:
            callback: Function to call when alert is triggered
        """
        self.alert_callbacks.append(callback)

    def _trigger_alerts(self, alert_type: str, alert_data: Dict[str, Any]):
        """
        Trigger alerts for performance issues.

        Args:
            alert_type: Type of alert
            alert_data: Alert data
        """
        current_time = time.time()

        # Check cooldown period
        if current_time - self.last_alert_time < self.alert_cooldown:
            return

        self.last_alert_time = current_time

        # Format alert message
        alert_message = self._format_alert_message(alert_type, alert_data)

        # Send alerts
        for callback in self.alert_callbacks:
            try:
                callback(
                    alert_type,
                    {"message": alert_message, "data": alert_data, "timestamp": current_time},
                )
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        # Built-in alert methods
        self._send_email_alert(alert_message)
        self._send_webhook_alert(alert_type, alert_data)

    def _format_alert_message(self, alert_type: str, alert_data: Dict) -> str:
        """Format alert message for notifications."""
        if alert_type == "performance_degradation":
            return (
                f"Model performance degradation detected!\n"
                f"Current accuracy: {alert_data.get('current_accuracy', 'N/A'):.3f}\n"
                f"Baseline accuracy: {alert_data.get('baseline_accuracy', 'N/A'):.3f}\n"
                f"Degradation: {alert_data.get('degradation', 'N/A'):.3f}"
            )
        elif alert_type == "concept_drift":
            return (
                f"Concept drift detected!\n"
                f"Model predictions may be unreliable.\n"
                f"Consider retraining the model."
            )
        else:
            return f"Alert: {alert_type} - {alert_data}"

    def _send_email_alert(self, message: str):
        """Send email alert (placeholder implementation)."""
        # This would require email configuration
        logger.warning(f"EMAIL ALERT: {message}")

    def _send_webhook_alert(self, alert_type: str, alert_data: Dict):
        """Send webhook alert (placeholder implementation)."""
        # This would require webhook configuration
        logger.warning(f"WEBHOOK ALERT: {alert_type} - {alert_data}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of recent performance.

        Returns:
            Performance summary dictionary
        """
        if not self.performance_history:
            return {"status": "no_data"}

        recent_records = list(self.performance_history)[-10:]

        summary = {
            "status": "monitoring",
            "total_samples": sum(r["sample_count"] for r in self.performance_history),
            "recent_accuracy": np.mean([r["accuracy"] for r in recent_records]),
            "baseline_accuracy": self.baseline_accuracy,
            "performance_degradation": self.detect_performance_degradation(),
            "concept_drift": self.detect_concept_drift(),
            "monitoring_window_size": len(self.performance_history),
            "retraining_recommendation": self.recommend_retraining(),
        }

        return summary

    def export_performance_history(self, filepath: str):
        """
        Export performance history to file.

        Args:
            filepath: Path to save performance history
        """
        import json

        history_data = {
            "performance_history": list(self.performance_history),
            "baseline_accuracy": self.baseline_accuracy,
            "monitoring_config": {
                "window_size": self.window_size,
                "drift_threshold": self.drift_threshold,
                "alert_threshold": self.alert_threshold,
            },
        }

        with open(filepath, "w") as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"Performance history exported to {filepath}")
