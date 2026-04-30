#!/usr/bin/env python3
"""
Model Management System for Real-Time WSI Streaming

Tracks model performance over time, detects drift and degradation,
implements automated retraining triggers, and provides model security.
"""

import hashlib
import json
import logging
import queue
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from cryptography.fernet import Fernet

# Import BoundedQueue and GracefulThread for memory-safe queue operations and graceful shutdown
from src.utils.safe_threading import BoundedQueue, GracefulThread
from src.utils.safe_operations import safe_db_transaction

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetric:
    """Single model performance measurement."""

    timestamp: str
    model_version: str
    accuracy: float
    confidence_score: float
    processing_time_ms: int
    slide_type: str
    prediction_class: str
    ground_truth: Optional[str] = None
    uncertainty_score: Optional[float] = None


@dataclass
class ModelDriftAlert:
    """Model drift detection alert."""

    alert_id: str
    timestamp: str
    model_version: str
    drift_type: str  # 'accuracy', 'confidence', 'distribution'
    severity: str  # 'low', 'medium', 'high', 'critical'
    current_value: float
    baseline_value: float
    threshold_exceeded: float
    recommendation: str


@dataclass
class ModelSecurityStatus:
    """Model security and integrity status."""

    model_version: str
    integrity_verified: bool
    signature_valid: bool
    last_verification: str
    security_hash: str
    encryption_status: str
    adversarial_robustness_score: float


class ModelPerformanceTracker:
    """Tracks model accuracy and confidence over time."""

    def __init__(self, db_path: str = "model_performance.db"):
        """Initialize performance tracker.

        Args:
            db_path: Path to SQLite database for storing metrics
        """
        self.db_path = db_path
        self.drift_thresholds = {
            "accuracy_drop": 0.05,  # 5% accuracy drop triggers alert
            "confidence_shift": 0.1,  # 10% confidence shift
            "processing_time_increase": 0.2,  # 20% processing time increase
        }

        self._init_database()
        logger.info("ModelPerformanceTracker initialized")

    def record_performance_metric(self, metric: ModelPerformanceMetric):
        """Record a single performance metric.

        Args:
            metric: ModelPerformanceMetric to record
        """
        try:
            with safe_db_transaction(Path(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO performance_metrics 
                    (timestamp, model_version, accuracy, confidence_score, 
                     processing_time_ms, slide_type, prediction_class, 
                     ground_truth, uncertainty_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        metric.timestamp,
                        metric.model_version,
                        metric.accuracy,
                        metric.confidence_score,
                        metric.processing_time_ms,
                        metric.slide_type,
                        metric.prediction_class,
                        metric.ground_truth,
                        metric.uncertainty_score,
                    ),
                )

        except Exception as e:
            logger.error(f"Failed to record performance metric: {e}")

    def get_performance_history(
        self, model_version: str, days: int = 30
    ) -> List[ModelPerformanceMetric]:
        """Get performance history for a model version.

        Args:
            model_version: Model version to query
            days: Number of days of history to retrieve

        Returns:
            List of ModelPerformanceMetric objects
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            with safe_db_transaction(Path(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM performance_metrics 
                    WHERE model_version = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """,
                    (model_version, cutoff_date),
                )

                rows = cursor.fetchall()

                metrics = []
                for row in rows:
                    metric = ModelPerformanceMetric(
                        timestamp=row[1],
                        model_version=row[2],
                        accuracy=row[3],
                        confidence_score=row[4],
                        processing_time_ms=row[5],
                        slide_type=row[6],
                        prediction_class=row[7],
                        ground_truth=row[8],
                        uncertainty_score=row[9],
                    )
                    metrics.append(metric)

                return metrics

        except Exception as e:
            logger.error(f"Failed to get performance history: {e}")
            return []

    def calculate_performance_trends(self, model_version: str) -> Dict[str, float]:
        """Calculate performance trends for drift detection.

        Args:
            model_version: Model version to analyze

        Returns:
            Dictionary with trend analysis results
        """
        metrics = self.get_performance_history(model_version, days=30)

        if len(metrics) < 10:
            logger.warning(f"Insufficient data for trend analysis: {len(metrics)} metrics")
            return {}

        # Split into recent and baseline periods
        recent_metrics = metrics[: len(metrics) // 3]  # Most recent 1/3
        baseline_metrics = metrics[len(metrics) // 3 :]  # Older 2/3

        # Calculate averages
        recent_accuracy = np.mean([m.accuracy for m in recent_metrics])
        baseline_accuracy = np.mean([m.accuracy for m in baseline_metrics])

        recent_confidence = np.mean([m.confidence_score for m in recent_metrics])
        baseline_confidence = np.mean([m.confidence_score for m in baseline_metrics])

        recent_processing_time = np.mean([m.processing_time_ms for m in recent_metrics])
        baseline_processing_time = np.mean([m.processing_time_ms for m in baseline_metrics])

        # Calculate trends
        accuracy_trend = (recent_accuracy - baseline_accuracy) / baseline_accuracy
        confidence_trend = (recent_confidence - baseline_confidence) / baseline_confidence
        processing_time_trend = (
            recent_processing_time - baseline_processing_time
        ) / baseline_processing_time

        return {
            "accuracy_trend": accuracy_trend,
            "confidence_trend": confidence_trend,
            "processing_time_trend": processing_time_trend,
            "recent_accuracy": recent_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "recent_confidence": recent_confidence,
            "baseline_confidence": baseline_confidence,
            "data_points": len(metrics),
        }

    def _init_database(self):
        """Initialize SQLite database for performance tracking."""
        try:
            with safe_db_transaction(Path(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        accuracy REAL NOT NULL,
                        confidence_score REAL NOT NULL,
                        processing_time_ms INTEGER NOT NULL,
                        slide_type TEXT NOT NULL,
                        prediction_class TEXT NOT NULL,
                        ground_truth TEXT,
                        uncertainty_score REAL
                    )
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model_timestamp 
                    ON performance_metrics(model_version, timestamp)
                """)

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")


class ModelDriftDetector:
    """Detects model drift and performance degradation."""

    def __init__(self, performance_tracker: ModelPerformanceTracker):
        """Initialize drift detector.

        Args:
            performance_tracker: ModelPerformanceTracker instance
        """
        self.performance_tracker = performance_tracker
        # Bounded queue to prevent memory exhaustion from alert accumulation
        self.alert_queue = BoundedQueue(
            maxsize=1000,
            drop_policy='oldest',
            name='alert_queue'
        )
        self.monitoring_active = False
        self.monitor_thread: Optional[GracefulThread] = None

        logger.info("ModelDriftDetector initialized")

    def start_monitoring(self, model_version: str, check_interval_minutes: int = 60):
        """Start continuous drift monitoring.

        Args:
            model_version: Model version to monitor
            check_interval_minutes: How often to check for drift
        """
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        
        def cleanup_callback():
            """Cleanup callback for graceful shutdown."""
            logger.info("Drift monitoring cleanup completed")
        
        self.monitor_thread = GracefulThread(
            target=self._monitoring_loop,
            name="drift_monitor",
            daemon=False,
            cleanup_callback=cleanup_callback
        )
        # Pass arguments through a wrapper since GracefulThread target receives thread as first arg
        self._monitor_args = (model_version, check_interval_minutes)
        self.monitor_thread.start()

        logger.info(f"Started drift monitoring for {model_version}")

    def stop_monitoring(self):
        """Stop drift monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            if not self.monitor_thread.stop(timeout=5.0):
                logger.warning("Drift monitor thread did not stop within timeout")
            self.monitor_thread = None

        logger.info("Stopped drift monitoring")

    def check_for_drift(self, model_version: str) -> List[ModelDriftAlert]:
        """Check for model drift and generate alerts.

        Args:
            model_version: Model version to check

        Returns:
            List of ModelDriftAlert objects
        """
        alerts = []

        try:
            trends = self.performance_tracker.calculate_performance_trends(model_version)

            if not trends:
                return alerts

            # Check accuracy drift
            if (
                trends["accuracy_trend"]
                < -self.performance_tracker.drift_thresholds["accuracy_drop"]
            ):
                severity = self._calculate_severity(abs(trends["accuracy_trend"]), 0.05, 0.1, 0.15)
                alert = ModelDriftAlert(
                    alert_id=f"drift_{int(time.time())}",
                    timestamp=datetime.now().isoformat(),
                    model_version=model_version,
                    drift_type="accuracy",
                    severity=severity,
                    current_value=trends["recent_accuracy"],
                    baseline_value=trends["baseline_accuracy"],
                    threshold_exceeded=abs(trends["accuracy_trend"]),
                    recommendation=self._get_accuracy_recommendation(trends["accuracy_trend"]),
                )
                alerts.append(alert)

            # Check confidence drift
            if (
                abs(trends["confidence_trend"])
                > self.performance_tracker.drift_thresholds["confidence_shift"]
            ):
                severity = self._calculate_severity(abs(trends["confidence_trend"]), 0.1, 0.2, 0.3)
                alert = ModelDriftAlert(
                    alert_id=f"drift_{int(time.time())}_conf",
                    timestamp=datetime.now().isoformat(),
                    model_version=model_version,
                    drift_type="confidence",
                    severity=severity,
                    current_value=trends["recent_confidence"],
                    baseline_value=trends["baseline_confidence"],
                    threshold_exceeded=abs(trends["confidence_trend"]),
                    recommendation=self._get_confidence_recommendation(trends["confidence_trend"]),
                )
                alerts.append(alert)

            # Check processing time drift
            if (
                trends["processing_time_trend"]
                > self.performance_tracker.drift_thresholds["processing_time_increase"]
            ):
                severity = self._calculate_severity(trends["processing_time_trend"], 0.2, 0.4, 0.6)
                alert = ModelDriftAlert(
                    alert_id=f"drift_{int(time.time())}_time",
                    timestamp=datetime.now().isoformat(),
                    model_version=model_version,
                    drift_type="processing_time",
                    severity=severity,
                    current_value=trends.get("recent_processing_time", 0),
                    baseline_value=trends.get("baseline_processing_time", 0),
                    threshold_exceeded=trends["processing_time_trend"],
                    recommendation="Investigate performance degradation and optimize model inference",
                )
                alerts.append(alert)

        except Exception as e:
            logger.error(f"Failed to check for drift: {e}")

        return alerts

    def _monitoring_loop(self, thread: GracefulThread):
        """Continuous monitoring loop.
        
        Args:
            thread: GracefulThread instance for shutdown coordination
        """
        # Get monitoring arguments
        model_version, check_interval_minutes = self._monitor_args
        
        while not thread.should_stop():
            try:
                alerts = self.check_for_drift(model_version)

                for alert in alerts:
                    self.alert_queue.put(alert)
                    logger.warning(f"Drift alert: {alert.drift_type} - {alert.severity} severity")

                # Sleep for check interval - exit immediately if stop requested
                if thread.wait_or_stop(check_interval_minutes * 60):
                    break

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                if thread.wait_or_stop(60):  # Wait 1 minute before retrying
                    break

    def _calculate_severity(self, value: float, low: float, medium: float, high: float) -> str:
        """Calculate alert severity based on thresholds."""
        if value >= high:
            return "critical"
        elif value >= medium:
            return "high"
        elif value >= low:
            return "medium"
        else:
            return "low"

    def _get_accuracy_recommendation(self, trend: float) -> str:
        """Get recommendation for accuracy drift."""
        if trend < -0.15:
            return "Critical accuracy drop detected. Immediate model retraining required."
        elif trend < -0.1:
            return "Significant accuracy drop. Schedule model retraining within 24 hours."
        elif trend < -0.05:
            return "Moderate accuracy drop. Monitor closely and prepare for retraining."
        else:
            return "Minor accuracy variation. Continue monitoring."

    def _get_confidence_recommendation(self, trend: float) -> str:
        """Get recommendation for confidence drift."""
        if abs(trend) > 0.3:
            return "Severe confidence calibration drift. Recalibrate model immediately."
        elif abs(trend) > 0.2:
            return "Significant confidence drift. Review model calibration."
        else:
            return "Moderate confidence drift. Monitor calibration quality."

    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get alert queue statistics.
        
        Returns:
            Dictionary with queue statistics including size, maxsize, and dropped count
        """
        return self.alert_queue.get_stats()


class AutomatedRetrainingManager:
    """Manages automated model retraining triggers."""

    def __init__(self, drift_detector: ModelDriftDetector):
        """Initialize retraining manager.

        Args:
            drift_detector: ModelDriftDetector instance
        """
        self.drift_detector = drift_detector
        self.retraining_thresholds = {
            "critical_alerts": 1,  # 1 critical alert triggers retraining
            "high_alerts": 3,  # 3 high alerts trigger retraining
            "medium_alerts": 5,  # 5 medium alerts trigger retraining
        }
        # Bounded queue to prevent memory exhaustion from retraining request accumulation
        self.retraining_queue = BoundedQueue(
            maxsize=1000,
            drop_policy='oldest',
            name='retraining_queue'
        )

        logger.info("AutomatedRetrainingManager initialized")

    def evaluate_retraining_need(self, model_version: str) -> Dict[str, Any]:
        """Evaluate if model needs retraining based on alerts.

        Args:
            model_version: Model version to evaluate

        Returns:
            Dictionary with retraining recommendation
        """
        # Get recent alerts (last 24 hours)
        alerts = self._get_recent_alerts(model_version, hours=24)

        # Count alerts by severity
        alert_counts = {
            "critical": len([a for a in alerts if a.severity == "critical"]),
            "high": len([a for a in alerts if a.severity == "high"]),
            "medium": len([a for a in alerts if a.severity == "medium"]),
            "low": len([a for a in alerts if a.severity == "low"]),
        }

        # Determine if retraining is needed
        retraining_needed = (
            alert_counts["critical"] >= self.retraining_thresholds["critical_alerts"]
            or alert_counts["high"] >= self.retraining_thresholds["high_alerts"]
            or alert_counts["medium"] >= self.retraining_thresholds["medium_alerts"]
        )

        # Determine priority
        if alert_counts["critical"] > 0:
            priority = "immediate"
        elif alert_counts["high"] >= 2:
            priority = "urgent"
        elif retraining_needed:
            priority = "scheduled"
        else:
            priority = "none"

        recommendation = {
            "retraining_needed": retraining_needed,
            "priority": priority,
            "alert_counts": alert_counts,
            "total_alerts": len(alerts),
            "recommendation_reason": self._get_retraining_reason(alert_counts),
            "estimated_completion_hours": self._estimate_retraining_time(priority),
        }

        if retraining_needed:
            logger.info(f"Retraining recommended for {model_version}: {priority} priority")

        return recommendation

    def trigger_automated_retraining(self, model_version: str, priority: str = "scheduled"):
        """Trigger automated model retraining.

        Args:
            model_version: Model version to retrain
            priority: Retraining priority ('immediate', 'urgent', 'scheduled')
        """
        retraining_request = {
            "model_version": model_version,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "trigger_reason": "automated_drift_detection",
            "estimated_duration_hours": self._estimate_retraining_time(priority),
        }

        self.retraining_queue.put(retraining_request)

        logger.info(f"Automated retraining triggered for {model_version} with {priority} priority")

    def _get_recent_alerts(self, model_version: str, hours: int = 24) -> List[ModelDriftAlert]:
        """Get recent drift alerts for a model."""
        # This would query the alert storage system
        # For now, return empty list as placeholder
        return []

    def _get_retraining_reason(self, alert_counts: Dict[str, int]) -> str:
        """Get human-readable reason for retraining recommendation."""
        if alert_counts["critical"] > 0:
            return f"Critical performance degradation detected ({alert_counts['critical']} critical alerts)"
        elif alert_counts["high"] >= 3:
            return f"Multiple high-severity issues detected ({alert_counts['high']} high alerts)"
        elif alert_counts["medium"] >= 5:
            return f"Persistent medium-severity issues detected ({alert_counts['medium']} medium alerts)"
        else:
            return "No retraining needed based on current alerts"

    def _estimate_retraining_time(self, priority: str) -> int:
        """Estimate retraining time based on priority."""
        time_estimates = {
            "immediate": 2,  # 2 hours for emergency retraining
            "urgent": 6,  # 6 hours for urgent retraining
            "scheduled": 24,  # 24 hours for scheduled retraining
        }
        return time_estimates.get(priority, 24)

    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get retraining queue statistics.
        
        Returns:
            Dictionary with queue statistics including size, maxsize, and dropped count
        """
        return self.retraining_queue.get_stats()


class ModelSecurityManager:
    """Manages model security, integrity verification, and adversarial protection."""

    def __init__(self, encryption_key: Optional[bytes] = None):
        """Initialize model security manager.

        Args:
            encryption_key: Encryption key for model storage (generates new if None)
        """
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)

        logger.info("ModelSecurityManager initialized")

    def verify_model_integrity(self, model_path: str, expected_hash: str) -> bool:
        """Verify model file integrity using cryptographic hash.

        Args:
            model_path: Path to model file
            expected_hash: Expected SHA-256 hash of the model

        Returns:
            True if integrity is verified, False otherwise
        """
        try:
            # Calculate SHA-256 hash of model file
            sha256_hash = hashlib.sha256()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)

            actual_hash = sha256_hash.hexdigest()

            integrity_verified = actual_hash == expected_hash

            if integrity_verified:
                logger.info(f"Model integrity verified: {model_path}")
            else:
                logger.error(f"Model integrity check failed: {model_path}")
                logger.error(f"Expected: {expected_hash}")
                logger.error(f"Actual: {actual_hash}")

            return integrity_verified

        except Exception as e:
            logger.error(f"Failed to verify model integrity: {e}")
            return False

    def sign_model(self, model_path: str) -> str:
        """Generate cryptographic signature for model.

        Args:
            model_path: Path to model file

        Returns:
            Cryptographic signature (SHA-256 hash)
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)

            signature = sha256_hash.hexdigest()

            # Store signature in metadata file
            signature_path = f"{model_path}.signature"
            with open(signature_path, "w") as f:
                json.dump(
                    {
                        "model_path": model_path,
                        "signature": signature,
                        "timestamp": datetime.now().isoformat(),
                        "algorithm": "SHA-256",
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Model signed: {model_path}")
            return signature

        except Exception as e:
            logger.error(f"Failed to sign model: {e}")
            return ""

    def encrypt_model(self, model_path: str, output_path: str) -> bool:
        """Encrypt model file for secure storage.

        Args:
            model_path: Path to original model file
            output_path: Path for encrypted model file

        Returns:
            True if encryption successful, False otherwise
        """
        try:
            # Read model file
            with open(model_path, "rb") as f:
                model_data = f.read()

            # Encrypt data
            encrypted_data = self.cipher.encrypt(model_data)

            # Write encrypted file
            with open(output_path, "wb") as f:
                f.write(encrypted_data)

            logger.info(f"Model encrypted: {model_path} -> {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to encrypt model: {e}")
            return False

    def decrypt_model(self, encrypted_path: str, output_path: str) -> bool:
        """Decrypt model file for use.

        Args:
            encrypted_path: Path to encrypted model file
            output_path: Path for decrypted model file

        Returns:
            True if decryption successful, False otherwise
        """
        try:
            # Read encrypted file
            with open(encrypted_path, "rb") as f:
                encrypted_data = f.read()

            # Decrypt data
            decrypted_data = self.cipher.decrypt(encrypted_data)

            # Write decrypted file
            with open(output_path, "wb") as f:
                f.write(decrypted_data)

            logger.info(f"Model decrypted: {encrypted_path} -> {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to decrypt model: {e}")
            return False

    def test_adversarial_robustness(self, model, test_samples: List[torch.Tensor]) -> float:
        """Test model robustness against adversarial attacks.

        Args:
            model: PyTorch model to test
            test_samples: List of test input tensors

        Returns:
            Robustness score (0-1, higher is better)
        """
        try:
            model.eval()
            robust_predictions = 0
            total_tests = 0

            for sample in test_samples:
                # Generate adversarial example using FGSM
                sample.requires_grad = True

                # Forward pass
                output = model(sample.unsqueeze(0))
                original_pred = torch.argmax(output, dim=1)

                # Calculate loss
                loss = torch.nn.functional.cross_entropy(output, original_pred)

                # Backward pass
                model.zero_grad()
                loss.backward()

                # Generate adversarial example
                epsilon = 0.01  # Small perturbation
                adversarial_sample = sample + epsilon * sample.grad.sign()
                adversarial_sample = torch.clamp(adversarial_sample, 0, 1)

                # Test adversarial example
                with torch.no_grad():
                    adversarial_output = model(adversarial_sample.unsqueeze(0))
                    adversarial_pred = torch.argmax(adversarial_output, dim=1)

                # Check if prediction is consistent
                if original_pred == adversarial_pred:
                    robust_predictions += 1

                total_tests += 1

            robustness_score = robust_predictions / total_tests if total_tests > 0 else 0.0

            logger.info(f"Adversarial robustness score: {robustness_score:.3f}")
            return robustness_score

        except Exception as e:
            logger.error(f"Failed to test adversarial robustness: {e}")
            return 0.0

    def get_security_status(self, model_path: str) -> ModelSecurityStatus:
        """Get comprehensive security status for a model.

        Args:
            model_path: Path to model file

        Returns:
            ModelSecurityStatus with security information
        """
        # Check if signature file exists
        signature_path = f"{model_path}.signature"
        signature_valid = Path(signature_path).exists()

        # Verify integrity if signature exists
        integrity_verified = False
        if signature_valid:
            try:
                with open(signature_path, "r") as f:
                    signature_data = json.load(f)
                expected_hash = signature_data["signature"]
                integrity_verified = self.verify_model_integrity(model_path, expected_hash)
            except Exception as e:
                logger.error(f"Failed to verify signature: {e}")

        # Calculate current hash
        try:
            sha256_hash = hashlib.sha256()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            current_hash = sha256_hash.hexdigest()
        except Exception:
            current_hash = "unknown"

        # Check encryption status
        encryption_status = "encrypted" if model_path.endswith(".encrypted") else "unencrypted"

        return ModelSecurityStatus(
            model_version=Path(model_path).stem,
            integrity_verified=integrity_verified,
            signature_valid=signature_valid,
            last_verification=datetime.now().isoformat(),
            security_hash=current_hash,
            encryption_status=encryption_status,
            adversarial_robustness_score=0.0,  # Would be populated by actual test
        )


def main():
    """Run model management system example."""
    print("Model Management System for Real-Time WSI Streaming")
    print("Tracks performance, detects drift, manages retraining, ensures security")


if __name__ == "__main__":
    main()
