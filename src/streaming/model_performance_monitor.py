"""
Model Performance Monitoring for Real-Time WSI Streaming

This module implements comprehensive model performance monitoring including
accuracy tracking, drift detection, and automated retraining triggers for
the streaming WSI processing system.

**Validates: Requirements 9.1.2**
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .config_manager import StreamingConfig
from .metrics import ProcessingMetrics


class ModelDriftType(Enum):
    """Types of model drift detection."""

    ACCURACY_DRIFT = "accuracy_drift"
    CONFIDENCE_DRIFT = "confidence_drift"
    PREDICTION_DRIFT = "prediction_drift"
    FEATURE_DRIFT = "feature_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RetrainingTrigger(Enum):
    """Triggers for automated model retraining."""

    ACCURACY_THRESHOLD = "accuracy_threshold"
    DRIFT_DETECTION = "drift_detection"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SCHEDULED_RETRAINING = "scheduled_retraining"
    MANUAL_TRIGGER = "manual_trigger"


@dataclass
class ModelPerformanceMetrics:
    """Comprehensive model performance metrics."""

    timestamp: datetime
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    confidence_mean: float
    confidence_std: float
    processing_time_mean: float
    processing_time_std: float
    memory_usage_mean: float
    memory_usage_std: float
    throughput_patches_per_second: float
    total_predictions: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_score": self.auc_score,
            "confidence_mean": self.confidence_mean,
            "confidence_std": self.confidence_std,
            "processing_time_mean": self.processing_time_mean,
            "processing_time_std": self.processing_time_std,
            "memory_usage_mean": self.memory_usage_mean,
            "memory_usage_std": self.memory_usage_std,
            "throughput_patches_per_second": self.throughput_patches_per_second,
            "total_predictions": self.total_predictions,
        }


@dataclass
class DriftDetectionResult:
    """Results from model drift detection analysis."""

    drift_type: ModelDriftType
    drift_detected: bool
    drift_score: float
    threshold: float
    confidence_level: float
    detection_method: str
    affected_metrics: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert drift result to dictionary."""
        return {
            "drift_type": self.drift_type.value,
            "drift_detected": self.drift_detected,
            "drift_score": self.drift_score,
            "threshold": self.threshold,
            "confidence_level": self.confidence_level,
            "detection_method": self.detection_method,
            "affected_metrics": self.affected_metrics,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PerformanceAlert:
    """Performance monitoring alert."""

    alert_id: str
    severity: AlertSeverity
    alert_type: str
    message: str
    metrics: Dict[str, float]
    recommended_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "alert_type": self.alert_type,
            "message": self.message,
            "metrics": self.metrics,
            "recommended_actions": self.recommended_actions,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


@dataclass
class RetrainingRecommendation:
    """Recommendation for model retraining."""

    trigger: RetrainingTrigger
    priority: int  # 1-10 scale
    justification: str
    estimated_improvement: float
    training_data_requirements: Dict[str, Any]
    estimated_training_time: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary."""
        return {
            "trigger": self.trigger.value,
            "priority": self.priority,
            "justification": self.justification,
            "estimated_improvement": self.estimated_improvement,
            "training_data_requirements": self.training_data_requirements,
            "estimated_training_time": self.estimated_training_time,
            "timestamp": self.timestamp.isoformat(),
        }


class ModelPerformanceMonitor:
    """
    Comprehensive model performance monitoring system.

    This class implements continuous monitoring of model performance including:
    - Real-time accuracy and confidence tracking
    - Model drift detection using statistical methods
    - Performance degradation alerts
    - Automated retraining trigger recommendations
    """

    def __init__(self, config: StreamingConfig, monitoring_config: Optional[Dict] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Monitoring configuration
        self.monitoring_config = monitoring_config or {
            "metrics_window_size": 1000,  # Number of recent predictions to track
            "drift_detection_window": 500,  # Window for drift detection
            "accuracy_threshold": 0.85,  # Minimum acceptable accuracy
            "drift_threshold": 0.05,  # Threshold for drift detection
            "alert_cooldown_minutes": 30,  # Minimum time between similar alerts
            "retraining_threshold": 0.80,  # Accuracy threshold for retraining
        }

        # Performance tracking
        self.performance_history: List[ModelPerformanceMetrics] = []
        self.recent_predictions: deque = deque(maxlen=self.monitoring_config["metrics_window_size"])
        self.recent_ground_truth: deque = deque(
            maxlen=self.monitoring_config["metrics_window_size"]
        )
        self.recent_confidences: deque = deque(maxlen=self.monitoring_config["metrics_window_size"])
        self.recent_processing_times: deque = deque(
            maxlen=self.monitoring_config["metrics_window_size"]
        )
        self.recent_memory_usage: deque = deque(
            maxlen=self.monitoring_config["metrics_window_size"]
        )

        # Drift detection
        self.drift_detection_results: List[DriftDetectionResult] = []
        self.baseline_metrics: Optional[ModelPerformanceMetrics] = None

        # Alerting
        self.active_alerts: List[PerformanceAlert] = []
        self.alert_history: List[PerformanceAlert] = []
        self.last_alert_times: Dict[str, datetime] = {}

        # Retraining
        self.retraining_recommendations: List[RetrainingRecommendation] = []
        self.current_model_version = "1.0.0"

        # Callbacks for external integration
        self.alert_callbacks: List[Callable] = []
        self.retraining_callbacks: List[Callable] = []

    def track_prediction(
        self,
        prediction: int,
        confidence: float,
        ground_truth: Optional[int] = None,
        processing_time: float = 0.0,
        memory_usage: float = 0.0,
        model_version: Optional[str] = None,
    ) -> None:
        """
        Track a single prediction for performance monitoring.

        **Validates: Requirements 9.1.2.1**

        Args:
            prediction: Model prediction (0 or 1)
            confidence: Prediction confidence (0.0 to 1.0)
            ground_truth: True label if available
            processing_time: Time taken for prediction in seconds
            memory_usage: Memory used for prediction in GB
            model_version: Version of model used for prediction
        """
        # Store recent data
        self.recent_predictions.append(prediction)
        self.recent_confidences.append(confidence)
        self.recent_processing_times.append(processing_time)
        self.recent_memory_usage.append(memory_usage)

        if ground_truth is not None:
            self.recent_ground_truth.append(ground_truth)

        # Update model version
        if model_version:
            self.current_model_version = model_version

        # Trigger periodic analysis
        if len(self.recent_predictions) % 100 == 0:  # Every 100 predictions
            asyncio.create_task(self._analyze_recent_performance())

    async def _analyze_recent_performance(self) -> None:
        """Analyze recent performance and detect issues."""
        try:
            # Calculate current metrics
            current_metrics = self._calculate_current_metrics()

            if current_metrics:
                self.performance_history.append(current_metrics)

                # Detect drift
                await self._detect_model_drift(current_metrics)

                # Check for performance degradation
                await self._check_performance_degradation(current_metrics)

                # Generate retraining recommendations
                await self._evaluate_retraining_needs(current_metrics)

        except Exception as e:
            self.logger.error(f"Error in performance analysis: {e}")

    def _calculate_current_metrics(self) -> Optional[ModelPerformanceMetrics]:
        """Calculate current performance metrics from recent data."""
        if len(self.recent_predictions) < 10:  # Need minimum data
            return None

        predictions = list(self.recent_predictions)
        confidences = list(self.recent_confidences)
        processing_times = list(self.recent_processing_times)
        memory_usage = list(self.recent_memory_usage)

        # Calculate metrics only if we have ground truth
        if len(self.recent_ground_truth) < 10:
            # Return partial metrics without accuracy-based measures
            return ModelPerformanceMetrics(
                timestamp=datetime.now(),
                model_version=self.current_model_version,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                auc_score=0.0,
                confidence_mean=np.mean(confidences),
                confidence_std=np.std(confidences),
                processing_time_mean=np.mean(processing_times),
                processing_time_std=np.std(processing_times),
                memory_usage_mean=np.mean(memory_usage),
                memory_usage_std=np.std(memory_usage),
                throughput_patches_per_second=(
                    1.0 / np.mean(processing_times) if np.mean(processing_times) > 0 else 0.0
                ),
                total_predictions=len(predictions),
            )

        ground_truth = list(self.recent_ground_truth)

        # Ensure we have matching lengths
        min_length = min(len(predictions), len(ground_truth))
        predictions = predictions[-min_length:]
        ground_truth = ground_truth[-min_length:]
        confidences = confidences[-min_length:]

        # Calculate performance metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average="binary", zero_division=0
        )

        try:
            auc = roc_auc_score(ground_truth, confidences)
        except ValueError:
            auc = 0.0  # Handle case where all labels are the same class

        return ModelPerformanceMetrics(
            timestamp=datetime.now(),
            model_version=self.current_model_version,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            confidence_mean=np.mean(confidences),
            confidence_std=np.std(confidences),
            processing_time_mean=np.mean(processing_times),
            processing_time_std=np.std(processing_times),
            memory_usage_mean=np.mean(memory_usage),
            memory_usage_std=np.std(memory_usage),
            throughput_patches_per_second=(
                1.0 / np.mean(processing_times) if np.mean(processing_times) > 0 else 0.0
            ),
            total_predictions=len(predictions),
        )

    async def _detect_model_drift(self, current_metrics: ModelPerformanceMetrics) -> None:
        """
        Detect model drift using statistical methods.

        **Validates: Requirements 9.1.2.2**
        """
        if not self.baseline_metrics:
            # Set first metrics as baseline
            self.baseline_metrics = current_metrics
            return

        drift_results = []

        # Accuracy drift detection
        accuracy_drift = abs(current_metrics.accuracy - self.baseline_metrics.accuracy)
        if accuracy_drift > self.monitoring_config["drift_threshold"]:
            drift_results.append(
                DriftDetectionResult(
                    drift_type=ModelDriftType.ACCURACY_DRIFT,
                    drift_detected=True,
                    drift_score=accuracy_drift,
                    threshold=self.monitoring_config["drift_threshold"],
                    confidence_level=0.95,
                    detection_method="threshold_comparison",
                    affected_metrics=["accuracy"],
                )
            )

        # Confidence drift detection
        confidence_drift = abs(
            current_metrics.confidence_mean - self.baseline_metrics.confidence_mean
        )
        if confidence_drift > 0.1:  # 10% confidence drift threshold
            drift_results.append(
                DriftDetectionResult(
                    drift_type=ModelDriftType.CONFIDENCE_DRIFT,
                    drift_detected=True,
                    drift_score=confidence_drift,
                    threshold=0.1,
                    confidence_level=0.95,
                    detection_method="threshold_comparison",
                    affected_metrics=["confidence_mean"],
                )
            )

        # Performance degradation detection
        performance_degradation = (
            (self.baseline_metrics.accuracy - current_metrics.accuracy)
            + (self.baseline_metrics.auc_score - current_metrics.auc_score)
        ) / 2

        if performance_degradation > 0.05:  # 5% performance degradation
            drift_results.append(
                DriftDetectionResult(
                    drift_type=ModelDriftType.PERFORMANCE_DEGRADATION,
                    drift_detected=True,
                    drift_score=performance_degradation,
                    threshold=0.05,
                    confidence_level=0.95,
                    detection_method="composite_metric",
                    affected_metrics=["accuracy", "auc_score"],
                )
            )

        # Store drift results
        self.drift_detection_results.extend(drift_results)

        # Generate alerts for detected drift
        for drift_result in drift_results:
            if drift_result.drift_detected:
                await self._generate_drift_alert(drift_result, current_metrics)

    async def _check_performance_degradation(
        self, current_metrics: ModelPerformanceMetrics
    ) -> None:
        """Check for performance degradation and generate alerts."""
        # Check accuracy threshold
        if current_metrics.accuracy < self.monitoring_config["accuracy_threshold"]:
            await self._generate_alert(
                alert_type="accuracy_degradation",
                severity=AlertSeverity.CRITICAL,
                message=f"Model accuracy ({current_metrics.accuracy:.3f}) below threshold ({self.monitoring_config['accuracy_threshold']:.3f})",
                metrics={
                    "accuracy": current_metrics.accuracy,
                    "threshold": self.monitoring_config["accuracy_threshold"],
                },
                recommended_actions=[
                    "Review recent training data quality",
                    "Check for data distribution changes",
                    "Consider model retraining",
                    "Investigate potential model corruption",
                ],
            )

        # Check processing time degradation
        if len(self.performance_history) > 10:
            recent_avg_time = np.mean(
                [m.processing_time_mean for m in self.performance_history[-10:]]
            )
            baseline_avg_time = np.mean(
                [m.processing_time_mean for m in self.performance_history[:10]]
            )

            if recent_avg_time > baseline_avg_time * 1.5:  # 50% slower
                await self._generate_alert(
                    alert_type="processing_time_degradation",
                    severity=AlertSeverity.WARNING,
                    message=f"Processing time increased by {((recent_avg_time / baseline_avg_time - 1) * 100):.1f}%",
                    metrics={"current_time": recent_avg_time, "baseline_time": baseline_avg_time},
                    recommended_actions=[
                        "Check GPU utilization",
                        "Monitor memory usage patterns",
                        "Review system resource availability",
                        "Consider model optimization",
                    ],
                )

    async def _evaluate_retraining_needs(self, current_metrics: ModelPerformanceMetrics) -> None:
        """
        Evaluate need for model retraining and generate recommendations.

        **Validates: Requirements 9.1.2.3**
        """
        recommendations = []

        # Accuracy-based retraining trigger
        if current_metrics.accuracy < self.monitoring_config["retraining_threshold"]:
            recommendations.append(
                RetrainingRecommendation(
                    trigger=RetrainingTrigger.ACCURACY_THRESHOLD,
                    priority=9,  # High priority
                    justification=f"Accuracy ({current_metrics.accuracy:.3f}) below retraining threshold ({self.monitoring_config['retraining_threshold']:.3f})",
                    estimated_improvement=0.05,  # Estimated 5% improvement
                    training_data_requirements={
                        "minimum_samples": 10000,
                        "class_balance": "balanced",
                        "data_freshness_days": 30,
                    },
                    estimated_training_time=4.0,  # 4 hours
                )
            )

        # Drift-based retraining trigger
        recent_drift = [d for d in self.drift_detection_results[-10:] if d.drift_detected]
        if len(recent_drift) >= 3:  # Multiple drift detections
            recommendations.append(
                RetrainingRecommendation(
                    trigger=RetrainingTrigger.DRIFT_DETECTION,
                    priority=7,  # Medium-high priority
                    justification=f"Multiple drift detections ({len(recent_drift)}) in recent monitoring window",
                    estimated_improvement=0.03,  # Estimated 3% improvement
                    training_data_requirements={
                        "minimum_samples": 15000,
                        "class_balance": "representative",
                        "data_freshness_days": 14,
                    },
                    estimated_training_time=6.0,  # 6 hours
                )
            )

        # Performance degradation trigger
        if len(self.performance_history) > 20:
            recent_performance = np.mean([m.accuracy for m in self.performance_history[-10:]])
            historical_performance = np.mean(
                [m.accuracy for m in self.performance_history[-20:-10]]
            )

            if recent_performance < historical_performance - 0.02:  # 2% degradation
                recommendations.append(
                    RetrainingRecommendation(
                        trigger=RetrainingTrigger.PERFORMANCE_DEGRADATION,
                        priority=6,  # Medium priority
                        justification=f"Performance degraded by {((historical_performance - recent_performance) * 100):.1f}% over recent period",
                        estimated_improvement=0.04,  # Estimated 4% improvement
                        training_data_requirements={
                            "minimum_samples": 12000,
                            "class_balance": "balanced",
                            "data_freshness_days": 21,
                        },
                        estimated_training_time=5.0,  # 5 hours
                    )
                )

        # Store recommendations
        self.retraining_recommendations.extend(recommendations)

        # Trigger retraining callbacks for high-priority recommendations
        for rec in recommendations:
            if rec.priority >= 8:  # High priority threshold
                for callback in self.retraining_callbacks:
                    try:
                        await callback(rec)
                    except Exception as e:
                        self.logger.error(f"Error in retraining callback: {e}")

    async def _generate_drift_alert(
        self, drift_result: DriftDetectionResult, current_metrics: ModelPerformanceMetrics
    ) -> None:
        """Generate alert for detected model drift."""
        severity = AlertSeverity.WARNING
        if drift_result.drift_score > drift_result.threshold * 2:
            severity = AlertSeverity.CRITICAL

        await self._generate_alert(
            alert_type=f"model_drift_{drift_result.drift_type.value}",
            severity=severity,
            message=f"Model drift detected: {drift_result.drift_type.value} (score: {drift_result.drift_score:.4f})",
            metrics={
                "drift_score": drift_result.drift_score,
                "threshold": drift_result.threshold,
                "affected_metrics": drift_result.affected_metrics,
            },
            recommended_actions=[
                "Investigate data distribution changes",
                "Review recent model updates",
                "Consider model retraining",
                "Validate input data quality",
            ],
        )

    async def _generate_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        message: str,
        metrics: Dict[str, Any],
        recommended_actions: List[str],
    ) -> None:
        """Generate and process performance alert."""
        # Check alert cooldown
        if alert_type in self.last_alert_times:
            time_since_last = datetime.now() - self.last_alert_times[alert_type]
            cooldown = timedelta(minutes=self.monitoring_config["alert_cooldown_minutes"])
            if time_since_last < cooldown:
                return  # Skip alert due to cooldown

        # Create alert
        alert = PerformanceAlert(
            alert_id=f"{alert_type}_{int(time.time())}",
            severity=severity,
            alert_type=alert_type,
            message=message,
            metrics=metrics,
            recommended_actions=recommended_actions,
        )

        # Store alert
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        self.last_alert_times[alert_type] = datetime.now()

        # Log alert
        self.logger.warning(f"Performance Alert [{severity.value.upper()}]: {message}")

        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self.logger.info(f"Alert {alert_id} acknowledged")
                return True
        return False

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.performance_history if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {"error": "No metrics available for the specified period"}

        # Calculate summary statistics
        accuracies = [m.accuracy for m in recent_metrics]
        confidences = [m.confidence_mean for m in recent_metrics]
        processing_times = [m.processing_time_mean for m in recent_metrics]
        throughputs = [m.throughput_patches_per_second for m in recent_metrics]

        summary = {
            "time_period_hours": hours,
            "total_predictions": sum(m.total_predictions for m in recent_metrics),
            "accuracy": {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies),
                "min": np.min(accuracies),
                "max": np.max(accuracies),
            },
            "confidence": {"mean": np.mean(confidences), "std": np.std(confidences)},
            "processing_time": {"mean": np.mean(processing_times), "std": np.std(processing_times)},
            "throughput": {"mean": np.mean(throughputs), "std": np.std(throughputs)},
            "active_alerts": len([a for a in self.active_alerts if not a.acknowledged]),
            "drift_detections": len(
                [
                    d
                    for d in self.drift_detection_results
                    if d.timestamp >= cutoff_time and d.drift_detected
                ]
            ),
            "retraining_recommendations": len(
                [r for r in self.retraining_recommendations if r.timestamp >= cutoff_time]
            ),
        }

        return summary

    def export_monitoring_data(self, filepath: str) -> None:
        """Export monitoring data to file for analysis."""
        data = {
            "performance_history": [m.to_dict() for m in self.performance_history],
            "drift_detection_results": [d.to_dict() for d in self.drift_detection_results],
            "alert_history": [a.to_dict() for a in self.alert_history],
            "retraining_recommendations": [r.to_dict() for r in self.retraining_recommendations],
            "monitoring_config": self.monitoring_config,
            "export_timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Monitoring data exported to {filepath}")

    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)

    def add_retraining_callback(self, callback: Callable) -> None:
        """Add callback function for retraining recommendations."""
        self.retraining_callbacks.append(callback)


# Example usage and testing functions
async def example_alert_handler(alert: PerformanceAlert) -> None:
    """Example alert handler function."""
    print(f"ALERT [{alert.severity.value.upper()}]: {alert.message}")
    print(f"Recommended actions: {', '.join(alert.recommended_actions)}")


async def example_retraining_handler(recommendation: RetrainingRecommendation) -> None:
    """Example retraining recommendation handler."""
    print(
        f"RETRAINING RECOMMENDED [Priority {recommendation.priority}]: {recommendation.justification}"
    )
    print(f"Estimated improvement: {recommendation.estimated_improvement:.1%}")


async def run_model_monitoring_demo():
    """Run model performance monitoring demonstration."""
    config = StreamingConfig(
        tile_size=1024,
        batch_size=32,
        memory_budget_gb=2.0,
        target_time=30.0,
        confidence_threshold=0.95,
    )

    monitor = ModelPerformanceMonitor(config)

    # Add example callbacks
    monitor.add_alert_callback(example_alert_handler)
    monitor.add_retraining_callback(example_retraining_handler)

    print("Starting Model Performance Monitoring Demo...")

    # Simulate predictions over time
    print("\nSimulating model predictions...")
    for i in range(1000):
        # Simulate gradual performance degradation
        base_accuracy = 0.95 - (i / 10000)  # Gradual decline

        # Generate synthetic prediction
        confidence = np.random.uniform(0.7, 0.99)
        prediction = 1 if confidence > 0.8 else 0
        ground_truth = np.random.choice([0, 1], p=[0.3, 0.7])  # Imbalanced dataset

        # Add some noise to simulate real-world conditions
        if np.random.random() < 0.1:  # 10% chance of incorrect prediction
            prediction = 1 - ground_truth

        processing_time = np.random.uniform(0.02, 0.05)  # 20-50ms per patch
        memory_usage = np.random.uniform(1.5, 1.9)  # Within 2GB limit

        monitor.track_prediction(
            prediction=prediction,
            confidence=confidence,
            ground_truth=ground_truth,
            processing_time=processing_time,
            memory_usage=memory_usage,
            model_version="1.0.0",
        )

        # Simulate real-time processing
        if i % 100 == 0:
            print(f"  Processed {i+1}/1000 predictions...")
            await asyncio.sleep(0.1)  # Brief pause

    # Get performance summary
    print("\nGenerating performance summary...")
    summary = monitor.get_performance_summary(hours=1)

    print(f"Performance Summary:")
    print(f"  Total predictions: {summary['total_predictions']}")
    print(
        f"  Average accuracy: {summary['accuracy']['mean']:.3f} ± {summary['accuracy']['std']:.3f}"
    )
    print(f"  Average confidence: {summary['confidence']['mean']:.3f}")
    print(f"  Average processing time: {summary['processing_time']['mean']:.3f}s")
    print(f"  Active alerts: {summary['active_alerts']}")
    print(f"  Drift detections: {summary['drift_detections']}")
    print(f"  Retraining recommendations: {summary['retraining_recommendations']}")

    # Export monitoring data
    monitor.export_monitoring_data("model_monitoring_demo.json")
    print(f"\nMonitoring data exported to model_monitoring_demo.json")

    return monitor


if __name__ == "__main__":
    # Run the monitoring demo
    asyncio.run(run_model_monitoring_demo())
