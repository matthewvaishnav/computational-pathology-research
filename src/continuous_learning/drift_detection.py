"""
Model drift detection for federated learning.

Monitors prediction confidence distributions, accuracy metrics, and data
distribution shifts to detect model drift in medical AI systems.
"""

import logging
import numpy as np
import torch
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Represents a model drift alert."""
    timestamp: float
    drift_type: str  # "confidence", "accuracy", "distribution", "performance"
    severity: str    # "warning", "critical"
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    description: str
    affected_diseases: List[str]


@dataclass
class ConfidenceDistribution:
    """Represents confidence distribution statistics."""
    timestamp: float
    round_number: int
    disease_type: str
    mean_confidence: float
    std_confidence: float
    min_confidence: float
    max_confidence: float
    percentiles: Dict[int, float]  # 25th, 50th, 75th, 90th, 95th percentiles
    entropy: float
    sample_count: int


class ModelDriftDetector:
    """
    Detects model drift in federated learning systems.
    
    Monitors multiple indicators:
    - Prediction confidence distributions
    - Accuracy metrics over time
    - Feature distribution shifts
    - Performance degradation patterns
    """
    
    def __init__(
        self,
        window_size: int = 100,
        confidence_threshold: float = 0.1,
        accuracy_threshold: float = 0.05,
        alert_cooldown: int = 300,  # 5 minutes
        supported_diseases: List[str] = None
    ):
        """
        Initialize drift detector.
        
        Args:
            window_size: Size of sliding window for statistics
            confidence_threshold: Threshold for confidence drift detection
            accuracy_threshold: Threshold for accuracy drift detection
            alert_cooldown: Cooldown period between alerts (seconds)
            supported_diseases: List of supported disease types
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.accuracy_threshold = accuracy_threshold
        self.alert_cooldown = alert_cooldown
        self.supported_diseases = supported_diseases or ["breast", "lung", "prostate", "colon", "melanoma"]
        
        # Tracking data structures
        self.confidence_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.accuracy_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.prediction_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        # Baseline statistics
        self.baseline_confidence: Dict[str, ConfidenceDistribution] = {}
        self.baseline_accuracy: Dict[str, float] = {}
        
        # Alert management
        self.alerts: List[DriftAlert] = []
        self.last_alert_time: Dict[str, float] = {}
        self.alert_callbacks = []
        
        # Drift detection state
        self.drift_detected: Dict[str, bool] = defaultdict(bool)
        self.drift_start_time: Dict[str, Optional[float]] = defaultdict(lambda: None)
        
        logger.info(
            f"Initialized drift detector: window={window_size}, "
            f"conf_thresh={confidence_threshold}, acc_thresh={accuracy_threshold}"
        )
    
    def update_predictions(
        self,
        round_number: int,
        predictions: Dict[str, List[Dict]],
        ground_truth: Optional[Dict[str, List]] = None
    ) -> None:
        """
        Update drift detector with new predictions.
        
        Args:
            round_number: Current round number
            predictions: Dict of disease_type -> list of prediction dicts
                        Each prediction dict should have 'confidence' and 'prediction' keys
            ground_truth: Optional ground truth labels for accuracy calculation
        """
        timestamp = datetime.now().timestamp()
        
        for disease_type, disease_predictions in predictions.items():
            if disease_type not in self.supported_diseases:
                continue
            
            # Extract confidence values
            confidences = [pred.get('confidence', 0.0) for pred in disease_predictions]
            if not confidences:
                continue
            
            # Update confidence distribution
            conf_dist = self._compute_confidence_distribution(
                timestamp, round_number, disease_type, confidences
            )
            self.confidence_history[disease_type].append(conf_dist)
            
            # Update prediction history
            self.prediction_history[disease_type].extend(disease_predictions)
            
            # Update accuracy if ground truth available
            if ground_truth and disease_type in ground_truth:
                gt_labels = ground_truth[disease_type]
                if len(gt_labels) == len(disease_predictions):
                    predictions_labels = [pred.get('prediction', 0) for pred in disease_predictions]
                    accuracy = self._compute_accuracy(predictions_labels, gt_labels)
                    self.accuracy_history[disease_type].append((timestamp, accuracy))
            
            # Set baseline if not established
            if disease_type not in self.baseline_confidence:
                self._establish_baseline(disease_type)
            
            # Check for drift
            self._check_drift(disease_type, conf_dist, timestamp)
        
        logger.debug(f"Updated drift detector for round {round_number} with {len(predictions)} disease types")
    
    def _compute_confidence_distribution(
        self,
        timestamp: float,
        round_number: int,
        disease_type: str,
        confidences: List[float]
    ) -> ConfidenceDistribution:
        """Compute confidence distribution statistics."""
        confidences = np.array(confidences)
        
        # Basic statistics
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        min_conf = np.min(confidences)
        max_conf = np.max(confidences)
        
        # Percentiles
        percentiles = {
            25: np.percentile(confidences, 25),
            50: np.percentile(confidences, 50),
            75: np.percentile(confidences, 75),
            90: np.percentile(confidences, 90),
            95: np.percentile(confidences, 95)
        }
        
        # Entropy (measure of uncertainty distribution)
        # Bin confidences and compute entropy
        hist, _ = np.histogram(confidences, bins=10, range=(0, 1))
        hist = hist / np.sum(hist)  # Normalize
        entropy = -np.sum(hist * np.log(hist + 1e-10))  # Add small epsilon to avoid log(0)
        
        return ConfidenceDistribution(
            timestamp=timestamp,
            round_number=round_number,
            disease_type=disease_type,
            mean_confidence=mean_conf,
            std_confidence=std_conf,
            min_confidence=min_conf,
            max_confidence=max_conf,
            percentiles=percentiles,
            entropy=entropy,
            sample_count=len(confidences)
        )
    
    def _compute_accuracy(self, predictions: List, ground_truth: List) -> float:
        """Compute accuracy from predictions and ground truth."""
        if len(predictions) != len(ground_truth):
            return 0.0
        
        correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
        return correct / len(predictions)
    
    def _establish_baseline(self, disease_type: str) -> None:
        """Establish baseline statistics for a disease type."""
        if len(self.confidence_history[disease_type]) < 5:
            return  # Need more data
        
        # Use first few distributions as baseline
        baseline_distributions = list(self.confidence_history[disease_type])[:5]
        
        # Aggregate baseline statistics
        mean_confidences = [dist.mean_confidence for dist in baseline_distributions]
        baseline_mean = np.mean(mean_confidences)
        
        # Create baseline distribution (use most recent as template)
        latest_dist = baseline_distributions[-1]
        self.baseline_confidence[disease_type] = ConfidenceDistribution(
            timestamp=latest_dist.timestamp,
            round_number=latest_dist.round_number,
            disease_type=disease_type,
            mean_confidence=baseline_mean,
            std_confidence=np.mean([d.std_confidence for d in baseline_distributions]),
            min_confidence=np.mean([d.min_confidence for d in baseline_distributions]),
            max_confidence=np.mean([d.max_confidence for d in baseline_distributions]),
            percentiles={
                k: np.mean([d.percentiles[k] for d in baseline_distributions])
                for k in latest_dist.percentiles.keys()
            },
            entropy=np.mean([d.entropy for d in baseline_distributions]),
            sample_count=sum(d.sample_count for d in baseline_distributions)
        )
        
        # Establish baseline accuracy
        if self.accuracy_history[disease_type]:
            recent_accuracies = [acc for _, acc in list(self.accuracy_history[disease_type])[:5]]
            self.baseline_accuracy[disease_type] = np.mean(recent_accuracies)
        
        logger.info(f"Established baseline for {disease_type}: conf={baseline_mean:.4f}")
    
    def _check_drift(
        self,
        disease_type: str,
        current_dist: ConfidenceDistribution,
        timestamp: float
    ) -> None:
        """Check for drift in current distribution."""
        if disease_type not in self.baseline_confidence:
            return
        
        baseline = self.baseline_confidence[disease_type]
        
        # Check confidence distribution drift
        conf_drift = self._detect_confidence_drift(baseline, current_dist)
        if conf_drift:
            self._trigger_alert(
                timestamp=timestamp,
                drift_type="confidence",
                severity="warning" if abs(conf_drift) < 2 * self.confidence_threshold else "critical",
                metric_name="mean_confidence",
                current_value=current_dist.mean_confidence,
                baseline_value=baseline.mean_confidence,
                threshold=self.confidence_threshold,
                description=f"Confidence distribution drift detected for {disease_type}",
                affected_diseases=[disease_type]
            )
        
        # Check accuracy drift
        if self.accuracy_history[disease_type] and disease_type in self.baseline_accuracy:
            recent_accuracy = self.accuracy_history[disease_type][-1][1]
            baseline_accuracy = self.baseline_accuracy[disease_type]
            
            accuracy_drift = abs(recent_accuracy - baseline_accuracy)
            if accuracy_drift > self.accuracy_threshold:
                self._trigger_alert(
                    timestamp=timestamp,
                    drift_type="accuracy",
                    severity="warning" if accuracy_drift < 2 * self.accuracy_threshold else "critical",
                    metric_name="accuracy",
                    current_value=recent_accuracy,
                    baseline_value=baseline_accuracy,
                    threshold=self.accuracy_threshold,
                    description=f"Accuracy drift detected for {disease_type}",
                    affected_diseases=[disease_type]
                )
        
        # Check entropy drift (uncertainty pattern changes)
        entropy_drift = abs(current_dist.entropy - baseline.entropy)
        entropy_threshold = 0.5  # Entropy threshold
        if entropy_drift > entropy_threshold:
            self._trigger_alert(
                timestamp=timestamp,
                drift_type="distribution",
                severity="warning",
                metric_name="entropy",
                current_value=current_dist.entropy,
                baseline_value=baseline.entropy,
                threshold=entropy_threshold,
                description=f"Prediction uncertainty pattern changed for {disease_type}",
                affected_diseases=[disease_type]
            )
    
    def _detect_confidence_drift(
        self,
        baseline: ConfidenceDistribution,
        current: ConfidenceDistribution
    ) -> Optional[float]:
        """Detect drift in confidence distribution."""
        # Statistical test for mean difference
        mean_diff = abs(current.mean_confidence - baseline.mean_confidence)
        
        if mean_diff > self.confidence_threshold:
            return mean_diff
        
        # Additional checks for distribution shape changes
        # Check if percentile patterns have changed significantly
        percentile_diffs = []
        for percentile in [25, 50, 75, 90, 95]:
            diff = abs(current.percentiles[percentile] - baseline.percentiles[percentile])
            percentile_diffs.append(diff)
        
        max_percentile_diff = max(percentile_diffs)
        if max_percentile_diff > self.confidence_threshold * 1.5:
            return max_percentile_diff
        
        return None
    
    def _trigger_alert(
        self,
        timestamp: float,
        drift_type: str,
        severity: str,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        threshold: float,
        description: str,
        affected_diseases: List[str]
    ) -> None:
        """Trigger a drift alert."""
        # Check cooldown
        alert_key = f"{drift_type}_{metric_name}_{affected_diseases[0]}"
        if (alert_key in self.last_alert_time and 
            timestamp - self.last_alert_time[alert_key] < self.alert_cooldown):
            return
        
        # Create alert
        alert = DriftAlert(
            timestamp=timestamp,
            drift_type=drift_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline_value,
            threshold=threshold,
            description=description,
            affected_diseases=affected_diseases
        )
        
        self.alerts.append(alert)
        self.last_alert_time[alert_key] = timestamp
        
        # Update drift state
        for disease in affected_diseases:
            if not self.drift_detected[disease]:
                self.drift_detected[disease] = True
                self.drift_start_time[disease] = timestamp
        
        # Notify callbacks
        self._notify_alert(alert)
        
        logger.warning(f"Drift alert: {description} (severity: {severity})")
    
    def _notify_alert(self, alert: DriftAlert) -> None:
        """Notify registered callbacks of drift alert."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in drift alert callback: {e}")
    
    def add_alert_callback(self, callback) -> None:
        """Add callback for drift alerts."""
        self.alert_callbacks.append(callback)
    
    def get_drift_status(self) -> Dict[str, Dict]:
        """Get current drift status for all disease types."""
        status = {}
        
        for disease_type in self.supported_diseases:
            recent_alerts = [
                alert for alert in self.alerts[-10:]  # Last 10 alerts
                if disease_type in alert.affected_diseases
            ]
            
            status[disease_type] = {
                "drift_detected": self.drift_detected[disease_type],
                "drift_start_time": self.drift_start_time[disease_type],
                "recent_alerts": len(recent_alerts),
                "last_alert": recent_alerts[-1] if recent_alerts else None,
                "confidence_samples": len(self.confidence_history[disease_type]),
                "accuracy_samples": len(self.accuracy_history[disease_type]),
                "baseline_established": disease_type in self.baseline_confidence
            }
        
        return status
    
    def get_confidence_trends(self, disease_type: str, hours: int = 24) -> Dict:
        """Get confidence trends for a disease type."""
        if disease_type not in self.confidence_history:
            return {"error": f"No data for disease type: {disease_type}"}
        
        # Filter recent data
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        recent_distributions = [
            dist for dist in self.confidence_history[disease_type]
            if dist.timestamp > cutoff_time
        ]
        
        if not recent_distributions:
            return {"error": "No recent data available"}
        
        # Compute trends
        timestamps = [dist.timestamp for dist in recent_distributions]
        mean_confidences = [dist.mean_confidence for dist in recent_distributions]
        entropies = [dist.entropy for dist in recent_distributions]
        
        # Linear trend analysis
        if len(mean_confidences) > 1:
            conf_slope, _, conf_r_value, _, _ = stats.linregress(range(len(mean_confidences)), mean_confidences)
            entropy_slope, _, entropy_r_value, _, _ = stats.linregress(range(len(entropies)), entropies)
        else:
            conf_slope = entropy_slope = conf_r_value = entropy_r_value = 0.0
        
        return {
            "disease_type": disease_type,
            "time_range_hours": hours,
            "sample_count": len(recent_distributions),
            "confidence_trend": {
                "slope": conf_slope,
                "r_squared": conf_r_value ** 2,
                "current_mean": mean_confidences[-1] if mean_confidences else 0.0,
                "baseline_mean": self.baseline_confidence[disease_type].mean_confidence if disease_type in self.baseline_confidence else 0.0
            },
            "entropy_trend": {
                "slope": entropy_slope,
                "r_squared": entropy_r_value ** 2,
                "current_entropy": entropies[-1] if entropies else 0.0,
                "baseline_entropy": self.baseline_confidence[disease_type].entropy if disease_type in self.baseline_confidence else 0.0
            },
            "recent_distributions": recent_distributions[-10:]  # Last 10 for detailed analysis
        }
    
    def plot_confidence_trends(
        self, 
        disease_type: str, 
        save_path: Optional[str] = None
    ) -> None:
        """Plot confidence trends for a disease type."""
        if disease_type not in self.confidence_history:
            logger.warning(f"No data for disease type: {disease_type}")
            return
        
        distributions = list(self.confidence_history[disease_type])
        if not distributions:
            logger.warning(f"No distributions for disease type: {disease_type}")
            return
        
        # Extract data for plotting
        timestamps = [dist.timestamp for dist in distributions]
        mean_confidences = [dist.mean_confidence for dist in distributions]
        entropies = [dist.entropy for dist in distributions]
        
        # Convert timestamps to relative hours
        start_time = timestamps[0]
        time_hours = [(t - start_time) / 3600 for t in timestamps]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Confidence plot
        ax1.plot(time_hours, mean_confidences, 'b-', linewidth=2, label='Mean Confidence')
        if disease_type in self.baseline_confidence:
            baseline_conf = self.baseline_confidence[disease_type].mean_confidence
            ax1.axhline(y=baseline_conf, color='g', linestyle='--', alpha=0.7, label='Baseline')
            ax1.axhline(y=baseline_conf + self.confidence_threshold, 
                       color='orange', linestyle='--', alpha=0.7, label='Warning threshold')
            ax1.axhline(y=baseline_conf - self.confidence_threshold, 
                       color='orange', linestyle='--', alpha=0.7)
        
        ax1.set_ylabel('Mean Confidence')
        ax1.set_title(f'Confidence Trends - {disease_type.title()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Entropy plot
        ax2.plot(time_hours, entropies, 'r-', linewidth=2, label='Entropy')
        if disease_type in self.baseline_confidence:
            baseline_entropy = self.baseline_confidence[disease_type].entropy
            ax2.axhline(y=baseline_entropy, color='g', linestyle='--', alpha=0.7, label='Baseline')
        
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Entropy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confidence trends plot to {save_path}")
        else:
            plt.show()
    
    def reset_drift_state(self, disease_type: Optional[str] = None) -> None:
        """Reset drift detection state."""
        if disease_type:
            self.drift_detected[disease_type] = False
            self.drift_start_time[disease_type] = None
            logger.info(f"Reset drift state for {disease_type}")
        else:
            self.drift_detected.clear()
            self.drift_start_time.clear()
            logger.info("Reset drift state for all disease types")
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """Get summary of recent alerts."""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        recent_alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
        
        # Group by type and severity
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[f"{alert.drift_type}_{alert.severity}"] += 1
        
        return {
            "total_alerts": len(recent_alerts),
            "time_range_hours": hours,
            "alert_breakdown": dict(alert_counts),
            "affected_diseases": list(set(
                disease for alert in recent_alerts 
                for disease in alert.affected_diseases
            )),
            "most_recent_alert": recent_alerts[-1] if recent_alerts else None
        }


if __name__ == "__main__":
    # Demo: Model drift detection
    
    print("=== Model Drift Detection Demo ===\n")
    
    # Create drift detector
    detector = ModelDriftDetector(
        window_size=50,
        confidence_threshold=0.05,
        accuracy_threshold=0.03
    )
    
    # Add alert callback
    def alert_handler(alert: DriftAlert):
        print(f"🚨 DRIFT ALERT: {alert.description} (severity: {alert.severity})")
    
    detector.add_alert_callback(alert_handler)
    
    print("Drift detector initialized")
    
    # Simulate federated learning rounds with gradual drift
    np.random.seed(42)
    
    for round_num in range(1, 101):
        # Simulate predictions for different diseases
        predictions = {}
        ground_truth = {}
        
        for disease in ["breast", "lung", "prostate"]:
            # Simulate gradual confidence drift for breast cancer
            if disease == "breast":
                base_confidence = 0.85
                # Introduce drift after round 50
                if round_num > 50:
                    drift_factor = (round_num - 50) * 0.002  # Gradual decrease
                    base_confidence -= drift_factor
            else:
                base_confidence = 0.80
            
            # Generate predictions
            num_samples = np.random.randint(20, 50)
            confidences = np.random.normal(base_confidence, 0.1, num_samples)
            confidences = np.clip(confidences, 0.0, 1.0)
            
            predictions[disease] = [
                {"confidence": conf, "prediction": int(conf > 0.5)}
                for conf in confidences
            ]
            
            # Generate ground truth (with some noise)
            ground_truth[disease] = [
                int(conf > 0.5 + np.random.normal(0, 0.1))
                for conf in confidences
            ]
        
        # Update detector
        detector.update_predictions(round_num, predictions, ground_truth)
        
        # Print status every 20 rounds
        if round_num % 20 == 0:
            print(f"\n--- Round {round_num} ---")
            status = detector.get_drift_status()
            for disease, info in status.items():
                if info["baseline_established"]:
                    print(f"{disease}: drift={info['drift_detected']}, "
                          f"alerts={info['recent_alerts']}")
    
    # Final analysis
    print(f"\n--- Final Analysis ---")
    
    # Drift status
    final_status = detector.get_drift_status()
    print(f"Drift Status:")
    for disease, info in final_status.items():
        print(f"  {disease}: {'DRIFT DETECTED' if info['drift_detected'] else 'STABLE'}")
    
    # Alert summary
    alert_summary = detector.get_alert_summary(hours=24)
    print(f"\nAlert Summary (24h):")
    print(f"  Total alerts: {alert_summary['total_alerts']}")
    print(f"  Affected diseases: {alert_summary['affected_diseases']}")
    print(f"  Alert breakdown: {alert_summary['alert_breakdown']}")
    
    # Confidence trends
    print(f"\nConfidence Trends:")
    for disease in ["breast", "lung", "prostate"]:
        trends = detector.get_confidence_trends(disease, hours=24)
        if "error" not in trends:
            conf_trend = trends["confidence_trend"]
            print(f"  {disease}: slope={conf_trend['slope']:.6f}, "
                  f"current={conf_trend['current_mean']:.4f}, "
                  f"baseline={conf_trend['baseline_mean']:.4f}")
    
    print("\n=== Demo Complete ===")