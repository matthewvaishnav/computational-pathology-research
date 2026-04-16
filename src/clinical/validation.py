"""
Model validation and monitoring infrastructure for clinical deployment.

This module provides comprehensive model validation including accuracy validation,
AUC validation, bootstrap confidence intervals, and performance monitoring for
concept drift detection and distribution shift.

Example:
    >>> from src.clinical.validation import ModelValidator
    >>> from src.clinical.classifier import MultiClassDiseaseClassifier
    >>>
    >>> validator = ModelValidator()
    >>> model = MultiClassDiseaseClassifier(...)
    >>>
    >>> # Validate model performance
    >>> results = validator.validate_model(model, validation_loader)
    >>> print(f"Accuracy: {results['accuracy']:.3f}")
    >>> print(f"AUC: {results['auc']:.3f}")
    >>>
    >>> # Monitor performance over time
    >>> validator.track_performance(model, new_data_loader)
    >>> if validator.detect_performance_degradation():
    ...     print("Model retraining recommended")
"""

import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.utils import resample
from torch.utils.data import DataLoader

from src.clinical.classifier import MultiClassDiseaseClassifier
from src.clinical.taxonomy import DiseaseTaxonomy

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Comprehensive model validation and monitoring for clinical deployment.

    This class provides:
    - Model accuracy validation (>90% threshold)
    - AUC validation (>0.95 for binary classification)
    - Bootstrap confidence intervals for performance metrics
    - Performance validation by disease taxonomy and patient subpopulation
    - Concept drift and distribution shift detection
    - Performance degradation alerts and retraining recommendations

    Attributes:
        accuracy_threshold: Minimum required accuracy (default: 0.90)
        auc_threshold: Minimum required AUC for binary classification (default: 0.95)
        performance_history: Historical performance metrics for drift detection
        alert_callbacks: Functions to call when performance degrades
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.90,
        auc_threshold: float = 0.95,
        max_history_length: int = 100,
        drift_detection_window: int = 10,
        performance_degradation_threshold: float = 0.05,
    ):
        """
        Initialize ModelValidator.

        Args:
            accuracy_threshold: Minimum required accuracy (0.0-1.0)
            auc_threshold: Minimum required AUC for binary classification (0.0-1.0)
            max_history_length: Maximum number of performance records to keep
            drift_detection_window: Number of recent evaluations to use for drift detection
            performance_degradation_threshold: Threshold for detecting significant performance drops
        """
        self.accuracy_threshold = accuracy_threshold
        self.auc_threshold = auc_threshold
        self.max_history_length = max_history_length
        self.drift_detection_window = drift_detection_window
        self.performance_degradation_threshold = performance_degradation_threshold

        # Performance tracking
        self.performance_history = deque(maxlen=max_history_length)
        self.subpopulation_history = defaultdict(lambda: deque(maxlen=max_history_length))
        self.taxonomy_history = defaultdict(lambda: deque(maxlen=max_history_length))

        # Alert system
        self.alert_callbacks = []

        logger.info(
            f"ModelValidator initialized with accuracy_threshold={accuracy_threshold}, "
            f"auc_threshold={auc_threshold}"
        )

    def validate_model(
        self,
        model: MultiClassDiseaseClassifier,
        validation_loader: DataLoader,
        device: Optional[torch.device] = None,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Comprehensive model validation with bootstrap confidence intervals.

        Args:
            model: Model to validate
            validation_loader: DataLoader with validation data
            device: Device to run validation on
            bootstrap_samples: Number of bootstrap samples for confidence intervals
            confidence_level: Confidence level for intervals (default: 0.95)

        Returns:
            Dictionary containing validation results:
            - accuracy: Overall accuracy
            - auc: Area under ROC curve (for binary classification)
            - precision: Precision scores per class
            - recall: Recall scores per class
            - f1_score: F1 scores per class
            - confusion_matrix: Confusion matrix
            - bootstrap_ci: Bootstrap confidence intervals
            - validation_passed: Whether validation thresholds are met
            - taxonomy_results: Results by disease taxonomy
            - subpopulation_results: Results by patient subpopulation
        """
        if device is None:
            device = next(iter(model.parameters())).device

        model.eval()

        # Collect predictions and ground truth
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_metadata = []

        logger.info("Starting model validation...")
        start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(validation_loader):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                # Get model predictions
                outputs = model(batch)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

                # Store metadata if available
                if "metadata" in batch:
                    all_metadata.extend(batch["metadata"])

                if (batch_idx + 1) % 10 == 0:
                    logger.debug(f"Processed {batch_idx + 1} validation batches")

        validation_time = time.time() - start_time
        logger.info(f"Validation completed in {validation_time:.2f} seconds")

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        labels = np.array(all_labels)

        # Calculate basic metrics
        accuracy = accuracy_score(labels, predictions)

        # Calculate AUC (handle both binary and multi-class)
        num_classes = len(np.unique(labels))
        if num_classes == 2:
            # Binary classification - use binary AUC
            auc_score = roc_auc_score(labels, probabilities[:, 1])
        else:
            # Multi-class - use macro-averaged AUC
            try:
                auc_score = roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
            except ValueError:
                # Handle case where some classes are missing
                auc_score = 0.0
                logger.warning(
                    "Could not calculate AUC - some classes may be missing from validation set"
                )

        # Calculate detailed metrics
        conf_matrix = confusion_matrix(labels, predictions)

        # Calculate per-class metrics
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for class_idx in range(num_classes):
            if class_idx in labels:
                # True positives, false positives, false negatives
                tp = conf_matrix[class_idx, class_idx]
                fp = conf_matrix[:, class_idx].sum() - tp
                fn = conf_matrix[class_idx, :].sum() - tp

                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
            else:
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                f1_scores.append(0.0)

        # Bootstrap confidence intervals
        bootstrap_ci = self._calculate_bootstrap_confidence_intervals(
            labels, predictions, probabilities, bootstrap_samples, confidence_level
        )

        # Validation results
        results = {
            "accuracy": accuracy,
            "auc": auc_score,
            "precision": precision_scores,
            "recall": recall_scores,
            "f1_score": f1_scores,
            "confusion_matrix": conf_matrix.tolist(),
            "bootstrap_ci": bootstrap_ci,
            "num_samples": len(labels),
            "num_classes": num_classes,
            "validation_time": validation_time,
            "timestamp": time.time(),
        }

        # Check validation thresholds
        accuracy_passed = accuracy >= self.accuracy_threshold
        auc_passed = (
            auc_score >= self.auc_threshold if num_classes == 2 else True
        )  # Only check AUC for binary

        results["validation_passed"] = bool(accuracy_passed and auc_passed)
        results["accuracy_passed"] = bool(accuracy_passed)
        results["auc_passed"] = bool(auc_passed)

        # Validate by disease taxonomy if model has taxonomy
        if hasattr(model, "taxonomy") and model.taxonomy is not None:
            results["taxonomy_results"] = self._validate_by_taxonomy(
                model.taxonomy, labels, predictions, probabilities
            )

        # Validate by patient subpopulation if metadata available
        if all_metadata:
            results["subpopulation_results"] = self._validate_by_subpopulation(
                all_metadata, labels, predictions, probabilities
            )

        # Store in performance history
        self.performance_history.append(results)

        # Log results
        logger.info("Validation Results:")
        logger.info(f"  Accuracy: {accuracy:.4f} (threshold: {self.accuracy_threshold})")
        logger.info(f"  AUC: {auc_score:.4f} (threshold: {self.auc_threshold})")
        logger.info(f"  Validation Passed: {results['validation_passed']}")

        if not results["validation_passed"]:
            logger.warning("Model validation FAILED - performance below thresholds")
            self._trigger_alerts("validation_failed", results)

        return results

    def _calculate_bootstrap_confidence_intervals(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        n_bootstrap: int,
        confidence_level: float,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals for performance metrics.

        Args:
            labels: True labels
            predictions: Model predictions
            probabilities: Model probability outputs
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            Dictionary with confidence intervals for each metric
        """
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        bootstrap_accuracies = []
        bootstrap_aucs = []

        logger.debug(f"Calculating bootstrap confidence intervals with {n_bootstrap} samples...")

        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = resample(range(len(labels)), n_samples=len(labels), random_state=i)
            boot_labels = labels[indices]
            boot_predictions = predictions[indices]
            boot_probabilities = probabilities[indices]

            # Calculate metrics
            boot_accuracy = accuracy_score(boot_labels, boot_predictions)
            bootstrap_accuracies.append(boot_accuracy)

            # Calculate AUC if binary classification
            if len(np.unique(boot_labels)) == 2 and len(np.unique(boot_predictions)) == 2:
                try:
                    boot_auc = roc_auc_score(boot_labels, boot_probabilities[:, 1])
                    bootstrap_aucs.append(boot_auc)
                except ValueError:
                    # Skip this bootstrap sample if AUC can't be calculated
                    pass

        # Calculate confidence intervals
        ci_results = {}

        if bootstrap_accuracies:
            ci_results["accuracy"] = (
                np.percentile(bootstrap_accuracies, lower_percentile),
                np.percentile(bootstrap_accuracies, upper_percentile),
            )

        if bootstrap_aucs:
            ci_results["auc"] = (
                np.percentile(bootstrap_aucs, lower_percentile),
                np.percentile(bootstrap_aucs, upper_percentile),
            )

        logger.debug(f"Bootstrap confidence intervals calculated: {ci_results}")
        return ci_results

    def _validate_by_taxonomy(
        self,
        taxonomy: DiseaseTaxonomy,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Validate performance separately for each disease taxonomy level.

        Args:
            taxonomy: Disease taxonomy
            labels: True labels
            predictions: Model predictions
            probabilities: Model probability outputs

        Returns:
            Dictionary with performance metrics by taxonomy level
        """
        results = {}

        # Get taxonomy hierarchy
        taxonomy_info = taxonomy.get_taxonomy_info()

        for level_name, level_classes in taxonomy_info.items():
            if level_name == "hierarchy":
                continue

            # Map labels to this taxonomy level
            level_labels = []
            level_predictions = []

            for true_label, pred_label in zip(labels, predictions):
                # Find which level class this belongs to
                true_level_class = None
                pred_level_class = None

                for class_name, class_ids in level_classes.items():
                    if true_label in class_ids:
                        true_level_class = class_name
                    if pred_label in class_ids:
                        pred_level_class = class_name

                if true_level_class is not None and pred_level_class is not None:
                    level_labels.append(true_level_class)
                    level_predictions.append(pred_level_class)

            if level_labels:
                # Calculate metrics for this taxonomy level
                level_accuracy = accuracy_score(level_labels, level_predictions)

                results[level_name] = {
                    "accuracy": level_accuracy,
                    "num_samples": len(level_labels),
                    "num_classes": len(set(level_labels)),
                }

                # Store in taxonomy history
                self.taxonomy_history[level_name].append(
                    {"accuracy": level_accuracy, "timestamp": time.time()}
                )

        return results

    def _validate_by_subpopulation(
        self,
        metadata: List[Dict],
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Validate performance separately for patient subpopulations.

        Args:
            metadata: List of metadata dictionaries for each sample
            labels: True labels
            predictions: Model predictions
            probabilities: Model probability outputs

        Returns:
            Dictionary with performance metrics by subpopulation
        """
        results = {}

        # Define subpopulations based on metadata
        subpopulations = {"age_groups": {}, "sex": {}, "smoking_status": {}, "disease_history": {}}

        # Group samples by subpopulation
        for i, meta in enumerate(metadata):
            if i >= len(labels):
                break

            # Age groups
            if "age" in meta and meta["age"] is not None:
                age = meta["age"]
                if age < 40:
                    age_group = "under_40"
                elif age < 65:
                    age_group = "40_to_65"
                else:
                    age_group = "over_65"

                if age_group not in subpopulations["age_groups"]:
                    subpopulations["age_groups"][age_group] = {"labels": [], "predictions": []}
                subpopulations["age_groups"][age_group]["labels"].append(labels[i])
                subpopulations["age_groups"][age_group]["predictions"].append(predictions[i])

            # Sex
            if "sex" in meta and meta["sex"] is not None:
                sex = meta["sex"]
                if sex not in subpopulations["sex"]:
                    subpopulations["sex"][sex] = {"labels": [], "predictions": []}
                subpopulations["sex"][sex]["labels"].append(labels[i])
                subpopulations["sex"][sex]["predictions"].append(predictions[i])

            # Smoking status
            if "smoking_status" in meta and meta["smoking_status"] is not None:
                smoking = meta["smoking_status"]
                if smoking not in subpopulations["smoking_status"]:
                    subpopulations["smoking_status"][smoking] = {"labels": [], "predictions": []}
                subpopulations["smoking_status"][smoking]["labels"].append(labels[i])
                subpopulations["smoking_status"][smoking]["predictions"].append(predictions[i])

        # Calculate metrics for each subpopulation
        for pop_type, populations in subpopulations.items():
            results[pop_type] = {}

            for pop_name, pop_data in populations.items():
                if len(pop_data["labels"]) > 0:
                    pop_accuracy = accuracy_score(pop_data["labels"], pop_data["predictions"])

                    results[pop_type][pop_name] = {
                        "accuracy": pop_accuracy,
                        "num_samples": len(pop_data["labels"]),
                    }

                    # Store in subpopulation history
                    subpop_key = f"{pop_type}_{pop_name}"
                    self.subpopulation_history[subpop_key].append(
                        {"accuracy": pop_accuracy, "timestamp": time.time()}
                    )

        return results

    def track_performance(
        self,
        model: MultiClassDiseaseClassifier,
        data_loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Track model performance over time for concept drift detection.

        Args:
            model: Model to evaluate
            data_loader: DataLoader with new data
            device: Device to run evaluation on

        Returns:
            Dictionary containing performance metrics and drift detection results
        """
        # Run validation on new data
        results = self.validate_model(model, data_loader, device, bootstrap_samples=100)

        # Detect concept drift and distribution shift
        drift_results = self.detect_concept_drift()
        results["drift_detection"] = drift_results

        # Check for performance degradation
        degradation_detected = self.detect_performance_degradation()
        results["performance_degradation"] = degradation_detected

        if degradation_detected:
            logger.warning("Performance degradation detected - model retraining recommended")
            self._trigger_alerts("performance_degradation", results)

        if drift_results["drift_detected"]:
            logger.warning(f"Concept drift detected: {drift_results['drift_type']}")
            self._trigger_alerts("concept_drift", results)

        return results

    def detect_concept_drift(self) -> Dict[str, Any]:
        """
        Detect concept drift and distribution shift in model performance.

        Returns:
            Dictionary containing drift detection results:
            - drift_detected: Whether drift was detected
            - drift_type: Type of drift (accuracy, auc, distribution)
            - drift_magnitude: Magnitude of the drift
            - confidence: Confidence in drift detection
        """
        if len(self.performance_history) < self.drift_detection_window:
            return {
                "drift_detected": False,
                "drift_type": None,
                "drift_magnitude": 0.0,
                "confidence": 0.0,
                "message": f"Insufficient data for drift detection (need {self.drift_detection_window} samples)",
            }

        # Get recent performance metrics
        recent_metrics = list(self.performance_history)[-self.drift_detection_window:]
        older_metrics = list(self.performance_history)[: -self.drift_detection_window]

        if not older_metrics:
            return {
                "drift_detected": False,
                "drift_type": None,
                "drift_magnitude": 0.0,
                "confidence": 0.0,
                "message": "No historical data for comparison",
            }

        # Calculate average performance for recent vs older periods
        recent_accuracy = np.mean([m["accuracy"] for m in recent_metrics])
        older_accuracy = np.mean([m["accuracy"] for m in older_metrics])

        recent_auc = np.mean([m["auc"] for m in recent_metrics if m["auc"] > 0])
        older_auc = np.mean([m["auc"] for m in older_metrics if m["auc"] > 0])

        # Detect accuracy drift
        accuracy_drift = abs(recent_accuracy - older_accuracy)
        accuracy_drift_detected = accuracy_drift > self.performance_degradation_threshold

        # Detect AUC drift (if applicable)
        auc_drift = abs(recent_auc - older_auc) if recent_auc > 0 and older_auc > 0 else 0
        auc_drift_detected = auc_drift > self.performance_degradation_threshold

        # Determine drift type and magnitude
        drift_detected = accuracy_drift_detected or auc_drift_detected

        if accuracy_drift_detected and auc_drift_detected:
            drift_type = "combined"
            drift_magnitude = max(accuracy_drift, auc_drift)
        elif accuracy_drift_detected:
            drift_type = "accuracy"
            drift_magnitude = accuracy_drift
        elif auc_drift_detected:
            drift_type = "auc"
            drift_magnitude = auc_drift
        else:
            drift_type = None
            drift_magnitude = 0.0

        # Calculate confidence based on consistency of drift across window
        if drift_detected:
            # Check if drift is consistent across the window
            accuracy_trend = [m["accuracy"] for m in recent_metrics]
            if len(accuracy_trend) > 1:
                # Simple trend analysis - check if performance is consistently lower
                declining_count = sum(
                    1
                    for i in range(1, len(accuracy_trend))
                    if accuracy_trend[i] < accuracy_trend[i - 1]
                )
                confidence = declining_count / (len(accuracy_trend) - 1)
            else:
                confidence = 0.5
        else:
            confidence = 0.0

        return {
            "drift_detected": drift_detected,
            "drift_type": drift_type,
            "drift_magnitude": drift_magnitude,
            "confidence": confidence,
            "recent_accuracy": recent_accuracy,
            "older_accuracy": older_accuracy,
            "recent_auc": recent_auc,
            "older_auc": older_auc,
            "accuracy_drift": accuracy_drift,
            "auc_drift": auc_drift,
        }

    def detect_performance_degradation(self) -> bool:
        """
        Detect if model performance has degraded below acceptable thresholds.

        Returns:
            True if performance degradation is detected, False otherwise
        """
        if len(self.performance_history) < 2:
            return False

        # Get most recent performance
        latest_performance = self.performance_history[-1]

        # Check if current performance is below thresholds
        accuracy_degraded = latest_performance["accuracy"] < self.accuracy_threshold
        auc_degraded = (
            latest_performance["auc"] < self.auc_threshold
            if latest_performance["auc"] > 0
            else False
        )

        # Also check for significant drop from historical average
        if len(self.performance_history) >= 5:
            historical_accuracy = np.mean(
                [m["accuracy"] for m in list(self.performance_history)[:-1]]
            )
            accuracy_drop = historical_accuracy - latest_performance["accuracy"]
            significant_drop = accuracy_drop > self.performance_degradation_threshold
        else:
            significant_drop = False

        return accuracy_degraded or auc_degraded or significant_drop

    def recommend_retraining(self) -> Dict[str, Any]:
        """
        Generate recommendations for model retraining based on performance analysis.

        Returns:
            Dictionary containing retraining recommendations:
            - should_retrain: Whether retraining is recommended
            - urgency: Urgency level (low, medium, high, critical)
            - reasons: List of reasons for retraining
            - suggested_actions: List of suggested actions
        """
        if not self.performance_history:
            return {
                "should_retrain": False,
                "urgency": "none",
                "reasons": [],
                "suggested_actions": ["Collect validation data and run initial validation"],
            }

        latest_performance = self.performance_history[-1]
        drift_results = self.detect_concept_drift()
        degradation_detected = self.detect_performance_degradation()

        reasons = []
        urgency = "none"
        should_retrain = False

        # Check validation failure
        if not latest_performance.get("validation_passed", True):
            should_retrain = True
            urgency = "critical"
            reasons.append(
                f"Model failed validation thresholds (accuracy: {latest_performance.get('accuracy', 0):.3f}, "
                f"AUC: {latest_performance.get('auc', 0):.3f})"
            )

        # Check concept drift
        if drift_results["drift_detected"]:
            should_retrain = True
            if drift_results["confidence"] > 0.8:
                urgency = "high" if urgency != "critical" else urgency
            else:
                urgency = "medium" if urgency not in ["critical", "high"] else urgency
            reasons.append(
                f"Concept drift detected ({drift_results['drift_type']}) with "
                f"magnitude {drift_results['drift_magnitude']:.3f}"
            )

        # Check performance degradation
        if degradation_detected:
            should_retrain = True
            urgency = "high" if urgency not in ["critical"] else urgency
            reasons.append("Performance degradation below acceptable thresholds")

        # Check subpopulation performance
        if "subpopulation_results" in latest_performance:
            poor_subpop_performance = []
            for pop_type, populations in latest_performance["subpopulation_results"].items():
                for pop_name, metrics in populations.items():
                    if metrics.get("accuracy", 1.0) < self.accuracy_threshold:
                        poor_subpop_performance.append(f"{pop_type}_{pop_name}")

            if poor_subpop_performance:
                should_retrain = True
                urgency = "medium" if urgency not in ["critical", "high"] else urgency
                reasons.append(
                    f"Poor performance in subpopulations: {', '.join(poor_subpop_performance)}"
                )

        # Generate suggested actions
        suggested_actions = []

        if should_retrain:
            suggested_actions.extend(
                [
                    "Collect additional training data, especially for underperforming subpopulations",
                    "Review data quality and preprocessing pipeline",
                    "Consider model architecture updates or hyperparameter tuning",
                    "Implement data augmentation strategies",
                    "Schedule model retraining with expanded dataset",
                ]
            )

            if drift_results["drift_detected"]:
                suggested_actions.append("Investigate sources of distribution shift in input data")

            if "subpopulation_results" in latest_performance:
                suggested_actions.append("Focus on collecting more diverse training data")
        else:
            suggested_actions.extend(
                [
                    "Continue monitoring model performance",
                    "Maintain current validation schedule",
                    "Consider proactive data collection for future retraining",
                ]
            )

        return {
            "should_retrain": should_retrain,
            "urgency": urgency,
            "reasons": reasons,
            "suggested_actions": suggested_actions,
            "latest_accuracy": latest_performance.get("accuracy", 0),
            "latest_auc": latest_performance.get("auc", 0),
            "drift_confidence": drift_results.get("confidence", 0.0),
        }

    def add_alert_callback(self, callback: callable):
        """
        Add a callback function to be called when performance alerts are triggered.

        Args:
            callback: Function to call with (alert_type, data) arguments
        """
        self.alert_callbacks.append(callback)

    def _trigger_alerts(self, alert_type: str, data: Dict[str, Any]):
        """
        Trigger all registered alert callbacks.

        Args:
            alert_type: Type of alert (validation_failed, performance_degradation, concept_drift)
            data: Alert data to pass to callbacks
        """
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of model performance history and current status.

        Returns:
            Dictionary containing performance summary
        """
        if not self.performance_history:
            return {"status": "no_data", "message": "No performance data available"}

        latest = self.performance_history[-1]

        # Calculate trends
        if len(self.performance_history) >= 5:
            recent_accuracies = [m["accuracy"] for m in list(self.performance_history)[-5:]]
            accuracy_trend = (
                "improving" if recent_accuracies[-1] > recent_accuracies[0] else "declining"
            )

            recent_aucs = [m["auc"] for m in list(self.performance_history)[-5:] if m["auc"] > 0]
            auc_trend = (
                "improving" if recent_aucs and recent_aucs[-1] > recent_aucs[0] else "declining"
            )
        else:
            accuracy_trend = "stable"
            auc_trend = "stable"

        # Determine overall status
        if latest.get("validation_passed", True):
            if self.detect_performance_degradation():
                status = "degraded"
            elif self.detect_concept_drift()["drift_detected"]:
                status = "drift_detected"
            else:
                status = "healthy"
        else:
            status = "failed"

        return {
            "status": status,
            "latest_accuracy": latest.get("accuracy", 0),
            "latest_auc": latest.get("auc", 0),
            "accuracy_trend": accuracy_trend,
            "auc_trend": auc_trend,
            "validation_passed": latest.get("validation_passed", True),
            "total_evaluations": len(self.performance_history),
            "drift_detected": self.detect_concept_drift()["drift_detected"],
            "performance_degradation": self.detect_performance_degradation(),
            "retraining_recommendation": self.recommend_retraining(),
        }

    def export_performance_history(self, filepath: str):
        """
        Export performance history to file for analysis.

        Args:
            filepath: Path to save performance history
        """
        import json

        # Convert deque to list and handle numpy types
        history_data = {
            "performance_history": [
                {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in record.items()}
                for record in self.performance_history
            ],
            "subpopulation_history": {k: list(v) for k, v in self.subpopulation_history.items()},
            "taxonomy_history": {k: list(v) for k, v in self.taxonomy_history.items()},
            "configuration": {
                "accuracy_threshold": self.accuracy_threshold,
                "auc_threshold": self.auc_threshold,
                "max_history_length": self.max_history_length,
                "drift_detection_window": self.drift_detection_window,
                "performance_degradation_threshold": self.performance_degradation_threshold,
            },
        }

        with open(filepath, "w") as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"Performance history exported to {filepath}")


class PerformanceMonitor:
    """
    Real-time performance monitoring system for production deployment.

    This class provides continuous monitoring of model performance with
    automatic alerting and retraining recommendations.
    """

    def __init__(
        self,
        validator: ModelValidator,
        monitoring_interval: float = 3600.0,  # 1 hour
        alert_email: Optional[str] = None,
        alert_webhook: Optional[str] = None,
    ):
        """
        Initialize PerformanceMonitor.

        Args:
            validator: ModelValidator instance
            monitoring_interval: Interval between monitoring checks (seconds)
            alert_email: Email address for alerts
            alert_webhook: Webhook URL for alerts
        """
        self.validator = validator
        self.monitoring_interval = monitoring_interval
        self.alert_email = alert_email
        self.alert_webhook = alert_webhook

        # Add alert callbacks
        self.validator.add_alert_callback(self._handle_alert)

        self.is_monitoring = False
        self.last_check_time = 0

        logger.info(f"PerformanceMonitor initialized with {monitoring_interval}s interval")

    def start_monitoring(self):
        """Start continuous performance monitoring."""
        self.is_monitoring = True
        self.last_check_time = time.time()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop continuous performance monitoring."""
        self.is_monitoring = False
        logger.info("Performance monitoring stopped")

    def check_performance(
        self,
        model: MultiClassDiseaseClassifier,
        data_loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> bool:
        """
        Check if it's time to run performance monitoring and execute if needed.

        Args:
            model: Model to monitor
            data_loader: DataLoader with monitoring data
            device: Device to run monitoring on

        Returns:
            True if monitoring was performed, False otherwise
        """
        if not self.is_monitoring:
            return False

        current_time = time.time()
        if current_time - self.last_check_time < self.monitoring_interval:
            return False

        logger.info("Running scheduled performance monitoring...")

        try:
            # Track performance
            self.validator.track_performance(model, data_loader, device)

            # Log summary
            summary = self.validator.get_performance_summary()
            logger.info(f"Performance monitoring completed - Status: {summary['status']}")

            self.last_check_time = current_time
            return True

        except Exception as e:
            logger.error(f"Error during performance monitoring: {e}")
            return False

    def _handle_alert(self, alert_type: str, data: Dict[str, Any]):
        """
        Handle performance alerts by sending notifications.

        Args:
            alert_type: Type of alert
            data: Alert data
        """
        message = self._format_alert_message(alert_type, data)

        logger.warning(f"PERFORMANCE ALERT: {message}")

        # Send email alert if configured
        if self.alert_email:
            self._send_email_alert(alert_type, message)

        # Send webhook alert if configured
        if self.alert_webhook:
            self._send_webhook_alert(alert_type, message, data)

    def _format_alert_message(self, alert_type: str, data: Dict[str, Any]) -> str:
        """Format alert message for notifications."""
        if alert_type == "validation_failed":
            return (
                f"Model validation failed - Accuracy: {data['accuracy']:.3f}, "
                f"AUC: {data['auc']:.3f}"
            )
        elif alert_type == "performance_degradation":
            return f"Performance degradation detected - Current accuracy: {data['accuracy']:.3f}"
        elif alert_type == "concept_drift":
            drift_info = data.get("drift_detection", {})
            return (
                f"Concept drift detected - Type: {drift_info.get('drift_type', 'unknown')}, "
                f"Magnitude: {drift_info.get('drift_magnitude', 0):.3f}"
            )
        else:
            return f"Unknown alert type: {alert_type}"

    def _send_email_alert(self, alert_type: str, message: str):
        """Send email alert (placeholder implementation)."""
        # This would integrate with an email service
        logger.info(f"EMAIL ALERT: {message}")
        # TODO: Implement actual email sending

    def _send_webhook_alert(self, alert_type: str, message: str, data: Dict[str, Any]):
        """Send webhook alert (placeholder implementation)."""
        # This would send HTTP POST to webhook URL
        logger.info(f"WEBHOOK ALERT: {message}")
        # TODO: Implement actual webhook sending
