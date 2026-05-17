"""
Model validation infrastructure for clinical deployment.

Provides comprehensive model validation including accuracy validation,
AUC validation, and bootstrap confidence intervals.
"""

import logging
import time
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
    Comprehensive model validation for clinical deployment.

    Provides methods for validating model performance including accuracy,
    AUC, bootstrap confidence intervals, and subpopulation analysis.
    """

    def __init__(self, taxonomy: Optional[DiseaseTaxonomy] = None):
        """
        Initialize model validator.

        Args:
            taxonomy: Disease taxonomy for hierarchical validation
        """
        self.taxonomy = taxonomy
        self.validation_history = []

    def validate_model(
        self,
        model: MultiClassDiseaseClassifier,
        validation_loader: DataLoader,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Comprehensive model validation with bootstrap confidence intervals.

        Args:
            model: Model to validate
            validation_loader: Validation data loader
            bootstrap_samples: Number of bootstrap samples
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary containing validation metrics
        """
        logger.info("Starting comprehensive model validation")

        # Collect predictions and ground truth
        all_predictions = []
        all_probabilities = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for batch in validation_loader:
                features = batch["features"]
                labels = batch["labels"]

                probabilities = model.predict_proba(features)
                predictions = model.predict(features)

                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
                all_labels.extend(labels.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)

        # Calculate base metrics
        results = self._calculate_base_metrics(all_labels, all_predictions, all_probabilities)

        # Calculate bootstrap confidence intervals
        bootstrap_results = self._calculate_bootstrap_confidence_intervals(
            all_labels, all_predictions, all_probabilities, bootstrap_samples, confidence_level
        )
        results.update(bootstrap_results)

        # Validate by taxonomy if available
        if self.taxonomy:
            taxonomy_results = self._validate_by_taxonomy(
                all_labels, all_predictions, all_probabilities
            )
            results["taxonomy_validation"] = taxonomy_results

        # Validate by subpopulation
        subpop_results = self._validate_by_subpopulation(
            all_labels, all_predictions, all_probabilities, validation_loader
        )
        results["subpopulation_validation"] = subpop_results

        # Store validation history
        self.validation_history.append({"timestamp": time.time(), "results": results})

        logger.info(f"Validation complete. Accuracy: {results['accuracy']:.3f}")
        return results

    def _calculate_base_metrics(
        self, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
    ) -> Dict[str, float]:
        """Calculate base validation metrics."""
        metrics = {}

        # Accuracy
        metrics["accuracy"] = accuracy_score(labels, predictions)

        # AUC (handle binary and multiclass)
        if probabilities.shape[1] == 2:
            metrics["auc"] = roc_auc_score(labels, probabilities[:, 1])
        else:
            metrics["auc"] = roc_auc_score(
                labels, probabilities, multi_class="ovr", average="macro"
            )

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics["confusion_matrix"] = cm.tolist()

        # Per-class metrics
        num_classes = probabilities.shape[1]
        for class_idx in range(num_classes):
            class_mask = labels == class_idx
            if class_mask.sum() > 0:
                class_predictions = predictions[class_mask]
                class_accuracy = (class_predictions == class_idx).mean()
                metrics[f"class_{class_idx}_accuracy"] = class_accuracy

        return metrics

    def _calculate_bootstrap_confidence_intervals(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        n_bootstrap: int,
        confidence_level: float,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate bootstrap confidence intervals for metrics."""
        bootstrap_accuracies = []
        bootstrap_aucs = []

        n_samples = len(labels)

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = resample(range(n_samples), n_samples=n_samples)
            boot_labels = labels[indices]
            boot_predictions = predictions[indices]
            boot_probabilities = probabilities[indices]

            # Calculate metrics
            boot_accuracy = accuracy_score(boot_labels, boot_predictions)
            bootstrap_accuracies.append(boot_accuracy)

            try:
                if probabilities.shape[1] == 2:
                    boot_auc = roc_auc_score(boot_labels, boot_probabilities[:, 1])
                else:
                    boot_auc = roc_auc_score(
                        boot_labels, boot_probabilities, multi_class="ovr", average="macro"
                    )
                bootstrap_aucs.append(boot_auc)
            except ValueError:
                # Skip if bootstrap sample doesn't have all classes
                continue

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        results = {
            "bootstrap_confidence_intervals": {
                "accuracy": {
                    "lower": np.percentile(bootstrap_accuracies, lower_percentile),
                    "upper": np.percentile(bootstrap_accuracies, upper_percentile),
                    "mean": np.mean(bootstrap_accuracies),
                    "std": np.std(bootstrap_accuracies),
                }
            }
        }

        if bootstrap_aucs:
            results["bootstrap_confidence_intervals"]["auc"] = {
                "lower": np.percentile(bootstrap_aucs, lower_percentile),
                "upper": np.percentile(bootstrap_aucs, upper_percentile),
                "mean": np.mean(bootstrap_aucs),
                "std": np.std(bootstrap_aucs),
            }

        return results

    def _validate_by_taxonomy(
        self, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """Validate performance by disease taxonomy hierarchy."""
        results = {}

        # Get all taxonomy levels
        for level in range(self.taxonomy.get_max_level() + 1):
            level_diseases = self.taxonomy.get_diseases_at_level(level)
            level_results = {}

            for disease in level_diseases:
                disease_id = disease.disease_id
                disease_mask = labels == disease_id

                if disease_mask.sum() > 0:
                    disease_predictions = predictions[disease_mask]
                    disease_accuracy = (disease_predictions == disease_id).mean()
                    level_results[disease.name] = {
                        "accuracy": disease_accuracy,
                        "sample_count": disease_mask.sum(),
                    }

            results[f"level_{level}"] = level_results

        return results

    def _validate_by_subpopulation(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        validation_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Validate performance by patient subpopulations."""
        # This is a simplified implementation
        # In practice, you'd extract demographic/clinical metadata
        results = {
            "overall_performance": {
                "accuracy": accuracy_score(labels, predictions),
                "sample_count": len(labels),
            }
        }

        # Add age-based validation if metadata available
        # Add gender-based validation if metadata available
        # Add comorbidity-based validation if metadata available

        return results
