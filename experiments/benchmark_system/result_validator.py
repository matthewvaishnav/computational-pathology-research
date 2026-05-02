"""
Result validation and anomaly detection for the Competitor Benchmark System.

This module provides validation of training results, sanity checks for metrics,
anomaly detection, and quality assurance flags for suspicious results.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from experiments.benchmark_system.models import TrainingResult


logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue found in training results."""
    
    severity: str  # "error", "warning", "info"
    category: str  # "metric_range", "anomaly", "progress", "resource"
    message: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None


@dataclass
class ValidationReport:
    """Report of validation results for a training run."""
    
    valid: bool
    issues: List[ValidationIssue]
    requires_manual_review: bool
    qa_flags: List[str]
    
    def __post_init__(self):
        """Ensure lists are initialized."""
        if self.issues is None:
            self.issues = []
        if self.qa_flags is None:
            self.qa_flags = []


class ResultValidator:
    """
    Validates training results for sanity and detects anomalies.
    
    Implements:
    - Metric range validation (accuracy in [0, 1])
    - Anomaly detection (NaN, accuracy below random chance)
    - Training progress verification (loss decreasing)
    - Resource usage sanity checks
    - QA flags for suspicious results
    
    Requirements: 8.5, 8.6, 10.1, 10.2, 10.4, 10.5, 10.6
    """
    
    def __init__(
        self,
        min_accuracy_threshold: float = 0.4,
        max_throughput_samples_per_sec: float = 10000.0,
        max_gpu_memory_mb: float = 24000.0,
        loss_decrease_tolerance: float = 0.01
    ):
        """
        Initialize result validator.
        
        Args:
            min_accuracy_threshold: Minimum acceptable accuracy (below is anomaly)
            max_throughput_samples_per_sec: Maximum plausible throughput
            max_gpu_memory_mb: Maximum plausible GPU memory usage
            loss_decrease_tolerance: Minimum expected loss decrease
        """
        self.min_accuracy_threshold = min_accuracy_threshold
        self.max_throughput_samples_per_sec = max_throughput_samples_per_sec
        self.max_gpu_memory_mb = max_gpu_memory_mb
        self.loss_decrease_tolerance = loss_decrease_tolerance
    
    def validate_training_result(
        self,
        result: TrainingResult
    ) -> ValidationReport:
        """
        Validate a training result for sanity and quality.
        
        Args:
            result: Training result to validate
            
        Returns:
            ValidationReport with validation status and issues
            
        Requirements: 8.5 (Result validation), 8.6 (Anomaly detection),
                     10.1 (Training progress verification), 10.2 (Invalid output detection),
                     10.4 (Resource usage sanity checks), 10.6 (QA flags)
        """
        issues = []
        qa_flags = []
        
        # Validate metric ranges
        issues.extend(self._validate_metric_ranges(result))
        
        # Detect anomalies
        anomaly_issues, anomaly_flags = self._detect_anomalies(result)
        issues.extend(anomaly_issues)
        qa_flags.extend(anomaly_flags)
        
        # Verify training progress
        progress_issues, progress_flags = self._verify_training_progress(result)
        issues.extend(progress_issues)
        qa_flags.extend(progress_flags)
        
        # Check resource usage
        resource_issues, resource_flags = self._check_resource_usage(result)
        issues.extend(resource_issues)
        qa_flags.extend(resource_flags)
        
        # Determine if valid
        error_count = sum(1 for issue in issues if issue.severity == "error")
        valid = error_count == 0
        
        # Determine if manual review needed
        requires_manual_review = (
            error_count > 0 or
            len(qa_flags) > 0 or
            any(issue.severity == "warning" for issue in issues)
        )
        
        return ValidationReport(
            valid=valid,
            issues=issues,
            requires_manual_review=requires_manual_review,
            qa_flags=qa_flags
        )
    
    def _validate_metric_ranges(
        self,
        result: TrainingResult
    ) -> List[ValidationIssue]:
        """
        Validate that metrics are within valid ranges.
        
        Args:
            result: Training result to validate
            
        Returns:
            List of validation issues found
            
        Requirement: 8.5 (Metric range validation), 10.2 (Invalid output detection)
        """
        issues = []
        
        # Check accuracy metrics (should be in [0, 1])
        accuracy_metrics = {
            "test_accuracy": result.test_accuracy,
            "test_precision": result.test_precision,
            "test_recall": result.test_recall,
            "test_f1": result.test_f1,
        }
        
        for metric_name, value in accuracy_metrics.items():
            if not (0.0 <= value <= 1.0):
                issues.append(ValidationIssue(
                    severity="error",
                    category="metric_range",
                    message=f"{metric_name} out of valid range [0, 1]",
                    metric_name=metric_name,
                    metric_value=value
                ))
        
        # Check AUC (should be in [0, 1])
        if not (0.0 <= result.test_auc <= 1.0):
            issues.append(ValidationIssue(
                severity="error",
                category="metric_range",
                message="test_auc out of valid range [0, 1]",
                metric_name="test_auc",
                metric_value=result.test_auc
            ))
        
        # Check for NaN or Inf values
        all_metrics = {
            "test_accuracy": result.test_accuracy,
            "test_auc": result.test_auc,
            "test_f1": result.test_f1,
            "test_precision": result.test_precision,
            "test_recall": result.test_recall,
            "final_train_loss": result.final_train_loss,
            "final_val_loss": result.final_val_loss,
            "training_time_seconds": result.training_time_seconds,
            "samples_per_second": result.samples_per_second,
        }
        
        for metric_name, value in all_metrics.items():
            if math.isnan(value):
                issues.append(ValidationIssue(
                    severity="error",
                    category="metric_range",
                    message=f"{metric_name} is NaN",
                    metric_name=metric_name,
                    metric_value=value
                ))
            elif math.isinf(value):
                issues.append(ValidationIssue(
                    severity="error",
                    category="metric_range",
                    message=f"{metric_name} is Inf",
                    metric_name=metric_name,
                    metric_value=value
                ))
        
        # Check for negative values where they shouldn't be
        non_negative_metrics = {
            "training_time_seconds": result.training_time_seconds,
            "samples_per_second": result.samples_per_second,
            "inference_time_ms": result.inference_time_ms,
            "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
            "avg_gpu_utilization": result.avg_gpu_utilization,
            "model_parameters": result.model_parameters,
        }
        
        for metric_name, value in non_negative_metrics.items():
            if value < 0:
                issues.append(ValidationIssue(
                    severity="error",
                    category="metric_range",
                    message=f"{metric_name} is negative",
                    metric_name=metric_name,
                    metric_value=value
                ))
        
        return issues
    
    def _detect_anomalies(
        self,
        result: TrainingResult
    ) -> tuple[List[ValidationIssue], List[str]]:
        """
        Detect anomalous results that may indicate problems.
        
        Args:
            result: Training result to check
            
        Returns:
            Tuple of (validation issues, QA flags)
            
        Requirement: 8.6 (Anomaly detection), 10.2 (Invalid output detection),
                     10.6 (QA flags)
        """
        issues = []
        qa_flags = []
        
        # Check for accuracy below random chance
        num_classes = result.task_spec.num_classes
        random_chance = 1.0 / num_classes
        
        if result.test_accuracy < random_chance:
            issues.append(ValidationIssue(
                severity="error",
                category="anomaly",
                message=(
                    f"Accuracy ({result.test_accuracy:.3f}) below random chance "
                    f"({random_chance:.3f}) for {num_classes}-class problem"
                ),
                metric_name="test_accuracy",
                metric_value=result.test_accuracy
            ))
            qa_flags.append("accuracy_below_random_chance")
        
        # Check for suspiciously low accuracy
        if result.test_accuracy < self.min_accuracy_threshold:
            issues.append(ValidationIssue(
                severity="warning",
                category="anomaly",
                message=(
                    f"Accuracy ({result.test_accuracy:.3f}) below threshold "
                    f"({self.min_accuracy_threshold:.3f})"
                ),
                metric_name="test_accuracy",
                metric_value=result.test_accuracy
            ))
            qa_flags.append("low_accuracy")
        
        # Check for AUC significantly below 0.5 (worse than random)
        if result.test_auc < 0.5:
            issues.append(ValidationIssue(
                severity="warning",
                category="anomaly",
                message=f"AUC ({result.test_auc:.3f}) below 0.5 (worse than random)",
                metric_name="test_auc",
                metric_value=result.test_auc
            ))
            qa_flags.append("auc_below_random")
        
        # Check for suspiciously high loss values
        if result.final_train_loss > 10.0 or result.final_val_loss > 10.0:
            issues.append(ValidationIssue(
                severity="warning",
                category="anomaly",
                message=(
                    f"High loss values (train: {result.final_train_loss:.3f}, "
                    f"val: {result.final_val_loss:.3f})"
                ),
                metric_name="final_val_loss",
                metric_value=result.final_val_loss
            ))
            qa_flags.append("high_loss")
        
        # Check for perfect accuracy (suspicious)
        if result.test_accuracy >= 0.999:
            issues.append(ValidationIssue(
                severity="warning",
                category="anomaly",
                message=f"Suspiciously high accuracy ({result.test_accuracy:.3f})",
                metric_name="test_accuracy",
                metric_value=result.test_accuracy
            ))
            qa_flags.append("perfect_accuracy")
        
        # Check for zero variance in confidence intervals
        acc_ci_width = result.accuracy_ci[1] - result.accuracy_ci[0]
        if acc_ci_width < 0.001:
            issues.append(ValidationIssue(
                severity="warning",
                category="anomaly",
                message="Confidence interval width suspiciously small",
                metric_name="accuracy_ci",
                metric_value=acc_ci_width
            ))
            qa_flags.append("narrow_confidence_interval")
        
        return issues, qa_flags
    
    def _verify_training_progress(
        self,
        result: TrainingResult
    ) -> tuple[List[ValidationIssue], List[str]]:
        """
        Verify that training made progress (loss decreased).
        
        Args:
            result: Training result to check
            
        Returns:
            Tuple of (validation issues, QA flags)
            
        Requirement: 10.1 (Training progress verification)
        """
        issues = []
        qa_flags = []
        
        # Check if training completed expected epochs
        expected_epochs = result.task_spec.num_epochs
        if result.epochs_completed < expected_epochs:
            issues.append(ValidationIssue(
                severity="warning",
                category="progress",
                message=(
                    f"Training incomplete: {result.epochs_completed}/{expected_epochs} "
                    f"epochs completed"
                ),
                metric_name="epochs_completed",
                metric_value=result.epochs_completed
            ))
            qa_flags.append("incomplete_training")
        
        # Check if validation loss is reasonable compared to training loss
        # (val loss should not be significantly lower than train loss)
        if result.final_val_loss < result.final_train_loss * 0.5:
            issues.append(ValidationIssue(
                severity="warning",
                category="progress",
                message=(
                    f"Validation loss ({result.final_val_loss:.3f}) significantly "
                    f"lower than training loss ({result.final_train_loss:.3f})"
                ),
                metric_name="final_val_loss",
                metric_value=result.final_val_loss
            ))
            qa_flags.append("suspicious_val_loss")
        
        # Check if training time is reasonable
        if result.training_time_seconds < 10.0:
            issues.append(ValidationIssue(
                severity="warning",
                category="progress",
                message=(
                    f"Training time suspiciously short "
                    f"({result.training_time_seconds:.1f}s)"
                ),
                metric_name="training_time_seconds",
                metric_value=result.training_time_seconds
            ))
            qa_flags.append("short_training_time")
        
        return issues, qa_flags
    
    def _check_resource_usage(
        self,
        result: TrainingResult
    ) -> tuple[List[ValidationIssue], List[str]]:
        """
        Check resource usage for sanity.
        
        Args:
            result: Training result to check
            
        Returns:
            Tuple of (validation issues, QA flags)
            
        Requirement: 10.4 (Resource usage sanity checks)
        """
        issues = []
        qa_flags = []
        
        # Check throughput is within plausible range
        if result.samples_per_second > self.max_throughput_samples_per_sec:
            issues.append(ValidationIssue(
                severity="error",
                category="resource",
                message=(
                    f"Throughput ({result.samples_per_second:.1f} samples/s) "
                    f"exceeds theoretical limit ({self.max_throughput_samples_per_sec:.1f})"
                ),
                metric_name="samples_per_second",
                metric_value=result.samples_per_second
            ))
            qa_flags.append("implausible_throughput")
        
        # Check GPU memory usage is within plausible range
        if result.peak_gpu_memory_mb > self.max_gpu_memory_mb:
            issues.append(ValidationIssue(
                severity="error",
                category="resource",
                message=(
                    f"GPU memory ({result.peak_gpu_memory_mb:.1f} MB) "
                    f"exceeds plausible limit ({self.max_gpu_memory_mb:.1f} MB)"
                ),
                metric_name="peak_gpu_memory_mb",
                metric_value=result.peak_gpu_memory_mb
            ))
            qa_flags.append("implausible_gpu_memory")
        
        # Check GPU utilization is in valid range [0, 100]
        if not (0.0 <= result.avg_gpu_utilization <= 100.0):
            issues.append(ValidationIssue(
                severity="error",
                category="resource",
                message=(
                    f"GPU utilization ({result.avg_gpu_utilization:.1f}%) "
                    f"out of valid range [0, 100]"
                ),
                metric_name="avg_gpu_utilization",
                metric_value=result.avg_gpu_utilization
            ))
        
        # Check for suspiciously low GPU utilization
        if result.avg_gpu_utilization < 10.0:
            issues.append(ValidationIssue(
                severity="warning",
                category="resource",
                message=(
                    f"Low GPU utilization ({result.avg_gpu_utilization:.1f}%) "
                    f"may indicate inefficient training"
                ),
                metric_name="avg_gpu_utilization",
                metric_value=result.avg_gpu_utilization
            ))
            qa_flags.append("low_gpu_utilization")
        
        # Check GPU temperature is in valid range
        if not (0.0 <= result.peak_gpu_temperature <= 100.0):
            issues.append(ValidationIssue(
                severity="error",
                category="resource",
                message=(
                    f"GPU temperature ({result.peak_gpu_temperature:.1f}°C) "
                    f"out of valid range [0, 100]"
                ),
                metric_name="peak_gpu_temperature",
                metric_value=result.peak_gpu_temperature
            ))
        
        # Check for high GPU temperature
        if result.peak_gpu_temperature > 85.0:
            issues.append(ValidationIssue(
                severity="warning",
                category="resource",
                message=(
                    f"High GPU temperature ({result.peak_gpu_temperature:.1f}°C) "
                    f"may indicate thermal throttling"
                ),
                metric_name="peak_gpu_temperature",
                metric_value=result.peak_gpu_temperature
            ))
            qa_flags.append("high_gpu_temperature")
        
        # Check inference time is reasonable
        if result.inference_time_ms < 0.01:
            issues.append(ValidationIssue(
                severity="warning",
                category="resource",
                message=(
                    f"Inference time suspiciously low ({result.inference_time_ms:.3f} ms)"
                ),
                metric_name="inference_time_ms",
                metric_value=result.inference_time_ms
            ))
            qa_flags.append("low_inference_time")
        
        return issues, qa_flags
    
    def validate_statistical_significance(
        self,
        result1: TrainingResult,
        result2: TrainingResult,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Validate statistical significance between two results.
        
        Args:
            result1: First training result
            result2: Second training result
            confidence_level: Confidence level for significance testing
            
        Returns:
            Dictionary with significance test results
            
        Requirement: 10.5 (Statistical significance validation)
        """
        # Check if confidence intervals overlap
        acc_ci_overlap = not (
            result1.accuracy_ci[1] < result2.accuracy_ci[0] or
            result2.accuracy_ci[1] < result1.accuracy_ci[0]
        )
        
        auc_ci_overlap = not (
            result1.auc_ci[1] < result2.auc_ci[0] or
            result2.auc_ci[1] < result1.auc_ci[0]
        )
        
        f1_ci_overlap = not (
            result1.f1_ci[1] < result2.f1_ci[0] or
            result2.f1_ci[1] < result1.f1_ci[0]
        )
        
        # Calculate differences
        acc_diff = result1.test_accuracy - result2.test_accuracy
        auc_diff = result1.test_auc - result2.test_auc
        f1_diff = result1.test_f1 - result2.test_f1
        
        return {
            "accuracy": {
                "result1": result1.test_accuracy,
                "result2": result2.test_accuracy,
                "difference": acc_diff,
                "ci_overlap": acc_ci_overlap,
                "significant": not acc_ci_overlap
            },
            "auc": {
                "result1": result1.test_auc,
                "result2": result2.test_auc,
                "difference": auc_diff,
                "ci_overlap": auc_ci_overlap,
                "significant": not auc_ci_overlap
            },
            "f1": {
                "result1": result1.test_f1,
                "result2": result2.test_f1,
                "difference": f1_diff,
                "ci_overlap": f1_ci_overlap,
                "significant": not f1_ci_overlap
            },
            "confidence_level": confidence_level
        }
