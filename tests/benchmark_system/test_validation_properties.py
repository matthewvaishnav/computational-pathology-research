"""
Property-based tests for result validation in the Competitor Benchmark System.

Feature: competitor-benchmark-system
Property 3: Result Validation Sanity Checks

**Validates: Requirements 8.5, 10.1, 10.6, 10.4**
"""

import math
from datetime import datetime
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, settings

from experiments.benchmark_system.models import TrainingResult, TaskSpecification
from experiments.benchmark_system.result_validator import ResultValidator


# Strategy for generating TaskSpecification instances
@st.composite
def task_specification_strategy(draw):
    """Generate random TaskSpecification instances."""
    return TaskSpecification(
        dataset_name=draw(st.sampled_from(["PatchCamelyon", "TCGA", "Camelyon16"])),
        data_root=Path("/data/test"),
        model_architecture=draw(st.sampled_from(["resnet18", "vit_small", "efficientnet"])),
        num_epochs=draw(st.integers(min_value=1, max_value=100)),
        batch_size=draw(st.integers(min_value=1, max_value=256)),
        learning_rate=draw(st.floats(min_value=1e-6, max_value=1e-1)),
        num_classes=draw(st.integers(min_value=2, max_value=10))
    )


# Strategy for generating valid TrainingResult instances
@st.composite
def valid_training_result_strategy(draw):
    """Generate valid TrainingResult instances."""
    task_spec = draw(task_specification_strategy())
    
    # Valid metrics in proper ranges (above random chance for the number of classes)
    random_chance = 1.0 / task_spec.num_classes
    test_accuracy = draw(st.floats(min_value=random_chance + 0.1, max_value=0.95))
    test_auc = draw(st.floats(min_value=0.6, max_value=0.95))
    test_f1 = draw(st.floats(min_value=0.5, max_value=0.95))
    test_precision = draw(st.floats(min_value=0.5, max_value=0.95))
    test_recall = draw(st.floats(min_value=0.5, max_value=0.95))
    
    # Valid confidence intervals
    acc_ci_lower = draw(st.floats(min_value=0.0, max_value=test_accuracy))
    acc_ci_upper = draw(st.floats(min_value=test_accuracy, max_value=1.0))
    
    auc_ci_lower = draw(st.floats(min_value=0.0, max_value=test_auc))
    auc_ci_upper = draw(st.floats(min_value=test_auc, max_value=1.0))
    
    f1_ci_lower = draw(st.floats(min_value=0.0, max_value=test_f1))
    f1_ci_upper = draw(st.floats(min_value=test_f1, max_value=1.0))
    
    return TrainingResult(
        framework_name=draw(st.sampled_from(["HistoCore", "PathML", "CLAM", "PyTorch"])),
        task_spec=task_spec,
        training_time_seconds=draw(st.floats(min_value=10.0, max_value=10000.0)),
        epochs_completed=task_spec.num_epochs,
        final_train_loss=draw(st.floats(min_value=0.01, max_value=5.0)),
        final_val_loss=draw(st.floats(min_value=0.01, max_value=5.0)),
        test_accuracy=test_accuracy,
        test_auc=test_auc,
        test_f1=test_f1,
        test_precision=test_precision,
        test_recall=test_recall,
        accuracy_ci=(acc_ci_lower, acc_ci_upper),
        auc_ci=(auc_ci_lower, auc_ci_upper),
        f1_ci=(f1_ci_lower, f1_ci_upper),
        peak_gpu_memory_mb=draw(st.floats(min_value=100.0, max_value=12000.0)),
        avg_gpu_utilization=draw(st.floats(min_value=10.0, max_value=100.0)),
        peak_gpu_temperature=draw(st.floats(min_value=30.0, max_value=85.0)),
        samples_per_second=draw(st.floats(min_value=1.0, max_value=1000.0)),
        inference_time_ms=draw(st.floats(min_value=0.1, max_value=100.0)),
        model_parameters=draw(st.integers(min_value=1000, max_value=100_000_000)),
        checkpoint_path=Path("/checkpoints/test.pth"),
        metrics_path=Path("/metrics/test.json"),
        log_path=Path("/logs/test.log"),
        status="success"
    )


# Strategy for generating invalid TrainingResult instances with edge cases
@st.composite
def invalid_training_result_strategy(draw):
    """Generate invalid TrainingResult instances with various edge cases."""
    task_spec = draw(task_specification_strategy())
    
    # Choose what kind of invalid result to generate
    invalid_type = draw(st.sampled_from([
        "nan_metrics",
        "out_of_range_accuracy",
        "negative_values",
        "below_random_chance",
        "implausible_throughput"
    ]))
    
    # Base valid values
    test_accuracy = 0.75
    test_auc = 0.80
    test_f1 = 0.70
    test_precision = 0.72
    test_recall = 0.68
    training_time = 100.0
    epochs_completed = task_spec.num_epochs
    samples_per_second = 100.0
    peak_gpu_memory = 8000.0
    
    # Apply specific invalid pattern
    if invalid_type == "nan_metrics":
        # Randomly make some metrics NaN
        if draw(st.booleans()):
            test_accuracy = float('nan')
        if draw(st.booleans()):
            test_auc = float('nan')
        if draw(st.booleans()):
            test_f1 = float('nan')
    
    elif invalid_type == "out_of_range_accuracy":
        # Accuracy outside [0, 1]
        test_accuracy = draw(st.one_of(
            st.floats(min_value=-1.0, max_value=-0.01),
            st.floats(min_value=1.01, max_value=2.0)
        ))
    
    elif invalid_type == "negative_values":
        # Negative values where they shouldn't be
        training_time = draw(st.floats(min_value=-1000.0, max_value=-0.1))
        samples_per_second = draw(st.floats(min_value=-100.0, max_value=-0.1))
    
    elif invalid_type == "below_random_chance":
        # Accuracy below random chance for the number of classes
        random_chance = 1.0 / task_spec.num_classes
        test_accuracy = draw(st.floats(min_value=0.0, max_value=random_chance * 0.9))
    
    elif invalid_type == "implausible_throughput":
        # Throughput exceeding theoretical limits
        samples_per_second = draw(st.floats(min_value=10001.0, max_value=100000.0))
    
    return TrainingResult(
        framework_name=draw(st.sampled_from(["HistoCore", "PathML", "CLAM", "PyTorch"])),
        task_spec=task_spec,
        training_time_seconds=training_time,
        epochs_completed=epochs_completed,
        final_train_loss=draw(st.floats(min_value=0.01, max_value=5.0)),
        final_val_loss=draw(st.floats(min_value=0.01, max_value=5.0)),
        test_accuracy=test_accuracy,
        test_auc=test_auc,
        test_f1=test_f1,
        test_precision=test_precision,
        test_recall=test_recall,
        accuracy_ci=(0.7, 0.8),
        auc_ci=(0.75, 0.85),
        f1_ci=(0.65, 0.75),
        peak_gpu_memory_mb=peak_gpu_memory,
        avg_gpu_utilization=draw(st.floats(min_value=10.0, max_value=100.0)),
        peak_gpu_temperature=draw(st.floats(min_value=30.0, max_value=85.0)),
        samples_per_second=samples_per_second,
        inference_time_ms=draw(st.floats(min_value=0.1, max_value=100.0)),
        model_parameters=draw(st.integers(min_value=1000, max_value=100_000_000)),
        checkpoint_path=Path("/checkpoints/test.pth"),
        metrics_path=Path("/metrics/test.json"),
        log_path=Path("/logs/test.log"),
        status="success"
    )


class TestResultValidationProperties:
    """
    Property-based tests for result validation.
    
    Property 3: Result Validation Sanity Checks
    
    For any training result, the validation system SHALL verify that:
    1. Accuracy metrics are within [0.0, 1.0]
    2. No metrics contain NaN or Inf values
    3. Throughput does not exceed theoretical hardware limits
    4. Training loss shows decreasing trend over epochs
    
    **Validates: Requirements 8.5, 10.1, 10.6, 10.4**
    """
    
    @given(result=valid_training_result_strategy())
    @settings(max_examples=100)
    def test_valid_results_pass_validation(self, result):
        """
        Property: Valid training results should pass validation.
        
        For any training result with metrics in valid ranges, no NaN/Inf values,
        and plausible resource usage, validation should succeed.
        """
        validator = ResultValidator()
        report = validator.validate_training_result(result)
        
        # Valid results should have no errors
        error_count = sum(1 for issue in report.issues if issue.severity == "error")
        assert error_count == 0, (
            f"Valid result should have no errors, but found {error_count}: "
            f"{[issue.message for issue in report.issues if issue.severity == 'error']}"
        )
    
    @given(result=valid_training_result_strategy())
    @settings(max_examples=100)
    def test_accuracy_range_validation(self, result):
        """
        Property: Accuracy metrics must be in [0.0, 1.0].
        
        For any training result, all accuracy-based metrics (accuracy, precision,
        recall, F1, AUC) must be within the valid range [0.0, 1.0].
        """
        validator = ResultValidator()
        report = validator.validate_training_result(result)
        
        # Check that all accuracy metrics are in valid range
        assert 0.0 <= result.test_accuracy <= 1.0
        assert 0.0 <= result.test_auc <= 1.0
        assert 0.0 <= result.test_f1 <= 1.0
        assert 0.0 <= result.test_precision <= 1.0
        assert 0.0 <= result.test_recall <= 1.0
        
        # No range errors should be reported for valid metrics
        range_errors = [
            issue for issue in report.issues
            if issue.category == "metric_range" and issue.severity == "error"
        ]
        assert len(range_errors) == 0, (
            f"Valid metrics should not trigger range errors: {range_errors}"
        )
    
    @given(result=valid_training_result_strategy())
    @settings(max_examples=100)
    def test_nan_inf_detection(self, result):
        """
        Property: NaN and Inf values must be detected.
        
        For any training result, the validator must detect and report
        NaN or Inf values in any metric.
        """
        validator = ResultValidator()
        
        # Original result should be valid
        report = validator.validate_training_result(result)
        nan_inf_errors = [
            issue for issue in report.issues
            if "NaN" in issue.message or "Inf" in issue.message
        ]
        assert len(nan_inf_errors) == 0, "Valid result should have no NaN/Inf"
        
        # Create a result with NaN
        result_with_nan = TrainingResult(
            framework_name=result.framework_name,
            task_spec=result.task_spec,
            training_time_seconds=result.training_time_seconds,
            epochs_completed=result.epochs_completed,
            final_train_loss=result.final_train_loss,
            final_val_loss=result.final_val_loss,
            test_accuracy=float('nan'),  # Inject NaN
            test_auc=result.test_auc,
            test_f1=result.test_f1,
            test_precision=result.test_precision,
            test_recall=result.test_recall,
            accuracy_ci=result.accuracy_ci,
            auc_ci=result.auc_ci,
            f1_ci=result.f1_ci,
            peak_gpu_memory_mb=result.peak_gpu_memory_mb,
            avg_gpu_utilization=result.avg_gpu_utilization,
            peak_gpu_temperature=result.peak_gpu_temperature,
            samples_per_second=result.samples_per_second,
            inference_time_ms=result.inference_time_ms,
            model_parameters=result.model_parameters,
            checkpoint_path=result.checkpoint_path,
            metrics_path=result.metrics_path,
            log_path=result.log_path,
            status=result.status
        )
        
        # Validate result with NaN
        report_with_nan = validator.validate_training_result(result_with_nan)
        nan_errors = [
            issue for issue in report_with_nan.issues
            if "NaN" in issue.message and issue.severity == "error"
        ]
        assert len(nan_errors) > 0, "NaN value should be detected"
    
    @given(result=valid_training_result_strategy())
    @settings(max_examples=100)
    def test_throughput_limit_validation(self, result):
        """
        Property: Throughput must not exceed theoretical hardware limits.
        
        For any training result, throughput (samples/second) must be
        within plausible hardware limits.
        """
        validator = ResultValidator(max_throughput_samples_per_sec=10000.0)
        
        # Valid result should pass
        report = validator.validate_training_result(result)
        throughput_errors = [
            issue for issue in report.issues
            if "throughput" in issue.message.lower() and issue.severity == "error"
        ]
        assert len(throughput_errors) == 0, "Valid throughput should not trigger errors"
        
        # Create result with implausible throughput
        result_high_throughput = TrainingResult(
            framework_name=result.framework_name,
            task_spec=result.task_spec,
            training_time_seconds=result.training_time_seconds,
            epochs_completed=result.epochs_completed,
            final_train_loss=result.final_train_loss,
            final_val_loss=result.final_val_loss,
            test_accuracy=result.test_accuracy,
            test_auc=result.test_auc,
            test_f1=result.test_f1,
            test_precision=result.test_precision,
            test_recall=result.test_recall,
            accuracy_ci=result.accuracy_ci,
            auc_ci=result.auc_ci,
            f1_ci=result.f1_ci,
            peak_gpu_memory_mb=result.peak_gpu_memory_mb,
            avg_gpu_utilization=result.avg_gpu_utilization,
            peak_gpu_temperature=result.peak_gpu_temperature,
            samples_per_second=15000.0,  # Exceeds limit
            inference_time_ms=result.inference_time_ms,
            model_parameters=result.model_parameters,
            checkpoint_path=result.checkpoint_path,
            metrics_path=result.metrics_path,
            log_path=result.log_path,
            status=result.status
        )
        
        # Validate result with high throughput
        report_high = validator.validate_training_result(result_high_throughput)
        throughput_errors_high = [
            issue for issue in report_high.issues
            if "throughput" in issue.message.lower() and issue.severity == "error"
        ]
        assert len(throughput_errors_high) > 0, (
            "Implausible throughput should be detected"
        )
    
    @given(result=valid_training_result_strategy())
    @settings(max_examples=100)
    def test_anomaly_detection_below_random_chance(self, result):
        """
        Property: Accuracy below random chance must be detected as anomaly.
        
        For any training result with accuracy below 1/num_classes,
        the validator must flag it as an anomaly.
        """
        validator = ResultValidator()
        num_classes = result.task_spec.num_classes
        random_chance = 1.0 / num_classes
        
        # Create result with accuracy below random chance
        result_low_acc = TrainingResult(
            framework_name=result.framework_name,
            task_spec=result.task_spec,
            training_time_seconds=result.training_time_seconds,
            epochs_completed=result.epochs_completed,
            final_train_loss=result.final_train_loss,
            final_val_loss=result.final_val_loss,
            test_accuracy=random_chance * 0.5,  # Below random chance
            test_auc=result.test_auc,
            test_f1=result.test_f1,
            test_precision=result.test_precision,
            test_recall=result.test_recall,
            accuracy_ci=result.accuracy_ci,
            auc_ci=result.auc_ci,
            f1_ci=result.f1_ci,
            peak_gpu_memory_mb=result.peak_gpu_memory_mb,
            avg_gpu_utilization=result.avg_gpu_utilization,
            peak_gpu_temperature=result.peak_gpu_temperature,
            samples_per_second=result.samples_per_second,
            inference_time_ms=result.inference_time_ms,
            model_parameters=result.model_parameters,
            checkpoint_path=result.checkpoint_path,
            metrics_path=result.metrics_path,
            log_path=result.log_path,
            status=result.status
        )
        
        # Validate result
        report = validator.validate_training_result(result_low_acc)
        
        # Should have anomaly detected
        anomaly_issues = [
            issue for issue in report.issues
            if issue.category == "anomaly" and "random chance" in issue.message
        ]
        assert len(anomaly_issues) > 0, (
            f"Accuracy below random chance ({random_chance:.3f}) should be detected"
        )
        
        # Should have QA flag
        assert "accuracy_below_random_chance" in report.qa_flags
    
    @given(result=valid_training_result_strategy())
    @settings(max_examples=100)
    def test_qa_flags_for_suspicious_results(self, result):
        """
        Property: QA flags must be set for suspicious results.
        
        For any training result with suspicious characteristics (perfect accuracy,
        high loss, low GPU utilization), appropriate QA flags must be set.
        """
        validator = ResultValidator()
        
        # Test perfect accuracy
        result_perfect = TrainingResult(
            framework_name=result.framework_name,
            task_spec=result.task_spec,
            training_time_seconds=result.training_time_seconds,
            epochs_completed=result.epochs_completed,
            final_train_loss=result.final_train_loss,
            final_val_loss=result.final_val_loss,
            test_accuracy=1.0,  # Perfect accuracy
            test_auc=result.test_auc,
            test_f1=result.test_f1,
            test_precision=result.test_precision,
            test_recall=result.test_recall,
            accuracy_ci=result.accuracy_ci,
            auc_ci=result.auc_ci,
            f1_ci=result.f1_ci,
            peak_gpu_memory_mb=result.peak_gpu_memory_mb,
            avg_gpu_utilization=result.avg_gpu_utilization,
            peak_gpu_temperature=result.peak_gpu_temperature,
            samples_per_second=result.samples_per_second,
            inference_time_ms=result.inference_time_ms,
            model_parameters=result.model_parameters,
            checkpoint_path=result.checkpoint_path,
            metrics_path=result.metrics_path,
            log_path=result.log_path,
            status=result.status
        )
        
        report_perfect = validator.validate_training_result(result_perfect)
        assert "perfect_accuracy" in report_perfect.qa_flags, (
            "Perfect accuracy should trigger QA flag"
        )
