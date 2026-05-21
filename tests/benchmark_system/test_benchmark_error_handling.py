"""
Unit tests for error handling and result validation in the Competitor Benchmark System.

Tests error isolation, error classification, fatal error handling, invalid output detection,
and anomaly detection.

Requirements: 8.1, 8.3, 8.4, 8.5, 8.6, 10.2
"""

from pathlib import Path


from experiments.benchmark_system.error_handler import (
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    RecoveryAction,
)
from experiments.benchmark_system.models import TaskSpecification, TrainingResult
from experiments.benchmark_system.result_validator import ResultValidator


class TestErrorClassification:
    """
    Test error classification functionality.

    Requirement: 8.3 (Error classification - recoverable vs fatal)
    """

    def test_classify_installation_error(self):
        """Test classification of installation errors."""
        handler = ErrorHandler()

        # Test various installation error messages
        errors = [
            Exception("Module not found: pathml"),
            Exception("Failed to install dependency"),
            Exception("ImportError: cannot import name 'foo'"),
            Exception("No module named 'numpy'"),
        ]

        for error in errors:
            category = handler.classify_error(error)
            assert (
                category == ErrorCategory.INSTALLATION
            ), f"Should classify as INSTALLATION: {error}"

    def test_classify_configuration_error(self):
        """Test classification of configuration errors."""
        handler = ErrorHandler()

        # Test various configuration error types
        errors = [
            ValueError("Invalid parameter value"),
            TypeError("Expected int, got str"),
            KeyError("Missing required configuration key"),
            Exception("Invalid configuration: batch_size must be positive"),
        ]

        for error in errors:
            category = handler.classify_error(error)
            assert (
                category == ErrorCategory.CONFIGURATION
            ), f"Should classify as CONFIGURATION: {error}"

    def test_classify_timeout_error(self):
        """Test classification of timeout errors."""
        handler = ErrorHandler()

        # Test timeout errors
        errors = [
            TimeoutError("Operation timed out"),
            Exception("Timeout exceeded"),
            Exception("Training timeout after 3600s"),
        ]

        for error in errors:
            category = handler.classify_error(error)
            assert category == ErrorCategory.TIMEOUT, f"Should classify as TIMEOUT: {error}"

    def test_classify_resource_error(self):
        """Test classification of resource errors."""
        handler = ErrorHandler()

        # Test resource errors
        errors = [
            Exception("CUDA out of memory"),
            Exception("GPU memory allocation failed"),
            Exception("Out of memory error"),
            Exception("OOM: cannot allocate tensor"),
        ]

        for error in errors:
            category = handler.classify_error(error)
            assert category == ErrorCategory.RESOURCE, f"Should classify as RESOURCE: {error}"

    def test_classify_data_error(self):
        """Test classification of data errors."""
        handler = ErrorHandler()

        # Test data errors
        errors = [
            Exception("File not found: dataset.h5"),
            Exception("Corrupted data file"),
            Exception("IO error reading data"),
            Exception("Data loading failed"),
        ]

        for error in errors:
            category = handler.classify_error(error)
            assert category == ErrorCategory.DATA, f"Should classify as DATA: {error}"

    def test_classify_runtime_error(self):
        """Test classification of generic runtime errors."""
        handler = ErrorHandler()

        # Test generic runtime errors
        errors = [
            Exception("Unexpected error during training"),
            RuntimeError("Training failed"),
            Exception("Unknown error occurred"),
        ]

        for error in errors:
            category = handler.classify_error(error)
            assert category == ErrorCategory.RUNTIME, f"Should classify as RUNTIME: {error}"

    def test_is_recoverable(self):
        """Test recoverable vs fatal error determination."""
        handler = ErrorHandler()

        # Recoverable errors
        assert handler.is_recoverable(ErrorCategory.RESOURCE)
        assert handler.is_recoverable(ErrorCategory.TIMEOUT)
        assert handler.is_recoverable(ErrorCategory.RUNTIME)

        # Fatal errors
        assert not handler.is_recoverable(ErrorCategory.CONFIGURATION)
        assert not handler.is_recoverable(ErrorCategory.INSTALLATION)
        assert not handler.is_recoverable(ErrorCategory.DATA)


class TestErrorIsolation:
    """
    Test error isolation functionality.

    Requirement: 8.1 (Error isolation - one framework failure doesn't stop others)
    """

    def test_handle_error_returns_continue_action(self):
        """Test that runtime errors return LOG_AND_CONTINUE action."""
        handler = ErrorHandler()

        error = Exception("Runtime error in framework")
        context = ErrorContext(
            framework_name="PathML",
            error=error,
            error_category=ErrorCategory.RUNTIME,
            retry_count=3,  # Exceeded retries
            max_retries=3,
        )

        action = handler.handle_error(error, context)

        # Should continue with next framework
        assert (
            action == RecoveryAction.LOG_AND_CONTINUE
        ), "Runtime error after max retries should continue with next framework"

    def test_handle_error_logs_error(self):
        """Test that errors are logged for later reporting."""
        handler = ErrorHandler()

        error = Exception("Test error")
        context = ErrorContext(
            framework_name="CLAM", error=error, error_category=ErrorCategory.RUNTIME
        )

        handler.handle_error(error, context)

        # Error should be logged
        assert len(handler.error_log) == 1
        assert handler.error_log[0].framework_name == "CLAM"
        assert handler.error_log[0].error == error

    def test_error_summary_generation(self):
        """Test error summary report generation."""
        handler = ErrorHandler()

        # Log multiple errors
        errors = [
            (Exception("Install failed"), "PathML", ErrorCategory.INSTALLATION),
            (Exception("Runtime error"), "CLAM", ErrorCategory.RUNTIME),
            (Exception("Timeout"), "PyTorch", ErrorCategory.TIMEOUT),
            (Exception("Another runtime error"), "CLAM", ErrorCategory.RUNTIME),
        ]

        for error, framework, category in errors:
            context = ErrorContext(framework_name=framework, error=error, error_category=category)
            handler.handle_error(error, context)

        # Get summary
        summary = handler.get_error_summary()

        # Check summary contents
        assert summary["total_errors"] == 4
        assert summary["by_category"]["installation"] == 1
        assert summary["by_category"]["runtime"] == 2
        assert summary["by_category"]["timeout"] == 1
        assert summary["by_framework"]["PathML"] == 1
        assert summary["by_framework"]["CLAM"] == 2
        assert summary["by_framework"]["PyTorch"] == 1
        assert len(summary["errors"]) == 4


class TestFatalErrorHandling:
    """
    Test fatal error handling functionality.

    Requirement: 8.4 (Fatal error handling)
    """

    def test_configuration_error_halts_benchmark(self):
        """Test that configuration errors halt the entire benchmark."""
        handler = ErrorHandler()

        error = ValueError("Invalid configuration parameter")
        context = ErrorContext(
            framework_name="HistoCore", error=error, error_category=ErrorCategory.CONFIGURATION
        )

        action = handler.handle_error(error, context)

        # Should halt benchmark
        assert action == RecoveryAction.HALT_BENCHMARK, "Configuration error should halt benchmark"

    def test_installation_error_skips_framework(self):
        """Test that installation errors skip the framework after retries."""
        handler = ErrorHandler()

        error = Exception("Failed to install framework")
        context = ErrorContext(
            framework_name="PathML",
            error=error,
            error_category=ErrorCategory.INSTALLATION,
            retry_count=3,  # Exceeded retries
            max_retries=3,
        )

        action = handler.handle_error(error, context)

        # Should skip framework
        assert (
            action == RecoveryAction.SKIP_FRAMEWORK
        ), "Installation error after max retries should skip framework"

    def test_timeout_saves_partial_results(self):
        """Test that timeout errors save partial results."""
        handler = ErrorHandler()

        error = TimeoutError("Training timeout")
        context = ErrorContext(
            framework_name="CLAM", error=error, error_category=ErrorCategory.TIMEOUT
        )

        action = handler.handle_error(error, context)

        # Should save partial results and continue
        assert (
            action == RecoveryAction.SAVE_PARTIAL_AND_CONTINUE
        ), "Timeout error should save partial results"


class TestInvalidOutputDetection:
    """
    Test invalid output detection functionality.

    Requirements: 8.5 (Result validation), 8.6 (Anomaly detection), 10.2 (Invalid output detection)
    """

    def create_valid_result(self) -> TrainingResult:
        """Helper to create a valid training result."""
        task_spec = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18",
            num_epochs=10,
            batch_size=32,
            learning_rate=1e-4,
            num_classes=2,
        )

        return TrainingResult(
            framework_name="HistoCore",
            task_spec=task_spec,
            training_time_seconds=100.0,
            epochs_completed=10,
            final_train_loss=0.3,
            final_val_loss=0.35,
            test_accuracy=0.85,
            test_auc=0.90,
            test_f1=0.83,
            test_precision=0.82,
            test_recall=0.84,
            accuracy_ci=(0.83, 0.87),
            auc_ci=(0.88, 0.92),
            f1_ci=(0.81, 0.85),
            peak_gpu_memory_mb=8000.0,
            avg_gpu_utilization=75.0,
            peak_gpu_temperature=70.0,
            samples_per_second=100.0,
            inference_time_ms=10.0,
            model_parameters=11_000_000,
            checkpoint_path=Path("/checkpoints/model.pth"),
            metrics_path=Path("/metrics/metrics.json"),
            log_path=Path("/logs/training.log"),
            status="success",
        )

    def test_detect_nan_metrics(self):
        """Test detection of NaN metrics."""
        validator = ResultValidator()
        result = self.create_valid_result()

        # Create result with NaN
        result.test_accuracy = float("nan")

        report = validator.validate_training_result(result)

        # Should detect NaN
        assert not report.valid
        nan_errors = [
            issue for issue in report.issues if "NaN" in issue.message and issue.severity == "error"
        ]
        assert len(nan_errors) > 0, "Should detect NaN metric"

    def test_detect_inf_metrics(self):
        """Test detection of Inf metrics."""
        validator = ResultValidator()
        result = self.create_valid_result()

        # Create result with Inf
        result.training_time_seconds = float("inf")

        report = validator.validate_training_result(result)

        # Should detect Inf
        assert not report.valid
        inf_errors = [
            issue for issue in report.issues if "Inf" in issue.message and issue.severity == "error"
        ]
        assert len(inf_errors) > 0, "Should detect Inf metric"

    def test_detect_out_of_range_accuracy(self):
        """Test detection of accuracy outside [0, 1]."""
        validator = ResultValidator()
        result = self.create_valid_result()

        # Test accuracy > 1
        result.test_accuracy = 1.5
        report = validator.validate_training_result(result)
        assert not report.valid

        # Test accuracy < 0
        result.test_accuracy = -0.1
        report = validator.validate_training_result(result)
        assert not report.valid

    def test_detect_negative_values(self):
        """Test detection of negative values where they shouldn't be."""
        validator = ResultValidator()
        result = self.create_valid_result()

        # Test negative training time
        result.training_time_seconds = -10.0
        report = validator.validate_training_result(result)
        assert not report.valid

        # Test negative throughput
        result.training_time_seconds = 100.0  # Reset
        result.samples_per_second = -50.0
        report = validator.validate_training_result(result)
        assert not report.valid


class TestAnomalyDetection:
    """
    Test anomaly detection functionality.

    Requirement: 8.6 (Anomaly detection)
    """

    def create_valid_result(self) -> TrainingResult:
        """Helper to create a valid training result."""
        task_spec = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18",
            num_epochs=10,
            batch_size=32,
            learning_rate=1e-4,
            num_classes=2,
        )

        return TrainingResult(
            framework_name="HistoCore",
            task_spec=task_spec,
            training_time_seconds=100.0,
            epochs_completed=10,
            final_train_loss=0.3,
            final_val_loss=0.35,
            test_accuracy=0.85,
            test_auc=0.90,
            test_f1=0.83,
            test_precision=0.82,
            test_recall=0.84,
            accuracy_ci=(0.83, 0.87),
            auc_ci=(0.88, 0.92),
            f1_ci=(0.81, 0.85),
            peak_gpu_memory_mb=8000.0,
            avg_gpu_utilization=75.0,
            peak_gpu_temperature=70.0,
            samples_per_second=100.0,
            inference_time_ms=10.0,
            model_parameters=11_000_000,
            checkpoint_path=Path("/checkpoints/model.pth"),
            metrics_path=Path("/metrics/metrics.json"),
            log_path=Path("/logs/training.log"),
            status="success",
        )

    def test_detect_accuracy_below_random_chance(self):
        """Test detection of accuracy below random chance."""
        validator = ResultValidator()
        result = self.create_valid_result()

        # Binary classification: random chance = 0.5
        result.test_accuracy = 0.3  # Below 0.5

        report = validator.validate_training_result(result)

        # Should detect anomaly
        anomaly_issues = [
            issue
            for issue in report.issues
            if issue.category == "anomaly" and "random chance" in issue.message
        ]
        assert len(anomaly_issues) > 0, "Should detect accuracy below random chance"
        assert "accuracy_below_random_chance" in report.qa_flags

    def test_detect_low_accuracy(self):
        """Test detection of suspiciously low accuracy."""
        validator = ResultValidator(min_accuracy_threshold=0.6)
        result = self.create_valid_result()

        result.test_accuracy = 0.55  # Below threshold

        report = validator.validate_training_result(result)

        # Should flag low accuracy
        assert "low_accuracy" in report.qa_flags

    def test_detect_auc_below_random(self):
        """Test detection of AUC below 0.5."""
        validator = ResultValidator()
        result = self.create_valid_result()

        result.test_auc = 0.45  # Below 0.5

        report = validator.validate_training_result(result)

        # Should flag AUC below random
        assert "auc_below_random" in report.qa_flags

    def test_detect_high_loss(self):
        """Test detection of suspiciously high loss values."""
        validator = ResultValidator()
        result = self.create_valid_result()

        result.final_val_loss = 15.0  # Very high

        report = validator.validate_training_result(result)

        # Should flag high loss
        assert "high_loss" in report.qa_flags

    def test_detect_perfect_accuracy(self):
        """Test detection of suspiciously perfect accuracy."""
        validator = ResultValidator()
        result = self.create_valid_result()

        result.test_accuracy = 1.0  # Perfect

        report = validator.validate_training_result(result)

        # Should flag perfect accuracy
        assert "perfect_accuracy" in report.qa_flags

    def test_detect_implausible_throughput(self):
        """Test detection of implausible throughput."""
        validator = ResultValidator(max_throughput_samples_per_sec=10000.0)
        result = self.create_valid_result()

        result.samples_per_second = 15000.0  # Exceeds limit

        report = validator.validate_training_result(result)

        # Should detect implausible throughput
        assert not report.valid
        assert "implausible_throughput" in report.qa_flags

    def test_detect_incomplete_training(self):
        """Test detection of incomplete training."""
        validator = ResultValidator()
        result = self.create_valid_result()

        result.epochs_completed = 5  # Expected 10

        report = validator.validate_training_result(result)

        # Should flag incomplete training
        assert "incomplete_training" in report.qa_flags

    def test_detect_low_gpu_utilization(self):
        """Test detection of low GPU utilization."""
        validator = ResultValidator()
        result = self.create_valid_result()

        result.avg_gpu_utilization = 5.0  # Very low

        report = validator.validate_training_result(result)

        # Should flag low GPU utilization
        assert "low_gpu_utilization" in report.qa_flags

    def test_detect_high_gpu_temperature(self):
        """Test detection of high GPU temperature."""
        validator = ResultValidator()
        result = self.create_valid_result()

        result.peak_gpu_temperature = 90.0  # Above 85°C

        report = validator.validate_training_result(result)

        # Should flag high temperature
        assert "high_gpu_temperature" in report.qa_flags


class TestValidResultsPassValidation:
    """Test that valid results pass validation without issues."""

    def test_valid_result_passes(self):
        """Test that a valid result passes all validation checks."""
        validator = ResultValidator()

        task_spec = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18",
            num_epochs=10,
            batch_size=32,
            learning_rate=1e-4,
            num_classes=2,
        )

        result = TrainingResult(
            framework_name="HistoCore",
            task_spec=task_spec,
            training_time_seconds=100.0,
            epochs_completed=10,
            final_train_loss=0.3,
            final_val_loss=0.35,
            test_accuracy=0.85,
            test_auc=0.90,
            test_f1=0.83,
            test_precision=0.82,
            test_recall=0.84,
            accuracy_ci=(0.83, 0.87),
            auc_ci=(0.88, 0.92),
            f1_ci=(0.81, 0.85),
            peak_gpu_memory_mb=8000.0,
            avg_gpu_utilization=75.0,
            peak_gpu_temperature=70.0,
            samples_per_second=100.0,
            inference_time_ms=10.0,
            model_parameters=11_000_000,
            checkpoint_path=Path("/checkpoints/model.pth"),
            metrics_path=Path("/metrics/metrics.json"),
            log_path=Path("/logs/training.log"),
            status="success",
        )

        report = validator.validate_training_result(result)

        # Should be valid
        assert report.valid, f"Valid result should pass validation: {report.issues}"

        # Should have no errors
        error_count = sum(1 for issue in report.issues if issue.severity == "error")
        assert error_count == 0, f"Should have no errors: {report.issues}"


class TestStatisticalSignificanceValidation:
    """
    Test statistical significance validation.

    Requirement: 10.5 (Statistical significance validation)
    """

    def create_result(self, accuracy: float, auc: float, f1: float) -> TrainingResult:
        """Helper to create a training result with specific metrics."""
        task_spec = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18",
            num_epochs=10,
            batch_size=32,
            learning_rate=1e-4,
            num_classes=2,
        )

        return TrainingResult(
            framework_name="HistoCore",
            task_spec=task_spec,
            training_time_seconds=100.0,
            epochs_completed=10,
            final_train_loss=0.3,
            final_val_loss=0.35,
            test_accuracy=accuracy,
            test_auc=auc,
            test_f1=f1,
            test_precision=0.82,
            test_recall=0.84,
            accuracy_ci=(accuracy - 0.02, accuracy + 0.02),
            auc_ci=(auc - 0.02, auc + 0.02),
            f1_ci=(f1 - 0.02, f1 + 0.02),
            peak_gpu_memory_mb=8000.0,
            avg_gpu_utilization=75.0,
            peak_gpu_temperature=70.0,
            samples_per_second=100.0,
            inference_time_ms=10.0,
            model_parameters=11_000_000,
            checkpoint_path=Path("/checkpoints/model.pth"),
            metrics_path=Path("/metrics/metrics.json"),
            log_path=Path("/logs/training.log"),
            status="success",
        )

    def test_significance_with_non_overlapping_ci(self):
        """Test significance detection with non-overlapping confidence intervals."""
        validator = ResultValidator()

        result1 = self.create_result(accuracy=0.90, auc=0.92, f1=0.88)
        result2 = self.create_result(accuracy=0.80, auc=0.82, f1=0.78)

        significance = validator.validate_statistical_significance(result1, result2)

        # CIs should not overlap
        assert not significance["accuracy"]["ci_overlap"]
        assert significance["accuracy"]["significant"]

    def test_significance_with_overlapping_ci(self):
        """Test significance detection with overlapping confidence intervals."""
        validator = ResultValidator()

        result1 = self.create_result(accuracy=0.85, auc=0.87, f1=0.83)
        result2 = self.create_result(accuracy=0.84, auc=0.86, f1=0.82)

        significance = validator.validate_statistical_significance(result1, result2)

        # CIs should overlap
        assert significance["accuracy"]["ci_overlap"]
        assert not significance["accuracy"]["significant"]
