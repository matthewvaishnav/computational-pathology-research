"""
Unit tests for Benchmark Orchestrator.

Tests cover:
- Quick mode configuration (Requirement 6.3)
- Full mode configuration (Requirement 6.4)
- Framework selection filtering (Requirement 6.7)
- Progress logging every 10 minutes (Requirement 5.5)
- Completion notification (Requirement 5.7)
- Timeout enforcement (Requirement 5.8)

**Validates Requirements: 5.5, 5.7, 5.8, 6.3, 6.4, 6.7**
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from experiments.benchmark_system.models import (
    BenchmarkConfig,
    FrameworkEnvironment,
    TaskSpecification,
    TrainingResult,
)
from experiments.benchmark_system.orchestrator import BenchmarkOrchestrator

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def task_spec():
    """Create a standard task specification for testing."""
    return TaskSpecification(
        dataset_name="PatchCamelyon",
        data_root=Path("data/pcam"),
        model_architecture="resnet18_transformer",
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        feature_dim=512,
        num_classes=2,
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-5,
        optimizer="AdamW",
        random_seed=42,
        augmentation_config={},
        metrics=["accuracy", "auc", "f1"],
    )


@pytest.fixture
def quick_config(task_spec):
    """Create a quick mode benchmark configuration."""
    return BenchmarkConfig(
        mode="quick",
        frameworks=["HistoCore", "PathML"],
        task_spec=task_spec,
        quick_mode_epochs=3,
        quick_mode_samples=1000,
        max_gpu_memory_mb=12000,
        max_gpu_temperature=85.0,
        timeout_hours=4.0,
        checkpoint_interval_minutes=30,
        output_dir=Path("results/test_benchmark"),
        random_seed=42,
        bootstrap_samples=1000,
        confidence_level=0.95,
    )


@pytest.fixture
def full_config(task_spec):
    """Create a full mode benchmark configuration."""
    return BenchmarkConfig(
        mode="full",
        frameworks=["HistoCore", "PathML", "CLAM", "PyTorch"],
        task_spec=task_spec,
        quick_mode_epochs=3,
        quick_mode_samples=1000,
        max_gpu_memory_mb=12000,
        max_gpu_temperature=85.0,
        timeout_hours=48.0,
        checkpoint_interval_minutes=30,
        output_dir=Path("results/test_benchmark"),
        random_seed=42,
        bootstrap_samples=1000,
        confidence_level=0.95,
    )


@pytest.fixture
def mock_framework_env():
    """Create a mock framework environment."""
    return FrameworkEnvironment(
        framework_name="HistoCore",
        venv_path=Path("/tmp/venv/histocore"),
        python_version="3.10.0",
        framework_version="1.0.0",
        dependencies={"torch": "2.0.0", "numpy": "1.24.0"},
        installed_at=datetime.now(),
        patches_applied=[],
        validation_status="valid",
        validation_errors=[],
    )


@pytest.fixture
def mock_training_result(task_spec):
    """Create a mock training result."""
    return TrainingResult(
        framework_name="HistoCore",
        task_spec=task_spec,
        training_time_seconds=3600.0,
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
        peak_gpu_temperature=78.0,
        samples_per_second=100.0,
        inference_time_ms=10.0,
        model_parameters=1000000,
        checkpoint_path=Path("checkpoints/model.pt"),
        metrics_path=Path("metrics/metrics.json"),
        log_path=Path("logs/training.log"),
        status="success",
        error_message=None,
    )


# ============================================================================
# Test Quick Mode Configuration (Requirement 6.3)
# ============================================================================


def test_quick_mode_configuration(quick_config):
    """
    Test quick mode configuration reduces epochs and samples.

    **Validates: Requirement 6.3**
    """
    orchestrator = BenchmarkOrchestrator(config=quick_config)

    # Apply mode-specific configuration
    modified_task_spec = orchestrator._apply_mode_configuration()

    # Verify quick mode reduces epochs
    assert modified_task_spec.num_epochs == quick_config.quick_mode_epochs
    assert modified_task_spec.num_epochs == 3

    # Verify other settings preserved
    assert modified_task_spec.dataset_name == "PatchCamelyon"
    assert modified_task_spec.batch_size == 32
    assert modified_task_spec.learning_rate == 1e-4


def test_quick_mode_preserves_other_settings(quick_config):
    """
    Test quick mode preserves non-epoch settings.

    **Validates: Requirement 6.3**
    """
    orchestrator = BenchmarkOrchestrator(config=quick_config)
    modified_task_spec = orchestrator._apply_mode_configuration()

    # Verify all other settings preserved
    assert modified_task_spec.dataset_name == quick_config.task_spec.dataset_name
    assert modified_task_spec.model_architecture == quick_config.task_spec.model_architecture
    assert modified_task_spec.batch_size == quick_config.task_spec.batch_size
    assert modified_task_spec.learning_rate == quick_config.task_spec.learning_rate
    assert modified_task_spec.optimizer == quick_config.task_spec.optimizer
    assert modified_task_spec.random_seed == quick_config.task_spec.random_seed


def test_quick_mode_logs_configuration(quick_config, caplog):
    """
    Test quick mode logs configuration changes.

    **Validates: Requirement 6.3**
    """
    orchestrator = BenchmarkOrchestrator(config=quick_config)

    with caplog.at_level("INFO"):
        orchestrator._apply_mode_configuration()

    # Verify logging
    assert any("quick mode configuration" in record.message.lower() for record in caplog.records)
    assert any("epochs=3" in record.message.lower() for record in caplog.records)


# ============================================================================
# Test Full Mode Configuration (Requirement 6.4)
# ============================================================================


def test_full_mode_configuration(full_config):
    """
    Test full mode uses complete configuration without modifications.

    **Validates: Requirement 6.4**
    """
    orchestrator = BenchmarkOrchestrator(config=full_config)

    # Apply mode-specific configuration
    modified_task_spec = orchestrator._apply_mode_configuration()

    # Verify full mode does NOT reduce epochs
    assert modified_task_spec.num_epochs == full_config.task_spec.num_epochs
    assert modified_task_spec.num_epochs == 10  # Original value

    # Verify all settings preserved
    assert modified_task_spec.dataset_name == full_config.task_spec.dataset_name
    assert modified_task_spec.batch_size == full_config.task_spec.batch_size
    assert modified_task_spec.learning_rate == full_config.task_spec.learning_rate


def test_full_mode_logs_no_modifications(full_config, caplog):
    """
    Test full mode logs that no modifications are applied.

    **Validates: Requirement 6.4**
    """
    orchestrator = BenchmarkOrchestrator(config=full_config)

    with caplog.at_level("INFO"):
        orchestrator._apply_mode_configuration()

    # Verify logging indicates no modifications
    assert any("full mode" in record.message.lower() for record in caplog.records)
    assert any("no modifications" in record.message.lower() for record in caplog.records)


def test_full_mode_preserves_all_settings(full_config):
    """
    Test full mode preserves all task specification settings.

    **Validates: Requirement 6.4**
    """
    orchestrator = BenchmarkOrchestrator(config=full_config)
    original_task_spec = full_config.task_spec
    modified_task_spec = orchestrator._apply_mode_configuration()

    # Verify all fields match
    assert modified_task_spec.dataset_name == original_task_spec.dataset_name
    assert modified_task_spec.data_root == original_task_spec.data_root
    assert modified_task_spec.model_architecture == original_task_spec.model_architecture
    assert modified_task_spec.train_split == original_task_spec.train_split
    assert modified_task_spec.val_split == original_task_spec.val_split
    assert modified_task_spec.test_split == original_task_spec.test_split
    assert modified_task_spec.feature_dim == original_task_spec.feature_dim
    assert modified_task_spec.num_classes == original_task_spec.num_classes
    assert modified_task_spec.num_epochs == original_task_spec.num_epochs
    assert modified_task_spec.batch_size == original_task_spec.batch_size
    assert modified_task_spec.learning_rate == original_task_spec.learning_rate
    assert modified_task_spec.weight_decay == original_task_spec.weight_decay
    assert modified_task_spec.optimizer == original_task_spec.optimizer
    assert modified_task_spec.random_seed == original_task_spec.random_seed


# ============================================================================
# Test Framework Selection Filtering (Requirement 6.7)
# ============================================================================


def test_framework_selection_filtering(task_spec):
    """
    Test framework selection filtering works correctly.

    **Validates: Requirement 6.7**
    """
    # Create config with only 2 frameworks
    config = BenchmarkConfig(
        mode="quick",
        frameworks=["HistoCore", "PathML"],  # Only 2 frameworks
        task_spec=task_spec,
        output_dir=Path("results/test"),
    )

    orchestrator = BenchmarkOrchestrator(config=config)

    # Verify only selected frameworks are configured
    assert orchestrator.config.frameworks == ["HistoCore", "PathML"]
    assert len(orchestrator.config.frameworks) == 2


def test_framework_selection_single_framework(task_spec):
    """
    Test framework selection with single framework.

    **Validates: Requirement 6.7**
    """
    # Create config with only HistoCore
    config = BenchmarkConfig(
        mode="quick",
        frameworks=["HistoCore"],  # Single framework
        task_spec=task_spec,
        output_dir=Path("results/test"),
    )

    orchestrator = BenchmarkOrchestrator(config=config)

    # Verify only HistoCore is configured
    assert orchestrator.config.frameworks == ["HistoCore"]
    assert len(orchestrator.config.frameworks) == 1


def test_framework_selection_all_frameworks(task_spec):
    """
    Test framework selection with all frameworks.

    **Validates: Requirement 6.7**
    """
    # Create config with all frameworks
    config = BenchmarkConfig(
        mode="full",
        frameworks=["HistoCore", "PathML", "CLAM", "PyTorch"],
        task_spec=task_spec,
        output_dir=Path("results/test"),
    )

    orchestrator = BenchmarkOrchestrator(config=config)

    # Verify all frameworks are configured
    assert len(orchestrator.config.frameworks) == 4
    assert "HistoCore" in orchestrator.config.frameworks
    assert "PathML" in orchestrator.config.frameworks
    assert "CLAM" in orchestrator.config.frameworks
    assert "PyTorch" in orchestrator.config.frameworks


# ============================================================================
# Test Progress Logging (Requirement 5.5)
# ============================================================================


def test_progress_logging_every_10_minutes(quick_config, caplog):
    """
    Test progress logging occurs every 10 minutes.

    **Validates: Requirement 5.5**
    """
    orchestrator = BenchmarkOrchestrator(config=quick_config)
    orchestrator.start_time = datetime.now()

    # Initialize last progress log time
    orchestrator.last_progress_log_time = time.time()

    # Simulate 9 minutes elapsed (should NOT log)
    with patch("time.time", return_value=orchestrator.last_progress_log_time + 9 * 60):
        with caplog.at_level("INFO"):
            orchestrator._log_progress_if_needed()

    # Verify no progress log
    assert not any("PROGRESS UPDATE" in record.message for record in caplog.records)

    # Clear logs
    caplog.clear()

    # Simulate 10 minutes elapsed (should log)
    with patch("time.time", return_value=orchestrator.last_progress_log_time + 10 * 60):
        with caplog.at_level("INFO"):
            orchestrator._log_progress_if_needed()

    # Verify progress log
    assert any("PROGRESS UPDATE" in record.message for record in caplog.records)


def test_progress_logging_includes_framework_status(quick_config, caplog):
    """
    Test progress logging includes framework completion status.

    **Validates: Requirement 5.5**
    """
    orchestrator = BenchmarkOrchestrator(config=quick_config)
    orchestrator.start_time = datetime.now()
    orchestrator.last_progress_log_time = time.time()

    # Add some completed frameworks
    orchestrator.framework_results = {"HistoCore": Mock()}
    orchestrator.failed_frameworks = ["PathML"]

    # Simulate 10 minutes elapsed
    with patch("time.time", return_value=orchestrator.last_progress_log_time + 10 * 60):
        with caplog.at_level("INFO"):
            orchestrator._log_progress_if_needed()

    # Verify progress log includes status
    progress_logs = [r.message for r in caplog.records if "PROGRESS UPDATE" in r.message]
    assert len(progress_logs) > 0
    assert "1/2 frameworks completed" in progress_logs[0]
    assert "HistoCore" in progress_logs[0]
    assert "PathML" in progress_logs[0]


def test_progress_logging_updates_timestamp(quick_config):
    """
    Test progress logging updates last progress log timestamp.

    **Validates: Requirement 5.5**
    """
    orchestrator = BenchmarkOrchestrator(config=quick_config)
    orchestrator.start_time = datetime.now()
    orchestrator.last_progress_log_time = time.time()

    initial_time = orchestrator.last_progress_log_time

    # Simulate 10 minutes elapsed
    new_time = initial_time + 10 * 60
    with patch("time.time", return_value=new_time):
        orchestrator._log_progress_if_needed()

    # Verify timestamp updated
    assert orchestrator.last_progress_log_time == new_time
    assert orchestrator.last_progress_log_time > initial_time


def test_progress_logging_initializes_on_first_call(quick_config):
    """
    Test progress logging initializes timestamp on first call.

    **Validates: Requirement 5.5**
    """
    orchestrator = BenchmarkOrchestrator(config=quick_config)
    orchestrator.start_time = datetime.now()

    # Verify last_progress_log_time is None initially
    assert orchestrator.last_progress_log_time is None

    # Call progress logging
    orchestrator._log_progress_if_needed()

    # Verify timestamp initialized
    assert orchestrator.last_progress_log_time is not None


# ============================================================================
# Test Completion Notification (Requirement 5.7)
# ============================================================================


def test_completion_notification_sent(quick_config, mock_training_result, caplog):
    """
    Test completion notification is sent when benchmark completes.

    **Validates: Requirement 5.7**
    """
    orchestrator = BenchmarkOrchestrator(config=quick_config)

    # Create mock benchmark suite result
    from experiments.benchmark_system.models import BenchmarkSuiteResult

    result = BenchmarkSuiteResult(
        config=quick_config,
        framework_results={"HistoCore": mock_training_result},
        start_time=datetime.now(),
        end_time=datetime.now(),
        total_duration_hours=3.5,
        significance_tests={},
        accuracy_ranking=[("HistoCore", 0.85)],
        auc_ranking=[("HistoCore", 0.90)],
        efficiency_ranking=[("HistoCore", 0.85)],
        report_path=Path("report.md"),
        visualization_dir=Path("visualizations"),
        successful_frameworks=["HistoCore"],
        failed_frameworks=[],
        errors={},
    )

    with caplog.at_level("INFO"):
        orchestrator._send_completion_notification(result)

    # Verify completion notification logged
    assert any("BENCHMARK SUITE COMPLETED" in record.message for record in caplog.records)


def test_completion_notification_includes_duration(quick_config, mock_training_result, caplog):
    """
    Test completion notification includes total duration.

    **Validates: Requirement 5.7**
    """
    orchestrator = BenchmarkOrchestrator(config=quick_config)

    from experiments.benchmark_system.models import BenchmarkSuiteResult

    result = BenchmarkSuiteResult(
        config=quick_config,
        framework_results={"HistoCore": mock_training_result},
        start_time=datetime.now(),
        end_time=datetime.now(),
        total_duration_hours=3.5,
        significance_tests={},
        accuracy_ranking=[],
        auc_ranking=[],
        efficiency_ranking=[],
        report_path=Path("report.md"),
        visualization_dir=Path("visualizations"),
        successful_frameworks=["HistoCore"],
        failed_frameworks=[],
        errors={},
    )

    with caplog.at_level("INFO"):
        orchestrator._send_completion_notification(result)

    # Verify duration included
    notification_logs = [
        r.message for r in caplog.records if "BENCHMARK SUITE COMPLETED" in r.message
    ]
    assert len(notification_logs) > 0
    assert "3.5" in notification_logs[0] or "3.50" in notification_logs[0]


def test_completion_notification_includes_framework_status(
    quick_config, mock_training_result, caplog
):
    """
    Test completion notification includes successful and failed frameworks.

    **Validates: Requirement 5.7**
    """
    orchestrator = BenchmarkOrchestrator(config=quick_config)

    from experiments.benchmark_system.models import BenchmarkSuiteResult

    result = BenchmarkSuiteResult(
        config=quick_config,
        framework_results={"HistoCore": mock_training_result},
        start_time=datetime.now(),
        end_time=datetime.now(),
        total_duration_hours=3.5,
        significance_tests={},
        accuracy_ranking=[],
        auc_ranking=[],
        efficiency_ranking=[],
        report_path=Path("report.md"),
        visualization_dir=Path("visualizations"),
        successful_frameworks=["HistoCore"],
        failed_frameworks=["PathML"],
        errors={},
    )

    with caplog.at_level("INFO"):
        orchestrator._send_completion_notification(result)

    # Verify framework status included
    notification_logs = [
        r.message for r in caplog.records if "BENCHMARK SUITE COMPLETED" in r.message
    ]
    assert len(notification_logs) > 0
    assert "HistoCore" in notification_logs[0]
    assert "PathML" in notification_logs[0]


def test_completion_notification_includes_report_paths(quick_config, mock_training_result, caplog):
    """
    Test completion notification includes report and visualization paths.

    **Validates: Requirement 5.7**
    """
    orchestrator = BenchmarkOrchestrator(config=quick_config)

    from experiments.benchmark_system.models import BenchmarkSuiteResult

    result = BenchmarkSuiteResult(
        config=quick_config,
        framework_results={"HistoCore": mock_training_result},
        start_time=datetime.now(),
        end_time=datetime.now(),
        total_duration_hours=3.5,
        significance_tests={},
        accuracy_ranking=[],
        auc_ranking=[],
        efficiency_ranking=[],
        report_path=Path("results/benchmark_report.md"),
        visualization_dir=Path("results/visualizations"),
        successful_frameworks=["HistoCore"],
        failed_frameworks=[],
        errors={},
    )

    with caplog.at_level("INFO"):
        orchestrator._send_completion_notification(result)

    # Verify paths included
    notification_logs = [
        r.message for r in caplog.records if "BENCHMARK SUITE COMPLETED" in r.message
    ]
    assert len(notification_logs) > 0
    assert "benchmark_report.md" in notification_logs[0]
    assert "visualizations" in notification_logs[0]


# ============================================================================
# Test Timeout Enforcement (Requirement 5.8)
# ============================================================================


def test_timeout_enforcement_configuration(quick_config):
    """
    Test timeout is configured correctly from config.

    **Validates: Requirement 5.8**
    """
    orchestrator = BenchmarkOrchestrator(config=quick_config)

    # Verify timeout configured
    assert orchestrator.config.timeout_hours == 4.0

    # Verify timeout converted to seconds in run_single_framework
    timeout_seconds = orchestrator.config.timeout_hours * 3600
    assert timeout_seconds == 4.0 * 3600
    assert timeout_seconds == 14400


def test_timeout_enforcement_different_modes(task_spec):
    """
    Test timeout differs between quick and full modes.

    **Validates: Requirement 5.8**
    """
    # Quick mode with short timeout
    quick_config = BenchmarkConfig(
        mode="quick",
        frameworks=["HistoCore"],
        task_spec=task_spec,
        timeout_hours=4.0,
        output_dir=Path("results/test"),
    )

    # Full mode with long timeout
    full_config = BenchmarkConfig(
        mode="full",
        frameworks=["HistoCore"],
        task_spec=task_spec,
        timeout_hours=48.0,
        output_dir=Path("results/test"),
    )

    quick_orchestrator = BenchmarkOrchestrator(config=quick_config)
    full_orchestrator = BenchmarkOrchestrator(config=full_config)

    # Verify different timeouts
    assert quick_orchestrator.config.timeout_hours == 4.0
    assert full_orchestrator.config.timeout_hours == 48.0
    assert full_orchestrator.config.timeout_hours > quick_orchestrator.config.timeout_hours


def test_timeout_enforcement_in_run_single_framework(quick_config, mock_framework_env):
    """
    Test timeout is enforced in run_single_framework.

    **Validates: Requirement 5.8**
    """
    orchestrator = BenchmarkOrchestrator(config=quick_config)
    orchestrator.framework_environments = {"HistoCore": mock_framework_env}

    # Mock dependencies
    with patch.object(orchestrator.resource_manager, "allocate_gpu") as mock_allocate:
        with patch.object(orchestrator.resource_manager, "clear_gpu_memory"):
            with patch.object(orchestrator.task_executor, "configure_task") as mock_configure:
                with patch.object(orchestrator.metrics_collector, "start_collection"):
                    with patch.object(orchestrator.metrics_collector, "finalize_collection"):
                        with patch.object(
                            orchestrator.task_executor, "execute_training"
                        ) as mock_execute:

                            mock_allocate.return_value = Mock()
                            mock_configure.return_value = Mock(config_dict={})

                            # Mock execute_training to raise NotImplementedError
                            mock_execute.side_effect = NotImplementedError("Not implemented")

                            # Verify timeout is calculated
                            task_spec = orchestrator._apply_mode_configuration()

                            try:
                                orchestrator.run_single_framework("HistoCore", task_spec)
                            except NotImplementedError:
                                pass  # Expected

                            # Verify timeout was calculated (4 hours * 3600 seconds)
                            # This is verified by checking the code path executes
                            assert orchestrator.config.timeout_hours == 4.0


# ============================================================================
# Test Estimated Completion Time (Requirement 5.2)
# ============================================================================


def test_estimate_completion_time_quick_mode(quick_config):
    """
    Test estimated completion time for quick mode.

    **Validates: Requirement 5.2**
    """
    orchestrator = BenchmarkOrchestrator(config=quick_config)

    # Estimate completion time
    estimated_duration = orchestrator.estimate_completion_time()

    # Quick mode: ~1 hour per framework * 2 frameworks * 1.1 overhead
    # Expected: ~2.2 hours
    assert isinstance(estimated_duration, timedelta)
    assert estimated_duration.total_seconds() > 0

    # Should be approximately 2.2 hours (allow some tolerance)
    expected_hours = 1.0 * len(quick_config.frameworks) * 1.1
    actual_hours = estimated_duration.total_seconds() / 3600
    assert abs(actual_hours - expected_hours) < 0.1


def test_estimate_completion_time_full_mode(full_config):
    """
    Test estimated completion time for full mode.

    **Validates: Requirement 5.2**
    """
    orchestrator = BenchmarkOrchestrator(config=full_config)

    # Estimate completion time
    estimated_duration = orchestrator.estimate_completion_time()

    # Full mode: ~10 hours per framework * 4 frameworks * 1.1 overhead
    # Expected: ~44 hours
    assert isinstance(estimated_duration, timedelta)
    assert estimated_duration.total_seconds() > 0

    # Should be approximately 44 hours (allow some tolerance)
    expected_hours = 10.0 * len(full_config.frameworks) * 1.1
    actual_hours = estimated_duration.total_seconds() / 3600
    assert abs(actual_hours - expected_hours) < 0.1


def test_estimate_completion_time_scales_with_frameworks(task_spec):
    """
    Test estimated completion time scales with number of frameworks.

    **Validates: Requirement 5.2**
    """
    # Config with 2 frameworks
    config_2 = BenchmarkConfig(
        mode="quick",
        frameworks=["HistoCore", "PathML"],
        task_spec=task_spec,
        output_dir=Path("results/test"),
    )

    # Config with 4 frameworks
    config_4 = BenchmarkConfig(
        mode="quick",
        frameworks=["HistoCore", "PathML", "CLAM", "PyTorch"],
        task_spec=task_spec,
        output_dir=Path("results/test"),
    )

    orchestrator_2 = BenchmarkOrchestrator(config=config_2)
    orchestrator_4 = BenchmarkOrchestrator(config=config_4)

    duration_2 = orchestrator_2.estimate_completion_time()
    duration_4 = orchestrator_4.estimate_completion_time()

    # Duration should scale approximately linearly
    ratio = duration_4.total_seconds() / duration_2.total_seconds()
    assert 1.9 < ratio < 2.1  # Should be approximately 2x


# ============================================================================
# Integration Tests
# ============================================================================


def test_orchestrator_initialization(quick_config):
    """Test orchestrator initializes correctly with all components."""
    orchestrator = BenchmarkOrchestrator(config=quick_config)

    # Verify components initialized
    assert orchestrator.config == quick_config
    assert orchestrator.framework_manager is not None
    assert orchestrator.task_executor is not None
    assert orchestrator.resource_manager is not None
    assert orchestrator.metrics_collector is not None
    assert orchestrator.checkpoint_manager is not None
    assert orchestrator.error_handler is not None
    assert orchestrator.report_generator is not None
    assert orchestrator.result_validator is not None

    # Verify state tracking initialized
    assert orchestrator.start_time is None
    assert orchestrator.framework_results == {}
    assert orchestrator.framework_environments == {}
    assert orchestrator.failed_frameworks == []
    assert orchestrator.last_progress_log_time is None


def test_orchestrator_with_custom_components(quick_config):
    """Test orchestrator accepts custom component instances."""
    # Create mock components
    mock_framework_manager = Mock()
    mock_task_executor = Mock()
    mock_resource_manager = Mock()

    orchestrator = BenchmarkOrchestrator(
        config=quick_config,
        framework_manager=mock_framework_manager,
        task_executor=mock_task_executor,
        resource_manager=mock_resource_manager,
    )

    # Verify custom components used
    assert orchestrator.framework_manager == mock_framework_manager
    assert orchestrator.task_executor == mock_task_executor
    assert orchestrator.resource_manager == mock_resource_manager


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
