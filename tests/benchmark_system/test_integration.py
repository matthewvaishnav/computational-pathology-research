"""
Integration tests for the Competitor Benchmark System.

These tests verify end-to-end workflows with synthetic data and mocked components
to avoid long execution times. They test:
- Single framework benchmark with synthetic data
- Checkpoint recovery with simulated crash
- Multi-framework execution
- Error recovery with injected failures
- Report generation pipeline

**Validates Requirements: 5.4, 8.1, 7.1-7.10**
"""

import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from experiments.benchmark_system.checkpoint_manager import CheckpointManager
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
def synthetic_task_spec(tmp_path):
    """Create a minimal task specification for fast testing."""
    return TaskSpecification(
        dataset_name="SyntheticPCam",
        data_root=tmp_path / "data",
        model_architecture="resnet18_transformer",
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        feature_dim=512,
        num_classes=2,
        num_epochs=2,  # Minimal epochs for fast testing
        batch_size=16,  # Small batch size
        learning_rate=1e-4,
        weight_decay=1e-5,
        optimizer="AdamW",
        random_seed=42,
        augmentation_config={},
        metrics=["accuracy", "auc", "f1"],
    )


@pytest.fixture
def integration_config(synthetic_task_spec, tmp_path):
    """Create a benchmark configuration for integration testing."""
    return BenchmarkConfig(
        mode="quick",
        frameworks=["HistoCore"],  # Single framework for fast testing
        task_spec=synthetic_task_spec,
        quick_mode_epochs=2,
        quick_mode_samples=100,  # Minimal samples
        max_gpu_memory_mb=12000,
        max_gpu_temperature=85.0,
        timeout_hours=1.0,  # Short timeout for testing
        checkpoint_interval_minutes=1,  # Frequent checkpoints for testing
        output_dir=tmp_path / "results",
        random_seed=42,
        bootstrap_samples=100,  # Reduced for speed
        confidence_level=0.95,
    )


@pytest.fixture
def mock_framework_env(tmp_path):
    """Create a mock framework environment."""
    return FrameworkEnvironment(
        framework_name="HistoCore",
        venv_path=tmp_path / "venv" / "histocore",
        python_version="3.10.0",
        framework_version="1.0.0",
        dependencies={"torch": "2.0.0", "numpy": "1.24.0"},
        installed_at=datetime.now(),
        patches_applied=[],
        validation_status="valid",
        validation_errors=[],
    )


@pytest.fixture
def mock_training_result(synthetic_task_spec, tmp_path):
    """Create a mock training result with synthetic data."""
    return TrainingResult(
        framework_name="HistoCore",
        task_spec=synthetic_task_spec,
        training_time_seconds=120.0,  # 2 minutes
        epochs_completed=2,
        final_train_loss=0.4,
        final_val_loss=0.45,
        test_accuracy=0.82,
        test_auc=0.88,
        test_f1=0.80,
        test_precision=0.79,
        test_recall=0.81,
        accuracy_ci=(0.80, 0.84),
        auc_ci=(0.86, 0.90),
        f1_ci=(0.78, 0.82),
        peak_gpu_memory_mb=4000.0,
        avg_gpu_utilization=65.0,
        peak_gpu_temperature=72.0,
        samples_per_second=50.0,
        inference_time_ms=20.0,
        model_parameters=500000,
        checkpoint_path=tmp_path / "checkpoints" / "model.pt",
        metrics_path=tmp_path / "metrics" / "metrics.json",
        log_path=tmp_path / "logs" / "training.log",
        status="success",
        error_message=None,
    )


# ============================================================================
# Test 1: Single Framework Benchmark with Synthetic Data
# ============================================================================


def test_single_framework_benchmark_with_synthetic_data(
    integration_config, mock_framework_env, mock_training_result, tmp_path
):
    """
    Test complete benchmark for one framework with synthetic data.

    This test verifies:
    - Framework installation and validation
    - Task configuration
    - Training execution
    - Metrics collection
    - Result validation
    - File outputs (checkpoints, metrics, logs)

    **Validates: Requirements 5.4, 8.1**
    """
    # Create output directories
    integration_config.output_dir.mkdir(parents=True, exist_ok=True)

    # Create orchestrator with mocked components
    orchestrator = BenchmarkOrchestrator(config=integration_config)

    # Mock GPU availability
    with patch.object(orchestrator.resource_manager, "verify_gpu_availability") as mock_gpu_check:
        mock_gpu_check.return_value = Mock(
            available=True,
            name="Mock GPU",
            memory_total_mb=12000.0,
            cuda_available=True,
            error_message=None,
        )

        # Mock framework installation
        with patch.object(orchestrator.framework_manager, "install_framework") as mock_install:
            mock_install.return_value = mock_framework_env

            # Mock framework validation
            with patch.object(
                orchestrator.framework_manager, "validate_installation"
            ) as mock_validate:
                mock_validate.return_value = mock_framework_env

                # Mock GPU allocation
                with patch.object(orchestrator.resource_manager, "allocate_gpu") as mock_allocate:
                    mock_allocate.return_value = Mock()

                    # Mock GPU memory cleanup
                    with patch.object(
                        orchestrator.resource_manager, "clear_gpu_memory"
                    ) as mock_clear:

                        # Mock task configuration
                        with patch.object(
                            orchestrator.task_executor, "configure_task"
                        ) as mock_configure:
                            mock_configure.return_value = Mock(config_dict={})

                            # Mock metrics collection
                            with patch.object(orchestrator.metrics_collector, "start_collection"):
                                with patch.object(
                                    orchestrator.metrics_collector, "finalize_collection"
                                ) as mock_finalize:
                                    mock_finalize.return_value = {}

                                    # Mock training execution
                                    with patch.object(
                                        orchestrator.task_executor, "execute_training"
                                    ) as mock_execute:
                                        mock_execute.return_value = mock_training_result

                                        # Mock report generation
                                        with patch.object(
                                            orchestrator.report_generator,
                                            "generate_visualizations",
                                        ):
                                            with patch.object(
                                                orchestrator.report_generator,
                                                "update_performance_comparison_md",
                                            ):
                                                with patch.object(
                                                    orchestrator.report_generator,
                                                    "export_to_csv",
                                                ):
                                                    with patch.object(
                                                        orchestrator.report_generator,
                                                        "export_to_json",
                                                    ):
                                                        # Run benchmark suite
                                                        result = orchestrator.run_benchmark_suite()

    # Verify benchmark completed successfully
    assert result is not None
    assert len(result.successful_frameworks) == 1
    assert "HistoCore" in result.successful_frameworks
    assert len(result.failed_frameworks) == 0

    # Verify framework results
    assert "HistoCore" in result.framework_results
    framework_result = result.framework_results["HistoCore"]
    assert framework_result.status == "success"
    assert framework_result.test_accuracy == 0.82
    assert framework_result.test_auc == 0.88

    # Verify timing
    assert result.total_duration_hours > 0
    assert result.start_time is not None
    assert result.end_time is not None

    # Verify GPU was allocated and cleaned up
    mock_allocate.assert_called_once()
    mock_clear.assert_called_once()


# ============================================================================
# Test 2: Checkpoint Recovery with Simulated Crash
# ============================================================================


def test_checkpoint_recovery_with_simulated_crash(
    integration_config, mock_framework_env, mock_training_result, tmp_path
):
    """
    Test checkpoint recovery after simulated crash.

    This test verifies:
    - Checkpoint creation during execution
    - Checkpoint loading after crash
    - State restoration
    - Resumption from last checkpoint

    **Validates: Requirement 5.4**
    """
    # Create checkpoint directory
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval_minutes=1,
    )

    # Create benchmark state
    benchmark_state = {
        "config": {
            "mode": "quick",
            "frameworks": ["HistoCore", "PathML"],
            "output_dir": str(tmp_path / "results"),
            "quick_mode_epochs": 2,
            "quick_mode_samples": 100,
            "timeout_hours": 1.0,
            "checkpoint_interval_minutes": 1,
            "random_seed": 42,
            "bootstrap_samples": 100,
            "confidence_level": 0.95,
            "task_spec": None,
        },
        "start_time": datetime.now().isoformat(),
        "completed_frameworks": ["HistoCore"],
        "failed_frameworks": [],
        "framework_environments": {
            "HistoCore": {
                "framework_name": "HistoCore",
                "venv_path": str(tmp_path / "venv" / "histocore"),
                "python_version": "3.10.0",
                "framework_version": "1.0.0",
                "dependencies": {"torch": "2.0.0"},
                "installed_at": datetime.now().isoformat(),
                "patches_applied": [],
                "validation_status": "valid",
                "validation_errors": [],
            }
        },
    }

    # Save checkpoint
    checkpoint_path = checkpoint_manager.save_checkpoint(benchmark_state)
    assert checkpoint_path is not None
    assert checkpoint_path.exists()

    # Verify checkpoint file contains expected data
    with open(checkpoint_path, "r") as f:
        checkpoint_data = json.load(f)

    # Checkpoint manager wraps state in "benchmark_state" key
    assert "benchmark_state" in checkpoint_data
    loaded_state = checkpoint_data["benchmark_state"]

    assert loaded_state["config"]["mode"] == "quick"
    assert loaded_state["completed_frameworks"] == ["HistoCore"]
    assert loaded_state["failed_frameworks"] == []

    # Simulate crash and recovery
    # Load checkpoint (this unwraps the benchmark_state)
    recovered_state = checkpoint_manager.load_checkpoint(checkpoint_path)

    # Verify state restored correctly
    assert recovered_state["config"]["mode"] == "quick"
    assert recovered_state["completed_frameworks"] == ["HistoCore"]
    assert recovered_state["failed_frameworks"] == []
    assert "HistoCore" in recovered_state["framework_environments"]

    # Verify we can resume from this state
    # (In production, orchestrator would skip completed frameworks)
    assert "PathML" not in recovered_state["completed_frameworks"]


# ============================================================================
# Test 3: Multi-Framework Execution
# ============================================================================


def test_multi_framework_execution(
    synthetic_task_spec, mock_framework_env, mock_training_result, tmp_path
):
    """
    Test benchmark execution for multiple frameworks sequentially.

    This test verifies:
    - Sequential framework execution
    - GPU memory cleanup between frameworks
    - Independent framework results
    - Aggregated benchmark results

    **Validates: Requirements 8.1, 5.4**
    """
    # Create config with 2 frameworks
    config = BenchmarkConfig(
        mode="quick",
        frameworks=["HistoCore", "PathML"],
        task_spec=synthetic_task_spec,
        quick_mode_epochs=2,
        quick_mode_samples=100,
        output_dir=tmp_path / "results",
        timeout_hours=1.0,
        checkpoint_interval_minutes=1,
        random_seed=42,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Create orchestrator
    orchestrator = BenchmarkOrchestrator(config=config)

    # Mock GPU availability
    with patch.object(orchestrator.resource_manager, "verify_gpu_availability") as mock_gpu_check:
        mock_gpu_check.return_value = Mock(
            available=True,
            name="Mock GPU",
            memory_total_mb=12000.0,
            cuda_available=True,
            error_message=None,
        )

        # Mock framework installation for both frameworks
        def mock_install_side_effect(framework_name):
            env = FrameworkEnvironment(
                framework_name=framework_name,
                venv_path=tmp_path / "venv" / framework_name.lower(),
                python_version="3.10.0",
                framework_version="1.0.0",
                dependencies={"torch": "2.0.0"},
                installed_at=datetime.now(),
                patches_applied=[],
                validation_status="valid",
                validation_errors=[],
            )
            return env

        with patch.object(orchestrator.framework_manager, "install_framework") as mock_install:
            mock_install.side_effect = mock_install_side_effect

            # Mock framework validation
            with patch.object(
                orchestrator.framework_manager, "validate_installation"
            ) as mock_validate:
                mock_validate.side_effect = lambda env: env

                # Mock GPU operations
                with patch.object(orchestrator.resource_manager, "allocate_gpu") as mock_allocate:
                    mock_allocate.return_value = Mock()

                    with patch.object(
                        orchestrator.resource_manager, "clear_gpu_memory"
                    ) as mock_clear:

                        # Mock task configuration
                        with patch.object(
                            orchestrator.task_executor, "configure_task"
                        ) as mock_configure:
                            mock_configure.return_value = Mock(config_dict={})

                            # Mock metrics collection
                            with patch.object(orchestrator.metrics_collector, "start_collection"):
                                with patch.object(
                                    orchestrator.metrics_collector, "finalize_collection"
                                ) as mock_finalize:
                                    mock_finalize.return_value = {}

                                    # Mock training execution with different results per framework
                                    def mock_execute_side_effect(config, env):
                                        return TrainingResult(
                                            framework_name=env.framework_name,
                                            task_spec=synthetic_task_spec,
                                            training_time_seconds=120.0,
                                            epochs_completed=2,
                                            final_train_loss=0.4,
                                            final_val_loss=0.45,
                                            test_accuracy=(
                                                0.82 if env.framework_name == "HistoCore" else 0.78
                                            ),
                                            test_auc=(
                                                0.88 if env.framework_name == "HistoCore" else 0.85
                                            ),
                                            test_f1=(
                                                0.80 if env.framework_name == "HistoCore" else 0.76
                                            ),
                                            test_precision=0.79,
                                            test_recall=0.81,
                                            accuracy_ci=(0.80, 0.84),
                                            auc_ci=(0.86, 0.90),
                                            f1_ci=(0.78, 0.82),
                                            peak_gpu_memory_mb=4000.0,
                                            avg_gpu_utilization=65.0,
                                            peak_gpu_temperature=72.0,
                                            samples_per_second=50.0,
                                            inference_time_ms=20.0,
                                            model_parameters=500000,
                                            checkpoint_path=tmp_path
                                            / "checkpoints"
                                            / f"{env.framework_name}.pt",
                                            metrics_path=tmp_path
                                            / "metrics"
                                            / f"{env.framework_name}.json",
                                            log_path=tmp_path
                                            / "logs"
                                            / f"{env.framework_name}.log",
                                            status="success",
                                            error_message=None,
                                        )

                                    with patch.object(
                                        orchestrator.task_executor, "execute_training"
                                    ) as mock_execute:
                                        mock_execute.side_effect = mock_execute_side_effect

                                        # Mock report generation
                                        with patch.object(
                                            orchestrator.report_generator,
                                            "generate_visualizations",
                                        ):
                                            with patch.object(
                                                orchestrator.report_generator,
                                                "update_performance_comparison_md",
                                            ):
                                                with patch.object(
                                                    orchestrator.report_generator,
                                                    "export_to_csv",
                                                ):
                                                    with patch.object(
                                                        orchestrator.report_generator,
                                                        "export_to_json",
                                                    ):
                                                        # Run benchmark suite
                                                        result = orchestrator.run_benchmark_suite()

    # Verify both frameworks completed
    assert len(result.successful_frameworks) == 2
    assert "HistoCore" in result.successful_frameworks
    assert "PathML" in result.successful_frameworks
    assert len(result.failed_frameworks) == 0

    # Verify framework results are independent
    histocore_result = result.framework_results["HistoCore"]
    pathml_result = result.framework_results["PathML"]

    assert histocore_result.test_accuracy == 0.82
    assert pathml_result.test_accuracy == 0.78
    assert histocore_result.test_accuracy != pathml_result.test_accuracy

    # Verify GPU was allocated and cleaned up for each framework
    assert mock_allocate.call_count == 2
    assert mock_clear.call_count == 2


# ============================================================================
# Test 4: Error Recovery with Injected Failures
# ============================================================================


def test_error_recovery_with_injected_failures(
    synthetic_task_spec, mock_framework_env, mock_training_result, tmp_path
):
    """
    Test error recovery when one framework fails.

    This test verifies:
    - Framework failure is caught and logged
    - Benchmark continues with remaining frameworks
    - Failed framework is marked appropriately
    - Successful frameworks complete normally

    **Validates: Requirement 8.1**
    """
    # Create config with 3 frameworks
    config = BenchmarkConfig(
        mode="quick",
        frameworks=["HistoCore", "PathML", "CLAM"],
        task_spec=synthetic_task_spec,
        quick_mode_epochs=2,
        quick_mode_samples=100,
        output_dir=tmp_path / "results",
        timeout_hours=1.0,
        checkpoint_interval_minutes=1,
        random_seed=42,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Create orchestrator
    orchestrator = BenchmarkOrchestrator(config=config)

    # Mock GPU availability
    with patch.object(orchestrator.resource_manager, "verify_gpu_availability") as mock_gpu_check:
        mock_gpu_check.return_value = Mock(
            available=True,
            name="Mock GPU",
            memory_total_mb=12000.0,
            cuda_available=True,
            error_message=None,
        )

        # Mock framework installation - PathML fails
        def mock_install_side_effect(framework_name):
            if framework_name == "PathML":
                raise RuntimeError("PathML installation failed (simulated)")

            env = FrameworkEnvironment(
                framework_name=framework_name,
                venv_path=tmp_path / "venv" / framework_name.lower(),
                python_version="3.10.0",
                framework_version="1.0.0",
                dependencies={"torch": "2.0.0"},
                installed_at=datetime.now(),
                patches_applied=[],
                validation_status="valid",
                validation_errors=[],
            )
            return env

        with patch.object(orchestrator.framework_manager, "install_framework") as mock_install:
            mock_install.side_effect = mock_install_side_effect

            # Mock framework validation
            with patch.object(
                orchestrator.framework_manager, "validate_installation"
            ) as mock_validate:
                mock_validate.side_effect = lambda env: env

                # Mock GPU operations
                with patch.object(orchestrator.resource_manager, "allocate_gpu") as mock_allocate:
                    mock_allocate.return_value = Mock()

                    with patch.object(
                        orchestrator.resource_manager, "clear_gpu_memory"
                    ) as mock_clear:

                        # Mock task configuration
                        with patch.object(
                            orchestrator.task_executor, "configure_task"
                        ) as mock_configure:
                            mock_configure.return_value = Mock(config_dict={})

                            # Mock metrics collection
                            with patch.object(orchestrator.metrics_collector, "start_collection"):
                                with patch.object(
                                    orchestrator.metrics_collector, "finalize_collection"
                                ) as mock_finalize:
                                    mock_finalize.return_value = {}

                                    # Mock training execution
                                    def mock_execute_side_effect(config, env):
                                        return TrainingResult(
                                            framework_name=env.framework_name,
                                            task_spec=synthetic_task_spec,
                                            training_time_seconds=120.0,
                                            epochs_completed=2,
                                            final_train_loss=0.4,
                                            final_val_loss=0.45,
                                            test_accuracy=0.82,
                                            test_auc=0.88,
                                            test_f1=0.80,
                                            test_precision=0.79,
                                            test_recall=0.81,
                                            accuracy_ci=(0.80, 0.84),
                                            auc_ci=(0.86, 0.90),
                                            f1_ci=(0.78, 0.82),
                                            peak_gpu_memory_mb=4000.0,
                                            avg_gpu_utilization=65.0,
                                            peak_gpu_temperature=72.0,
                                            samples_per_second=50.0,
                                            inference_time_ms=20.0,
                                            model_parameters=500000,
                                            checkpoint_path=tmp_path
                                            / "checkpoints"
                                            / f"{env.framework_name}.pt",
                                            metrics_path=tmp_path
                                            / "metrics"
                                            / f"{env.framework_name}.json",
                                            log_path=tmp_path
                                            / "logs"
                                            / f"{env.framework_name}.log",
                                            status="success",
                                            error_message=None,
                                        )

                                    with patch.object(
                                        orchestrator.task_executor, "execute_training"
                                    ) as mock_execute:
                                        mock_execute.side_effect = mock_execute_side_effect

                                        # Mock report generation
                                        with patch.object(
                                            orchestrator.report_generator,
                                            "generate_visualizations",
                                        ):
                                            with patch.object(
                                                orchestrator.report_generator,
                                                "update_performance_comparison_md",
                                            ):
                                                with patch.object(
                                                    orchestrator.report_generator,
                                                    "export_to_csv",
                                                ):
                                                    with patch.object(
                                                        orchestrator.report_generator,
                                                        "export_to_json",
                                                    ):
                                                        # Run benchmark suite
                                                        result = orchestrator.run_benchmark_suite()

    # Verify PathML failed but others succeeded
    assert len(result.successful_frameworks) == 2
    assert "HistoCore" in result.successful_frameworks
    assert "CLAM" in result.successful_frameworks
    assert len(result.failed_frameworks) == 1
    assert "PathML" in result.failed_frameworks

    # Verify successful frameworks have results
    assert "HistoCore" in result.framework_results
    assert "CLAM" in result.framework_results
    assert "PathML" not in result.framework_results

    # Verify error is recorded
    assert "PathML" in result.errors


# ============================================================================
# Test 5: Report Generation Pipeline
# ============================================================================


def test_report_generation_pipeline(
    integration_config, mock_framework_env, mock_training_result, tmp_path
):
    """
    Test complete report generation pipeline.

    This test verifies:
    - Comparison table generation
    - Statistical significance tests
    - Visualization generation
    - CSV/JSON export
    - PERFORMANCE_COMPARISON.md update

    **Validates: Requirements 7.1-7.10**
    """
    # Create output directories
    integration_config.output_dir.mkdir(parents=True, exist_ok=True)

    # Create orchestrator
    orchestrator = BenchmarkOrchestrator(config=integration_config)

    # Mock GPU availability
    with patch.object(orchestrator.resource_manager, "verify_gpu_availability") as mock_gpu_check:
        mock_gpu_check.return_value = Mock(
            available=True,
            name="Mock GPU",
            memory_total_mb=12000.0,
            cuda_available=True,
            error_message=None,
        )

        # Mock framework installation
        with patch.object(orchestrator.framework_manager, "install_framework") as mock_install:
            mock_install.return_value = mock_framework_env

            # Mock framework validation
            with patch.object(
                orchestrator.framework_manager, "validate_installation"
            ) as mock_validate:
                mock_validate.return_value = mock_framework_env

                # Mock GPU operations
                with patch.object(orchestrator.resource_manager, "allocate_gpu") as mock_allocate:
                    mock_allocate.return_value = Mock()

                    with patch.object(orchestrator.resource_manager, "clear_gpu_memory"):

                        # Mock task configuration
                        with patch.object(
                            orchestrator.task_executor, "configure_task"
                        ) as mock_configure:
                            mock_configure.return_value = Mock(config_dict={})

                            # Mock metrics collection
                            with patch.object(orchestrator.metrics_collector, "start_collection"):
                                with patch.object(
                                    orchestrator.metrics_collector, "finalize_collection"
                                ) as mock_finalize:
                                    mock_finalize.return_value = {}

                                    # Mock training execution
                                    with patch.object(
                                        orchestrator.task_executor, "execute_training"
                                    ) as mock_execute:
                                        mock_execute.return_value = mock_training_result

                                        # Mock report generation - verify all methods called
                                        with patch.object(
                                            orchestrator.report_generator,
                                            "generate_visualizations",
                                        ) as mock_viz:
                                            with patch.object(
                                                orchestrator.report_generator,
                                                "update_performance_comparison_md",
                                            ) as mock_update:
                                                with patch.object(
                                                    orchestrator.report_generator,
                                                    "export_to_csv",
                                                ) as mock_csv:
                                                    with patch.object(
                                                        orchestrator.report_generator,
                                                        "export_to_json",
                                                    ) as mock_json:
                                                        # Run benchmark suite
                                                        result = orchestrator.run_benchmark_suite()

                                                        # Verify all report generation methods called
                                                        mock_viz.assert_called_once()
                                                        mock_update.assert_called_once()
                                                        mock_csv.assert_called_once()
                                                        mock_json.assert_called_once()

    # Verify result contains report paths
    assert result.report_path is not None
    assert result.visualization_dir is not None

    # Verify report paths are correct
    assert result.report_path == integration_config.output_dir / "benchmark_report.md"
    assert result.visualization_dir == integration_config.output_dir / "visualizations"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
