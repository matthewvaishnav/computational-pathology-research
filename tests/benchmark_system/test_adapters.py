"""
Unit tests for all framework adapters.

Tests the adapters' ability to execute training tasks and extract metrics
using each framework's native APIs. This file consolidates tests for all
four adapters (HistoCore, PathML, CLAM, PyTorch) for easier comparison.

Requirements: 2.1, 2.4, 4.1-4.8
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import torch

from experiments.benchmark_system.adapters.clam_adapter import CLAMAdapter
from experiments.benchmark_system.adapters.histocore_adapter import HistoCoreAdapter
from experiments.benchmark_system.adapters.pathml_adapter import PathMLAdapter
from experiments.benchmark_system.adapters.pytorch_adapter import PyTorchAdapter
from experiments.benchmark_system.models import (
    FrameworkEnvironment,
    TaskSpecification,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(params=["HistoCore", "PathML", "CLAM", "PyTorch"])
def framework_name(request):
    """Parametrize tests across all frameworks."""
    return request.param


@pytest.fixture
def framework_env(framework_name):
    """Create a mock framework environment for any framework."""
    return FrameworkEnvironment(
        framework_name=framework_name,
        venv_path=Path(f"/mock/venv/{framework_name.lower()}"),
        python_version="3.10.0",
        framework_version="1.0.0" if framework_name != "PyTorch" else "2.0.0",
        dependencies={"torch": "2.0.0", "numpy": "1.24.0"},
        installed_at=datetime.now(),
        patches_applied=[],
        validation_status="valid",
        validation_errors=[],
    )


@pytest.fixture
def adapter(framework_name, framework_env):
    """Create the appropriate adapter based on framework name."""
    adapters = {
        "HistoCore": HistoCoreAdapter,
        "PathML": PathMLAdapter,
        "CLAM": CLAMAdapter,
        "PyTorch": PyTorchAdapter,
    }
    return adapters[framework_name](framework_env)


@pytest.fixture
def task_spec():
    """Create a simple task specification."""
    return TaskSpecification(
        dataset_name="TestDataset",
        data_root=Path("/mock/data"),
        model_architecture="resnet18_transformer",
        feature_dim=128,
        num_classes=2,
        num_epochs=2,
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=1e-5,
        optimizer="AdamW",
        random_seed=42,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
    )


@pytest.fixture
def config_dict(framework_name):
    """Create framework-specific configuration dictionary."""
    configs = {
        "HistoCore": {},
        "PathML": {
            "model_config": {},
            "train_config": {"optimizer": "adam", "lr": 1e-4},
        },
        "CLAM": {
            "model_type": "resnet18",
            "model_size": "small",
            "opt": "adam",
            "lr": 1e-4,
        },
        "PyTorch": {},
    }
    return configs[framework_name]


# ============================================================================
# Test 1: Adapter Initialization
# ============================================================================


def test_adapter_initialization(adapter, framework_env, framework_name):
    """Test adapter can be initialized for all frameworks."""
    assert adapter.env == framework_env
    assert adapter.device is not None

    # Verify framework-specific initialization
    if framework_name == "PyTorch":
        # PyTorch adapter should log version
        assert torch.__version__ is not None


# ============================================================================
# Test 2: Training Execution with Mock Data
# ============================================================================


def test_adapter_training_execution(adapter, task_spec, config_dict, framework_name):
    """Test adapter can execute training and return results for all frameworks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Execute training
        result = adapter.execute_training(
            task_spec=task_spec,
            config_dict=config_dict,
            output_dir=output_dir,
        )

        # Verify result structure
        assert result.framework_name == framework_name
        assert result.task_spec == task_spec
        assert result.status == "success"
        assert result.error_message is None

        # Verify training metrics
        assert result.epochs_completed == task_spec.num_epochs
        assert result.training_time_seconds > 0
        assert result.final_train_loss >= 0
        assert result.final_val_loss >= 0

        # Verify performance metrics (Requirements 4.5, 4.6, 4.7)
        assert 0.0 <= result.test_accuracy <= 1.0
        assert 0.0 <= result.test_auc <= 1.0
        assert 0.0 <= result.test_f1 <= 1.0
        assert 0.0 <= result.test_precision <= 1.0
        assert 0.0 <= result.test_recall <= 1.0

        # Verify confidence intervals (Requirement 4.10)
        assert len(result.accuracy_ci) == 2
        assert result.accuracy_ci[0] <= result.test_accuracy <= result.accuracy_ci[1]
        assert len(result.auc_ci) == 2
        assert len(result.f1_ci) == 2

        # Verify resource metrics (Requirements 4.3, 4.4)
        assert result.peak_gpu_memory_mb >= 0
        assert result.avg_gpu_utilization >= 0
        assert result.peak_gpu_temperature >= 0

        # Verify throughput metrics (Requirements 4.4, 4.8)
        assert result.samples_per_second > 0
        assert result.inference_time_ms > 0

        # Verify model info
        assert result.model_parameters > 0

        # Verify paths exist
        assert result.checkpoint_path.exists()
        assert result.metrics_path.exists()


# ============================================================================
# Test 3: Metrics Extraction
# ============================================================================


def test_adapter_metrics_extraction(adapter, task_spec, config_dict, framework_name):
    """Test metrics extraction for each framework."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = adapter.execute_training(
            task_spec=task_spec,
            config_dict=config_dict,
            output_dir=output_dir,
        )

        # Verify all required metrics are present
        required_metrics = [
            "test_accuracy",
            "test_auc",
            "test_f1",
            "test_precision",
            "test_recall",
        ]

        for metric in required_metrics:
            assert hasattr(result, metric)
            value = getattr(result, metric)
            assert isinstance(value, (int, float))
            assert 0.0 <= value <= 1.0, f"{metric} out of range: {value}"


# ============================================================================
# Test 4: Configuration Translation
# ============================================================================


def test_adapter_configuration_handling(adapter, task_spec, config_dict, framework_name):
    """Test configuration translation for each framework."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Test with framework-specific configuration
        result = adapter.execute_training(
            task_spec=task_spec,
            config_dict=config_dict,
            output_dir=output_dir,
        )

        assert result.status == "success"

        # Verify configuration was applied correctly
        # (All adapters should handle their specific config formats)
        assert result.framework_name == framework_name


# ============================================================================
# Test 5: Random Seed Reproducibility
# ============================================================================


def test_adapter_random_seed_reproducibility(adapter, task_spec, config_dict, framework_name):
    """Test that same random seed produces similar results for all frameworks."""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:

        # Run training twice with same seed
        result1 = adapter.execute_training(
            task_spec=task_spec,
            config_dict=config_dict,
            output_dir=Path(tmpdir1),
        )

        result2 = adapter.execute_training(
            task_spec=task_spec,
            config_dict=config_dict,
            output_dir=Path(tmpdir2),
        )

        # Results should be similar (not exactly equal due to GPU non-determinism)
        # but should be close
        assert abs(result1.test_accuracy - result2.test_accuracy) < 0.1
        assert abs(result1.test_auc - result2.test_auc) < 0.1


# ============================================================================
# Test 6: Different Optimizer Support
# ============================================================================


@pytest.mark.parametrize("optimizer", ["Adam", "AdamW", "SGD"])
def test_adapter_different_optimizers(adapter, task_spec, config_dict, framework_name, optimizer):
    """Test adapter works with different optimizers for all frameworks."""
    # Create task spec with specific optimizer
    task_spec_copy = TaskSpecification(
        dataset_name=task_spec.dataset_name,
        data_root=task_spec.data_root,
        model_architecture=task_spec.model_architecture,
        feature_dim=task_spec.feature_dim,
        num_classes=task_spec.num_classes,
        num_epochs=1,  # Quick test
        batch_size=task_spec.batch_size,
        learning_rate=task_spec.learning_rate,
        weight_decay=task_spec.weight_decay,
        optimizer=optimizer,
        random_seed=task_spec.random_seed,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = adapter.execute_training(
            task_spec=task_spec_copy,
            config_dict=config_dict,
            output_dir=Path(tmpdir),
        )

        assert result.status == "success"
        assert result.epochs_completed == 1


# ============================================================================
# Test 7: Checkpoint Creation and Loading
# ============================================================================


def test_adapter_checkpoint_creation(adapter, task_spec, config_dict, framework_name):
    """Test that checkpoints are created during training for all frameworks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = adapter.execute_training(
            task_spec=task_spec,
            config_dict=config_dict,
            output_dir=output_dir,
        )

        # Check that checkpoint directory exists
        checkpoint_dir = output_dir / "checkpoints"
        assert checkpoint_dir.exists()

        # Check that final checkpoint exists
        assert result.checkpoint_path.exists()

        # Check that checkpoint can be loaded
        checkpoint = torch.load(result.checkpoint_path)
        assert "epoch" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["epoch"] == task_spec.num_epochs


# ============================================================================
# Test 8: Metrics JSON Creation
# ============================================================================


def test_adapter_metrics_json_creation(adapter, task_spec, config_dict, framework_name):
    """Test that metrics JSON file is created for all frameworks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = adapter.execute_training(
            task_spec=task_spec,
            config_dict=config_dict,
            output_dir=output_dir,
        )

        # Check that metrics file exists
        assert result.metrics_path.exists()

        # Check that metrics can be loaded
        with open(result.metrics_path) as f:
            metrics = json.load(f)

        assert "train_losses" in metrics
        assert "val_losses" in metrics
        assert "epoch_times" in metrics
        assert "test_metrics" in metrics

        # Verify metrics structure
        assert len(metrics["train_losses"]) == task_spec.num_epochs
        assert len(metrics["val_losses"]) == task_spec.num_epochs
        assert len(metrics["epoch_times"]) == task_spec.num_epochs


# ============================================================================
# Test 9: Resource Monitoring
# ============================================================================


def test_adapter_resource_monitoring(adapter, task_spec, config_dict, framework_name):
    """Test that resource usage is monitored for all frameworks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = adapter.execute_training(
            task_spec=task_spec,
            config_dict=config_dict,
            output_dir=output_dir,
        )

        # Verify resource metrics are recorded
        assert result.peak_gpu_memory_mb >= 0
        assert result.avg_gpu_utilization >= 0
        assert result.peak_gpu_temperature >= 0

        # If GPU is available, memory should be non-zero
        if torch.cuda.is_available():
            assert result.peak_gpu_memory_mb > 0


# ============================================================================
# Test 10: Throughput Measurement
# ============================================================================


def test_adapter_throughput_measurement(adapter, task_spec, config_dict, framework_name):
    """Test that throughput is measured correctly for all frameworks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = adapter.execute_training(
            task_spec=task_spec,
            config_dict=config_dict,
            output_dir=output_dir,
        )

        # Verify throughput metrics
        assert result.samples_per_second > 0
        assert result.inference_time_ms > 0

        # Sanity check: throughput should be reasonable
        # (not exceeding theoretical hardware limits)
        assert result.samples_per_second < 1000000  # 1M samples/sec is unrealistic
        assert result.inference_time_ms < 10000  # 10 seconds per sample is unrealistic


# ============================================================================
# Test 11: Confidence Interval Computation
# ============================================================================


def test_adapter_confidence_intervals(adapter, task_spec, config_dict, framework_name):
    """Test that confidence intervals are computed correctly for all frameworks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = adapter.execute_training(
            task_spec=task_spec,
            config_dict=config_dict,
            output_dir=output_dir,
        )

        # Verify confidence intervals exist and are valid
        assert len(result.accuracy_ci) == 2
        assert result.accuracy_ci[0] <= result.accuracy_ci[1]
        assert result.accuracy_ci[0] <= result.test_accuracy <= result.accuracy_ci[1]

        assert len(result.auc_ci) == 2
        assert result.auc_ci[0] <= result.auc_ci[1]

        assert len(result.f1_ci) == 2
        assert result.f1_ci[0] <= result.f1_ci[1]


# ============================================================================
# Test 12: Multi-Class Classification
# ============================================================================


def test_adapter_multiclass_classification(adapter, config_dict, framework_name):
    """Test adapters work with multi-class classification."""
    # Create task spec with 3 classes
    task_spec = TaskSpecification(
        dataset_name="TestDataset",
        data_root=Path("/mock/data"),
        model_architecture="resnet18_transformer",
        feature_dim=128,
        num_classes=3,  # Multi-class
        num_epochs=1,
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=1e-5,
        optimizer="AdamW",
        random_seed=42,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = adapter.execute_training(
            task_spec=task_spec,
            config_dict=config_dict,
            output_dir=output_dir,
        )

        assert result.status == "success"
        assert 0.0 <= result.test_accuracy <= 1.0
        assert 0.0 <= result.test_auc <= 1.0


# ============================================================================
# Test 13: Framework-Specific Features
# ============================================================================


def test_histocore_specific_features():
    """Test HistoCore-specific features."""
    env = FrameworkEnvironment(
        framework_name="HistoCore",
        venv_path=Path("/mock/venv/histocore"),
        python_version="3.10.0",
        framework_version="1.0.0",
        dependencies={"torch": "2.0.0"},
        installed_at=datetime.now(),
        patches_applied=[],
        validation_status="valid",
        validation_errors=[],
    )

    adapter = HistoCoreAdapter(env)
    assert adapter.env.framework_name == "HistoCore"
    # HistoCore uses standard PyTorch training loop
    assert adapter.device is not None


def test_pathml_specific_features():
    """Test PathML-specific features."""
    env = FrameworkEnvironment(
        framework_name="PathML",
        venv_path=Path("/mock/venv/pathml"),
        python_version="3.10.0",
        framework_version="1.0.0",
        dependencies={"torch": "2.0.0"},
        installed_at=datetime.now(),
        patches_applied=[],
        validation_status="valid",
        validation_errors=[],
    )

    adapter = PathMLAdapter(env)
    assert adapter.env.framework_name == "PathML"
    # PathML has specific configuration format
    assert adapter.device is not None


def test_clam_specific_features():
    """Test CLAM-specific features."""
    env = FrameworkEnvironment(
        framework_name="CLAM",
        venv_path=Path("/mock/venv/clam"),
        python_version="3.10.0",
        framework_version="1.0.0",
        dependencies={"torch": "2.0.0"},
        installed_at=datetime.now(),
        patches_applied=[],
        validation_status="valid",
        validation_errors=[],
    )

    adapter = CLAMAdapter(env)
    assert adapter.env.framework_name == "CLAM"
    # CLAM uses attention-based MIL architecture
    assert adapter.device is not None


def test_pytorch_baseline_simplicity():
    """Test that PyTorch adapter uses minimal configuration."""
    env = FrameworkEnvironment(
        framework_name="PyTorch",
        venv_path=Path("/mock/venv/pytorch"),
        python_version="3.10.0",
        framework_version="2.0.0",
        dependencies={"torch": "2.0.0"},
        installed_at=datetime.now(),
        patches_applied=[],
        validation_status="valid",
        validation_errors=[],
    )

    adapter = PyTorchAdapter(env)

    task_spec = TaskSpecification(
        dataset_name="TestDataset",
        data_root=Path("/mock/data"),
        model_architecture="resnet18_transformer",
        feature_dim=128,
        num_classes=2,
        num_epochs=1,
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=1e-5,
        optimizer="AdamW",
        random_seed=42,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Execute with empty config dict (baseline should work with minimal config)
        result = adapter.execute_training(
            task_spec=task_spec,
            config_dict={},
            output_dir=output_dir,
        )

        assert result.status == "success"
        assert result.framework_name == "PyTorch"


# ============================================================================
# Test 14: Error Handling
# ============================================================================


def test_adapter_invalid_optimizer(adapter, task_spec, config_dict, framework_name):
    """Test that adapters handle invalid optimizer gracefully."""
    task_spec_invalid = TaskSpecification(
        dataset_name=task_spec.dataset_name,
        data_root=task_spec.data_root,
        model_architecture=task_spec.model_architecture,
        feature_dim=task_spec.feature_dim,
        num_classes=task_spec.num_classes,
        num_epochs=1,
        batch_size=task_spec.batch_size,
        learning_rate=task_spec.learning_rate,
        weight_decay=task_spec.weight_decay,
        optimizer="InvalidOptimizer",  # Invalid
        random_seed=task_spec.random_seed,
    )

    # Create invalid config dict for frameworks that read optimizer from config
    invalid_config = config_dict.copy()
    if framework_name == "PathML":
        invalid_config["train_config"] = {"optimizer": "InvalidOptimizer"}
    elif framework_name == "CLAM":
        invalid_config["opt"] = "InvalidOptimizer"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Should raise ValueError for invalid optimizer
        with pytest.raises(ValueError):
            adapter.execute_training(
                task_spec=task_spec_invalid,
                config_dict=invalid_config,
                output_dir=output_dir,
            )


# ============================================================================
# Test 15: Comparison Across Frameworks
# ============================================================================


def test_all_adapters_produce_comparable_results():
    """Test that all adapters produce results with the same structure."""
    task_spec = TaskSpecification(
        dataset_name="TestDataset",
        data_root=Path("/mock/data"),
        model_architecture="resnet18_transformer",
        feature_dim=128,
        num_classes=2,
        num_epochs=1,
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=1e-5,
        optimizer="AdamW",
        random_seed=42,
    )

    results = {}

    for framework_name in ["HistoCore", "PathML", "CLAM", "PyTorch"]:
        env = FrameworkEnvironment(
            framework_name=framework_name,
            venv_path=Path(f"/mock/venv/{framework_name.lower()}"),
            python_version="3.10.0",
            framework_version="1.0.0" if framework_name != "PyTorch" else "2.0.0",
            dependencies={"torch": "2.0.0"},
            installed_at=datetime.now(),
            patches_applied=[],
            validation_status="valid",
            validation_errors=[],
        )

        adapters = {
            "HistoCore": HistoCoreAdapter,
            "PathML": PathMLAdapter,
            "CLAM": CLAMAdapter,
            "PyTorch": PyTorchAdapter,
        }

        adapter = adapters[framework_name](env)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            result = adapter.execute_training(
                task_spec=task_spec,
                config_dict={},
                output_dir=output_dir,
            )

            results[framework_name] = result

    # Verify all results have the same structure
    required_attributes = [
        "framework_name",
        "test_accuracy",
        "test_auc",
        "test_f1",
        "test_precision",
        "test_recall",
        "training_time_seconds",
        "peak_gpu_memory_mb",
        "samples_per_second",
        "inference_time_ms",
    ]

    for framework_name, result in results.items():
        for attr in required_attributes:
            assert hasattr(result, attr), f"{framework_name} missing {attr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
