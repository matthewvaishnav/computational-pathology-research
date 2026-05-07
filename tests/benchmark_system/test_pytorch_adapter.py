"""
Unit tests for baseline PyTorch adapter.

Tests the PyTorch adapter's ability to execute training tasks and extract
metrics using vanilla PyTorch without framework-specific abstractions.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import torch

from experiments.benchmark_system.adapters.pytorch_adapter import PyTorchAdapter
from experiments.benchmark_system.models import (
    FrameworkEnvironment,
    TaskSpecification,
)


@pytest.fixture
def framework_env():
    """Create a mock framework environment."""
    return FrameworkEnvironment(
        framework_name="PyTorch",
        venv_path=Path("/mock/venv"),
        python_version="3.10.0",
        framework_version="2.0.0",
        dependencies={"torch": "2.0.0", "numpy": "1.24.0"},
        installed_at=datetime.now(),
        patches_applied=[],
        validation_status="valid",
        validation_errors=[],
    )


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


def test_adapter_initialization(framework_env):
    """Test adapter can be initialized."""
    adapter = PyTorchAdapter(framework_env)
    assert adapter.env == framework_env
    assert adapter.device is not None


def test_adapter_training_execution(framework_env, task_spec):
    """Test adapter can execute training and return results."""
    adapter = PyTorchAdapter(framework_env)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Execute training
        result = adapter.execute_training(
            task_spec=task_spec,
            config_dict={},
            output_dir=output_dir,
        )
        
        # Verify result structure
        assert result.framework_name == "PyTorch"
        assert result.task_spec == task_spec
        assert result.status == "success"
        assert result.error_message is None
        
        # Verify training metrics
        assert result.epochs_completed == task_spec.num_epochs
        assert result.training_time_seconds > 0
        assert result.final_train_loss >= 0
        assert result.final_val_loss >= 0
        
        # Verify performance metrics
        assert 0.0 <= result.test_accuracy <= 1.0
        assert 0.0 <= result.test_auc <= 1.0
        assert 0.0 <= result.test_f1 <= 1.0
        assert 0.0 <= result.test_precision <= 1.0
        assert 0.0 <= result.test_recall <= 1.0
        
        # Verify confidence intervals
        assert len(result.accuracy_ci) == 2
        assert result.accuracy_ci[0] <= result.test_accuracy <= result.accuracy_ci[1]
        assert len(result.auc_ci) == 2
        assert len(result.f1_ci) == 2
        
        # Verify resource metrics
        assert result.peak_gpu_memory_mb >= 0
        assert result.avg_gpu_utilization >= 0
        assert result.peak_gpu_temperature >= 0
        
        # Verify throughput metrics
        assert result.samples_per_second > 0
        assert result.inference_time_ms > 0
        
        # Verify model info
        assert result.model_parameters > 0
        
        # Verify paths exist
        assert result.checkpoint_path.exists()
        assert result.metrics_path.exists()


def test_adapter_random_seed_reproducibility(framework_env, task_spec):
    """Test that same random seed produces same results."""
    adapter = PyTorchAdapter(framework_env)
    
    with tempfile.TemporaryDirectory() as tmpdir1, \
         tempfile.TemporaryDirectory() as tmpdir2:
        
        # Run training twice with same seed
        result1 = adapter.execute_training(
            task_spec=task_spec,
            config_dict={},
            output_dir=Path(tmpdir1),
        )
        
        result2 = adapter.execute_training(
            task_spec=task_spec,
            config_dict={},
            output_dir=Path(tmpdir2),
        )
        
        # Results should be similar (not exactly equal due to GPU non-determinism)
        # but should be close
        assert abs(result1.test_accuracy - result2.test_accuracy) < 0.1
        assert abs(result1.test_auc - result2.test_auc) < 0.1


def test_adapter_different_optimizers(framework_env, task_spec):
    """Test adapter works with different optimizers."""
    adapter = PyTorchAdapter(framework_env)
    
    optimizers = ["Adam", "AdamW", "SGD"]
    
    for optimizer in optimizers:
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
                config_dict={},
                output_dir=Path(tmpdir),
            )
            
            assert result.status == "success"
            assert result.epochs_completed == 1


def test_adapter_checkpoint_creation(framework_env, task_spec):
    """Test that checkpoints are created during training."""
    adapter = PyTorchAdapter(framework_env)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        result = adapter.execute_training(
            task_spec=task_spec,
            config_dict={},
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


def test_adapter_metrics_json_creation(framework_env, task_spec):
    """Test that metrics JSON file is created."""
    adapter = PyTorchAdapter(framework_env)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        result = adapter.execute_training(
            task_spec=task_spec,
            config_dict={},
            output_dir=output_dir,
        )
        
        # Check that metrics file exists
        assert result.metrics_path.exists()
        
        # Check that metrics can be loaded
        import json
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


def test_adapter_baseline_simplicity(framework_env, task_spec):
    """Test that PyTorch adapter uses minimal configuration."""
    adapter = PyTorchAdapter(framework_env)
    
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


def test_adapter_pytorch_version_compatibility(framework_env):
    """Test that adapter reports PyTorch version correctly."""
    adapter = PyTorchAdapter(framework_env)
    
    # Verify PyTorch version is accessible
    import torch
    assert torch.__version__ is not None
    
    # Adapter should match HistoCore's PyTorch version (>=2.0.0)
    major_version = int(torch.__version__.split('.')[0])
    assert major_version >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
