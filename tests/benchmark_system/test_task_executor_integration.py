"""
Integration tests for Task Executor with HistoCore adapter.

Tests that the task executor can successfully delegate to the HistoCore adapter
and execute training tasks end-to-end.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from experiments.benchmark_system.models import (
    FrameworkEnvironment,
    TaskSpecification,
)
from experiments.benchmark_system.task_executor import TrainingTaskExecutor


@pytest.fixture
def framework_env():
    """Create a mock framework environment."""
    return FrameworkEnvironment(
        framework_name="HistoCore",
        venv_path=Path("/mock/venv"),
        python_version="3.10.0",
        framework_version="1.0.0",
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


def test_task_executor_histocore_integration(framework_env, task_spec):
    """Test that task executor can execute HistoCore training."""
    executor = TrainingTaskExecutor()
    
    # Configure task for HistoCore
    config = executor.configure_task(task_spec, "HistoCore")
    
    assert config.framework_name == "HistoCore"
    assert config.task_spec == task_spec
    assert config.random_seed == task_spec.random_seed
    
    # Execute training
    with tempfile.TemporaryDirectory() as tmpdir:
        result = executor.execute_training(
            config=config,
            env=framework_env,
            output_dir=Path(tmpdir),
        )
        
        # Verify result
        assert result.framework_name == "HistoCore"
        assert result.status == "success"
        assert result.epochs_completed == task_spec.num_epochs
        assert 0.0 <= result.test_accuracy <= 1.0
        assert result.checkpoint_path.exists()


def test_task_executor_pathml_integration(task_spec):
    """Test that task executor can execute PathML training."""
    executor = TrainingTaskExecutor()
    
    # Create PathML framework environment
    pathml_env = FrameworkEnvironment(
        framework_name="PathML",
        venv_path=Path("/mock/venv"),
        python_version="3.10.0",
        framework_version="2.0.0",
        dependencies={"torch": "2.0.0", "numpy": "1.24.0"},
        installed_at=datetime.now(),
        patches_applied=[],
        validation_status="valid",
        validation_errors=[],
    )
    
    # Configure task for PathML
    config = executor.configure_task(task_spec, "PathML")
    
    assert config.framework_name == "PathML"
    assert config.task_spec == task_spec
    assert config.random_seed == task_spec.random_seed
    
    # Execute training
    with tempfile.TemporaryDirectory() as tmpdir:
        result = executor.execute_training(
            config=config,
            env=pathml_env,
            output_dir=Path(tmpdir),
        )
        
        # Verify result
        assert result.framework_name == "PathML"
        assert result.status == "success"
        assert result.epochs_completed == task_spec.num_epochs
        assert 0.0 <= result.test_accuracy <= 1.0
        assert result.checkpoint_path.exists()


def test_task_executor_clam_integration(task_spec):
    """Test that task executor can execute CLAM training."""
    executor = TrainingTaskExecutor()
    
    # Create CLAM framework environment
    clam_env = FrameworkEnvironment(
        framework_name="CLAM",
        venv_path=Path("/mock/venv"),
        python_version="3.10.0",
        framework_version="1.0.0",
        dependencies={"torch": "2.0.0", "numpy": "1.24.0"},
        installed_at=datetime.now(),
        patches_applied=[],
        validation_status="valid",
        validation_errors=[],
    )
    
    # Configure task for CLAM
    config = executor.configure_task(task_spec, "CLAM")
    
    assert config.framework_name == "CLAM"
    assert config.task_spec == task_spec
    assert config.random_seed == task_spec.random_seed
    
    # Execute training
    with tempfile.TemporaryDirectory() as tmpdir:
        result = executor.execute_training(
            config=config,
            env=clam_env,
            output_dir=Path(tmpdir),
        )
        
        # Verify result
        assert result.framework_name == "CLAM"
        assert result.status == "success"
        assert result.epochs_completed == task_spec.num_epochs
        assert 0.0 <= result.test_accuracy <= 1.0
        assert result.checkpoint_path.exists()


def test_task_executor_unsupported_framework(framework_env, task_spec):
    """Test that task executor raises error for unsupported frameworks."""
    executor = TrainingTaskExecutor()
    
    # Configure task for PyTorch (not yet implemented)
    config = executor.configure_task(task_spec, "PyTorch")
    
    # Attempt to execute training should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="PyTorch adapter not yet implemented"):
        executor.execute_training(
            config=config,
            env=framework_env,
            output_dir=Path("/tmp/test"),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
