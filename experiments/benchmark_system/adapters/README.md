# Framework Adapters for Competitor Benchmark System

This directory contains framework-specific adapters that translate the generic `TaskSpecification` into framework-specific training execution. Each adapter implements the training loop using the framework's native APIs and extracts standardized metrics for fair comparison.

## Overview

The adapter pattern allows the benchmark system to execute identical training tasks across different frameworks while respecting each framework's unique APIs and conventions. All adapters implement a common interface:

```python
class FrameworkAdapter:
    def __init__(self, env: FrameworkEnvironment):
        """Initialize adapter with framework environment."""
        
    def execute_training(
        self,
        task_spec: TaskSpecification,
        config_dict: Dict[str, Any],
        output_dir: Path,
    ) -> TrainingResult:
        """Execute training task and return standardized results."""
```

## Available Adapters

### HistoCoreAdapter

**Status**: ✅ Implemented

**Location**: `experiments/benchmark_system/adapters/histocore_adapter.py`

**Description**: Adapter for executing training tasks using HistoCore's native APIs, including the MultimodalTrainer class and associated components.

**Features**:
- Uses HistoCore's training infrastructure
- Supports multiple optimizers (Adam, AdamW, SGD)
- Collects comprehensive metrics (accuracy, AUC, F1, precision, recall)
- Tracks resource usage (GPU memory, temperature, throughput)
- Implements checkpoint saving and recovery
- Computes bootstrap confidence intervals

**Requirements Satisfied**: 2.1, 2.4, 4.1-4.8

**Usage**:
```python
from experiments.benchmark_system.adapters.histocore_adapter import HistoCoreAdapter

adapter = HistoCoreAdapter(framework_env)
result = adapter.execute_training(task_spec, config_dict, output_dir)
```

### PathMLAdapter

**Status**: ⏳ Not Yet Implemented

**Location**: `experiments/benchmark_system/adapters/pathml_adapter.py` (planned)

**Description**: Adapter for executing training tasks using PathML's APIs.

**Requirements**: 2.1, 2.4, 4.1-4.8

### CLAMAdapter

**Status**: ⏳ Not Yet Implemented

**Location**: `experiments/benchmark_system/adapters/clam_adapter.py` (planned)

**Description**: Adapter for executing training tasks using CLAM's APIs.

**Requirements**: 2.1, 2.4, 4.1-4.8

### PyTorchAdapter

**Status**: ⏳ Not Yet Implemented

**Location**: `experiments/benchmark_system/adapters/pytorch_adapter.py` (planned)

**Description**: Adapter for executing training tasks using baseline PyTorch.

**Requirements**: 1.5, 2.1, 2.4, 4.1-4.8

## Integration with Task Executor

The `TrainingTaskExecutor` automatically delegates to the appropriate adapter based on the framework name:

```python
executor = TrainingTaskExecutor()

# Configure task for HistoCore
config = executor.configure_task(task_spec, "HistoCore")

# Execute training (automatically uses HistoCoreAdapter)
result = executor.execute_training(config, framework_env, output_dir)
```

## Metrics Collection

All adapters collect the following standardized metrics:

### Training Metrics
- `training_time_seconds`: Total training time (Requirement 4.1)
- `epochs_completed`: Number of epochs completed
- `final_train_loss`: Final training loss
- `final_val_loss`: Final validation loss

### Performance Metrics
- `test_accuracy`: Test set accuracy (Requirement 4.5)
- `test_auc`: Test set AUC (Requirement 4.6)
- `test_f1`: Test set F1 score (Requirement 4.7)
- `test_precision`: Test set precision
- `test_recall`: Test set recall

### Confidence Intervals
- `accuracy_ci`: Bootstrap confidence interval for accuracy (Requirement 4.10)
- `auc_ci`: Bootstrap confidence interval for AUC
- `f1_ci`: Bootstrap confidence interval for F1

### Resource Usage
- `peak_gpu_memory_mb`: Peak GPU memory usage in MB (Requirement 4.3)
- `avg_gpu_utilization`: Average GPU utilization percentage
- `peak_gpu_temperature`: Peak GPU temperature in Celsius

### Throughput
- `samples_per_second`: Training throughput (Requirement 4.4)
- `inference_time_ms`: Average inference time per sample in milliseconds (Requirement 4.8)

### Model Information
- `model_parameters`: Total number of model parameters

## Testing

Each adapter has comprehensive unit tests:

- **HistoCoreAdapter**: `tests/benchmark_system/test_histocore_adapter.py`
  - Tests initialization
  - Tests training execution
  - Tests random seed reproducibility
  - Tests different optimizers
  - Tests checkpoint creation
  - Tests metrics JSON creation

Integration tests verify the adapter works with the task executor:

- **Integration**: `tests/benchmark_system/test_task_executor_integration.py`
  - Tests task executor can delegate to HistoCore adapter
  - Tests unsupported frameworks raise appropriate errors

## Adding a New Adapter

To add a new framework adapter:

1. **Create adapter file**: `experiments/benchmark_system/adapters/<framework>_adapter.py`

2. **Implement adapter class**:
```python
class FrameworkAdapter:
    def __init__(self, env: FrameworkEnvironment):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def execute_training(
        self,
        task_spec: TaskSpecification,
        config_dict: Dict[str, Any],
        output_dir: Path,
    ) -> TrainingResult:
        # 1. Set random seeds
        # 2. Create data loaders
        # 3. Create model and optimizer
        # 4. Run training loop
        # 5. Evaluate on test set
        # 6. Compute confidence intervals
        # 7. Return TrainingResult
        pass
```

3. **Update task executor**: Add framework case to `execute_training()` method in `task_executor.py`

4. **Write tests**: Create `tests/benchmark_system/test_<framework>_adapter.py`

5. **Update `__init__.py`**: Add adapter to `experiments/benchmark_system/adapters/__init__.py`

## Design Principles

### Fairness
- All adapters use identical random seeds (Requirement 2.2)
- All adapters use identical data splits (Requirement 2.3)
- All adapters collect the same metrics for comparison

### Reproducibility
- Random seeds are set for PyTorch, NumPy, and CUDA
- Deterministic behavior is enabled where possible
- All configuration is logged and saved

### Robustness
- Adapters handle errors gracefully
- GPU resource monitoring prevents out-of-memory errors
- Checkpoints enable recovery from failures

### Extensibility
- New adapters can be added without modifying existing code
- Adapter interface is simple and well-defined
- Framework-specific logic is isolated in adapters

## References

- **Requirements**: `.kiro/specs/competitor-benchmark-system/requirements.md`
- **Design**: `.kiro/specs/competitor-benchmark-system/design.md`
- **Tasks**: `.kiro/specs/competitor-benchmark-system/tasks.md`
