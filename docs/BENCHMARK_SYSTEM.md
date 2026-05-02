---
layout: default
title: Competitor Benchmark System
---

# Competitor Benchmark System

Automated framework for comparing HistoCore against PathML, CLAM, and baseline PyTorch under identical, reproducible conditions.

---

## Overview

The benchmark system runs the same training task across all frameworks in isolated virtual environments, ensuring fair comparisons with identical data splits, random seeds, and hyperparameters.

**Frameworks compared:**
- **HistoCore** — this framework
- **PathML** — pathml library
- **CLAM** — Clustering-constrained Attention MIL
- **PyTorch** — baseline vanilla PyTorch

---

## Architecture

```
experiments/benchmark_system/
├── models.py                # TaskSpecification, FrameworkEnvironment, TrainingResult
├── framework_manager.py     # Installs & validates each framework in isolated venvs
├── task_executor.py         # Runs identical tasks, verifies config equivalence
├── orchestrator.py          # Coordinates full benchmark runs across frameworks
├── report_generator.py      # Generates JSON/HTML/Markdown reports with visualizations
└── adapters/
    ├── histocore_adapter.py # HistoCore framework integration
    ├── pathml_adapter.py    # PathML framework integration (in progress)
    ├── clam_adapter.py      # CLAM framework integration (in progress)
    └── pytorch_adapter.py   # Baseline PyTorch integration (in progress)
```

### `FrameworkManager`

Manages isolated virtual environments per framework — no dependency conflicts between PathML, CLAM, and HistoCore.

```python
from experiments.benchmark_system.framework_manager import FrameworkManager

manager = FrameworkManager(base_env_dir=Path("envs/benchmark_frameworks"))
env = manager.create_environment("PathML")
manager.install_framework(env)
manager.validate_installation(env)
```

### `TrainingTaskExecutor`

Translates a single `TaskSpecification` into framework-specific configs and verifies equivalence before running.

```python
from experiments.benchmark_system.task_executor import TrainingTaskExecutor
from experiments.benchmark_system.models import TaskSpecification
from pathlib import Path

task = TaskSpecification(
    dataset_name="PatchCamelyon",
    data_root=Path("data/pcam"),
    model_architecture="resnet18_transformer",
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-4,
    random_seed=42,
)

executor = TrainingTaskExecutor()
configs = executor.translate_task_for_all_frameworks(task)
report = executor.verify_equivalence(configs)
results = executor.run_all_frameworks(task, environments)
```

### `BenchmarkOrchestrator`

Coordinates full benchmark runs across multiple frameworks with checkpointing and error recovery.

```python
from experiments.benchmark_system.orchestrator import BenchmarkOrchestrator
from experiments.benchmark_system.models import TaskSpecification
from pathlib import Path

orchestrator = BenchmarkOrchestrator(
    base_env_dir=Path("envs/benchmark_frameworks"),
    results_dir=Path("results/benchmarks"),
    checkpoint_dir=Path("checkpoints/benchmarks")
)

task = TaskSpecification(
    dataset_name="PatchCamelyon",
    data_root=Path("data/pcam"),
    model_architecture="resnet18_transformer",
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-4,
    random_seed=42,
)

# Run benchmark across all frameworks
results = orchestrator.run_benchmark(
    task=task,
    frameworks=["HistoCore", "PathML", "CLAM", "PyTorch"]
)
```

### `ReportGenerator`

Generates comprehensive reports in JSON, HTML, and Markdown formats with visualizations.

```python
from experiments.benchmark_system.report_generator import ReportGenerator
from pathlib import Path

generator = ReportGenerator(output_dir=Path("results/reports"))

# Generate all report formats
report_paths = generator.generate_report(
    results=benchmark_results,
    task=task,
    formats=["json", "html", "markdown"]
)
```

---

## Task Specification

All frameworks receive identical configuration via `TaskSpecification`:

| Parameter | Default | Description |
|---|---|---|
| `dataset_name` | required | e.g. `"PatchCamelyon"` |
| `model_architecture` | required | e.g. `"resnet18_transformer"` |
| `num_epochs` | 10 | Training epochs |
| `batch_size` | 32 | Batch size |
| `learning_rate` | 1e-4 | Learning rate |
| `optimizer` | `"AdamW"` | Optimizer |
| `random_seed` | 42 | Reproducibility seed |
| `metrics` | `["accuracy", "auc", "f1"]` | Evaluation metrics |
| `train_split` | 0.8 | Train fraction |
| `val_split` | 0.1 | Validation fraction |
| `test_split` | 0.1 | Test fraction |

---

## Running Benchmarks

```bash
# Run full benchmark suite with orchestrator
python experiments/benchmark_system/example_orchestrator_usage.py

# Quick comparison (3 epochs)
python experiments/benchmark_system/example_orchestrator_usage.py --quick

# Single framework
python experiments/benchmark_system/example_orchestrator_usage.py --frameworks HistoCore

# Generate reports from existing results
python experiments/benchmark_system/example_orchestrator_usage.py --report-only
```

**Current Status:** 60% complete
- ✅ Core infrastructure (Tasks 1-9)
- ✅ Report Generator (Task 10)
- ✅ Benchmark Orchestrator (Task 11)
- ✅ HistoCore Adapter (Task 13.1)
- 🚧 PathML, CLAM, PyTorch adapters (Tasks 13.2-13.4)
- 🚧 CLI and main entry point (Task 15)

---

## Tests

```bash
# Run all benchmark system tests
pytest tests/benchmark_system/ -v

# Property-based tests
pytest tests/benchmark_system/test_configuration_properties.py \
       tests/benchmark_system/test_serialization_properties.py \
       --hypothesis-show-statistics
```

**Test coverage:**
- `test_framework_manager.py` — environment creation, installation, validation
- `test_task_executor.py` — task translation, equivalence verification, execution
- `test_orchestrator.py` — orchestration, checkpointing, error recovery
- `test_report_generator.py` — report generation, visualization, format validation
- `test_histocore_adapter.py` — HistoCore framework integration
- `test_configuration_properties.py` — property-based config correctness
- `test_serialization_properties.py` — property-based serialization round-trips

**Test Status:** 258 tests passing
