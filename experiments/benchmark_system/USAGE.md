# Competitor Benchmark System - Usage Guide

## Overview

The Competitor Benchmark System provides a main entry point (`run_benchmark.py`) that orchestrates fair performance comparisons between HistoCore and competing frameworks (PathML, CLAM, baseline PyTorch).

## Quick Start

### 1. Validate Setup

Before running benchmarks, validate your system setup:

```bash
python experiments/benchmark_system/run_benchmark.py validate
```

This checks:
- GPU availability (RTX 4070)
- CUDA and PyTorch installation
- Disk space
- Python version

### 2. List Available Frameworks

See which frameworks can be benchmarked:

```bash
python experiments/benchmark_system/run_benchmark.py list-frameworks
```

### 3. Run Quick Mode Benchmark (3-4 hours)

For rapid validation and testing:

```bash
python experiments/benchmark_system/run_benchmark.py run --mode quick
```

This runs a reduced benchmark with:
- 3 epochs (default)
- 1000 samples (default)
- All frameworks (HistoCore, PathML, CLAM, PyTorch)

### 4. Run Full Mode Benchmark (20-40+ hours)

For comprehensive evaluation:

```bash
python experiments/benchmark_system/run_benchmark.py run --mode full
```

This runs the complete benchmark with:
- 10 epochs (default)
- Full dataset
- All frameworks

## Advanced Usage

### Select Specific Frameworks

Benchmark only specific frameworks:

```bash
python experiments/benchmark_system/run_benchmark.py run \
    --mode quick \
    --frameworks HistoCore PathML
```

### Custom Configuration

Use a custom configuration file:

```bash
python experiments/benchmark_system/run_benchmark.py run \
    --config configs/custom_benchmark.yaml
```

### Resume from Checkpoint

Resume an interrupted benchmark:

```bash
python experiments/benchmark_system/run_benchmark.py resume \
    results/competitor_benchmarks/checkpoints/checkpoint_latest.json
```

### Custom Output Directory

Specify a custom output directory:

```bash
python experiments/benchmark_system/run_benchmark.py run \
    --mode quick \
    --output-dir results/my_benchmark
```

### Logging Options

Enable verbose logging:

```bash
python experiments/benchmark_system/run_benchmark.py run \
    --mode quick \
    --verbose
```

Save logs to file:

```bash
python experiments/benchmark_system/run_benchmark.py run \
    --mode quick \
    --log-file benchmark.log
```

Quiet mode (errors only):

```bash
python experiments/benchmark_system/run_benchmark.py run \
    --mode quick \
    --quiet
```

## Key Features

### 1. Component Integration

The entry point wires together all system components:
- **FrameworkManager**: Installs and validates frameworks
- **TaskExecutor**: Configures and executes training tasks
- **ResourceManager**: Manages GPU resources
- **MetricsCollector**: Collects performance metrics
- **CheckpointManager**: Handles crash recovery
- **ErrorHandler**: Manages errors gracefully
- **ReportGenerator**: Generates comparison reports
- **BenchmarkOrchestrator**: Coordinates the workflow

### 2. Error Summary Generation

After execution, the system automatically generates:
- **Error Log**: `results/competitor_benchmarks/errors.log`
  - Captures all warnings and errors
  - Includes stack traces and context
  
- **Error Summary**: `results/competitor_benchmarks/error_summary.txt`
  - Human-readable summary
  - Error and warning counts
  - Detailed log excerpts

### 3. Logging Configuration

The entry point configures comprehensive logging:
- Console output (INFO level by default)
- File logging (DEBUG level if `--log-file` specified)
- Error-specific logging (WARNING+ to errors.log)
- Structured format with timestamps and context

### 4. Graceful Error Handling

The system handles errors gracefully:
- **KeyboardInterrupt (Ctrl+C)**: Saves checkpoint and generates error summary
- **Unexpected Exceptions**: Logs critical error and generates summary
- **Framework Failures**: Continues with remaining frameworks
- **Timeout Enforcement**: Terminates hanging tasks

## Output Files

After running a benchmark, you'll find:

```
results/competitor_benchmarks/
├── benchmark_report.md          # Comprehensive comparison report
├── results.csv                  # Metrics in CSV format
├── results.json                 # Metrics in JSON format
├── errors.log                   # Detailed error log
├── error_summary.txt            # Human-readable error summary
├── visualizations/              # Performance plots
│   ├── training_curves.png
│   ├── efficiency_scatter.png
│   └── ...
├── checkpoints/                 # Recovery checkpoints
│   ├── checkpoint_latest.json
│   └── ...
└── HistoCore_metrics.json       # Per-framework metrics
    PathML_metrics.json
    CLAM_metrics.json
    PyTorch_metrics.json
```

## Requirements

**Hardware**:
- NVIDIA RTX 4070 GPU (12GB VRAM)
- 32GB+ system RAM
- 100GB+ free disk space

**Software**:
- Python 3.9-3.14
- CUDA 11.8+ and cuDNN 8.6+
- PyTorch with CUDA support

## Troubleshooting

### GPU Not Available

If validation fails with "GPU not available":
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure no other processes are using the GPU

### Out of Memory Errors

If benchmarks fail with OOM errors:
1. Reduce batch size: `--batch-size 16`
2. Use quick mode for testing: `--mode quick`
3. Close other GPU-intensive applications

### Framework Installation Failures

If framework installation fails:
1. Check error log: `results/competitor_benchmarks/errors.log`
2. Verify Python version compatibility
3. For PathML on Python 3.14, patches are applied automatically

### Checkpoint Corruption

If resume fails:
1. Try previous checkpoint: `checkpoints/checkpoint_<timestamp>.json`
2. Re-run from scratch if all checkpoints are corrupted

## Examples

### Example 1: Quick Validation

```bash
# Validate setup
python experiments/benchmark_system/run_benchmark.py validate

# Run quick benchmark with verbose logging
python experiments/benchmark_system/run_benchmark.py run \
    --mode quick \
    --verbose \
    --log-file quick_benchmark.log
```

### Example 2: Full Benchmark with Specific Frameworks

```bash
# Run full benchmark for HistoCore and PathML only
python experiments/benchmark_system/run_benchmark.py run \
    --mode full \
    --frameworks HistoCore PathML \
    --output-dir results/histocore_vs_pathml
```

### Example 3: Resume After Interruption

```bash
# Start full benchmark
python experiments/benchmark_system/run_benchmark.py run --mode full

# ... interrupted by Ctrl+C or crash ...

# Resume from checkpoint
python experiments/benchmark_system/run_benchmark.py resume \
    results/competitor_benchmarks/checkpoints/checkpoint_latest.json
```

## Requirements Validation

The entry point validates:
- **Requirement 5.1**: Long-running workload support (20-40+ hours)
- **Requirement 8.8**: Error summary generation

## See Also

- [Design Document](../../.kiro/specs/competitor-benchmark-system/design.md)
- [Requirements Document](../../.kiro/specs/competitor-benchmark-system/requirements.md)
- [CLI Documentation](cli.py)
- [Orchestrator Documentation](orchestrator.py)
