# Competitor Benchmark System

Automated fair performance comparison between HistoCore and competing frameworks (PathML, CLAM, baseline PyTorch) on RTX 4070 GPU hardware.

## Features

- **Fair Comparison**: Identical training tasks, datasets, hyperparameters across all frameworks
- **Isolated Environments**: Framework-specific virtual environments prevent dependency conflicts
- **Crash Recovery**: Automatic checkpointing every 30 minutes
- **Resource Management**: GPU memory monitoring, temperature throttling, exclusive execution
- **Comprehensive Metrics**: Training curves, efficiency metrics, statistical significance tests
- **Automated Reporting**: Updates PERFORMANCE_COMPARISON.md with real benchmark data

## Installation

### Prerequisites

**Hardware**:
- NVIDIA RTX 4070 GPU (12GB VRAM)
- 32GB+ system RAM
- 100GB+ free disk space

**Software**:
- Python 3.9-3.14
- CUDA 11.8+ and cuDNN 8.6+
- PyTorch with CUDA support

### Setup

1. **Clone repository**:
```bash
git clone https://github.com/yourusername/computational-pathology-research.git
cd computational-pathology-research
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Validate setup**:
```bash
python experiments/benchmark_system/run_benchmark.py validate
```

This checks GPU availability, CUDA installation, disk space, and Python version.

## Quick Start

### Run Quick Mode (3-4 hours)

For rapid validation:

```bash
python experiments/benchmark_system/run_benchmark.py run --mode quick
```

Runs reduced benchmark:
- 3 epochs
- 1000 samples
- All frameworks

### Run Full Mode (20-40+ hours)

For comprehensive evaluation:

```bash
python experiments/benchmark_system/run_benchmark.py run --mode full
```

Runs complete benchmark:
- 10 epochs
- Full dataset
- All frameworks

### Resume After Interruption

```bash
python experiments/benchmark_system/run_benchmark.py resume \
    results/competitor_benchmarks/checkpoints/checkpoint_latest.json
```

## Usage

See [USAGE.md](USAGE.md) for detailed usage instructions, including:
- Advanced configuration options
- Framework selection
- Custom configurations
- Logging options
- Troubleshooting

## Output

After running benchmark:

```
results/competitor_benchmarks/
├── benchmark_report.md          # Comprehensive comparison
├── results.csv                  # Metrics in CSV
├── results.json                 # Metrics in JSON
├── errors.log                   # Error log
├── error_summary.txt            # Error summary
├── visualizations/              # Performance plots
│   ├── training_curves.png
│   ├── efficiency_scatter.png
│   └── ...
├── checkpoints/                 # Recovery checkpoints
└── *_metrics.json               # Per-framework metrics
```

## Architecture

### Core Components

- **FrameworkManager**: Installs frameworks in isolated venvs, applies compatibility patches
- **TaskExecutor**: Configures and executes training tasks with equivalence validation
- **ResourceManager**: Manages GPU allocation, memory cleanup, temperature monitoring
- **MetricsCollector**: Collects training metrics, system metrics, confidence intervals
- **CheckpointManager**: Saves/loads benchmark state for crash recovery
- **ErrorHandler**: Classifies errors, implements retry logic with exponential backoff
- **ResultValidator**: Validates training results, detects anomalies
- **ReportGenerator**: Generates comparison tables, statistical tests, visualizations
- **BenchmarkOrchestrator**: Coordinates complete benchmark workflow

### Framework Adapters

- **HistoCoreAdapter**: HistoCore training loop and metrics extraction
- **PathMLAdapter**: PathML training loop with Python 3.14 compatibility patches
- **CLAMAdapter**: CLAM training loop and configuration handling
- **PyTorchAdapter**: Baseline PyTorch training loop

## Configuration

### Quick Mode Config

```yaml
mode: quick
epochs: 3
samples: 1000
frameworks:
  - HistoCore
  - PathML
  - CLAM
  - PyTorch
checkpoint_interval: 1800  # 30 minutes
```

### Full Mode Config

```yaml
mode: full
epochs: 10
samples: null  # Full dataset
frameworks:
  - HistoCore
  - PathML
  - CLAM
  - PyTorch
checkpoint_interval: 1800  # 30 minutes
```

See `configs/` directory for example configurations.

## Testing

Run test suite:

```bash
# All tests
pytest tests/benchmark_system/

# Unit tests only
pytest tests/benchmark_system/ -m "not integration"

# Integration tests only
pytest tests/benchmark_system/ -m integration

# Property tests only
pytest tests/benchmark_system/ -k property
```

Test coverage:
- **Unit tests**: Component-level validation
- **Property tests**: Universal correctness properties (configuration equivalence, serialization round-trip, result validation, exponential backoff)
- **Integration tests**: End-to-end workflow validation

## Troubleshooting

### GPU Not Available

```bash
# Check NVIDIA drivers
nvidia-smi

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

- Reduce batch size: `--batch-size 16`
- Use quick mode: `--mode quick`
- Close other GPU applications

### Framework Installation Failures

Check error log:
```bash
cat results/competitor_benchmarks/errors.log
```

For PathML on Python 3.14, compatibility patches are applied automatically.

### Checkpoint Corruption

Try previous checkpoint:
```bash
python experiments/benchmark_system/run_benchmark.py resume \
    results/competitor_benchmarks/checkpoints/checkpoint_<timestamp>.json
```

## Requirements Traceability

System validates:
- **1.x**: Framework installation and compatibility
- **2.x**: Configuration equivalence across frameworks
- **3.x**: GPU resource management
- **4.x**: Comprehensive metrics collection
- **5.x**: Long-running workload support and checkpointing
- **6.x**: Execution modes and progress tracking
- **7.x**: Automated reporting and visualization
- **8.x**: Error handling and recovery
- **9.x**: Versioning and reproducibility
- **10.x**: Result validation and quality assurance

See [requirements.md](../../.kiro/specs/competitor-benchmark-system/requirements.md) for complete requirements.

## Documentation

- [USAGE.md](USAGE.md) - Detailed usage guide
- [Design Document](../../.kiro/specs/competitor-benchmark-system/design.md) - System architecture
- [Requirements Document](../../.kiro/specs/competitor-benchmark-system/requirements.md) - Functional requirements
- [Tasks Document](../../.kiro/specs/competitor-benchmark-system/tasks.md) - Implementation plan

## License

See [LICENSE](../../LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for contribution guidelines.
