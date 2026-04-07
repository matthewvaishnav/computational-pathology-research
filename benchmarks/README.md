# Benchmark Manifest

Lightweight registry for tracking experiment results and metadata.

## Format

JSON Lines (`.jsonl`) - one JSON object per line, easy to append and parse.

## Structure

Each benchmark entry contains:

| Field | Description |
|-------|-------------|
| `experiment_name` | Unique identifier for the experiment |
| `dataset_name` | Dataset used (e.g., "PatchCamelyon") |
| `dataset_subset_size` | Number of samples used |
| `config_path` | Path to Hydra config |
| `train_command` | Command to reproduce training |
| `eval_command` | Command to reproduce evaluation |
| `final_metrics` | Key metrics (accuracy, AUC, etc.) |
| `artifact_paths` | Paths to checkpoints, results, logs |
| `caveats` | List of limitations and warnings |
| `notes` | Additional context |
| `date` | Run date (ISO 8601) |
| `status` | COMPLETE, FAILED, IN_PROGRESS |

## Current Entries

- **pcam_baseline**: PatchCamelyon framework validation (synthetic subset, 700 samples)

## Usage

### Reading the manifest

```python
from src.utils.benchmark_manifest import BenchmarkManifest

manifest = BenchmarkManifest()
entries = manifest.read_all()

for entry in entries:
    print(f"{entry.experiment_name}: {entry.final_metrics['test_accuracy']:.2%}")
```

### Adding a new entry

```python
from src.utils.benchmark_manifest import BenchmarkEntry, BenchmarkManifest
from datetime import datetime

entry = BenchmarkEntry(
    experiment_name="my_experiment",
    dataset_name="MyDataset",
    dataset_subset_size=1000,
    config_path="configs/my_config.yaml",
    train_command="python experiments/train.py --config configs/my_config.yaml",
    eval_command="python experiments/evaluate.py ...",
    final_metrics={"accuracy": 0.95},
    artifact_paths={"checkpoints": ["checkpoints/best.pth"]},
    caveats=["Synthetic data"],
    notes="Test run",
    date=datetime.now().isoformat(),
    status="COMPLETE"
)

manifest = BenchmarkManifest()
manifest.add_entry(entry)
```

### Generating Markdown summary

```python
manifest = BenchmarkManifest()
manifest.to_markdown("benchmarks/summary.md")
```

## Artifacts

Large artifacts (checkpoints, plots) are **gitignored** and not committed. The manifest records their paths so they can be reproduced by running the documented commands.

## See Also

- `PCAM_BENCHMARK_RESULTS.md` - Detailed PCam benchmark report
- `src/utils/benchmark_manifest.py` - Helper utilities
