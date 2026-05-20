# Repository Organization

## Directory Structure

### Core Code
```
models/                    # Model implementations
├── transnnmil_v2.py      # TransnnMIL v2.0 (3-branch architecture)
├── adaptive_pruning.py   # Adaptive pruning module
├── hierarchical_pooling.py # Hierarchical pooling branch
├── topology_branch.py    # Topology/graph branch
└── ...                   # Other models

src/                      # Source code modules
├── data/                 # Data loading, preprocessing
├── training/             # Training loops, optimizers
├── evaluation/           # Metrics, evaluation
└── utils/                # Utilities

scripts/                  # Executable scripts
├── train_v2_0.py        # TransnnMIL v2.0 training
├── prepare_panda_features.py # PANDA data preparation
├── verify_panda_features.py  # Feature verification
└── ...                   # Other scripts

utils/                    # Utility modules
├── data_utils.py        # Data utilities
├── model_utils.py       # Model utilities
└── ...                   # Other utilities
```

### Tests
```
tests/                    # Test suite
├── models/              # Model tests
│   ├── test_transnnmil_v2.py
│   ├── test_adaptive_pruning.py
│   └── ...
├── data/                # Data pipeline tests
└── integration/         # Integration tests
```

### Documentation
```
docs/                     # Documentation
├── TRANSNNMIL_V2_ARCHITECTURE.md  # Architecture guide
├── TRANSNNMIL_V2_TRAINING.md      # Training guide
├── TRANSNNMIL_V2_API.md           # API reference
├── MODEL_CARD_V2.md               # Model card
└── ...                            # Other docs

TRAINING_SETUP.md         # Training setup guide
TRANSFER_CHECKLIST.md     # Transfer guide for other PC
README.md                 # Main README
```

### Data
```
panda/                    # PANDA dataset
├── features_resnet50_300patches/  # Feature files (gitignored)
├── splits.json          # Train/val/test splits
└── train.csv            # Labels

data/                     # Other datasets (gitignored)
```

### Configuration
```
config/                   # Configuration files
configs/                  # Model configs
requirements*.txt         # Dependencies
pyproject.toml           # Project metadata
```

### CI/CD
```
.github/
└── workflows/           # GitHub Actions
    ├── ci.yml          # Continuous integration
    ├── cd.yml          # Continuous deployment
    └── ...             # Other workflows
```

## Gitignored Directories

These exist locally but are not tracked:

### Development
- `.venv*/`, `venv*/`, `envs/` - Virtual environments
- `.vscode/`, `.idea/`, `.cursor/` - IDE configs
- `.kilo/`, `.kiro/`, `.qodo/`, `.zap/` - AI assistant dirs

### Experiments & Outputs
- `experiments/` - Experiment runs
- `results/` - Training results
- `outputs/` - Model outputs
- `checkpoints/` - Model checkpoints
- `demo_*/` - Demo outputs
- `coverage_reports/` - Test coverage
- `dataset_test_results/` - Dataset tests

### Data & Cache
- `panda/features_resnet50_300patches/` - Feature files (large)
- `data/` - Datasets
- `pacs_cache/` - PACS cache
- `.cache/`, `__pycache__/` - Python cache

### Deployment
- `deploy/` - Deployment configs
- `kubernetes/`, `k8s/` - K8s configs
- `mobile/` - Mobile app
- `website/` - Website

### Private
- `business/`, `enterprise/`, `patents/` - Private content
- `ecosystem/` - Ecosystem projects

### Temporary
- `test_*.py`, `check_*.py`, `quick_*.py` (root only) - Temp scripts
- `*.log` - Log files
- `bandit*.json`, `security*.json` - Security scan results

## File Organization Guidelines

### What to Commit
✓ Source code (`.py`, `.js`, etc.)
✓ Tests
✓ Documentation (`.md`)
✓ Configuration (`.yaml`, `.json`, `.toml`)
✓ Small data files (<1MB)
✓ Requirements files
✓ CI/CD workflows

### What NOT to Commit
✗ Virtual environments
✗ IDE configs (personal)
✗ Large datasets (>10MB)
✗ Model checkpoints (`.pth`, `.pt`)
✗ Experiment outputs
✗ Temporary/debug scripts (root level)
✗ Security scan results
✗ Cache files
✗ Log files

## Cleaning Up

### Check what's ignored:
```bash
git status --ignored
```

### See untracked files:
```bash
git status --short
```

### Clean ignored files (dry run):
```bash
git clean -Xdn
```

### Clean ignored files (actual):
```bash
git clean -Xdf
```

## Current Status

✓ Gitignore updated to exclude:
  - Temp files and debug scripts
  - Experiment outputs
  - Tool directories
  - Virtual environments
  - Cache directories

✓ Repository organized:
  - Core code in `models/`, `src/`, `scripts/`
  - Tests in `tests/`
  - Docs in `docs/`
  - Data in `panda/`, `data/`

✓ No files deleted (all preserved locally)

## Next Steps

1. Review untracked files: `git status`
2. Add important files: `git add <file>`
3. Clean up if needed: `git clean -Xdf` (removes ignored files)
4. Keep working - gitignore handles the rest
