# Computational Pathology Research Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Coverage](https://img.shields.io/badge/coverage-62%25-yellow.svg)](htmlcov/index.html)

> **Research Framework**: Tested implementations for computational pathology with working benchmarks on PatchCamelyon and CAMELYON16-style slide-level classification.

> **📚 Documentation**: See [docs/](docs/) for all documentation. Start with [docs/DOCS_INDEX.md](docs/DOCS_INDEX.md) for navigation.

## Overview

This repository provides a tested PyTorch framework for computational pathology research, with working implementations for:

- ✅ **PatchCamelyon (PCam) Training**: 94% accuracy on synthetic subset
- ✅ **CAMELYON16 Slide-Level Pipeline**: Functional slide-level classification with mean/max pooling
- ✅ **Slide Predictions CSV Export**: Publication-ready prediction exports
- ✅ **Comprehensive Testing**: 62% code coverage with unit tests
- ✅ **Model Profiling**: Performance analysis and ONNX export tools

**Status**: This is a research codebase with working benchmarks on synthetic data. Not validated for clinical use.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/matthewvaishnav/computational-pathology-research.git
cd computational-pathology-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### PatchCamelyon (PCam) Training

Train on the PatchCamelyon benchmark:

```bash
# Generate synthetic data
python scripts/generate_synthetic_pcam.py

# Train model
python experiments/train_pcam.py --config experiments/configs/pcam.yaml

# Evaluate
python experiments/evaluate_pcam.py \
  --checkpoint checkpoints/pcam/best_model.pth \
  --data-root data/pcam \
  --output-dir results/pcam
```

**Results** (synthetic subset):
- Test Accuracy: 94.0%
- Test AUC: 1.0
- Training Time: ~40 seconds (8 epochs, CPU)

See [docs/PCAM_BENCHMARK_RESULTS.md](docs/PCAM_BENCHMARK_RESULTS.md) for details.

### CAMELYON16 Slide-Level Training

Train on CAMELYON16-style slide-level classification:

```bash
# Generate synthetic data
python scripts/generate_synthetic_camelyon.py

# Quick test (1 epoch)
python experiments/train_camelyon.py \
  --config experiments/configs/camelyon_quick_test.yaml

# Full training (50 epochs)
python experiments/train_camelyon.py \
  --config experiments/configs/camelyon.yaml

# Evaluate with CSV export
python experiments/evaluate_camelyon.py \
  --checkpoint checkpoints/camelyon/best_model.pth \
  --data-root data/camelyon \
  --output-dir results/camelyon \
  --save-predictions-csv
```

**Features**:
- Slide-level classification using pre-extracted HDF5 features
- Mean/max pooling aggregation methods
- CSV export for slide-level predictions
- Confusion matrix and ROC curve visualization

See [docs/CAMELYON_TRAINING_STATUS.md](docs/CAMELYON_TRAINING_STATUS.md) for details.

## Key Features

### 1. Pretrained Model Loading

Load pretrained models from torchvision and timm with automatic feature extraction:

```python
from src.models.pretrained import load_pretrained_encoder

# Load ResNet50 from torchvision
encoder = load_pretrained_encoder(
    model_name='resnet50',
    source='torchvision',
    pretrained=True,
    num_classes=2
)

# Load EfficientNet from timm
encoder = load_pretrained_encoder(
    model_name='efficientnet_b0',
    source='timm',
    pretrained=True,
    num_classes=2
)

# Get feature dimension
feature_dim = encoder.feature_dim  # e.g., 2048 for ResNet50
```

**Supported Sources**:
- `torchvision`: ResNet, DenseNet, EfficientNet, VGG, MobileNet, etc.
- `timm`: 1000+ models including Vision Transformers, ConvNeXt, etc.

**Features**:
- Automatic feature extraction layer detection
- Preserves pretrained weights
- Returns feature dimension for downstream tasks
- Handles both torchvision and timm model APIs

### 2. Slide-Level Predictions CSV Export

Export slide-level predictions to CSV for easy analysis:

```bash
python experiments/evaluate_camelyon.py \
  --checkpoint checkpoints/camelyon/best_model.pth \
  --split test \
  --save-predictions-csv
```

**CSV Format**:
- `slide_id`: Slide identifier
- `true_label`: Ground truth label (0/1)
- `predicted_label`: Model prediction (0/1)
- `probability`: Prediction probability
- `correct`: Whether prediction matches ground truth

### 3. Model Profiling

Profile model performance and export to ONNX:

```bash
# Profile inference time
python scripts/model_profiler.py \
  --checkpoint models/best_model.pth \
  --profile-type time

# Export to ONNX
python scripts/export_onnx.py \
  --checkpoint models/best_model.pth \
  --output models/model.onnx
```

### 4. Baseline Comparisons

Compare multiple model variants:

```bash
# Quick test (3 epochs)
python experiments/compare_pcam_baselines.py \
  --configs experiments/configs/pcam_comparison/*.yaml \
  --quick-test

# Full training
python experiments/compare_pcam_baselines.py \
  --configs experiments/configs/pcam_comparison/*.yaml
```

See [docs/PCAM_COMPARISON_GUIDE.md](docs/PCAM_COMPARISON_GUIDE.md) for details.

## Repository Structure

```
.
├── src/                    # Source code
│   ├── data/              # Data loading (PCam, CAMELYON)
│   ├── models/            # Model architectures
│   ├── training/          # Training infrastructure
│   └── utils/             # Utilities
├── experiments/           # Training and evaluation scripts
│   ├── train_pcam.py     # PCam training
│   ├── evaluate_pcam.py  # PCam evaluation
│   ├── train_camelyon.py # CAMELYON training
│   └── evaluate_camelyon.py  # CAMELYON evaluation
├── scripts/               # Utility scripts
│   ├── generate_synthetic_pcam.py
│   ├── generate_synthetic_camelyon.py
│   ├── model_profiler.py
│   └── export_onnx.py
├── examples/              # Demo and example scripts
├── tests/                 # Unit tests (62% coverage)
├── docs/                  # Documentation
│   ├── DOCS_INDEX.md     # Documentation index
│   ├── PCAM_BENCHMARK_RESULTS.md
│   ├── CAMELYON_TRAINING_STATUS.md
│   └── ...
├── configs/               # Configuration files
├── data/                  # Dataset directory
├── deploy/                # Deployment configurations
├── build/                 # Build scripts (Makefile)
└── README.md              # This file
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Documentation

See [docs/DOCS_INDEX.md](docs/DOCS_INDEX.md) for a complete documentation index.

**Key Documents**:
- [docs/PCAM_BENCHMARK_RESULTS.md](docs/PCAM_BENCHMARK_RESULTS.md) - PatchCamelyon benchmark results
- [docs/CAMELYON_TRAINING_STATUS.md](docs/CAMELYON_TRAINING_STATUS.md) - CAMELYON training guide
- [docs/PCAM_COMPARISON_GUIDE.md](docs/PCAM_COMPARISON_GUIDE.md) - Baseline comparison guide
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture details
- [docs/DOCKER.md](docs/DOCKER.md) - Docker deployment guide

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

See [requirements.txt](requirements.txt) for complete dependencies.

## Limitations

- **Synthetic Data**: Current benchmarks use synthetic data for testing
- **Feature-Cache Baseline**: CAMELYON uses pre-extracted features, not raw WSI
- **Research Code**: Not validated for clinical use
- **No Real Datasets**: Requires real pathology datasets for validation

## License

MIT License - See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{computational_pathology_research,
  title = {Computational Pathology Research Framework},
  author = {Matthew Vaishnav},
  year = {2026},
  url = {https://github.com/matthewvaishnav/computational-pathology-research}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
