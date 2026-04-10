# Computational Pathology Research Framework

[![CI](https://github.com/matthewvaishnav/computational-pathology-research/workflows/CI/badge.svg)](https://github.com/matthewvaishnav/computational-pathology-research/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/matthewvaishnav/computational-pathology-research/branch/main/graph/badge.svg)](https://codecov.io/gh/matthewvaishnav/computational-pathology-research)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Research Framework**: Tested implementations for computational pathology with working benchmarks on PatchCamelyon and CAMELYON16-style slide-level classification.

> **📚 Documentation**: See [docs/](docs/) for all documentation. Start with [docs/DOCS_INDEX.md](docs/DOCS_INDEX.md) for navigation.

## Overview

A production-grade PyTorch framework for computational pathology research, providing:

- 🔬 **Whole-Slide Image (WSI) Processing**: OpenSlide integration for .svs, .tiff, .ndpi formats
- 🧠 **Multiple Instance Learning (MIL)**: Slide-level classification with attention mechanisms
- 📊 **Benchmark Pipelines**: PatchCamelyon and CAMELYON16-compatible training/evaluation
- 🔧 **Analysis Tools**: Baseline comparison, metrics analysis, bootstrap confidence intervals
- 🚀 **Production Ready**: Docker/K8s deployment, ONNX export, model profiling, 62% test coverage
- 📦 **Pretrained Models**: Easy integration with torchvision and timm (1000+ architectures)

**Status**: Research framework with validated infrastructure. Real PCam dataset support included. Not validated for clinical use.

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

Train on the PatchCamelyon benchmark with real or synthetic data:

```bash
# Download real PCam dataset (7GB, 327K images)
python scripts/download_pcam.py --output-dir data/pcam_real

# Train model (RTX 4070 Laptop: ~18 min/epoch, 3.8 it/s)
python experiments/train_pcam.py --config experiments/configs/pcam_rtx4070_laptop.yaml

# Evaluate
python experiments/evaluate_pcam.py \
  --checkpoint checkpoints/pcam_real/best_model.pth \
  --data-root data/pcam_real \
  --output-dir results/pcam
```

**Real Dataset**: 262,144 train, 32,768 val, 32,768 test samples (96×96 RGB patches)

**Development/Testing**: Synthetic data generator available for pipeline validation:
```bash
python scripts/generate_synthetic_pcam.py  # Creates small test dataset
```

See [docs/PCAM_BENCHMARK_RESULTS.md](docs/PCAM_BENCHMARK_RESULTS.md) for details.

### Full-Scale PCam Experiments

Train on the complete 262K PCam dataset with GPU-optimized configurations:

```bash
# For 16GB GPU (RTX 4070, RTX 4080) - ~8 hours
python experiments/train_pcam.py \
  --config experiments/configs/pcam_fullscale/gpu_16gb.yaml

# For 24GB GPU (RTX 4090) - ~6 hours
python experiments/train_pcam.py \
  --config experiments/configs/pcam_fullscale/gpu_24gb.yaml

# Evaluate with bootstrap confidence intervals
python experiments/evaluate_pcam.py \
  --checkpoint checkpoints/pcam_fullscale/best_model.pth \
  --data-root data/pcam \
  --output-dir results/pcam_fullscale \
  --compute-bootstrap-ci \
  --bootstrap-samples 1000

# Compare baseline models (ResNet-50, DenseNet-121, EfficientNet-B0)
python experiments/compare_pcam_baselines.py \
  --configs experiments/configs/pcam_fullscale/baseline_*.yaml \
  --output results/pcam_comparison \
  --compute-bootstrap-ci
```

**Features**:
- GPU-optimized configurations for 16GB/24GB/40GB VRAM
- Mixed precision training (AMP) for 2x speedup
- Bootstrap confidence intervals for statistical validation
- Baseline model comparisons with comprehensive reports
- Automatic dataset download and validation

See [docs/PCAM_FULLSCALE_GUIDE.md](docs/PCAM_FULLSCALE_GUIDE.md) for complete guide.

### CAMELYON16 Slide-Level Training

Train on CAMELYON16-style slide-level classification:

```bash
# Generate synthetic slide-level data for testing
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

**Note**: Current implementation uses feature-cache baseline (pre-extracted HDF5 features). Raw WSI processing pipeline in development.

See [docs/CAMELYON_TRAINING_STATUS.md](docs/CAMELYON_TRAINING_STATUS.md) for details.

## Key Features

### Analysis Tools

**NEW**: Comprehensive analysis and comparison tools:

```bash
# Analyze training metrics
python experiments/analyze_metrics.py \
  --log-dir logs/pcam_real \
  --checkpoint checkpoints/pcam_real/best_model.pth \
  --output-dir results/metrics_analysis

# Compare baseline models
python experiments/compare_baselines.py \
  --results-dir results/baselines \
  --output-dir results/baseline_comparison
```

**Features**:
- Training curve visualization (loss, accuracy, AUC)
- Confusion matrix and ROC curves
- Baseline model comparison tables
- Efficiency analysis (accuracy vs parameters)
- Comprehensive markdown reports

See [experiments/README_ANALYSIS.md](experiments/README_ANALYSIS.md) for details.

### OpenSlide Integration

**NEW**: Whole-slide image reading support:

```python
from src.data.openslide_utils import WSIReader

# Read WSI file
with WSIReader("slide.svs") as reader:
    # Get thumbnail
    thumbnail = reader.get_thumbnail((512, 512))
    
    # Extract patches
    patches = reader.extract_patches(
        patch_size=256,
        level=1,
        tissue_threshold=0.5
    )
```

**Supported formats**: .svs, .tiff, .ndpi, and other OpenSlide-compatible formats

**Note**: Requires `openslide-python`: `pip install openslide-python`

### Core Features

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

- **Feature-Cache Baseline**: CAMELYON uses pre-extracted features, not raw WSI processing
- **Research Code**: Not validated for clinical use
- **Development Stage**: Active development, APIs may change
- **GPU Requirements**: Full-scale PCam training requires 16GB+ VRAM (synthetic mode available for testing)

## Roadmap

- [x] Full-scale PCam experiments with GPU optimization
- [x] Bootstrap confidence intervals for statistical validation
- [x] Baseline model comparison infrastructure
- [ ] Real PCam benchmark results (training in progress)
- [ ] Raw WSI processing pipeline for CAMELYON
- [ ] Attention-based MIL models
- [ ] Stain normalization integration
- [ ] Multi-GPU training support

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
