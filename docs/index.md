---
layout: default
title: Home
---

# Computational Pathology Research Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/LICENSE)
[![Code Coverage](https://img.shields.io/badge/coverage-62%25-yellow.svg)](https://github.com/matthewvaishnav/computational-pathology-research)

> **Research Framework**: Tested implementations for computational pathology with working benchmarks on PatchCamelyon and CAMELYON16-style slide-level classification.

## Overview

This repository provides a tested PyTorch framework for computational pathology research, with working implementations for:

- ✅ **PatchCamelyon (PCam) Training**: 94% accuracy on synthetic subset
- ✅ **CAMELYON16 Slide-Level Pipeline**: Functional slide-level classification with mean/max pooling
- ✅ **Slide Predictions CSV Export**: Publication-ready prediction exports
- ✅ **Comprehensive Testing**: 62% code coverage with unit tests
- ✅ **Model Profiling**: Performance analysis and ONNX export tools
- ✅ **Pretrained Model Loading**: Easy integration with torchvision and timm models

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

## Key Features

### Pretrained Model Loading

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

# Get feature dimension
feature_dim = encoder.feature_dim  # e.g., 2048 for ResNet50
```

### Slide-Level Predictions CSV Export

Export slide-level predictions to CSV for easy analysis:

```bash
python experiments/evaluate_camelyon.py \
  --checkpoint checkpoints/camelyon/best_model.pth \
  --split test \
  --save-predictions-csv
```

## Documentation

- [Documentation Index](DOCS_INDEX.html)
- [PatchCamelyon Benchmark Results](PCAM_BENCHMARK_RESULTS.html)
- [CAMELYON Training Guide](CAMELYON_TRAINING_STATUS.html)
- [Architecture Overview](ARCHITECTURE.html)
- [Docker Deployment](DOCKER.html)

## Repository Structure

```
.
├── src/                    # Source code
│   ├── data/              # Data loading (PCam, CAMELYON)
│   ├── models/            # Model architectures
│   ├── training/          # Training infrastructure
│   └── utils/             # Utilities
├── experiments/           # Training and evaluation scripts
├── scripts/               # Utility scripts
├── examples/              # Demo and example scripts
├── tests/                 # Unit tests (62% coverage)
├── docs/                  # Documentation
└── configs/               # Configuration files
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

See [requirements.txt](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/requirements.txt) for complete dependencies.

## Citation

```bibtex
@software{computational_pathology_research,
  title = {Computational Pathology Research Framework},
  author = {Matthew Vaishnav},
  year = {2026},
  url = {https://github.com/matthewvaishnav/computational-pathology-research}
}
```

## Links

- [GitHub Repository](https://github.com/matthewvaishnav/computational-pathology-research)
- [Issues](https://github.com/matthewvaishnav/computational-pathology-research/issues)
- [License](https://github.com/matthewvaishnav/computational-pathology-research/blob/main/LICENSE)
