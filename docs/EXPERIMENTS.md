---
layout: default
title: Experiments
---

# Experiments

Complete guide to running experiments on PatchCamelyon and CAMELYON16 datasets.

---

## PatchCamelyon Benchmark

The PatchCamelyon (PCam) dataset consists of 96×96 pixel patches extracted from histopathologic scans of lymph node sections.

### Generate Synthetic Data

```bash
python scripts/generate_synthetic_pcam.py
```

### Train Model

```bash
python experiments/train_pcam.py --config experiments/configs/pcam.yaml
```

### Evaluate

```bash
python experiments/evaluate_pcam.py \
  --checkpoint checkpoints/pcam/best_model.pth \
  --data-root data/pcam \
  --output-dir results/pcam
```

### Results

**Synthetic subset (8 epochs):**
- Test Accuracy: **94.0%**
- Test AUC: **1.0**
- Training Time: ~40 seconds (CPU)

See [PCAM_BENCHMARK_RESULTS.html](PCAM_BENCHMARK_RESULTS.html) for detailed analysis.

---

## CAMELYON16 Slide-Level Classification

Slide-level classification using pre-extracted patch features with attention-based aggregation.

### Generate Synthetic Data

```bash
python scripts/generate_synthetic_camelyon.py
```

### Train Slide Classifier

```bash
# Quick test (1 epoch)
python experiments/train_camelyon.py \
  --config experiments/configs/camelyon_quick_test.yaml

# Full training (50 epochs)
python experiments/train_camelyon.py \
  --config experiments/configs/camelyon.yaml
```

### Evaluate with CSV Export

```bash
python experiments/evaluate_camelyon.py \
  --checkpoint checkpoints/camelyon/best_model.pth \
  --data-root data/camelyon \
  --output-dir results/camelyon \
  --save-predictions-csv
```

### Features

- Multiple aggregation strategies (mean, max pooling)
- Variable-length slide handling with masking
- CSV export for downstream analysis
- ROC curves and confusion matrices

See [CAMELYON_TRAINING_STATUS.html](CAMELYON_TRAINING_STATUS.html) for detailed guide.

---

## Pretrained Model Integration

Load and fine-tune pretrained encoders:

```python
from src.models.pretrained import load_pretrained_encoder

# Load ResNet50 with ImageNet weights
encoder = load_pretrained_encoder(
    model_name='resnet50',
    source='torchvision',
    pretrained=True,
    num_classes=2
)

# Access feature dimension
feature_dim = encoder.feature_dim  # 2048 for ResNet50
```

**Supported Sources:**
- **torchvision**: ResNet, DenseNet, EfficientNet, VGG, MobileNet
- **timm**: 1000+ models including Vision Transformers, ConvNeXt, Swin

---

## Model Profiling

Analyze model performance and export for deployment:

```bash
# Profile inference latency
python scripts/model_profiler.py \
  --checkpoint models/best_model.pth \
  --profile-type time

# Export to ONNX format
python scripts/export_onnx.py \
  --checkpoint models/best_model.pth \
  --output models/model.onnx
```

---

## Baseline Comparisons

Compare multiple model architectures:

```bash
# Quick test (3 epochs)
python experiments/compare_pcam_baselines.py \
  --configs experiments/configs/pcam_comparison/*.yaml \
  --quick-test

# Full training
python experiments/compare_pcam_baselines.py \
  --configs experiments/configs/pcam_comparison/*.yaml
```

See [PCAM_COMPARISON_GUIDE.html](PCAM_COMPARISON_GUIDE.html) for detailed comparison guide.

---

## Configuration

Example training configuration:

```yaml
# experiments/configs/pcam.yaml
data:
  root_dir: "data/pcam"
  batch_size: 32
  num_workers: 4

model:
  architecture: "resnet18"
  num_classes: 2
  dropout: 0.5

training:
  epochs: 10
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "adam"
```

---

## Next Steps

- [Getting Started Guide](GETTING_STARTED.html) - Installation and setup
- [API Reference](API_REFERENCE.html) - Complete API documentation
- [Architecture Guide](ARCHITECTURE.html) - System design details
- [Documentation Index](DOCS_INDEX.html) - All documentation

---

<div class="footer-note">
  <p><em>Last updated: April 2026</em></p>
</div>
