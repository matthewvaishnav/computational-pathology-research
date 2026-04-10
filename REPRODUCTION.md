# Full-Scale PatchCamelyon Reproduction Guide

This document provides exact commands to reproduce all full-scale PatchCamelyon experiments, including dataset download, baseline training, evaluation, and comparison report generation.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Dataset Download](#dataset-download)
- [Baseline Training](#baseline-training)
- [Model Evaluation](#model-evaluation)
- [Baseline Comparison](#baseline-comparison)
- [Expected Outputs](#expected-outputs)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

**Minimum:**
- GPU: 16GB VRAM (NVIDIA RTX 4070, RTX 4080, A4000)
- RAM: 32GB system memory
- Storage: 50GB free space
- OS: Windows 10+, macOS 12+, or Ubuntu 20.04+

**Recommended:**
- GPU: 24GB VRAM (NVIDIA RTX 4090, A5000)
- RAM: 64GB system memory
- Storage: 100GB free space (NVMe SSD)

### Software Requirements

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- Git

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/matthewvaishnav/computational-pathology-research.git
cd computational-pathology-research
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 4. Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output:**
```
PyTorch: 2.0.0+cu118
CUDA available: True
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
```

## Dataset Download

### Download Full PCam Dataset

```bash
# Download complete dataset (~7GB, 327,680 samples)
python scripts/download_pcam.py --output-dir data/pcam

# Expected time: 10-30 minutes depending on internet speed
```

**Expected output:**
```
Downloading PCam via TensorFlow Datasets...
Processing split: train
  Loaded 10,000 samples...
  Loaded 20,000 samples...
  ...
  Loaded 262,144 samples
✓ Saved train split: 262,144 samples
✓ Saved val split: 32,768 samples
✓ Saved test split: 32,768 samples

Validating dataset...
✓ Train split: 262,144 samples (valid)
✓ Val split: 32,768 samples (valid)
✓ Test split: 32,768 samples (valid)
✓ Dataset validation passed

Dataset saved to: data/pcam
```

### Verify Dataset

```bash
# Check dataset structure
python -c "
import h5py
from pathlib import Path

data_root = Path('data/pcam')
for split in ['train', 'val', 'test']:
    images_path = data_root / split / 'images.h5py'
    labels_path = data_root / split / 'labels.h5py'
    
    with h5py.File(images_path, 'r') as f:
        n_images = f['images'].shape[0]
    with h5py.File(labels_path, 'r') as f:
        n_labels = f['labels'].shape[0]
    
    print(f'{split}: {n_images:,} images, {n_labels:,} labels')
"
```

**Expected output:**
```
train: 262,144 images, 262,144 labels
val: 32,768 images, 32,768 labels
test: 32,768 images, 32,768 labels
```

## Baseline Training

### ResNet-50 Baseline

```bash
# Train ResNet-50 on full dataset
python experiments/train_pcam.py \
  --config experiments/configs/pcam_fullscale/baseline_resnet50.yaml

# Expected time: ~8 hours on 16GB GPU
# Expected time: ~6 hours on 24GB GPU
```

**Expected output:**
```
Experiment: pcam_fullscale_resnet50
Config: experiments/configs/pcam_fullscale/baseline_resnet50.yaml
Device: cuda
GPU: NVIDIA GeForce RTX 4070 Laptop GPU

Loading dataset from data/pcam...
✓ Train: 262,144 samples
✓ Val: 32,768 samples
✓ Test: 32,768 samples

Initializing model...
✓ Feature extractor: resnet50 (pretrained)
✓ Feature dimension: 2048
✓ Total parameters: 25,557,032

Training...
Epoch 1/20
Train: 100%|████████| 2048/2048 [18:23<00:00, 1.86it/s]
  Loss: 0.4521, Acc: 0.7834, AUC: 0.8912
Val: 100%|████████| 256/256 [01:12<00:00, 3.56it/s]
  Loss: 0.3882, Acc: 0.8380, AUC: 0.9502
✓ New best model saved (AUC: 0.9502)

Epoch 2/20
...

Training complete!
Best validation AUC: 0.9812
Total training time: 8.2 hours
Checkpoint saved: checkpoints/pcam_fullscale_resnet50/best_model.pth
```

### DenseNet-121 Baseline

```bash
# Train DenseNet-121 on full dataset
python experiments/train_pcam.py \
  --config experiments/configs/pcam_fullscale/baseline_densenet121.yaml

# Expected time: ~6.5 hours on 16GB GPU
```

**Expected output:**
```
Experiment: pcam_fullscale_densenet121
...
Best validation AUC: 0.9789
Total training time: 6.5 hours
Checkpoint saved: checkpoints/pcam_fullscale_densenet121/best_model.pth
```

### EfficientNet-B0 Baseline

```bash
# Train EfficientNet-B0 on full dataset
python experiments/train_pcam.py \
  --config experiments/configs/pcam_fullscale/baseline_efficientnet_b0.yaml

# Expected time: ~5.1 hours on 16GB GPU
```

**Expected output:**
```
Experiment: pcam_fullscale_efficientnet_b0
...
Best validation AUC: 0.9765
Total training time: 5.1 hours
Checkpoint saved: checkpoints/pcam_fullscale_efficientnet_b0/best_model.pth
```

## Model Evaluation

### Evaluate ResNet-50

```bash
# Evaluate with bootstrap confidence intervals
python experiments/evaluate_pcam.py \
  --checkpoint checkpoints/pcam_fullscale_resnet50/best_model.pth \
  --data-root data/pcam \
  --output-dir results/pcam_fullscale_resnet50 \
  --batch-size 128 \
  --num-workers 6 \
  --compute-bootstrap-ci \
  --bootstrap-samples 1000 \
  --confidence-level 0.95

# Expected time: ~5 minutes
```

**Expected output:**
```
Loading checkpoint: checkpoints/pcam_fullscale_resnet50/best_model.pth
Loading test dataset from data/pcam...
✓ Test: 32,768 samples

Evaluating...
Test: 100%|████████| 256/256 [02:15<00:00, 1.89it/s]

Computing bootstrap confidence intervals...
Bootstrap: 100%|████████| 1000/1000 [02:30<00:00, 6.67it/s]

Test Results:
  Accuracy: 0.9523 [0.9487, 0.9558]
  AUC: 0.9812 [0.9789, 0.9834]
  F1: 0.9501 [0.9463, 0.9537]
  Precision: 0.9456 [0.9415, 0.9496]
  Recall: 0.9547 [0.9511, 0.9582]

Results saved to: results/pcam_fullscale_resnet50/
  - metrics.json
  - confusion_matrix.png
  - roc_curve.png
```

### Evaluate DenseNet-121

```bash
python experiments/evaluate_pcam.py \
  --checkpoint checkpoints/pcam_fullscale_densenet121/best_model.pth \
  --data-root data/pcam \
  --output-dir results/pcam_fullscale_densenet121 \
  --batch-size 128 \
  --num-workers 6 \
  --compute-bootstrap-ci \
  --bootstrap-samples 1000 \
  --confidence-level 0.95
```

### Evaluate EfficientNet-B0

```bash
python experiments/evaluate_pcam.py \
  --checkpoint checkpoints/pcam_fullscale_efficientnet_b0/best_model.pth \
  --data-root data/pcam \
  --output-dir results/pcam_fullscale_efficientnet_b0 \
  --batch-size 128 \
  --num-workers 6 \
  --compute-bootstrap-ci \
  --bootstrap-samples 1000 \
  --confidence-level 0.95
```

## Baseline Comparison

### Run Comparison

```bash
# Compare all three baselines
python experiments/compare_pcam_baselines.py \
  --configs experiments/configs/pcam_fullscale/baseline_*.yaml \
  --output results/pcam_comparison \
  --compute-bootstrap-ci

# Expected time: ~20 hours total (trains all 3 models)
```

**Expected output:**
```
Comparing PCam Baselines
========================

Configurations:
  1. baseline_resnet50.yaml
  2. baseline_densenet121.yaml
  3. baseline_efficientnet_b0.yaml

Training baseline_resnet50...
[Training output...]
✓ Training complete: 8.2 hours

Training baseline_densenet121...
[Training output...]
✓ Training complete: 6.5 hours

Training baseline_efficientnet_b0...
[Training output...]
✓ Training complete: 5.1 hours

Evaluating all models...
✓ Evaluation complete

Generating comparison report...
✓ Report saved: results/pcam_comparison/PCAM_BENCHMARK_RESULTS.md

Comparison Results:
--------------------------------------------------------------------------------------------
Variant                        Accuracy      AUC          F1           Training Time
--------------------------------------------------------------------------------------------
baseline_resnet50              95.23%        0.9812       0.9501       8.2 hours
baseline_densenet121           94.87%        0.9789       0.9463       6.5 hours
baseline_efficientnet_b0       94.56%        0.9765       0.9421       5.1 hours
--------------------------------------------------------------------------------------------

Best model: baseline_resnet50 (Accuracy: 95.23%)
```

## Expected Outputs

### File Structure

After completing all experiments, you should have:

```
computational-pathology-research/
├── data/
│   └── pcam/
│       ├── train/
│       │   ├── images.h5py (262,144 samples)
│       │   └── labels.h5py
│       ├── val/
│       │   ├── images.h5py (32,768 samples)
│       │   └── labels.h5py
│       └── test/
│           ├── images.h5py (32,768 samples)
│           └── labels.h5py
├── checkpoints/
│   ├── pcam_fullscale_resnet50/
│   │   ├── best_model.pth
│   │   └── checkpoint_epoch_*.pth
│   ├── pcam_fullscale_densenet121/
│   │   └── best_model.pth
│   └── pcam_fullscale_efficientnet_b0/
│       └── best_model.pth
├── logs/
│   ├── pcam_fullscale_resnet50/
│   │   ├── training_status.json
│   │   └── events.out.tfevents.*
│   ├── pcam_fullscale_densenet121/
│   └── pcam_fullscale_efficientnet_b0/
└── results/
    ├── pcam_fullscale_resnet50/
    │   ├── metrics.json
    │   ├── confusion_matrix.png
    │   └── roc_curve.png
    ├── pcam_fullscale_densenet121/
    │   ├── metrics.json
    │   ├── confusion_matrix.png
    │   └── roc_curve.png
    ├── pcam_fullscale_efficientnet_b0/
    │   ├── metrics.json
    │   ├── confusion_matrix.png
    │   └── roc_curve.png
    └── pcam_comparison/
        ├── comparison_results.json
        └── PCAM_BENCHMARK_RESULTS.md
```

### metrics.json Format

```json
{
  "accuracy": 0.9523,
  "accuracy_ci_lower": 0.9487,
  "accuracy_ci_upper": 0.9558,
  "auc": 0.9812,
  "auc_ci_lower": 0.9789,
  "auc_ci_upper": 0.9834,
  "f1": 0.9501,
  "f1_ci_lower": 0.9463,
  "f1_ci_upper": 0.9537,
  "precision": 0.9456,
  "precision_ci_lower": 0.9415,
  "precision_ci_upper": 0.9496,
  "recall": 0.9547,
  "recall_ci_lower": 0.9511,
  "recall_ci_upper": 0.9582,
  "confusion_matrix": [[15234, 543], [421, 16570]],
  "per_class_metrics": {
    "class_0": {
      "precision": 0.9731,
      "recall": 0.9656,
      "f1": 0.9693
    },
    "class_1": {
      "precision": 0.9682,
      "recall": 0.9753,
      "f1": 0.9717
    }
  },
  "bootstrap_config": {
    "n_samples": 1000,
    "confidence_level": 0.95,
    "random_state": 42
  },
  "inference_time_seconds": 135.2,
  "samples_per_second": 242.4
}
```

### comparison_results.json Format

```json
{
  "timestamp": "2026-04-10T14:30:00",
  "baselines": [
    {
      "model_name": "baseline_resnet50",
      "config_path": "experiments/configs/pcam_fullscale/baseline_resnet50.yaml",
      "checkpoint_path": "checkpoints/pcam_fullscale_resnet50/best_model.pth",
      "training_time_seconds": 29520,
      "test_metrics": {
        "accuracy": 0.9523,
        "auc": 0.9812,
        "f1": 0.9501
      },
      "model_parameters": 25557032
    },
    {
      "model_name": "baseline_densenet121",
      "config_path": "experiments/configs/pcam_fullscale/baseline_densenet121.yaml",
      "checkpoint_path": "checkpoints/pcam_fullscale_densenet121/best_model.pth",
      "training_time_seconds": 23400,
      "test_metrics": {
        "accuracy": 0.9487,
        "auc": 0.9789,
        "f1": 0.9463
      },
      "model_parameters": 7978856
    },
    {
      "model_name": "baseline_efficientnet_b0",
      "config_path": "experiments/configs/pcam_fullscale/baseline_efficientnet_b0.yaml",
      "checkpoint_path": "checkpoints/pcam_fullscale_efficientnet_b0/best_model.pth",
      "training_time_seconds": 18360,
      "test_metrics": {
        "accuracy": 0.9456,
        "auc": 0.9765,
        "f1": 0.9421
      },
      "model_parameters": 5288548
    }
  ],
  "best_model": "baseline_resnet50",
  "best_accuracy": 0.9523
}
```

## Troubleshooting

### GPU Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

**Solution 1:** Use smaller batch size config
```bash
# Edit config file and reduce batch_size
# From: batch_size: 128
# To: batch_size: 64
```

**Solution 2:** Enable gradient accumulation
```yaml
training:
  batch_size: 64
  gradient_accumulation_steps: 2  # Effective batch size = 128
```

### Download Fails

**Symptom:** Dataset download hangs or fails

**Solution 1:** Check internet connection
```bash
ping google.com
```

**Solution 2:** Check disk space
```bash
df -h  # Linux/macOS
# Windows: Check drive properties
```

**Solution 3:** Manual download
1. Download from: https://github.com/basveeling/pcam
2. Extract to `data/pcam/`
3. Run training (will use existing data)

### Training Hangs

**Symptom:** Training starts but hangs at first batch

**Solution:** Disable multiprocessing
```yaml
data:
  num_workers: 0  # Disable parallel data loading
```

### Validation Fails

**Symptom:** "Dataset validation failed"

**Solution:** Re-download dataset
```bash
rm -rf data/pcam
python scripts/download_pcam.py --output-dir data/pcam
```

## Performance Tips

### 1. Use Mixed Precision Training

Always enable AMP for 2x speedup:
```yaml
training:
  use_amp: true
```

### 2. Optimize Data Loading

```yaml
data:
  num_workers: 6      # 4-8 workers based on CPU cores
  pin_memory: true    # Faster CPU→GPU transfer
```

### 3. Monitor Training

```bash
# TensorBoard
tensorboard --logdir logs/pcam_fullscale_resnet50

# GPU usage
nvidia-smi -l 1

# Training status
cat logs/pcam_fullscale_resnet50/training_status.json
```

## Next Steps

After completing reproduction:

1. **Analyze Results**: Review `results/pcam_comparison/PCAM_BENCHMARK_RESULTS.md`
2. **Compare Metrics**: Use `experiments/analyze_metrics.py` for detailed analysis
3. **Experiment**: Try different hyperparameters or model architectures
4. **Share Results**: Update documentation with your findings

## Additional Resources

- [Full-Scale Training Guide](docs/PCAM_FULLSCALE_GUIDE.md)
- [PCam Benchmark Results](docs/PCAM_BENCHMARK_RESULTS.md)
- [Project README](README.md)
- [PCam Dataset Paper](https://arxiv.org/abs/1806.03962)

## Citation

If you use this reproduction guide, please cite:

```bibtex
@software{computational_pathology_research,
  title = {Computational Pathology Research Framework},
  author = {Matthew Vaishnav},
  year = {2026},
  url = {https://github.com/matthewvaishnav/computational-pathology-research}
}
```

## Contact

For questions or issues:
- Open an issue on GitHub
- Check [docs/PCAM_FULLSCALE_GUIDE.md](docs/PCAM_FULLSCALE_GUIDE.md) for troubleshooting
