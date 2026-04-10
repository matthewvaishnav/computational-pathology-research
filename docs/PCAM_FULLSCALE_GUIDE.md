# Full-Scale PatchCamelyon Training Guide

This guide provides step-by-step instructions for training and evaluating models on the complete PatchCamelyon (PCam) dataset with 262,144 training samples and 32,768 test samples.

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Software Dependencies](#software-dependencies)
- [Configuration Selection](#configuration-selection)
- [Step-by-Step Training](#step-by-step-training)
- [Evaluation](#evaluation)
- [Baseline Comparison](#baseline-comparison)
- [Troubleshooting](#troubleshooting)

## Hardware Requirements

### Minimum Requirements

- **GPU**: 16GB VRAM (NVIDIA RTX 4080, RTX 4070 Ti, A4000, or equivalent)
- **RAM**: 32GB system memory
- **Storage**: 50GB free space
  - Dataset: ~7GB
  - Checkpoints: ~5GB per model
  - Logs and results: ~5GB
- **OS**: Windows 10+, macOS 12+, or Ubuntu 20.04+

### Recommended Configuration

- **GPU**: 24GB VRAM (NVIDIA RTX 4090, A5000, or equivalent)
- **RAM**: 64GB system memory
- **Storage**: 100GB free space (NVMe SSD preferred)
- **OS**: Ubuntu 22.04 LTS

### Optimal Configuration

- **GPU**: 40GB+ VRAM (NVIDIA A100, A6000, or equivalent)
- **RAM**: 128GB system memory
- **Storage**: 200GB NVMe SSD
- **OS**: Ubuntu 22.04 LTS

### Expected Training Times

| GPU Memory | Batch Size | Time per Epoch | Total Time (20 epochs) |
|------------|------------|----------------|------------------------|
| 16GB       | 128        | ~24 minutes    | ~8 hours               |
| 24GB       | 256        | ~16 minutes    | ~6 hours               |
| 40GB+      | 512        | ~10 minutes    | ~4 hours               |

## Software Dependencies

### Required Packages

```bash
# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install h5py numpy pillow pyyaml tqdm

# For metrics and visualization
pip install scikit-learn matplotlib seaborn

# For dataset download (optional, recommended)
pip install tensorflow-datasets

# For development
pip install pytest black isort pre-commit
```

### Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Configuration Selection

Choose the configuration file based on your GPU memory:

### GPU-Optimized Configurations

**16GB GPU** (`experiments/configs/pcam_fullscale/gpu_16gb.yaml`):
- Batch size: 128
- Training time: ~8 hours
- Suitable for: RTX 4080, RTX 4070 Ti, A4000

**24GB GPU** (`experiments/configs/pcam_fullscale/gpu_24gb.yaml`):
- Batch size: 256
- Training time: ~6 hours
- Suitable for: RTX 4090, A5000

**40GB+ GPU** (`experiments/configs/pcam_fullscale/gpu_40gb.yaml`):
- Batch size: 512
- Training time: ~4 hours
- Suitable for: A100, A6000

### Baseline Model Configurations

**ResNet-50** (`experiments/configs/pcam_fullscale/baseline_resnet50.yaml`):
- Feature dimension: 2048
- Parameters: ~25M
- Best for: Strong baseline comparison

**DenseNet-121** (`experiments/configs/pcam_fullscale/baseline_densenet121.yaml`):
- Feature dimension: 1024
- Parameters: ~8M
- Best for: Memory-efficient training

**EfficientNet-B0** (`experiments/configs/pcam_fullscale/baseline_efficientnet_b0.yaml`):
- Feature dimension: 1280
- Parameters: ~5M
- Best for: Fast training and inference

## Step-by-Step Training

### 1. Download the Dataset

```bash
# Download full PCam dataset (~7GB)
python scripts/download_pcam.py --output-dir data/pcam

# The script will:
# - Download 262,144 training samples
# - Download 32,768 validation samples
# - Download 32,768 test samples
# - Validate dataset integrity
# - Save metadata
```

**Expected output:**
```
Downloading PCam via TensorFlow Datasets...
Processing split: train
  Loaded 10,000 samples...
  Loaded 20,000 samples...
  ...
✓ Saved train split: 262,144 samples
✓ Train split: 262,144 samples (valid)
✓ Val split: 32,768 samples (valid)
✓ Test split: 32,768 samples (valid)
✓ Dataset validation passed
```

### 2. Train the Model

Choose the appropriate configuration for your GPU:

```bash
# For 16GB GPU (RTX 4070, RTX 4080)
python experiments/train_pcam.py \
  --config experiments/configs/pcam_fullscale/gpu_16gb.yaml

# For 24GB GPU (RTX 4090)
python experiments/train_pcam.py \
  --config experiments/configs/pcam_fullscale/gpu_24gb.yaml

# For 40GB+ GPU (A100)
python experiments/train_pcam.py \
  --config experiments/configs/pcam_fullscale/gpu_40gb.yaml
```

**Training progress:**
```
Epoch 1/20
Train: 100%|████████| 2048/2048 [18:23<00:00, 3.8it/s]
  Loss: 0.4521, Acc: 0.7834, AUC: 0.8912
Val: 100%|████████| 256/256 [01:12<00:00, 4.2it/s]
  Loss: 0.3882, Acc: 0.8380, AUC: 0.9502
✓ New best model saved (AUC: 0.9502)

Epoch 2/20
...
```

### 3. Monitor Training

**Option A: TensorBoard**
```bash
tensorboard --logdir logs/pcam_fullscale
# Open http://localhost:6006 in browser
```

**Option B: Check training status**
```bash
# View real-time status
cat logs/pcam_fullscale/training_status.json

# Monitor GPU usage
nvidia-smi -l 1
```

### 4. Resume Training (if interrupted)

```bash
python experiments/train_pcam.py \
  --config experiments/configs/pcam_fullscale/gpu_16gb.yaml \
  --resume checkpoints/pcam_fullscale/best_model.pth
```

## Evaluation

### Basic Evaluation

```bash
python experiments/evaluate_pcam.py \
  --checkpoint checkpoints/pcam_fullscale_gpu16gb/best_model.pth \
  --data-root data/pcam \
  --output-dir results/pcam_fullscale \
  --batch-size 128 \
  --num-workers 6
```

### Evaluation with Bootstrap Confidence Intervals

For statistically robust results with uncertainty estimates:

```bash
python experiments/evaluate_pcam.py \
  --checkpoint checkpoints/pcam_fullscale_gpu16gb/best_model.pth \
  --data-root data/pcam \
  --output-dir results/pcam_fullscale \
  --batch-size 128 \
  --num-workers 6 \
  --compute-bootstrap-ci \
  --bootstrap-samples 1000 \
  --confidence-level 0.95
```

**Output files:**
- `results/pcam_fullscale/metrics.json` - All metrics with CIs
- `results/pcam_fullscale/confusion_matrix.png` - Confusion matrix
- `results/pcam_fullscale/roc_curve.png` - ROC curve

**Example metrics.json:**
```json
{
  "accuracy": 0.9523,
  "accuracy_ci_lower": 0.9487,
  "accuracy_ci_upper": 0.9558,
  "auc": 0.9812,
  "auc_ci_lower": 0.9789,
  "auc_ci_upper": 0.9834,
  ...
}
```

## Baseline Comparison

### Train Multiple Baselines

```bash
# Train ResNet-50 baseline
python experiments/train_pcam.py \
  --config experiments/configs/pcam_fullscale/baseline_resnet50.yaml

# Train DenseNet-121 baseline
python experiments/train_pcam.py \
  --config experiments/configs/pcam_fullscale/baseline_densenet121.yaml

# Train EfficientNet-B0 baseline
python experiments/train_pcam.py \
  --config experiments/configs/pcam_fullscale/baseline_efficientnet_b0.yaml
```

### Run Comparison

```bash
python experiments/compare_pcam_baselines.py \
  --configs experiments/configs/pcam_fullscale/baseline_*.yaml \
  --output results/pcam_comparison/comparison_results.json \
  --compute-bootstrap-ci
```

**Output:**
- `results/pcam_comparison/comparison_results.json` - Detailed comparison
- `results/pcam_comparison/PCAM_BENCHMARK_RESULTS.md` - Markdown report

**Example comparison table:**
```
Variant                        Accuracy      AUC          F1           Training Time
--------------------------------------------------------------------------------------------
baseline_resnet50              95.23%        0.9812       0.9501       8.2 hours
baseline_densenet121           94.87%        0.9789       0.9463       6.5 hours
baseline_efficientnet_b0       94.56%        0.9765       0.9421       5.1 hours
```

## Troubleshooting

### GPU Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

**Solution 1:** Reduce batch size
```yaml
# In your config file
training:
  batch_size: 64  # Reduce from 128
```

**Solution 2:** Enable gradient accumulation
```yaml
training:
  batch_size: 64
  gradient_accumulation_steps: 2  # Effective batch size = 128
```

**Solution 3:** Use automatic batch size reduction
The training script automatically reduces batch size on OOM and retries.

### Slow Training

**Symptom:** Training slower than expected

**Check 1:** Verify GPU is being used
```python
python -c "import torch; print(torch.cuda.is_available())"
```

**Check 2:** Verify mixed precision is enabled
```yaml
training:
  use_amp: true  # Should be true
```

**Check 3:** Increase num_workers
```yaml
data:
  num_workers: 6  # Try 4-8 based on CPU cores
```

### Download Failures

**Symptom:** Dataset download fails or hangs

**Solution 1:** Check internet connection and disk space
```bash
df -h  # Check disk space
ping google.com  # Check connection
```

**Solution 2:** Manual download
1. Download from: https://github.com/basveeling/pcam
2. Extract to `data/pcam/`
3. Run training (will use existing data)

**Solution 3:** Install TensorFlow Datasets
```bash
pip install tensorflow-datasets
```

### Validation Fails

**Symptom:** "Dataset validation failed"

**Check:** Verify sample counts
```bash
python -c "
import h5py
for split in ['train', 'val', 'test']:
    with h5py.File(f'data/pcam/{split}/images.h5py', 'r') as f:
        print(f'{split}: {f[\"images\"].shape[0]:,} samples')
"
```

**Expected:**
- train: 262,144 samples
- val: 32,768 samples
- test: 32,768 samples

### Training Hangs at Batch 0

**Symptom:** Training starts but hangs at first batch

**Solution:** Set num_workers to 0
```yaml
data:
  num_workers: 0  # Disable multiprocessing
```

This is often needed on Windows or when using h5py with multiprocessing.

## Performance Optimization Tips

### 1. Use Mixed Precision Training
Always enable AMP for 2x speedup:
```yaml
training:
  use_amp: true
```

### 2. Optimize Data Loading
```yaml
data:
  num_workers: 6      # 4-8 workers
  pin_memory: true    # Faster CPU→GPU transfer
```

### 3. Use Gradient Accumulation
For larger effective batch sizes:
```yaml
training:
  batch_size: 64
  gradient_accumulation_steps: 4  # Effective = 256
```

### 4. Enable Early Stopping
Avoid unnecessary epochs:
```yaml
early_stopping:
  enabled: true
  patience: 5
  min_delta: 0.001
```

## Next Steps

After completing full-scale training:

1. **Update Documentation**: Add your results to `docs/PCAM_BENCHMARK_RESULTS.md`
2. **Run Baselines**: Compare against ResNet-50, DenseNet-121, EfficientNet-B0
3. **Analyze Results**: Use `experiments/analyze_metrics.py` for detailed analysis
4. **Share Results**: Generate benchmark report with comparison runner

## Additional Resources

- [PCam Dataset Paper](https://arxiv.org/abs/1806.03962)
- [PCam GitHub](https://github.com/basveeling/pcam)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Project README](../README.md)
