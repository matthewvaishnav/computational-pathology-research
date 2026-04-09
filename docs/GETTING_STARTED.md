---
layout: default
title: Getting Started
---

# Getting Started

Complete guide to installing and using the Computational Pathology Research Framework.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Your First Model](#your-first-model)
5. [Working with Real Data](#working-with-real-data)
6. [Next Steps](#next-steps)

---

## System Requirements

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB free space

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA GPU with 8GB+ VRAM (for training)
- Storage: 50GB+ free space (for datasets)

### Software Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.9 or higher
- **CUDA**: 11.7+ (for GPU support)
- **Git**: For cloning the repository

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/matthewvaishnav/computational-pathology-research.git
cd computational-pathology-research
```

### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Step 4: Verify Installation

```bash
# Run quick test
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import src; print('Installation successful!')"

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Quick Start

### Generate Synthetic Data

For testing and development, generate synthetic datasets:

```bash
# Generate PCam synthetic data
python scripts/generate_synthetic_pcam.py

# Generate CAMELYON synthetic data
python scripts/generate_synthetic_camelyon.py
```

This creates:
- `data/pcam/` - Synthetic patch-level data
- `data/camelyon/features/` - Synthetic slide-level features

### Train Your First Model

Train a simple model on PCam:

```bash
python experiments/train_pcam.py \
  --config experiments/configs/pcam.yaml \
  --epochs 5
```

Expected output:
```
Epoch 1/5: Train Loss: 0.6234, Acc: 0.6500
Epoch 1/5: Val Loss: 0.5123, Acc: 0.7500
...
Training complete! Best accuracy: 0.9400
```

### Evaluate the Model

```bash
python experiments/evaluate_pcam.py \
  --checkpoint checkpoints/pcam/best_model.pth \
  --data-root data/pcam \
  --output-dir results/pcam
```

Results saved to `results/pcam/`:
- `metrics.json` - Evaluation metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curve.png` - ROC curve

---

## Your First Model

### Step-by-Step Tutorial

#### 1. Prepare Your Data

```python
from src.data import PatchCamelyonDataset
from torch.utils.data import DataLoader

# Create dataset
train_dataset = PatchCamelyonDataset(
    root_dir="data/pcam",
    split="train"
)

# Create data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

#### 2. Create Your Model

```python
from src.models import SimpleClassifier
import torch

# Create model
model = SimpleClassifier(num_classes=2, dropout=0.5)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

#### 3. Set Up Training

```python
from torch import nn, optim

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.1
)
```

#### 4. Training Loop

```python
from src.training import train_epoch, evaluate
from src.utils import save_checkpoint

best_acc = 0.0

for epoch in range(10):
    # Train
    train_metrics = train_epoch(
        model, train_loader, criterion, optimizer, device, epoch
    )
    
    # Validate
    val_metrics = evaluate(
        model, val_loader, criterion, device
    )
    
    # Update learning rate
    scheduler.step()
    
    # Save best model
    if val_metrics['accuracy'] > best_acc:
        best_acc = val_metrics['accuracy']
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            "checkpoints/best_model.pth"
        )
    
    # Print progress
    print(f"Epoch {epoch+1}/10")
    print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
    print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

print(f"Training complete! Best accuracy: {best_acc:.4f}")
```

#### 5. Evaluate and Visualize

```python
from src.utils import load_checkpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

# Load best model
checkpoint = load_checkpoint("checkpoints/best_model.pth", model)

# Evaluate
test_metrics = evaluate(model, test_loader, criterion, device)

# Confusion matrix
cm = confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')

# ROC curve
fpr, tpr, _ = roc_curve(test_metrics['labels'], test_metrics['probabilities'][:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png')

print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test AUC: {roc_auc:.4f}")
```

---

## Working with Real Data

### PatchCamelyon Dataset

#### Download the Dataset

```bash
# Download from official source
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz

# Extract
gunzip *.gz

# Move to data directory
mkdir -p data/pcam
mv *.h5 data/pcam/
```

#### Train on Real Data

```bash
python experiments/train_pcam.py \
  --config experiments/configs/pcam.yaml \
  --data-root data/pcam \
  --epochs 50 \
  --batch-size 64
```

### CAMELYON16 Dataset

#### Download and Prepare

```bash
# Download from CAMELYON16 challenge website
# https://camelyon16.grand-challenge.org/

# Extract features using your preferred method
# (e.g., pretrained ResNet50)

# Organize as HDF5 files
# data/camelyon/features/slide_000.h5
# data/camelyon/features/slide_001.h5
# ...
```

#### Train Slide Classifier

```bash
python experiments/train_camelyon.py \
  --config experiments/configs/camelyon.yaml \
  --data-root data/camelyon \
  --epochs 50
```

---

## Next Steps

### Explore Advanced Features

1. **Pretrained Models**
   - [Using Pretrained Encoders](API_REFERENCE.html#pretrained-models)
   - [Fine-tuning Strategies](ADVANCED_TRAINING.html#fine-tuning)

2. **Model Comparison**
   - [Baseline Comparisons](PCAM_COMPARISON_GUIDE.html)
   - [Architecture Search](ARCHITECTURE_SEARCH.html)

3. **Deployment**
   - [Docker Deployment](DOCKER.html)
   - [REST API](../deploy/README.html)
   - [ONNX Export](MODEL_EXPORT.html)

### Read the Documentation

- [API Reference](API_REFERENCE.html) - Complete API documentation
- [Architecture Guide](ARCHITECTURE.html) - System design details
- [Performance Optimization](PERFORMANCE.html) - Speed up training

### Join the Community

- [GitHub Issues](https://github.com/matthewvaishnav/computational-pathology-research/issues) - Report bugs or request features
- [Discussions](https://github.com/matthewvaishnav/computational-pathology-research/discussions) - Ask questions

---

## Troubleshooting

### Common Issues

#### ImportError: No module named 'src'

**Solution:** Install the package in development mode:
```bash
pip install -e .
```

#### CUDA Out of Memory

**Solution:** Reduce batch size in config:
```yaml
data:
  batch_size: 16  # Reduce from 32
```

#### Slow Data Loading

**Solution:** Increase number of workers:
```yaml
data:
  num_workers: 8  # Increase from 4
```

#### Poor Model Performance

**Solutions:**
1. Increase training epochs
2. Try different learning rates
3. Use data augmentation
4. Try pretrained models

---

<div class="footer-note">
  <p><em>Last updated: April 2026</em></p>
  <p>Need help? <a href="https://github.com/matthewvaishnav/computational-pathology-research/issues">Open an issue</a> on GitHub.</p>
</div>
