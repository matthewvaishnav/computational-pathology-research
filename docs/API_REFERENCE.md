---
layout: default
title: API Reference
---

# API Reference

Complete API documentation for the Computational Pathology Research Framework.

---

## Table of Contents

- [Data Loading](#data-loading)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Evaluation](#evaluation)
- [Utilities](#utilities)
- [Pretrained Models](#pretrained-models)

---

## Data Loading

### PatchCamelyonDataset

Loads 96x96 pixel patches from the PatchCamelyon dataset.

```python
from src.data import PatchCamelyonDataset

dataset = PatchCamelyonDataset(
    root_dir="data/pcam",
    split="train",  # "train", "val", or "test"
    transform=None
)
```

**Parameters:**
- `root_dir` (str): Path to PCam data directory
- `split` (str): Dataset split to load
- `transform` (callable, optional): Transform to apply to images

**Returns:**
- `image` (torch.Tensor): Image tensor of shape (3, 96, 96)
- `label` (int): Binary label (0 or 1)

---

### CAMELYONSlideDataset

Loads pre-extracted features for slide-level classification.

```python
from src.data import CAMELYONSlideDataset

dataset = CAMELYONSlideDataset(
    root_dir="data/camelyon/features",
    split="train",
    max_patches=None
)
```

**Parameters:**
- `root_dir` (str): Path to HDF5 feature directory
- `split` (str): Dataset split to load
- `max_patches` (int, optional): Maximum patches per slide

**Returns:**
- `features` (torch.Tensor): Feature tensor of shape (num_patches, feature_dim)
- `label` (int): Binary label (0 or 1)
- `slide_id` (str): Slide identifier

---

### collate_slide_bags

Collates variable-length slides into batched tensors with masking.

```python
from src.data import collate_slide_bags

batch = collate_slide_bags(samples)
```

**Parameters:**
- `samples` (list): List of (features, label, slide_id) tuples

**Returns:**
- `features` (torch.Tensor): Padded features of shape (batch, max_patches, feature_dim)
- `labels` (torch.Tensor): Labels of shape (batch,)
- `num_patches` (torch.Tensor): Number of patches per slide of shape (batch,)
- `slide_ids` (list): List of slide identifiers

---

## Model Architectures

### SimpleClassifier

Basic CNN classifier for patch-level classification.

```python
from src.models import SimpleClassifier

model = SimpleClassifier(
    num_classes=2,
    dropout=0.5
)
```

**Parameters:**
- `num_classes` (int): Number of output classes
- `dropout` (float): Dropout probability

**Forward:**
```python
output = model(images)  # images: (batch, 3, 96, 96)
# output: (batch, num_classes)
```

---

### SimpleSlideClassifier

Slide-level classifier with attention-based aggregation.

```python
from src.models import SimpleSlideClassifier

model = SimpleSlideClassifier(
    feature_dim=2048,
    hidden_dim=256,
    num_classes=2,
    pooling="attention",  # "mean", "max", or "attention"
    dropout=0.5
)
```

**Parameters:**
- `feature_dim` (int): Input feature dimension
- `hidden_dim` (int): Hidden layer dimension
- `num_classes` (int): Number of output classes
- `pooling` (str): Aggregation method
- `dropout` (float): Dropout probability

**Forward:**
```python
output = model(features, num_patches)
# features: (batch, max_patches, feature_dim)
# num_patches: (batch,)
# output: (batch, num_classes)
```

---

## Pretrained Models

### load_pretrained_encoder

Loads pretrained encoders from torchvision or timm.

```python
from src.models.pretrained import load_pretrained_encoder

encoder = load_pretrained_encoder(
    model_name="resnet50",
    source="torchvision",  # "torchvision" or "timm"
    pretrained=True,
    num_classes=2
)

# Access feature dimension
feature_dim = encoder.feature_dim
```

**Parameters:**
- `model_name` (str): Model architecture name
- `source` (str): Model source ("torchvision" or "timm")
- `pretrained` (bool): Load pretrained weights
- `num_classes` (int): Number of output classes

**Supported Models:**

**torchvision:**
- ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
- DenseNet: densenet121, densenet161, densenet169, densenet201
- EfficientNet: efficientnet_b0 through efficientnet_b7
- VGG: vgg11, vgg13, vgg16, vgg19
- MobileNet: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large

**timm:**
- Vision Transformers: vit_base_patch16_224, vit_large_patch16_224
- ConvNeXt: convnext_tiny, convnext_small, convnext_base
- Swin Transformer: swin_tiny_patch4_window7_224
- EfficientNet: efficientnet_b0 through efficientnet_b7
- And 1000+ more models

---

## Training

### train_epoch

Trains model for one epoch.

```python
from src.training import train_epoch

metrics = train_epoch(
    model=model,
    train_loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epoch=epoch
)
```

**Parameters:**
- `model` (nn.Module): Model to train
- `train_loader` (DataLoader): Training data loader
- `criterion` (nn.Module): Loss function
- `optimizer` (Optimizer): Optimizer
- `device` (torch.device): Device to train on
- `epoch` (int): Current epoch number

**Returns:**
- `metrics` (dict): Training metrics
  - `loss` (float): Average training loss
  - `accuracy` (float): Training accuracy
  - `time` (float): Epoch duration in seconds

---

## Evaluation

### evaluate

Evaluates model on validation/test set.

```python
from src.training import evaluate

metrics = evaluate(
    model=model,
    val_loader=val_loader,
    criterion=criterion,
    device=device
)
```

**Parameters:**
- `model` (nn.Module): Model to evaluate
- `val_loader` (DataLoader): Validation data loader
- `criterion` (nn.Module): Loss function
- `device` (torch.device): Device to evaluate on

**Returns:**
- `metrics` (dict): Evaluation metrics
  - `loss` (float): Average validation loss
  - `accuracy` (float): Validation accuracy
  - `auc` (float): Area under ROC curve
  - `predictions` (np.ndarray): Model predictions
  - `labels` (np.ndarray): Ground truth labels

---

## Utilities

### set_seed

Sets random seed for reproducibility.

```python
from src.utils import set_seed

set_seed(42)
```

**Parameters:**
- `seed` (int): Random seed value

---

### save_checkpoint

Saves model checkpoint.

```python
from src.utils import save_checkpoint

save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    metrics=metrics,
    path="checkpoints/model.pth"
)
```

**Parameters:**
- `model` (nn.Module): Model to save
- `optimizer` (Optimizer): Optimizer state
- `epoch` (int): Current epoch
- `metrics` (dict): Training metrics
- `path` (str): Save path

---

### load_checkpoint

Loads model checkpoint.

```python
from src.utils import load_checkpoint

checkpoint = load_checkpoint(
    path="checkpoints/model.pth",
    model=model,
    optimizer=optimizer
)
```

**Parameters:**
- `path` (str): Checkpoint path
- `model` (nn.Module, optional): Model to load weights into
- `optimizer` (Optimizer, optional): Optimizer to load state into

**Returns:**
- `checkpoint` (dict): Checkpoint dictionary
  - `epoch` (int): Saved epoch
  - `metrics` (dict): Saved metrics
  - `model_state_dict` (dict): Model weights
  - `optimizer_state_dict` (dict): Optimizer state

---

## Configuration

### Training Configuration

Example YAML configuration for training:

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
  scheduler: "step"
  step_size: 5
  gamma: 0.1

logging:
  log_interval: 10
  checkpoint_dir: "checkpoints/pcam"
  save_best: true
```

---

## Examples

### Training a PCam Model

```python
import torch
from torch.utils.data import DataLoader
from src.data import PatchCamelyonDataset
from src.models import SimpleClassifier
from src.training import train_epoch, evaluate
from src.utils import set_seed

# Set seed for reproducibility
set_seed(42)

# Create datasets
train_dataset = PatchCamelyonDataset("data/pcam", "train")
val_dataset = PatchCamelyonDataset("data/pcam", "val")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create model
model = SimpleClassifier(num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training setup
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
    val_metrics = evaluate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/10")
    print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
    print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
```

### Using Pretrained Models

```python
from src.models.pretrained import load_pretrained_encoder
import torch.nn as nn

# Load pretrained ResNet50
encoder = load_pretrained_encoder(
    model_name="resnet50",
    source="torchvision",
    pretrained=True,
    num_classes=2
)

# Get feature dimension
print(f"Feature dimension: {encoder.feature_dim}")

# Use in training
model = encoder.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

<div class="footer-note">
  <p><em>Last updated: April 2026</em></p>
  <p>For more examples, see the <a href="https://github.com/matthewvaishnav/computational-pathology-research/tree/main/examples">examples directory</a>.</p>
</div>
