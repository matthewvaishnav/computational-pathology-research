# TransnnMIL v2.0: Training Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision
pip install torch-geometric
pip install faiss-cpu  # or faiss-gpu
pip install h5py scikit-learn
```

### 2. Prepare Data
```python
# Feature extraction (run once)
python scripts/extract_features.py \
    --wsi_dir data/tcga/slides/ \
    --output_dir data/tcga/features/ \
    --encoder resnet50 \
    --batch_size 256
```

### 3. Train Model
```python
python scripts/train_v2_0.py \
    --data_dir data/tcga/features/ \
    --output_dir experiments/v2_0/run1/ \
    --num_epochs 100 \
    --batch_size 4 \
    --learning_rate 1e-4
```

---

## Training Script Usage

### Basic Training
```bash
python scripts/train_v2_0.py \
    --data_dir data/tcga/features/ \
    --output_dir experiments/v2_0/baseline/ \
    --config configs/transnnmil_v2_0.yaml
```

### Advanced Configuration
```bash
python scripts/train_v2_0.py \
    --data_dir data/tcga/features/ \
    --output_dir experiments/v2_0/ablation_gat/ \
    --num_epochs 100 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --num_regions 16 \
    --k_neighbors 8 \
    --gnn_type gat \
    --use_pruning false \
    --seed 42
```

### Command-Line Arguments
```
Data:
  --data_dir          Path to feature directory
  --split_file        Path to train/val/test split JSON
  --bag_length        Maximum patches per bag (default: 512)

Model:
  --feature_dim       Input feature dimension (default: 1024)
  --num_classes       Number of output classes (default: 2)
  --num_regions       Hierarchical regions (default: 16)
  --k_neighbors       k-NN graph neighbors (default: 8)
  --gnn_type          GNN architecture: gat/sage/gin (default: gat)
  --use_pruning       Enable adaptive pruning (default: false)
  --keep_ratio        Pruning keep ratio (default: 0.5)

Training:
  --num_epochs        Training epochs (default: 100)
  --batch_size        Batch size (default: 4)
  --learning_rate     Learning rate (default: 1e-4)
  --weight_decay      Weight decay (default: 1e-5)
  --warmup_epochs     LR warmup epochs (default: 5)
  --grad_clip         Gradient clipping (default: 1.0)

Optimization:
  --optimizer         Optimizer: adam/adamw/sgd (default: adamw)
  --scheduler         LR scheduler: cosine/step/plateau (default: cosine)
  --mixed_precision   Enable FP16 training (default: false)

Logging:
  --output_dir        Experiment output directory
  --log_interval      Log every N batches (default: 10)
  --val_interval      Validate every N epochs (default: 1)
  --save_interval     Save checkpoint every N epochs (default: 10)
  --wandb_project     Weights & Biases project name
  --wandb_run_name    W&B run name

System:
  --num_workers       DataLoader workers (default: 4)
  --seed              Random seed (default: 42)
  --device            Device: cuda/cpu (default: cuda)
```

---

## Configuration Files

### Example: `configs/transnnmil_v2_0.yaml`
```yaml
# Model Architecture
model:
  feature_dim: 1024
  num_classes: 2
  num_regions: 16
  k_neighbors: 8
  gnn_type: gat
  use_pruning: false
  dropout: 0.1

# Training
training:
  num_epochs: 100
  batch_size: 4
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  warmup_epochs: 5
  grad_clip: 1.0
  mixed_precision: false

# Optimizer
optimizer:
  type: adamw
  betas: [0.9, 0.999]
  eps: 1.0e-8

# Scheduler
scheduler:
  type: cosine
  T_max: 100
  eta_min: 1.0e-6

# Data
data:
  bag_length: 512
  num_workers: 4
  pin_memory: true

# Logging
logging:
  log_interval: 10
  val_interval: 1
  save_interval: 10
  wandb_project: transnnmil-v2
  wandb_run_name: baseline
```

---

## Data Format

### Feature Files (HDF5)
```
data/tcga/features/
├── slide_001.h5
│   ├── features  [N, D] float32
│   ├── coords    [N, 2] float32
│   └── label     scalar int
├── slide_002.h5
└── ...
```

### Split File (JSON)
```json
{
  "train": ["slide_001", "slide_002", ...],
  "val": ["slide_050", "slide_051", ...],
  "test": ["slide_100", "slide_101", ...]
}
```

---

## Training Pipeline

### 1. Data Loading
```python
from src.data.loaders import MILDataset, create_mil_dataloader

# Create dataset
train_dataset = MILDataset(
    data_dir='data/tcga/features/',
    split='train',
    bag_length=512
)

# Create dataloader
train_loader = create_mil_dataloader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)
```

### 2. Model Initialization
```python
from src.models.transnnmil_v2 import TransnnMILv2

model = TransnnMILv2(
    feature_dim=1024,
    num_classes=2,
    num_regions=16,
    k_neighbors=8,
    gnn_type='gat',
    use_pruning=False,
    dropout=0.1
)

model = model.cuda()
```

### 3. Optimizer & Scheduler
```python
import torch.optim as optim

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6
)
```

### 4. Training Loop
```python
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    
    for batch_idx, (features, coords, labels) in enumerate(train_loader):
        features = features.cuda()
        coords = coords.cuda()
        labels = labels.cuda()
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(features, coords)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
    
    # Validation
    val_auc = validate(model, val_loader)
    print(f'Epoch {epoch} Val AUC: {val_auc:.4f}')
    
    # Step scheduler
    scheduler.step()
    
    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_auc,
        }, f'checkpoint_epoch_{epoch}.pth')
```

### 5. Validation
```python
from sklearn.metrics import roc_auc_score

def validate(model, val_loader):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, coords, labels in val_loader:
            features = features.cuda()
            coords = coords.cuda()
            
            logits = model(features, coords)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            all_probs.append(probs.cpu())
            all_labels.append(labels)
    
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    auc = roc_auc_score(all_labels, all_probs)
    return auc
```

---

## Hyperparameter Tuning

### Grid Search
```python
# configs/grid_search.yaml
grid:
  num_regions: [8, 16, 32]
  k_neighbors: [4, 8, 16]
  learning_rate: [1e-5, 1e-4, 1e-3]
  gnn_type: [gat, sage, gin]
```

```bash
python scripts/grid_search.py \
    --config configs/grid_search.yaml \
    --data_dir data/tcga/features/ \
    --output_dir experiments/grid_search/
```

### Random Search
```bash
python scripts/random_search.py \
    --config configs/random_search.yaml \
    --data_dir data/tcga/features/ \
    --output_dir experiments/random_search/ \
    --num_trials 50
```

### Bayesian Optimization (Optuna)
```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    num_regions = trial.suggest_int('num_regions', 8, 32)
    k_neighbors = trial.suggest_int('k_neighbors', 4, 16)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    
    # Train model
    model = TransnnMILv2(num_regions=num_regions, k_neighbors=k_neighbors)
    val_auc = train_and_validate(model, lr)
    
    return val_auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f'Best AUC: {study.best_value:.4f}')
print(f'Best params: {study.best_params}')
```

---

## Multi-GPU Training

### DataParallel
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model = model.cuda()
```

### DistributedDataParallel
```bash
# Launch with torchrun
torchrun --nproc_per_node=4 scripts/train_v2_0_distributed.py \
    --data_dir data/tcga/features/ \
    --output_dir experiments/v2_0/distributed/
```

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Wrap model
model = TransnnMILv2(...)
model = model.cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler
from torch.utils.data.distributed import DistributedSampler
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, ...)
```

---

## Mixed Precision Training

### Automatic Mixed Precision (AMP)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for features, coords, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            logits = model(features, coords)
            loss = criterion(logits, labels)
        
        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
```

**Benefits**:
- 2x faster training
- 30-40% less GPU memory
- Minimal accuracy loss (<0.5% AUC)

---

## Monitoring & Logging

### TensorBoard
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/v2_0_baseline')

# Log scalars
writer.add_scalar('Loss/train', loss.item(), global_step)
writer.add_scalar('AUC/val', val_auc, epoch)

# Log histograms
writer.add_histogram('Attention/weights', attention_weights, epoch)

# Log images
writer.add_image('Regions/visualization', region_viz, epoch)

writer.close()
```

### Weights & Biases
```python
import wandb

wandb.init(
    project='transnnmil-v2',
    name='baseline',
    config={
        'num_regions': 16,
        'k_neighbors': 8,
        'learning_rate': 1e-4,
    }
)

# Log metrics
wandb.log({
    'train/loss': loss.item(),
    'val/auc': val_auc,
    'epoch': epoch
})

# Log model
wandb.watch(model, log='all', log_freq=100)

wandb.finish()
```

---

## Checkpointing

### Save Checkpoint
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'val_auc': val_auc,
    'config': config,
}

torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
```

### Load Checkpoint
```python
checkpoint = torch.load('checkpoint_epoch_50.pth')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Resume Training
```bash
python scripts/train_v2_0.py \
    --data_dir data/tcga/features/ \
    --output_dir experiments/v2_0/resume/ \
    --resume_from checkpoint_epoch_50.pth
```

---

## Troubleshooting

### Out of Memory (OOM)
**Solutions**:
1. Reduce batch size: `--batch_size 2`
2. Reduce bag length: `--bag_length 256`
3. Enable mixed precision: `--mixed_precision true`
4. Reduce num_regions: `--num_regions 8`
5. Use gradient checkpointing

### Slow Training
**Solutions**:
1. Increase num_workers: `--num_workers 8`
2. Enable pin_memory: `--pin_memory true`
3. Use mixed precision
4. Profile with PyTorch Profiler

### Poor Convergence
**Solutions**:
1. Increase warmup epochs: `--warmup_epochs 10`
2. Reduce learning rate: `--learning_rate 5e-5`
3. Increase weight decay: `--weight_decay 1e-4`
4. Check data quality (class balance, feature normalization)

### NaN Loss
**Solutions**:
1. Reduce learning rate
2. Enable gradient clipping: `--grad_clip 1.0`
3. Check for inf/nan in input data
4. Use mixed precision with loss scaling

---

## Best Practices

### 1. Data Preprocessing
- Normalize features (mean=0, std=1)
- Balance classes (weighted sampling or loss)
- Augment coordinates (random jitter)

### 2. Training
- Use warmup for first 5-10 epochs
- Monitor validation AUC, not just loss
- Save checkpoints every 10 epochs
- Use early stopping (patience=20)

### 3. Hyperparameters
- Start with defaults, then tune
- Tune learning rate first
- Then tune architecture (num_regions, k_neighbors)
- Finally tune regularization (dropout, weight_decay)

### 4. Evaluation
- Use 5-fold cross-validation
- Report mean ± std across folds
- Test on held-out test set
- Visualize predictions (attention, regions, graphs)

---

## Example Workflows

### Workflow 1: Quick Experiment
```bash
# Train with defaults
python scripts/train_v2_0.py \
    --data_dir data/tcga/features/ \
    --output_dir experiments/quick_test/ \
    --num_epochs 20 \
    --batch_size 4
```

### Workflow 2: Full Training
```bash
# Train with best hyperparameters
python scripts/train_v2_0.py \
    --config configs/best_config.yaml \
    --data_dir data/tcga/features/ \
    --output_dir experiments/final_model/ \
    --num_epochs 100 \
    --wandb_project transnnmil-v2 \
    --wandb_run_name final_model
```

### Workflow 3: Ablation Study
```bash
# Train 2-branch variants
for branches in AB AC BC; do
    python scripts/train_v2_0.py \
        --data_dir data/tcga/features/ \
        --output_dir experiments/ablation_${branches}/ \
        --branches ${branches} \
        --num_epochs 100
done
```

---

## See Also

- [Architecture Documentation](TRANSNNMIL_V2_ARCHITECTURE.md)
- [API Reference](TRANSNNMIL_V2_API.md)
- [Visualization Guide](TRANSNNMIL_V2_VISUALIZATION.md)
- [Ablation Results](../experiments/v2_0/ablation_results.md)
