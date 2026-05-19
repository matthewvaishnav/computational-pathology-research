# TransnnMIL v2.0: API Reference

## Models

### TransnnMILv2

Three-branch architecture combining TransMIL, hierarchical pooling, and topology branch.

```python
from src.models.transnnmil_v2 import TransnnMILv2

model = TransnnMILv2(
    feature_dim=1024,
    num_classes=2,
    num_regions=16,
    k_neighbors=8,
    gnn_type='gat',
    use_pruning=False,
    keep_ratio=0.5,
    dropout=0.1
)
```

**Parameters**:
- `feature_dim` (int): Input feature dimension
- `num_classes` (int): Number of output classes
- `num_regions` (int, default=16): Number of spatial regions for hierarchical pooling
- `k_neighbors` (int, default=8): Number of neighbors for k-NN graph
- `gnn_type` (str, default='gat'): GNN architecture ('gat', 'sage', 'gin')
- `use_pruning` (bool, default=False): Enable adaptive pruning
- `keep_ratio` (float, default=0.5): Fraction of patches to keep when pruning
- `dropout` (float, default=0.1): Dropout rate

**Methods**:

#### `forward(features, coords, mask=None)`
Forward pass through the model.

**Args**:
- `features` (Tensor): Patch features [batch_size, num_patches, feature_dim]
- `coords` (Tensor): Patch coordinates [batch_size, num_patches, 2]
- `mask` (Tensor, optional): Binary mask [batch_size, num_patches]

**Returns**:
- `logits` (Tensor): Class logits [batch_size, num_classes]

**Example**:
```python
import torch

features = torch.randn(4, 512, 1024)
coords = torch.rand(4, 512, 2)
mask = torch.ones(4, 512, dtype=torch.bool)

logits = model(features, coords, mask)
# logits.shape: [4, 2]
```

---

### TransnnMILv2TwoBranch

Two-branch variant for ablation studies.

```python
from src.models.transnnmil_v2 import TransnnMILv2TwoBranch

model = TransnnMILv2TwoBranch(
    feature_dim=1024,
    num_classes=2,
    branches='AB',  # 'AB', 'AC', or 'BC'
    num_regions=16,
    k_neighbors=8,
    gnn_type='gat',
    dropout=0.1
)
```

**Parameters**:
- `feature_dim` (int): Input feature dimension
- `num_classes` (int): Number of output classes
- `branches` (str): Branch combination ('AB', 'AC', 'BC')
  - 'AB': TransMIL + Hierarchical
  - 'AC': TransMIL + Topology
  - 'BC': Hierarchical + Topology
- `num_regions` (int, default=16): Number of spatial regions
- `k_neighbors` (int, default=8): Number of k-NN neighbors
- `gnn_type` (str, default='gat'): GNN architecture
- `dropout` (float, default=0.1): Dropout rate

**Methods**:

#### `forward(features, coords, mask=None)`
Same as TransnnMILv2.

---

## Components

### AdaptivePruning

Token sparsification module for efficiency.

```python
from src.models.adaptive_pruning import AdaptivePruning

pruning = AdaptivePruning(
    feature_dim=1024,
    keep_ratio=0.5,
    scoring_method='learned',
    dropout=0.1
)
```

**Parameters**:
- `feature_dim` (int): Input feature dimension
- `keep_ratio` (float, default=0.5): Fraction of patches to keep
- `scoring_method` (str, default='learned'): Scoring method ('learned', 'attention', 'confidence')
- `dropout` (float, default=0.1): Dropout rate

**Methods**:

#### `forward(features, mask=None, return_info=False)`
Prune uninformative patches.

**Args**:
- `features` (Tensor): Input features [B, N, D]
- `mask` (Tensor, optional): Binary mask [B, N]
- `return_info` (bool, default=False): Return pruning info

**Returns**:
- `pruned_features` (Tensor): Pruned features [B, N', D]
- `pruned_mask` (Tensor): Updated mask [B, N']
- `info` (dict, optional): Pruning statistics

**Example**:
```python
features = torch.randn(4, 512, 1024)
pruned_features, pruned_mask, info = pruning(features, return_info=True)

print(f"Original: {features.shape[1]} patches")
print(f"Pruned: {pruned_features.shape[1]} patches")
print(f"Speedup: {info['speedup']:.2f}x")
```

---

### HierarchicalPooling

Spatial clustering with region-level processing.

```python
from src.models.hierarchical_pooling import HierarchicalPooling

hierarchical = HierarchicalPooling(
    num_clusters=16,
    temperature=1.0,
    clustering_method='learnable',
    init_method='uniform'
)
```

**Parameters**:
- `num_clusters` (int): Number of spatial regions
- `temperature` (float, default=1.0): Softmax temperature
- `clustering_method` (str, default='learnable'): Clustering method ('learnable', 'kmeans', 'grid')
- `init_method` (str, default='uniform'): Initialization method ('uniform', 'random')

**Methods**:

#### `forward(coords, mask=None)`
Compute soft region assignments.

**Args**:
- `coords` (Tensor): Patch coordinates [B, N, 2]
- `mask` (Tensor, optional): Binary mask [B, N]

**Returns**:
- `assignments` (Tensor): Soft assignments [B, N, R]

#### `get_centers()`
Get cluster centers.

**Returns**:
- `centers` (Tensor): Cluster centers [R, 2]

**Example**:
```python
coords = torch.rand(4, 512, 2)
assignments = hierarchical(coords)

# Visualize regions
import matplotlib.pyplot as plt
region_ids = assignments[0].argmax(dim=1)
plt.scatter(coords[0, :, 0], coords[0, :, 1], c=region_ids)
plt.show()
```

---

### RegionAttentionPooling

Attention-based aggregation within regions.

```python
from src.models.hierarchical_pooling import RegionAttentionPooling

pooling = RegionAttentionPooling(
    feature_dim=1024,
    hidden_dim=512,
    dropout=0.1
)
```

**Parameters**:
- `feature_dim` (int): Input feature dimension
- `hidden_dim` (int, default=512): Hidden dimension
- `dropout` (float, default=0.1): Dropout rate

**Methods**:

#### `forward(features, assignments, mask=None)`
Pool features within regions.

**Args**:
- `features` (Tensor): Patch features [B, N, D]
- `assignments` (Tensor): Soft assignments [B, N, R]
- `mask` (Tensor, optional): Binary mask [B, N]

**Returns**:
- `region_features` (Tensor): Region features [B, R, D]

**Example**:
```python
features = torch.randn(4, 512, 1024)
assignments = torch.softmax(torch.randn(4, 512, 16), dim=2)

region_features = pooling(features, assignments)
# region_features.shape: [4, 16, 1024]
```

---

### TopologyBranch

k-NN graph construction with GNN processing.

```python
from src.models.topology_branch import TopologyBranch

topology = TopologyBranch(
    feature_dim=1024,
    hidden_dim=512,
    num_layers=2,
    k_neighbors=8,
    gnn_type='gat',
    pooling='attention',
    dropout=0.1
)
```

**Parameters**:
- `feature_dim` (int): Input feature dimension
- `hidden_dim` (int, default=512): Hidden dimension
- `num_layers` (int, default=2): Number of GNN layers
- `k_neighbors` (int, default=8): Number of k-NN neighbors
- `gnn_type` (str, default='gat'): GNN architecture ('gat', 'sage', 'gin')
- `pooling` (str, default='attention'): Global pooling method ('attention', 'mean', 'max')
- `dropout` (float, default=0.1): Dropout rate

**Methods**:

#### `forward(features, coords, mask=None)`
Process features through graph network.

**Args**:
- `features` (Tensor): Patch features [B, N, D]
- `coords` (Tensor): Patch coordinates [B, N, 2]
- `mask` (Tensor, optional): Binary mask [B, N]

**Returns**:
- `graph_features` (Tensor): Global graph features [B, hidden_dim]

**Example**:
```python
features = torch.randn(4, 512, 1024)
coords = torch.rand(4, 512, 2)

graph_features = topology(features, coords)
# graph_features.shape: [4, 512]
```

---

### GraphCache

Precomputed k-NN graph cache for efficiency.

```python
from src.models.graph_cache import GraphCache

cache = GraphCache(
    cache_dir='data/graphs/',
    k_neighbors=8,
    use_faiss=True
)
```

**Parameters**:
- `cache_dir` (str): Directory for cached graphs
- `k_neighbors` (int, default=8): Number of neighbors
- `use_faiss` (bool, default=True): Use FAISS for approximate k-NN

**Methods**:

#### `build_and_cache(slide_id, coords)`
Build and cache k-NN graph.

**Args**:
- `slide_id` (str): Slide identifier
- `coords` (Tensor): Patch coordinates [N, 2]

**Returns**:
- `edge_index` (Tensor): Edge indices [2, E]

#### `load(slide_id)`
Load cached graph.

**Args**:
- `slide_id` (str): Slide identifier

**Returns**:
- `edge_index` (Tensor): Edge indices [2, E]

**Example**:
```python
# Build and cache
coords = torch.rand(1000, 2)
edge_index = cache.build_and_cache('slide_001', coords)

# Load from cache
edge_index = cache.load('slide_001')
```

---

## Data

### MILDataset

Dataset for multiple instance learning.

```python
from src.data.loaders import MILDataset

dataset = MILDataset(
    data_dir='data/tcga/features/',
    split='train',
    bag_length=512,
    augment=True
)
```

**Parameters**:
- `data_dir` (str): Directory containing feature files
- `split` (str): Data split ('train', 'val', 'test')
- `bag_length` (int, default=512): Maximum patches per bag
- `augment` (bool, default=False): Enable data augmentation

**Methods**:

#### `__getitem__(idx)`
Get a single sample.

**Returns**:
- `features` (Tensor): Patch features [N, D]
- `coords` (Tensor): Patch coordinates [N, 2]
- `label` (int): Slide-level label

**Example**:
```python
features, coords, label = dataset[0]
print(f"Features: {features.shape}")
print(f"Coords: {coords.shape}")
print(f"Label: {label}")
```

---

### create_mil_dataloader

Create DataLoader with proper collation.

```python
from src.data.loaders import create_mil_dataloader

loader = create_mil_dataloader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

**Parameters**:
- `dataset` (Dataset): MIL dataset
- `batch_size` (int): Batch size
- `shuffle` (bool, default=False): Shuffle data
- `num_workers` (int, default=0): DataLoader workers
- `pin_memory` (bool, default=False): Pin memory for GPU

**Returns**:
- `loader` (DataLoader): Configured DataLoader

**Example**:
```python
for features, coords, labels in loader:
    # features: [B, N, D]
    # coords: [B, N, 2]
    # labels: [B]
    logits = model(features, coords)
    loss = criterion(logits, labels)
```

---

## Utilities

### extract_features

Extract patch features from WSI.

```python
from src.data.wsi_pipeline import extract_features

features, coords = extract_features(
    wsi_path='data/slides/slide_001.svs',
    encoder='resnet50',
    patch_size=256,
    stride=256,
    batch_size=256,
    device='cuda'
)
```

**Parameters**:
- `wsi_path` (str): Path to WSI file
- `encoder` (str): Feature encoder ('resnet50', 'phikon', 'uni')
- `patch_size` (int, default=256): Patch size in pixels
- `stride` (int, default=256): Stride between patches
- `batch_size` (int, default=256): Batch size for extraction
- `device` (str, default='cuda'): Device for extraction

**Returns**:
- `features` (Tensor): Patch features [N, D]
- `coords` (Tensor): Patch coordinates [N, 2]

---

### visualize_regions

Visualize hierarchical regions.

```python
from scripts.visualize_hierarchical import visualize_regions

fig = visualize_regions(
    coords=coords,
    assignments=assignments,
    centers=centers,
    title='Hierarchical Regions'
)
fig.savefig('regions.png')
```

**Parameters**:
- `coords` (Tensor): Patch coordinates [N, 2]
- `assignments` (Tensor): Soft assignments [N, R]
- `centers` (Tensor): Cluster centers [R, 2]
- `title` (str, optional): Plot title

**Returns**:
- `fig` (Figure): Matplotlib figure

---

### visualize_graph

Visualize k-NN graph.

```python
from scripts.visualize_graph import visualize_graph

fig = visualize_graph(
    coords=coords,
    edge_index=edge_index,
    node_colors=attention_weights,
    title='k-NN Graph (k=8)'
)
fig.savefig('graph.png')
```

**Parameters**:
- `coords` (Tensor): Patch coordinates [N, 2]
- `edge_index` (Tensor): Edge indices [2, E]
- `node_colors` (Tensor, optional): Node colors [N]
- `title` (str, optional): Plot title

**Returns**:
- `fig` (Figure): Matplotlib figure

---

## Training

### Trainer

High-level training interface.

```python
from src.training.unified_trainer import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    device='cuda',
    output_dir='experiments/run1/'
)

trainer.train(num_epochs=100)
```

**Parameters**:
- `model` (nn.Module): Model to train
- `train_loader` (DataLoader): Training data
- `val_loader` (DataLoader): Validation data
- `optimizer` (Optimizer): Optimizer
- `scheduler` (Scheduler, optional): LR scheduler
- `criterion` (nn.Module): Loss function
- `device` (str, default='cuda'): Device
- `output_dir` (str): Output directory

**Methods**:

#### `train(num_epochs)`
Train for specified epochs.

#### `validate()`
Run validation.

**Returns**:
- `metrics` (dict): Validation metrics

#### `save_checkpoint(epoch, metrics)`
Save model checkpoint.

---

## Metrics

### compute_metrics

Compute evaluation metrics.

```python
from src.utils.metrics import compute_metrics

metrics = compute_metrics(
    y_true=labels,
    y_pred=predictions,
    y_prob=probabilities
)

print(f"AUC: {metrics['auc']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
```

**Parameters**:
- `y_true` (array): True labels
- `y_pred` (array): Predicted labels
- `y_prob` (array): Predicted probabilities

**Returns**:
- `metrics` (dict): Dictionary of metrics
  - 'auc': Area under ROC curve
  - 'accuracy': Classification accuracy
  - 'precision': Precision score
  - 'recall': Recall score
  - 'f1': F1 score

---

## Configuration

### load_config

Load configuration from YAML.

```python
from src.config.experiment_config import load_config

config = load_config('configs/transnnmil_v2_0.yaml')

model = TransnnMILv2(**config['model'])
```

**Parameters**:
- `config_path` (str): Path to YAML config

**Returns**:
- `config` (dict): Configuration dictionary

---

## See Also

- [Architecture Documentation](TRANSNNMIL_V2_ARCHITECTURE.md)
- [Training Guide](TRANSNNMIL_V2_TRAINING.md)
- [Visualization Examples](../notebooks/transnnmil_v2_visualization.ipynb)
