# Design Document: PatchCamelyon Experiment

## Overview

This design specifies the implementation of an end-to-end experiment using the PatchCamelyon (PCam) dataset to demonstrate the computational pathology framework on real histopathology data. The experiment will train a binary classification model to distinguish metastatic tissue from normal tissue in 96x96 pixel lymph node patches.

The implementation leverages the existing multimodal fusion framework in single-modality mode, using only the WSI encoder path. This design focuses on minimal modifications to the existing codebase while adding new components for dataset handling, experiment configuration, and result visualization.

### Key Design Decisions

1. **Single-Modality Architecture**: Use only the WSI encoder path, bypassing genomic and clinical encoders
2. **Minimal Framework Changes**: Extend existing components rather than modifying core architecture
3. **Standard PyTorch Dataset**: Implement PCam as a standard Dataset class compatible with existing data loaders
4. **Configuration-Based**: Use YAML configuration following existing patterns in `experiments/configs/`
5. **Reuse Training Infrastructure**: Leverage existing `experiments/train.py` with minor adaptations
6. **Standalone Evaluation**: Create dedicated evaluation script for PCam-specific metrics and visualizations

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    PatchCamelyon Dataset                     │
│                    96x96 RGB Images                          │
│                    Binary Labels (0/1)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  PCamDataset (PyTorch)                       │
│  - Download from TensorFlow Datasets                         │
│  - Normalize to [0, 1]                                       │
│  - Apply augmentations                                       │
│  - Return dict: {'wsi_features': tensor, 'label': tensor}   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Extraction (ResNet-18)                  │
│  - Pretrained on ImageNet                                    │
│  - Extract features before final FC layer                    │
│  - Output: [batch, 512] feature vectors                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    WSI Encoder                               │
│  - Input: [batch, 1, 512] (single patch)                    │
│  - Simplified: no attention pooling needed                   │
│  - Output: [batch, embed_dim]                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Classification Head                             │
│  - Input: [batch, embed_dim]                                 │
│  - Output: [batch, 2] logits                                 │
│  - Binary classification                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                    Predictions
```

### Data Flow

1. **Raw Images** → PCamDataset loads 96x96 RGB images
2. **Preprocessing** → Normalize, augment, convert to tensor
3. **Feature Extraction** → ResNet-18 extracts 512-dim features
4. **Encoding** → WSI encoder processes single-patch sequence
5. **Classification** → Binary classification head produces logits
6. **Loss** → Binary cross-entropy with logits
7. **Optimization** → AdamW optimizer updates weights

## Components and Interfaces

### 1. PCamDataset Class

**Location**: `src/data/pcam_dataset.py`

**Purpose**: PyTorch Dataset for loading and preprocessing PatchCamelyon data

**Interface**:
```python
class PCamDataset(Dataset):
    """
    PatchCamelyon dataset for binary classification.
    
    Args:
        root_dir: Directory to download/store dataset
        split: One of 'train', 'val', 'test'
        transform: Optional torchvision transforms
        download: Whether to download if not present
        feature_extractor: Optional pretrained model for feature extraction
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        download: bool = True,
        feature_extractor: Optional[nn.Module] = None
    ):
        ...
    
    def __len__(self) -> int:
        """Returns number of samples in split."""
        ...
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'wsi_features': Tensor [1, feature_dim] or [3, 96, 96],
                'label': Tensor (scalar),
                'image_id': str
            }
        """
        ...
    
    def download(self):
        """Download PCam dataset from TensorFlow Datasets."""
        ...
```

**Key Methods**:
- `download()`: Downloads PCam from TensorFlow Datasets API
- `_load_split()`: Loads train/val/test split into memory or creates file index
- `_apply_transforms()`: Applies normalization and augmentation
- `_extract_features()`: Optionally extracts features using ResNet-18

### 2. Feature Extractor

**Location**: `src/models/feature_extractors.py`

**Purpose**: Extract features from raw images using pretrained CNN

**Interface**:
```python
class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based feature extractor for histopathology patches.
    
    Args:
        model_name: ResNet variant ('resnet18', 'resnet50')
        pretrained: Whether to use ImageNet pretrained weights
        feature_dim: Output feature dimension
    """
    
    def __init__(
        self,
        model_name: str = 'resnet18',
        pretrained: bool = True,
        feature_dim: int = 512
    ):
        ...
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            images: [batch, 3, H, W]
        
        Returns:
            features: [batch, feature_dim]
        """
        ...
```

### 3. Training Script Adapter

**Location**: `experiments/train_pcam.py`

**Purpose**: Training script specifically for PCam experiment

**Key Functions**:
```python
def create_pcam_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders for PCam."""
    ...

def create_single_modality_model(config: Dict) -> Tuple[nn.Module, nn.Module]:
    """
    Create WSI encoder and classification head.
    
    Returns:
        encoder: WSI encoder module
        head: Classification head module
    """
    ...

def train_epoch(
    encoder: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    config: Dict
) -> Dict[str, float]:
    """Train for one epoch."""
    ...

def validate(
    encoder: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """Validate model."""
    ...

def main():
    """Main training loop."""
    ...
```

### 4. Evaluation Script

**Location**: `experiments/evaluate_pcam.py`

**Purpose**: Comprehensive evaluation and metrics computation

**Key Functions**:
```python
def load_checkpoint(checkpoint_path: str, device: str) -> Tuple[nn.Module, nn.Module, Dict]:
    """Load trained model from checkpoint."""
    ...

def evaluate_model(
    encoder: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    device: str
) -> Dict[str, Any]:
    """
    Run inference and compute metrics.
    
    Returns:
        {
            'accuracy': float,
            'auc': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'confusion_matrix': np.ndarray,
            'predictions': List[int],
            'probabilities': List[float],
            'labels': List[int]
        }
    """
    ...

def compute_metrics(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """Compute classification metrics."""
    ...

def save_metrics(metrics: Dict, output_path: str):
    """Save metrics to JSON file."""
    ...

def main():
    """Main evaluation function."""
    ...
```

### 5. Visualization Notebook

**Location**: `experiments/notebooks/pcam_visualization.ipynb`

**Purpose**: Generate plots and visualizations

**Sections**:
1. Dataset Exploration
   - Sample image grid with labels
   - Class distribution
   - Image statistics

2. Training Curves
   - Loss curves (train/val)
   - Accuracy curves (train/val)
   - Learning rate schedule

3. Model Performance
   - Confusion matrix heatmap
   - ROC curve with AUC
   - Precision-recall curve

4. Prediction Analysis
   - Correct predictions examples
   - Incorrect predictions examples
   - Confidence distribution

5. Attention Visualization (if applicable)
   - Attention weights overlay
   - High-attention regions

## Data Models

### Dataset Structure

```
data/
└── pcam/
    ├── train/
    │   ├── images.npy          # [N, 96, 96, 3]
    │   └── labels.npy          # [N]
    ├── val/
    │   ├── images.npy
    │   └── labels.npy
    ├── test/
    │   ├── images.npy
    │   └── labels.npy
    └── metadata.json           # Dataset statistics
```

### Configuration Schema

```yaml
# experiments/configs/pcam.yaml

experiment:
  name: patchcamelyon
  description: Binary classification on PatchCamelyon dataset
  tags: [pcam, single-modality, baseline]

data:
  dataset: pcam
  root_dir: ./data/pcam
  download: true
  num_workers: 4
  pin_memory: true
  
  # Feature extraction
  extract_features: true
  feature_extractor:
    model: resnet18
    pretrained: true
    feature_dim: 512
  
  # Augmentation
  augmentation:
    enabled: true
    random_horizontal_flip: true
    random_vertical_flip: true
    color_jitter:
      brightness: 0.1
      contrast: 0.1
      saturation: 0.1
      hue: 0.05

model:
  # Single modality mode
  modalities: [wsi]
  embed_dim: 256
  
  wsi:
    input_dim: 512  # ResNet-18 feature dim
    hidden_dim: 256
    num_heads: 4
    num_layers: 1
    pooling: mean  # Simple pooling for single patch

task:
  type: classification
  num_classes: 2
  
  classification:
    hidden_dims: [128]
    dropout: 0.3

training:
  num_epochs: 20
  batch_size: 128
  learning_rate: 1e-3
  weight_decay: 1e-4
  
  optimizer:
    name: adamw
    betas: [0.9, 0.999]
  
  scheduler:
    name: cosine
    min_lr: 1e-6
    warmup_epochs: 2
  
  max_grad_norm: 1.0
  use_amp: true  # Mixed precision for faster training

validation:
  interval: 1
  metric: val_auc
  maximize: true

checkpoint:
  save_interval: 5
  save_best: true
  checkpoint_dir: ./checkpoints/pcam

early_stopping:
  enabled: true
  patience: 5
  min_delta: 0.001

logging:
  log_dir: ./logs/pcam
  log_interval: 50
  use_tensorboard: true

evaluation:
  generate_plots: true
  output_dir: ./results/pcam

seed: 42
device: cuda
```

### Checkpoint Format

```python
{
    'epoch': int,
    'global_step': int,
    'encoder_state_dict': OrderedDict,
    'head_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'scheduler_state_dict': OrderedDict,
    'metrics': {
        'train_loss': float,
        'train_accuracy': float,
        'val_loss': float,
        'val_accuracy': float,
        'val_auc': float
    },
    'config': Dict
}
```

### Metrics Output Format

```json
{
  "test_accuracy": 0.8542,
  "test_auc": 0.9123,
  "test_precision": 0.8456,
  "test_recall": 0.8621,
  "test_f1": 0.8537,
  "confusion_matrix": [[16234, 2766], [2311, 14689]],
  "per_class_metrics": {
    "class_0": {
      "precision": 0.8754,
      "recall": 0.8544,
      "f1": 0.8648
    },
    "class_1": {
      "precision": 0.8415,
      "recall": 0.8641,
      "f1": 0.8526
    }
  },
  "training_time_seconds": 1247.5,
  "inference_time_seconds": 23.4,
  "model_parameters": 1234567,
  "hardware": "NVIDIA RTX 3080",
  "pytorch_version": "2.0.1",
  "cuda_version": "11.8"
}
```

## Error Handling

### Dataset Download Errors

```python
class PCamDataset:
    def download(self):
        try:
            import tensorflow_datasets as tfds
            # Download logic
        except ImportError:
            raise ImportError(
                "tensorflow_datasets is required for PCam download. "
                "Install with: pip install tensorflow-datasets"
            )
        except Exception as e:
            logger.error(f"Failed to download PCam dataset: {e}")
            logger.info(
                "Troubleshooting steps:\n"
                "1. Check internet connection\n"
                "2. Verify disk space (need ~2GB)\n"
                "3. Try manual download from: "
                "https://github.com/basveeling/pcam\n"
                "4. Place files in: {self.root_dir}"
            )
            raise
```

### GPU Memory Errors

```python
def train_epoch(...):
    try:
        # Training loop
        ...
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(
                "GPU out of memory. Suggestions:\n"
                f"1. Reduce batch_size (current: {config['batch_size']})\n"
                "2. Reduce model size (embed_dim, hidden_dim)\n"
                "3. Disable mixed precision (use_amp: false)\n"
                "4. Use gradient accumulation"
            )
            # Attempt automatic batch size reduction
            if config.get('auto_batch_size', True):
                new_batch_size = config['batch_size'] // 2
                logger.warning(f"Automatically reducing batch_size to {new_batch_size}")
                config['batch_size'] = new_batch_size
                return train_epoch(...)  # Retry
        raise
```

### Checkpoint Loading Errors

```python
def load_checkpoint(checkpoint_path: str, device: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please provide a valid checkpoint path using --checkpoint argument"
        )
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint: {e}\n"
            "The checkpoint file may be corrupted. "
            "Try using a different checkpoint or retrain the model."
        )
    
    # Validate checkpoint structure
    required_keys = ['encoder_state_dict', 'head_state_dict', 'config']
    missing_keys = [k for k in required_keys if k not in checkpoint]
    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required keys: {missing_keys}\n"
            "This checkpoint may be from an incompatible version."
        )
    
    return checkpoint
```

### NaN Loss Detection

```python
def train_epoch(...):
    for batch_idx, batch in enumerate(dataloader):
        # Forward pass
        loss = criterion(logits, labels)
        
        # Check for NaN
        if torch.isnan(loss):
            logger.error(
                f"NaN loss detected at batch {batch_idx}\n"
                "Possible causes:\n"
                "1. Learning rate too high\n"
                "2. Gradient explosion\n"
                "3. Invalid input data\n"
                "4. Numerical instability\n"
                "\nSuggestions:\n"
                "1. Reduce learning_rate\n"
                "2. Enable gradient clipping\n"
                "3. Check input data for NaN/Inf values\n"
                "4. Use mixed precision training"
            )
            raise ValueError("NaN loss detected, terminating training")
        
        # Backward pass
        ...
```

### Missing Data Validation

```python
def validate_dataset(dataset: PCamDataset):
    """Validate dataset integrity before training."""
    logger.info("Validating dataset...")
    
    # Check dataset size
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    
    # Sample a few items
    for i in range(min(10, len(dataset))):
        try:
            sample = dataset[i]
        except Exception as e:
            logger.warning(f"Failed to load sample {i}: {e}")
            continue
        
        # Validate shapes
        if sample['wsi_features'].shape != (1, 512):
            raise ValueError(
                f"Invalid feature shape: {sample['wsi_features'].shape}, "
                "expected (1, 512)"
            )
        
        # Validate labels
        if sample['label'] not in [0, 1]:
            raise ValueError(f"Invalid label: {sample['label']}, expected 0 or 1")
    
    logger.info(f"Dataset validation passed: {len(dataset)} samples")
```

## Testing Strategy

This feature does not require property-based testing as it is primarily an integration of existing components for a specific dataset. The testing strategy focuses on:

### Unit Tests

1. **PCamDataset Tests** (`tests/test_pcam_dataset.py`)
   - Test dataset initialization
   - Test download functionality (mocked)
   - Test data loading and shapes
   - Test augmentation application
   - Test feature extraction
   - Test error handling for missing files

2. **Feature Extractor Tests** (`tests/test_feature_extractors.py`)
   - Test ResNet initialization
   - Test forward pass with various input sizes
   - Test feature dimension output
   - Test pretrained vs random initialization

3. **Training Script Tests** (`tests/test_train_pcam.py`)
   - Test dataloader creation
   - Test model initialization
   - Test single training step
   - Test checkpoint saving/loading
   - Test configuration validation

4. **Evaluation Script Tests** (`tests/test_evaluate_pcam.py`)
   - Test metrics computation
   - Test confusion matrix generation
   - Test checkpoint loading
   - Test output file creation

### Integration Tests

1. **End-to-End Training Test**
   - Run 1 epoch on small subset (100 samples)
   - Verify loss decreases
   - Verify checkpoint is saved
   - Verify logs are created

2. **End-to-End Evaluation Test**
   - Load checkpoint from training test
   - Run evaluation on test subset
   - Verify metrics are computed
   - Verify output files are created

3. **Visualization Test**
   - Run notebook cells programmatically
   - Verify plots are generated
   - Verify no errors in visualization code

### Manual Testing Checklist

- [ ] Download PCam dataset successfully
- [ ] Train for 1 epoch without errors
- [ ] Achieve >60% accuracy after 5 epochs
- [ ] Generate all visualizations
- [ ] Verify checkpoint can be loaded
- [ ] Run evaluation script successfully
- [ ] Verify metrics JSON is valid
- [ ] Test with different batch sizes
- [ ] Test with/without GPU
- [ ] Test resume from checkpoint

### Performance Benchmarks

- Training time: <30 minutes per epoch on RTX 3080
- Inference time: <1 minute for full test set
- Memory usage: <4GB GPU memory with batch_size=128
- Disk space: ~2GB for dataset, ~500MB for checkpoints

## Implementation Plan

### Phase 1: Dataset Implementation
1. Create `src/data/pcam_dataset.py`
2. Implement download functionality
3. Implement data loading and preprocessing
4. Add unit tests
5. Verify dataset loads correctly

### Phase 2: Feature Extraction
1. Create `src/models/feature_extractors.py`
2. Implement ResNet-based extractor
3. Add unit tests
4. Verify feature extraction works

### Phase 3: Training Script
1. Create `experiments/train_pcam.py`
2. Implement training loop
3. Add configuration file `experiments/configs/pcam.yaml`
4. Test training for 1 epoch
5. Verify checkpoints are saved

### Phase 4: Evaluation Script
1. Create `experiments/evaluate_pcam.py`
2. Implement metrics computation
3. Add visualization generation
4. Test evaluation on trained model
5. Verify metrics meet baseline (>60% accuracy)

### Phase 5: Visualization
1. Create `experiments/notebooks/pcam_visualization.ipynb`
2. Implement all visualization sections
3. Generate sample plots
4. Save plots to results directory

### Phase 6: Documentation
1. Update README with experiment section
2. Document training commands
3. Document evaluation commands
4. Add example results
5. Create troubleshooting guide

### Phase 7: Testing and Validation
1. Run full integration test
2. Verify reproducibility with fixed seed
3. Test on different hardware
4. Validate all error handling paths
5. Final code review

## Dependencies

### New Dependencies
```
tensorflow-datasets>=4.9.0  # For PCam download
matplotlib>=3.5.0           # For visualizations
seaborn>=0.12.0            # For heatmaps
scikit-learn>=1.0.0        # For metrics (already present)
jupyter>=1.0.0             # For notebooks
```

### Existing Dependencies (No Changes)
- torch>=2.0.0
- torchvision>=0.15.0
- numpy>=1.21.0
- tqdm>=4.62.0
- pyyaml>=6.0
- tensorboard>=2.10.0

## Deployment Considerations

### Hardware Requirements
- **Minimum**: CPU with 8GB RAM, 10GB disk space
- **Recommended**: GPU with 6GB VRAM, 16GB RAM, 20GB disk space
- **Optimal**: GPU with 8GB+ VRAM, 32GB RAM, 50GB disk space

### Environment Setup
```bash
# Create conda environment
conda create -n pcam python=3.9
conda activate pcam

# Install dependencies
pip install -r requirements.txt

# Download dataset (automatic on first run)
python experiments/train_pcam.py --config experiments/configs/pcam.yaml
```

### Training Time Estimates
- **1 epoch**: ~20-30 minutes (RTX 3080)
- **Full training (20 epochs)**: ~6-10 hours
- **Evaluation**: ~1-2 minutes

### Storage Requirements
- Dataset: ~2GB
- Checkpoints: ~500MB (5 checkpoints)
- Logs: ~100MB
- Results: ~50MB
- Total: ~3GB

## Future Enhancements

1. **Multi-Scale Features**: Extract features at multiple resolutions
2. **Ensemble Models**: Train multiple models and ensemble predictions
3. **Active Learning**: Identify uncertain samples for annotation
4. **Explainability**: Add GradCAM or attention visualization
5. **Transfer Learning**: Fine-tune on other histopathology datasets
6. **Data Efficiency**: Experiment with few-shot learning
7. **Uncertainty Quantification**: Add Bayesian or ensemble uncertainty
8. **Real-Time Inference**: Optimize for deployment with ONNX/TensorRT

## References

1. Veeling, B. S., et al. (2018). "Rotation equivariant CNNs for digital pathology." MICCAI.
2. PatchCamelyon Dataset: https://github.com/basveeling/pcam
3. TensorFlow Datasets: https://www.tensorflow.org/datasets/catalog/patch_camelyon

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-XX  
**Status**: Ready for Implementation
