# Configuration Files

This directory contains YAML configuration files for training and evaluation.

## Available Configurations

### 1. `default.yaml`
**Purpose**: Base configuration with sensible defaults for all settings.

**Use case**: Starting point for custom configurations, reference for all available options.

**Key settings**:
- Embed dim: 256
- Batch size: 16
- Learning rate: 1e-4
- Epochs: 100

### 2. `quick_demo.yaml`
**Purpose**: Fast training configuration for testing and demos.

**Use case**: Quick validation that code works, rapid prototyping, CI/CD testing.

**Key settings**:
- Embed dim: 128 (smaller model)
- Batch size: 16
- Learning rate: 5e-4 (higher for faster convergence)
- Epochs: 5 (very short)
- Early stopping: disabled

**Usage**:
```bash
python experiments/train.py --config-name quick_demo
```

### 3. `full_training.yaml`
**Purpose**: Production-ready configuration for best performance.

**Use case**: Training on real datasets for publication or deployment.

**Key settings**:
- Embed dim: 512 (larger model)
- Batch size: 32
- Learning rate: 5e-5 (lower for stability)
- Epochs: 200
- Mixed precision: enabled
- Gradient accumulation: 2 steps
- Data augmentation: enabled
- Weights & Biases: enabled

**Usage**:
```bash
python experiments/train.py --config-name full_training
```

### 4. `ablation.yaml`
**Purpose**: Configuration for systematic ablation studies.

**Use case**: Analyzing contribution of each component, understanding model behavior.

**Key settings**:
- Standard model size
- No missing modalities during training
- Ablation study: enabled
- Missing modality testing: enabled
- All evaluation metrics: enabled

**Usage**:
```bash
python experiments/train.py --config-name ablation
python experiments/evaluate.py --checkpoint checkpoints/ablation/best_model.pth --run-ablation
```

### 5. `survival.yaml`
**Purpose**: Configuration for survival analysis tasks.

**Use case**: Time-to-event prediction, prognosis modeling.

**Key settings**:
- Task type: survival
- Cox proportional hazards loss
- Time bins: 20
- Max follow-up: 60 months
- Concordance index metric

**Usage**:
```bash
python experiments/train.py --config-name survival
```

### 6. `stain_norm.yaml`
**Purpose**: Configuration for training stain normalization transformer.

**Use case**: Preprocessing pipeline, color normalization.

**Key settings**:
- Perceptual loss: enabled
- Color consistency loss: enabled
- SSIM metric
- Image augmentation: enabled

**Usage**:
```bash
python experiments/train_stain_norm.py --config-name stain_norm
```

## Configuration Structure

All configuration files follow this structure:

```yaml
# Model architecture
model:
  embed_dim: 256
  wsi: {...}
  genomic: {...}
  clinical: {...}
  fusion: {...}

# Task definition
task:
  type: classification  # or survival
  num_classes: 4
  classification: {...}
  survival: {...}

# Training hyperparameters
training:
  num_epochs: 100
  batch_size: 16
  learning_rate: 1e-4
  optimizer: {...}
  scheduler: {...}

# Data loading
data:
  data_dir: ./data
  num_workers: 4
  augmentation: {...}

# Validation and checkpointing
validation:
  metric: val_accuracy
  maximize: true

checkpoint:
  save_interval: 5
  checkpoint_dir: ./checkpoints

# Early stopping
early_stopping:
  enabled: true
  patience: 10

# Logging
logging:
  log_dir: ./logs
  use_tensorboard: true
  use_wandb: false

# Evaluation
evaluation:
  generate_plots: true
  run_ablation: false

# Reproducibility
seed: 42
device: cuda
```

## Using Configurations

### Method 1: Direct Usage (Recommended)

Use the configuration file name without extension:

```bash
python experiments/train.py --config-name quick_demo
```

### Method 2: Override Specific Parameters

Override individual parameters from command line:

```bash
python experiments/train.py \
    --config-name default \
    training.learning_rate=5e-4 \
    training.batch_size=32 \
    model.embed_dim=512
```

### Method 3: Create Custom Configuration

Create a new YAML file inheriting from an existing one:

```yaml
# experiments/configs/my_config.yaml
defaults:
  - default

# Override specific settings
training:
  num_epochs: 50
  learning_rate: 2e-4

model:
  embed_dim: 384
```

Then use it:

```bash
python experiments/train.py --config-name my_config
```

## Configuration Inheritance

Configurations can inherit from others using the `defaults` key:

```yaml
defaults:
  - default  # Inherit all settings from default.yaml

# Override only what you need
training:
  num_epochs: 50
```

This allows you to:
- Avoid duplication
- Maintain consistency
- Easily create variants

## Parameter Categories

### Model Parameters
- `model.embed_dim`: Common embedding dimension
- `model.dropout`: Dropout rate
- `model.wsi.*`: WSI encoder settings
- `model.genomic.*`: Genomic encoder settings
- `model.clinical.*`: Clinical text encoder settings
- `model.fusion.*`: Fusion layer settings

### Training Parameters
- `training.num_epochs`: Number of training epochs
- `training.batch_size`: Batch size
- `training.learning_rate`: Initial learning rate
- `training.weight_decay`: L2 regularization
- `training.optimizer.*`: Optimizer settings
- `training.scheduler.*`: LR scheduler settings
- `training.max_grad_norm`: Gradient clipping threshold

### Data Parameters
- `data.data_dir`: Path to data directory
- `data.num_workers`: Number of data loading workers
- `data.missing_modality_rate`: Probability of missing modality
- `data.augmentation.*`: Data augmentation settings

### Validation Parameters
- `validation.metric`: Metric to monitor
- `validation.maximize`: Whether to maximize metric
- `checkpoint.save_interval`: Save frequency
- `early_stopping.patience`: Early stopping patience

### Logging Parameters
- `logging.log_dir`: TensorBoard log directory
- `logging.use_tensorboard`: Enable TensorBoard
- `logging.use_wandb`: Enable Weights & Biases
- `logging.wandb.*`: W&B configuration

## Best Practices

1. **Start with `quick_demo.yaml`** for initial testing
2. **Use `default.yaml`** as reference for all options
3. **Use `full_training.yaml`** for production runs
4. **Create custom configs** by inheriting from defaults
5. **Document changes** in experiment tracking
6. **Version control** your custom configurations
7. **Use meaningful names** for custom configs
8. **Test configs** with quick_demo settings first

## Common Workflows

### Quick Test
```bash
python experiments/train.py --config-name quick_demo
```

### Full Training
```bash
python experiments/train.py --config-name full_training
```

### Ablation Study
```bash
# Train with ablation config
python experiments/train.py --config-name ablation

# Evaluate with ablation analysis
python experiments/evaluate.py \
    --checkpoint checkpoints/ablation/best_model.pth \
    --run-ablation \
    --test-missing-modalities
```

### Hyperparameter Tuning
```bash
# Try different learning rates
for lr in 1e-4 5e-5 1e-5; do
    python experiments/train.py \
        --config-name default \
        training.learning_rate=$lr \
        experiment.name="lr_${lr}"
done
```

### Resume Training
```bash
python experiments/train.py \
    --config-name full_training \
    --resume checkpoints/full_training/checkpoint_epoch_50.pth
```

## Troubleshooting

### Issue: Config file not found
**Solution**: Ensure you're using the config name without `.yaml` extension:
```bash
# Correct
python experiments/train.py --config-name quick_demo

# Incorrect
python experiments/train.py --config-name quick_demo.yaml
```

### Issue: Parameter override not working
**Solution**: Use dot notation for nested parameters:
```bash
python experiments/train.py \
    --config-name default \
    training.learning_rate=1e-4  # Correct
```

### Issue: Out of memory
**Solution**: Reduce batch size or model size:
```bash
python experiments/train.py \
    --config-name default \
    training.batch_size=8 \
    model.embed_dim=128
```

## Additional Resources

- See `experiments/train.py` for training script
- See `experiments/evaluate.py` for evaluation script
- See `experiments/train_stain_norm.py` for stain normalization training
- See main `README.md` for overall project documentation
