---
layout: default
title: Foundation Models
---

# Foundation Models

HistoCore integrates state-of-the-art foundation models pretrained on massive histopathology datasets, providing superior feature representations for downstream tasks.

## Overview

**Available Models:**
- **UNI** - Universal pathology model (1024-dim, ViT-L, 100k+ WSIs)
- **Phikon** - Histopathology specialist (768-dim, ViT-B/16, 500M+ patches)
- **GigaPath** - Gigapixel WSI model (1536-dim)
- **ResNet50** - Fast baseline (2048-dim, ImageNet)

**Benefits:**
- Better feature representations → improved accuracy
- Less training data required
- State-of-the-art pretrained weights
- Flexible model selection for speed/accuracy tradeoff

---

## Quick Start

### Load a Foundation Model

```python
from src.models.pretrained import PretrainedFeatureExtractor

# Load Phikon (recommended for histopathology)
extractor = PretrainedFeatureExtractor('phikon', freeze=True)

# Extract features from patches
patches = torch.randn(4, 3, 224, 224)  # [batch, channels, h, w]
features = extractor(patches)  # [4, 768]
```

### Download Model Weights

```bash
# Download all foundation models
python scripts/download_foundation_models.py --model all

# Download specific model
python scripts/download_foundation_models.py --model phikon

# List cached models
python scripts/download_foundation_models.py --list
```

---

## Model Comparison

### Performance Metrics

| Model | Output Dim | Speed | Memory | Accuracy | Use Case |
|-------|-----------|-------|--------|----------|----------|
| **UNI** | 1024 | Medium | High | Best | General pathology |
| **Phikon** | 768 | Fast | Medium | Excellent | Histopathology |
| **GigaPath** | 1536 | Slow | Very High | Excellent | Gigapixel WSIs |
| **ResNet50** | 2048 | Very Fast | Low | Good | Baseline |

### Benchmark Results

```bash
# Run comprehensive benchmark
python experiments/benchmark_foundation_models.py \
    --data-dir data/processed \
    --models uni phikon resnet50_imagenet \
    --output results/foundation_model_benchmark.json
```

**Expected Output:**

```
Foundation Model Benchmark Summary
================================================================================
Model                Speed (ms)      Memory (MB)     Accuracy    AUC         
--------------------------------------------------------------------------------
uni                  45.23           2048.5          0.9234      0.9567      
phikon               38.67           1536.2          0.9156      0.9489      
resnet50_imagenet    18.45           512.8           0.8567      0.8923      
================================================================================
```

---

## Model Selection Guide

### When to Use UNI

**Best for:** General pathology tasks across diverse tissue types

**Pros:**
- Best overall performance
- Trained on 100k+ WSIs
- Diverse tissue coverage

**Cons:**
- Slower inference
- Higher memory usage

**Example:**
```python
extractor = PretrainedFeatureExtractor('uni', freeze=True)
```

### When to Use Phikon

**Best for:** Histopathology-specific tasks

**Pros:**
- Fast inference (38 ms/sample)
- Specialized pretraining (500M+ patches)
- Good balance of speed/accuracy

**Cons:**
- Slightly lower performance than UNI

**Example:**
```python
extractor = PretrainedFeatureExtractor('phikon', freeze=True)
```

### When to Use GigaPath

**Best for:** Gigapixel-level WSI analysis

**Pros:**
- Designed for large WSIs
- Excellent performance

**Cons:**
- Very slow inference
- Very high memory usage

**Example:**
```python
extractor = PretrainedFeatureExtractor('gigapath', freeze=True)
```

### When to Use ResNet50

**Best for:** Fast prototyping, baselines

**Pros:**
- Very fast (18 ms/sample)
- Low memory
- Well-understood

**Cons:**
- Lower performance than foundation models

**Example:**
```python
extractor = PretrainedFeatureExtractor('resnet50_imagenet', freeze=True)
```

---

## Integration with Training Pipeline

### 1. Feature Extraction + WSI Encoder

```python
from src.models.pretrained import create_wsi_encoder_with_pretrained

# Create feature extractor + WSI encoder
extractor, wsi_encoder = create_wsi_encoder_with_pretrained(
    pretrained_model='phikon',
    output_dim=256,
    freeze_pretrained=True,
)

# Use in training
for batch in dataloader:
    # Extract patch features
    patch_features = extractor(batch['patches'])
    
    # Aggregate to slide-level
    slide_embedding = wsi_encoder(patch_features)
    
    # Classification
    logits = task_head(slide_embedding)
```

### 2. Training with Foundation Models

```bash
# Train classifier with Phikon features
python experiments/train.py \
    --feature-extractor phikon \
    --freeze-extractor \
    --batch-size 32 \
    --use-amp
```

### 3. Feature Caching for Faster Training

```bash
# Extract features once (saves time)
python scripts/extract_features.py \
    --model phikon \
    --input-dir data/slides/ \
    --output-dir data/features/phikon/ \
    --batch-size 64

# Train on cached features (much faster)
python experiments/train.py \
    --use-cached-features \
    --feature-dir data/features/phikon/
```

---

## Recommended Model Selection

Use the built-in recommendation system:

```python
from src.models.pretrained import get_recommended_model

# Get recommendation for task
model_name = get_recommended_model(task='histopathology')  # Returns 'phikon'
extractor = PretrainedFeatureExtractor(model_name)
```

**Available Tasks:**
- `'general'` → UNI (best overall)
- `'histopathology'` → Phikon (specialized)
- `'gigapixel'` → GigaPath (large WSIs)
- `'fast'` → ResNet50 (fastest)
- `'baseline'` → ResNet50 (standard baseline)

---

## Advanced Usage

### Fine-Tuning Foundation Models

```python
# Unfreeze for fine-tuning (requires large dataset)
extractor = PretrainedFeatureExtractor(
    'phikon',
    freeze=False,  # Allow gradient updates
)

# Use lower learning rate for pretrained weights
optimizer = torch.optim.Adam([
    {'params': extractor.parameters(), 'lr': 1e-5},
    {'params': task_head.parameters(), 'lr': 1e-3},
])
```

### Multi-Model Ensemble

```python
# Load multiple models
uni_extractor = PretrainedFeatureExtractor('uni')
phikon_extractor = PretrainedFeatureExtractor('phikon')

# Extract features from both
uni_features = uni_extractor(patches)
phikon_features = phikon_extractor(patches)

# Concatenate or average
combined_features = torch.cat([uni_features, phikon_features], dim=1)
```

---

## Configuration

### Model Registry

All models are registered in `src/models/pretrained.py`:

```python
PRETRAINED_MODELS = {
    "uni": {
        "name": "UNI",
        "source": "hf_hub:MahmoodLab/uni",
        "output_dim": 1024,
        "input_size": 224,
    },
    "phikon": {
        "name": "Phikon",
        "source": "hf_hub:owkin/phikon",
        "output_dim": 768,
        "input_size": 224,
    },
    # ... more models
}
```

### Download Configuration

Models are cached in `~/.cache/medical_ai_models/` by default:

```bash
# Use custom cache directory
python scripts/download_foundation_models.py \
    --model phikon \
    --cache-dir /path/to/cache
```

---

## Best Practices

### 1. Always Freeze Pretrained Weights

```python
# Recommended: freeze pretrained weights
extractor = PretrainedFeatureExtractor('phikon', freeze=True)

# Only unfreeze for fine-tuning with large datasets (>100k samples)
extractor = PretrainedFeatureExtractor('phikon', freeze=False)
```

### 2. Cache Features for Faster Training

```bash
# Extract features once
python scripts/extract_features.py --model phikon --input-dir data/slides/

# Train on cached features (10-50x faster)
python experiments/train.py --use-cached-features
```

### 3. Benchmark on Your Hardware

```bash
# Always benchmark before deployment
python experiments/benchmark_foundation_models.py \
    --data-dir data/processed \
    --models uni phikon \
    --device cuda
```

### 4. Use Appropriate Model for Task

- **Research/Best Accuracy:** UNI
- **Production/Speed:** Phikon
- **Prototyping:** ResNet50

---

## API Reference

### PretrainedFeatureExtractor

```python
from src.models.pretrained import PretrainedFeatureExtractor

extractor = PretrainedFeatureExtractor(
    model_name='phikon',           # Model to load
    cache_dir=None,                # Cache directory (default: ~/.cache)
    freeze=True,                   # Freeze weights
    device='cuda',                 # Device
)

# Extract features
features = extractor(patches)      # [batch_size, output_dim]

# Extract slide features
slide_features = extractor.extract_slide_features(
    patch_loader,                  # DataLoader for patches
    device='cuda',                 # Device
)
```

### Helper Functions

```python
from src.models.pretrained import (
    list_pretrained_models,
    get_recommended_model,
    create_wsi_encoder_with_pretrained,
)

# List available models
models = list_pretrained_models()

# Get recommendation
model_name = get_recommended_model(task='histopathology')

# Create extractor + encoder
extractor, encoder = create_wsi_encoder_with_pretrained(
    pretrained_model='phikon',
    output_dim=256,
    freeze_pretrained=True,
)
```

---

## Troubleshooting

### Model Download Fails

**Issue:** HuggingFace download fails

**Solution:**
```bash
# Set HuggingFace token for private models
export HUGGINGFACE_TOKEN=your_token_here
python scripts/download_foundation_models.py --model uni
```

### Out of Memory

**Issue:** GPU OOM when loading model

**Solution:**
- Use smaller batch size
- Use Phikon instead of UNI (lower memory)
- Use CPU for inference

### Slow Inference

**Issue:** Feature extraction is slow

**Solution:**
- Cache features once, reuse for training
- Use smaller model (Phikon or ResNet50)
- Increase batch size for better GPU utilization

---

## Citations

### UNI
```bibtex
@article{chen2024uni,
  title={Towards a general-purpose foundation model for computational pathology},
  author={Chen, Richard J and others},
  journal={Nature Medicine},
  year={2024}
}
```

### Phikon
```bibtex
@article{filiot2023phikon,
  title={Scaling self-supervised learning for histopathology with masked image modeling},
  author={Filiot, Alexandre and others},
  journal={medRxiv},
  year={2023}
}
```

---

## See Also

- [Training Optimizations](OPTIMIZATION_SUMMARY.html) - 2.5x faster training
- [Inference Optimization](INFERENCE_OPTIMIZATION.html) - 2-3x faster inference
- [Performance Comparison](PERFORMANCE_COMPARISON.html) - Benchmarks
- [Getting Started](GETTING_STARTED.html) - Setup guide
