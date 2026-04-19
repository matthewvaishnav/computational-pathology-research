---
layout: default
title: Model Interpretability
---

# Model Interpretability

Comprehensive interpretability tools for understanding deep learning model decisions in computational pathology.

---

## Overview

The interpretability system provides Grad-CAM visualizations, attention weight analysis, failure case identification, and feature importance computation. This is critical for clinical trust, debugging model failures, research publications, and regulatory compliance.

**Key Features:**
- Grad-CAM heatmaps for CNN feature extractors (ResNet, DenseNet, EfficientNet)
- Attention weight visualization for MIL models (AttentionMIL, CLAM, TransMIL)
- Failure case analysis and clustering
- Feature importance for clinical data
- Interactive visualization dashboard
- Publication-quality figure generation

---

## Grad-CAM Visualization

Generate gradient-weighted class activation maps for understanding which tissue regions influence predictions:

```python
from src.visualization.gradcam import GradCAMGenerator

# Initialize Grad-CAM generator
gradcam = GradCAMGenerator(
    model=trained_model,
    target_layers=['layer4'],  # ResNet layer
    transparency=0.4
)

# Generate heatmap for a patch
heatmap = gradcam.generate_heatmap(
    input_patch=patch_tensor,
    target_class=1,
    save_path='outputs/gradcam_patch_001.png'
)
```

**Supported Architectures:**
- ResNet-18/34/50/101: Target layers `layer4`, `layer3`
- DenseNet-121: Target layer `features.denseblock4`
- EfficientNet-B0: Target layer `features.7`

---

## Attention Weight Visualization

Visualize attention weights from Multiple Instance Learning models to understand which patches the model focuses on:

```python
from src.visualization.attention_heatmap import AttentionHeatmapGenerator

# Initialize attention visualizer
attention_viz = AttentionHeatmapGenerator(
    attention_dir='outputs/attention_weights',
    output_dir='outputs/heatmaps',
    colormap='viridis'
)

# Generate attention heatmap for slide
heatmap_path = attention_viz.generate_heatmap(
    slide_id='slide_001',
    thumbnail_size=(512, 512)
)
```

**Supported MIL Models:**
- **AttentionMIL**: Gated attention mechanism with instance/bag-level modes
- **CLAM**: Clustering-constrained attention with multi-branch support  
- **TransMIL**: Transformer encoder with positional encoding

---

## Failure Case Analysis

Analyze misclassified samples to identify systematic biases and model weaknesses:

```python
from src.interpretability.failure_analyzer import FailureAnalyzer

# Initialize failure analyzer
analyzer = FailureAnalyzer(
    model=trained_model,
    validation_loader=val_loader
)

# Analyze failure cases
failure_report = analyzer.analyze_failures(
    output_dir='results/failure_analysis',
    cluster_failures=True,
    export_csv=True
)
```

**Common Failure Patterns:**
1. **Staining Artifacts**: Model confused by unusual staining
2. **Tissue Folding**: Misclassification due to tissue preparation issues
3. **Boundary Cases**: Uncertainty at tumor-normal boundaries
4. **Rare Subtypes**: Poor performance on underrepresented classes

---

## Interactive Dashboard

Launch web-based interface for exploring model decisions:

```bash
# Start visualization server
python scripts/launch_dashboard.py \
  --checkpoint checkpoints/best_model.pth \
  --data-root data/validation \
  --port 8080
```

**Dashboard Features:**
- Sample browser with filtering options
- Side-by-side comparison (up to 4 samples)
- Export tools for publication-quality figures
- Keyboard navigation for efficient browsing

---

## Performance Benchmarks

| Operation | GPU Time | Memory Usage |
|-----------|----------|--------------|
| Grad-CAM (per patch) | 150ms | 2GB |
| Attention extraction | 80ms | 1GB |
| Feature importance | 3s | 4GB |

---

## Integration with Evaluation

Generate interpretability visualizations during standard evaluation:

```bash
# PCam evaluation with Grad-CAM
python experiments/evaluate_pcam.py \
  --checkpoint checkpoints/pcam/best_model.pth \
  --generate-gradcam \
  --gradcam-samples 100

# CAMELYON evaluation with attention heatmaps
python experiments/evaluate_camelyon.py \
  --checkpoint checkpoints/camelyon/best_model.pth \
  --generate-attention-heatmaps \
  --analyze-failures
```

---

<div class="footer-note">
  <p><em>Last updated: April 2026</em></p>
  <p>For questions about interpretability features, please <a href="https://github.com/matthewvaishnav/computational-pathology-research/issues">open an issue</a>.</p>
</div>