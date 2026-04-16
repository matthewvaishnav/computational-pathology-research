# Model Interpretability Module

Comprehensive interpretability tools for computational pathology models including Grad-CAM visualization, attention weight analysis, failure case identification, and feature importance computation.

## Features

- **Grad-CAM Visualization**: Generate gradient-weighted class activation maps for CNN feature extractors (ResNet, DenseNet, EfficientNet)
- **Attention Visualization**: Visualize attention weights for MIL architectures (AttentionMIL, CLAM, TransMIL)
- **Failure Analysis**: Identify and cluster systematic errors with k-means/DBSCAN/hierarchical clustering
- **Feature Importance**: Compute feature importance using permutation, SHAP, or gradient-based methods
- **Configuration Management**: YAML-based configuration with validation
- **Interactive Dashboard**: Flask-based web interface for exploring visualizations (optional)

## Installation

```bash
# Core dependencies
pip install torch torchvision numpy matplotlib scikit-learn pandas h5py pyyaml

# Optional: SHAP for feature importance
pip install shap

# Optional: Dashboard dependencies
pip install flask plotly redis
```

## Quick Start

### Grad-CAM Visualization

```python
from src.interpretability.gradcam import GradCAMGenerator
import torchvision.models as models
import torch

# Load model
model = models.resnet18(pretrained=True)

# Create Grad-CAM generator
generator = GradCAMGenerator(
    model=model,
    target_layers=['layer4'],
    device='cuda'
)

# Generate heatmaps
images = torch.randn(4, 3, 224, 224)
heatmaps = generator.generate(images)

# Visualize
import numpy as np
image = np.random.rand(224, 224, 3)
heatmap = heatmaps['layer4'][0].cpu().numpy()
generator.save_visualization(image, heatmap, 'gradcam.png')
```

### Failure Analysis

```python
from src.interpretability.failure_analysis import FailureAnalyzer
import numpy as np

# Create analyzer
analyzer = FailureAnalyzer(
    clustering_method='kmeans',
    n_clusters=5
)

# Identify failures
failures = analyzer.identify_failures(
    predictions=np.array([0, 1, 0, 1]),
    ground_truth=np.array([0, 0, 0, 1]),
    confidences=np.array([0.9, 0.7, 0.8, 0.95]),
    embeddings=np.random.randn(4, 512),
    slide_ids=['slide1', 'slide2', 'slide3', 'slide4']
)

# Cluster failures
clusters = analyzer.cluster_failures(failures)

# Analyze characteristics
stats = analyzer.analyze_cluster_characteristics(clusters)
print(stats)

# Export report
analyzer.export_failure_report('failures.csv')
```

### Feature Importance

```python
from src.interpretability.feature_importance import FeatureImportanceCalculator
import torch
import numpy as np

# Create calculator
calculator = FeatureImportanceCalculator(
    model=model,
    method='permutation',  # Options: permutation, shap, gradient
    device='cuda'
)

# Compute importance
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)
feature_names = [f'feature_{i}' for i in range(10)]

importance = calculator.compute_importance(X, y, feature_names, n_repeats=10)

# Rank features
ranked = calculator.rank_features(importance, top_k=5)
print(ranked)

# Visualize
calculator.visualize_importance(importance, 'importance.png')

# Export
calculator.export_importance_scores(importance, 'importance.csv')
```

### Configuration Management

```python
from src.interpretability.config import load_gradcam_config, save_gradcam_config
from pathlib import Path

# Load configuration
config = load_gradcam_config(Path('configs/interpretability/gradcam_default.yaml'))

# Modify configuration
config.alpha = 0.7
config.dpi = 600

# Save configuration
save_gradcam_config(config, Path('configs/interpretability/gradcam_custom.yaml'))
```

## Command-Line Usage

### Grad-CAM Generation

```bash
# Generate Grad-CAM for evaluation
python scripts/evaluate.py \
    --model resnet18 \
    --checkpoint checkpoints/best_model.pth \
    --data data/test \
    --enable-gradcam \
    --gradcam-layers layer4 \
    --interpretability-output-dir results/gradcam
```

### Failure Analysis

```bash
# Run failure analysis on evaluation results
python scripts/analyze_failures.py \
    --predictions results/predictions.csv \
    --embeddings results/embeddings.npy \
    --clustering-method kmeans \
    --n-clusters 5 \
    --output results/failure_analysis
```

### Feature Importance

```bash
# Compute feature importance for clinical data
python scripts/compute_feature_importance.py \
    --model checkpoints/multimodal_model.pth \
    --data data/clinical_features.csv \
    --method permutation \
    --n-repeats 10 \
    --output results/feature_importance
```

## Configuration Files

Default configurations are provided in `configs/interpretability/`:

- `gradcam_default.yaml`: Grad-CAM visualization settings
- `attention_default.yaml`: Attention visualization settings
- `dashboard_default.yaml`: Interactive dashboard settings

## API Documentation

### GradCAMGenerator

**Methods:**
- `__init__(model, target_layers, device)`: Initialize generator
- `generate(images, class_idx)`: Generate heatmaps for images
- `overlay_heatmap(image, heatmap, alpha, colormap)`: Overlay heatmap on image
- `save_visualization(image, heatmap, output_path, dpi)`: Save visualization to file

### FailureAnalyzer

**Methods:**
- `__init__(clustering_method, n_clusters, embedding_dim)`: Initialize analyzer
- `identify_failures(predictions, ground_truth, confidences, embeddings, slide_ids)`: Identify misclassified samples
- `cluster_failures(failures)`: Cluster failures by embeddings
- `analyze_cluster_characteristics(clusters)`: Compute cluster statistics
- `identify_systematic_biases(failures, subgroup_key)`: Analyze bias across subgroups
- `export_failure_report(output_path, failures)`: Export report to CSV

### FeatureImportanceCalculator

**Methods:**
- `__init__(model, method, device)`: Initialize calculator
- `compute_importance(X, y, feature_names, n_repeats)`: Compute importance scores
- `compute_permutation_importance(X, y, n_repeats)`: Permutation importance
- `compute_shap_values(X, y)`: SHAP values
- `compute_gradient_importance(X, y)`: Gradient-based importance
- `rank_features(importance, top_k)`: Rank features by importance
- `compute_confidence_intervals(X, y, feature_names, n_bootstrap)`: Bootstrap confidence intervals
- `visualize_importance(importance, output_path, top_k)`: Create bar plot
- `export_importance_scores(importance, output_path)`: Export to CSV

## Performance

### Computational Requirements

- **Grad-CAM**: < 200ms per patch on GPU, < 1s on CPU
- **Attention Extraction**: < 100ms per slide on GPU
- **Feature Importance**: < 5 seconds per model on CPU
- **Dashboard**: < 3 seconds initial load time

### GPU Acceleration

All components support GPU acceleration with automatic CPU fallback:

```python
# GPU mode (automatic fallback if unavailable)
generator = GradCAMGenerator(model, target_layers=['layer4'], device='cuda')

# CPU mode
generator = GradCAMGenerator(model, target_layers=['layer4'], device='cpu')
```

## Troubleshooting

### Common Issues

**1. Grad-CAM produces all-zero heatmaps**
- Check that target layers exist in model
- Verify model is in eval mode
- Ensure gradients are enabled for input

**2. SHAP computation is slow**
- Reduce number of background samples
- Use gradient method as fallback
- Consider using permutation importance instead

**3. Dashboard not loading**
- Check Flask server is running
- Verify port is not in use
- Check Redis connection (if using Redis cache)

**4. Out of memory errors**
- Reduce batch size
- Use CPU mode
- Enable gradient checkpointing

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

See `examples/interpretability/` for Jupyter notebook examples:

- `gradcam_example.ipynb`: Grad-CAM visualization walkthrough
- `attention_example.ipynb`: Attention visualization examples
- `failure_analysis_example.ipynb`: Failure analysis workflow
- `feature_importance_example.ipynb`: Feature importance computation

## Citation

If you use this interpretability module in your research, please cite:

```bibtex
@software{interpretability_module,
  title={Model Interpretability Tools for Computational Pathology},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/computational-pathology-research}
}
```

## References

- Grad-CAM: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
- SHAP: Lundberg & Lee "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017)
- Permutation Importance: Breiman "Random Forests" (Machine Learning 2001)

## License

MIT License - see LICENSE file for details
