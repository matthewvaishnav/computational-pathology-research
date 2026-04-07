# PCam Baseline Comparison Configurations

This directory contains configuration files for comparing different PCam model variants.

## Available Configurations

### 1. baseline_resnet18.yaml
**Description**: Current default configuration  
**Architecture**:
- Feature Extractor: ResNet-18 (pretrained, 11.2M params)
- WSI Encoder: Transformer (1 layer, 4 heads, 256 hidden dim)
- Classification Head: Hidden layer (128 dim) + output layer

**Purpose**: Baseline for comparison

---

### 2. resnet50.yaml
**Description**: Larger feature extractor  
**Architecture**:
- Feature Extractor: ResNet-50 (pretrained, 23.5M params)
- WSI Encoder: Transformer (1 layer, 4 heads, 256 hidden dim)
- Classification Head: Hidden layer (128 dim) + output layer

**Purpose**: Test whether a larger backbone improves performance

**Expected**: Higher capacity may improve accuracy but slower inference

---

### 3. simple_head.yaml
**Description**: Simpler classification head  
**Architecture**:
- Feature Extractor: ResNet-18 (pretrained, 11.2M params)
- WSI Encoder: Transformer (1 layer, 4 heads, 256 hidden dim)
- Classification Head: Direct linear layer (no hidden layer)

**Purpose**: Test whether the hidden layer is necessary

**Expected**: Fewer parameters, potentially faster but may reduce accuracy

---

## Running Comparisons

### Quick Test (3 epochs)
```bash
python experiments/compare_pcam_baselines.py \
  --configs experiments/configs/pcam_comparison/*.yaml \
  --quick-test
```

### Full Training (20 epochs)
```bash
python experiments/compare_pcam_baselines.py \
  --configs experiments/configs/pcam_comparison/*.yaml
```

### Specific Variants
```bash
python experiments/compare_pcam_baselines.py \
  --configs \
    experiments/configs/pcam_comparison/baseline_resnet18.yaml \
    experiments/configs/pcam_comparison/resnet50.yaml
```

## Output Structure

```
results/pcam_comparison/
├── comparison_results.json          # Aggregated comparison metrics
├── baseline_resnet18/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── resnet50/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── roc_curve.png
└── simple_head/
    ├── metrics.json
    ├── confusion_matrix.png
    └── roc_curve.png

checkpoints/pcam_comparison/
├── baseline_resnet18/
│   └── best_model.pth
├── resnet50/
│   └── best_model.pth
└── simple_head/
    └── best_model.pth

logs/pcam_comparison/
├── baseline_resnet18/
├── resnet50/
└── simple_head/
```

## Comparison Results Format

The `comparison_results.json` file contains:

```json
{
  "timestamp": "2026-04-07 12:00:00",
  "variants": [
    {
      "name": "baseline_resnet18",
      "config_path": "experiments/configs/pcam_comparison/baseline_resnet18.yaml",
      "training_status": "success",
      "evaluation_status": "success",
      "training_time_seconds": 45.2,
      "test_accuracy": 0.94,
      "test_auc": 1.0,
      "test_f1": 0.938,
      "test_precision": 0.951,
      "test_recall": 0.933,
      "model_parameters": {
        "feature_extractor": 11176512,
        "encoder": 987904,
        "head": 33281,
        "total": 12197697
      },
      "inference_time_seconds": 0.81,
      "samples_per_second": 123.5,
      "checkpoint_path": "checkpoints/pcam_comparison/baseline_resnet18/best_model.pth",
      "results_dir": "results/pcam_comparison/baseline_resnet18"
    }
  ]
}
```

## Important Notes

### Dataset
All comparisons use the same synthetic PCam subset:
- Train: 500 samples
- Val: 100 samples
- Test: 100 samples

### Reproducibility
- Fixed seed: 42
- Same data augmentation across all variants
- Same training hyperparameters (lr, weight decay, scheduler)

### Caveats
- Results on synthetic subset, not full PCam dataset
- Not comparable to published PCam baselines (different scale)
- Framework validation, not clinical validation
- Comparisons are fair within this repo but not to external methods

## Adding New Variants

To add a new variant:

1. Create a new YAML config file in this directory
2. Set a unique `experiment.name`
3. Modify the architecture parameters as needed
4. Update checkpoint/log/evaluation paths to use the variant name
5. Run the comparison script with the new config included

Example:
```yaml
experiment:
  name: my_new_variant
  description: Description of what's different
  tags: [pcam, my-tag]

# ... rest of config ...

checkpoint:
  checkpoint_dir: ./checkpoints/pcam_comparison/my_new_variant

logging:
  log_dir: ./logs/pcam_comparison/my_new_variant

evaluation:
  output_dir: ./results/pcam_comparison/my_new_variant
```
