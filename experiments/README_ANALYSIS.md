# Analysis Tools

This directory contains tools for analyzing trained models and comparing baselines.

## Available Tools

### 1. Baseline Comparison (`compare_baselines.py`)

Compare multiple baseline models on PCam dataset.

**Usage:**
```bash
python experiments/compare_baselines.py \
  --results-dir results/baselines \
  --output-dir results/baseline_comparison
```

**Features:**
- Loads results from multiple model experiments
- Generates comparison tables (CSV and Markdown)
- Creates visualization plots:
  - Accuracy/F1/AUC comparison bar chart
  - Efficiency plot (accuracy vs parameters)
  - Training time comparison
- Generates comprehensive markdown report

**Supported Models:**
- ResNet-18 (batch size 64)
- ResNet-50 (batch size 48)
- DenseNet-121 (batch size 48)
- EfficientNet-B0 (batch size 56)
- ViT-Base (batch size 24)

**Output:**
- `baseline_comparison.csv` - Comparison table
- `baseline_comparison.png` - Bar chart comparison
- `efficiency_plot.png` - Accuracy vs parameters
- `training_time_comparison.png` - Training time bars
- `baseline_comparison_report.md` - Full report

### 2. Metrics Analysis (`analyze_metrics.py`)

Comprehensive analysis of training logs and model metrics.

**Usage:**
```bash
python experiments/analyze_metrics.py \
  --log-dir logs/pcam_real \
  --checkpoint checkpoints/pcam_real/best_model.pth \
  --output-dir results/metrics_analysis
```

**Features:**
- Parses TensorBoard logs or JSON training logs
- Generates training curve plots:
  - Training loss over steps
  - Validation loss over epochs
  - Validation accuracy over epochs
  - Validation AUC over epochs
- Creates evaluation plots:
  - Confusion matrix
  - ROC curve with AUC
  - Precision-recall curve
- Generates classification report
- Creates comprehensive markdown report

**Output:**
- `training_curves.png` - Training/validation curves
- `confusion_matrix.png` - Confusion matrix heatmap
- `roc_curve.png` - ROC curve
- `precision_recall_curve.png` - PR curve
- `metrics.json` - Metrics dictionary
- `metrics_report.md` - Full analysis report

### 3. Baseline Training Script (`run_all_baselines.sh`)

**Coming Soon:** Script to train all baseline models sequentially.

```bash
# Will train all models with optimized batch sizes for RTX 4070 Laptop
./experiments/run_all_baselines.sh
```

## Example Workflow

### Step 1: Train Models

Train your baseline models (or use existing checkpoints):

```bash
# Train ResNet-18
python experiments/train_pcam.py \
  --config experiments/configs/pcam_rtx4070_laptop.yaml

# Train other models...
```

### Step 2: Analyze Individual Model

Analyze a single trained model:

```bash
python experiments/analyze_metrics.py \
  --log-dir logs/pcam_real \
  --checkpoint checkpoints/pcam_real/best_model.pth \
  --output-dir results/pcam_real_analysis
```

### Step 3: Compare Baselines

After training multiple models, compare them:

```bash
python experiments/compare_baselines.py \
  --results-dir results/baselines \
  --output-dir results/baseline_comparison
```

## Requirements

Additional packages needed for analysis:

```bash
pip install matplotlib seaborn pandas tensorboard scikit-learn
```

## Output Structure

```
results/
├── baselines/
│   ├── resnet18/
│   │   └── metrics.json
│   ├── resnet50/
│   │   └── metrics.json
│   └── ...
├── baseline_comparison/
│   ├── baseline_comparison.csv
│   ├── baseline_comparison.png
│   ├── efficiency_plot.png
│   ├── training_time_comparison.png
│   └── baseline_comparison_report.md
└── metrics_analysis/
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── metrics.json
    └── metrics_report.md
```

## Tips

1. **Batch Sizes:** The comparison script includes optimized batch sizes for RTX 4070 Laptop (8GB VRAM)
2. **Training Time:** Expect 3-6 hours per model depending on architecture
3. **Memory:** All models fit in 8GB VRAM with mixed precision enabled
4. **Reproducibility:** Set seed in config for reproducible results

## Troubleshooting

### TensorBoard Logs Not Found

If analyze_metrics.py can't find TensorBoard logs:
- Check that `log_dir` points to the correct directory
- Ensure TensorBoard logging was enabled during training
- Try using JSON logs instead

### Missing Results

If compare_baselines.py reports missing results:
- Ensure each model has a `metrics.json` file in its results directory
- Check that the results directory structure matches expectations
- Run individual model analysis first to generate metrics.json

### Out of Memory

If comparison plots fail due to memory:
- Close other applications
- Reduce plot DPI (change `dpi=300` to `dpi=150`)
- Process models one at a time
