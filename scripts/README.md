# Scripts Directory

Utility scripts for data processing, analysis, and model evaluation.

## Analysis Scripts

### Failure Analysis
**Script**: `analyze_pcam_failures.py`  
**Purpose**: Analyze misclassified samples to identify model weaknesses

```bash
python scripts/analyze_pcam_failures.py \
  --checkpoint checkpoints/pcam_real/best_model.pth \
  --data-root data/pcam_real \
  --output-dir results/pcam_real/failure_analysis \
  --batch-size 64
```

**Output**:
- `failure_analysis.json` - Complete analysis with misclassified indices
- `confidence_distribution.png` - Confidence analysis visualization
- `error_rates.png` - False positive/negative rate comparison

**Documentation**: [docs/PCAM_FAILURE_ANALYSIS.md](../docs/PCAM_FAILURE_ANALYSIS.md)

### Threshold Optimization
**Script**: `optimize_threshold.py`  
**Purpose**: Find optimal decision threshold for clinical deployment

```bash
python scripts/optimize_threshold.py \
  --results results/pcam_real/metrics.json \
  --output-dir results/pcam_real/threshold_optimization
```

**Output**:
- `threshold_optimization.json` - Complete optimization report
- `roc_curve_optimal.png` - ROC curve with optimal points
- `precision_recall_curve.png` - Precision-recall analysis
- `threshold_comparison.png` - Performance comparison

**Documentation**: [docs/THRESHOLD_OPTIMIZATION.md](../docs/THRESHOLD_OPTIMIZATION.md)

### Cross-Validation
**Script**: `cross_validate_pcam.py`  
**Purpose**: K-fold cross-validation for robustness assessment

```bash
# Quick test (15-20 minutes)
bash scripts/run_cv_quick_test.sh

# Full 5-fold CV (30-40 hours)
bash scripts/run_cv_full.sh

# Custom configuration
python scripts/cross_validate_pcam.py \
  --data-root data/pcam_real \
  --output-dir results/pcam_cv \
  --n-folds 5 \
  --num-epochs 20 \
  --batch-size 128 \
  --use-amp
```

**Output**:
- `fold_X_best_model.pth` - Best checkpoint for each fold
- `fold_X_results.json` - Detailed results per fold
- `cross_validation_results.json` - Aggregated statistics

**Documentation**: [docs/PCAM_CROSS_VALIDATION.md](../docs/PCAM_CROSS_VALIDATION.md)

## Data Generation Scripts

### Synthetic PCam Data
**Script**: `generate_synthetic_pcam.py`  
**Purpose**: Generate synthetic PCam data for testing

```bash
python scripts/generate_synthetic_pcam.py \
  --output-dir data/pcam_synthetic \
  --num-samples 1000
```

### Synthetic CAMELYON Data
**Script**: `generate_synthetic_camelyon.py`  
**Purpose**: Generate synthetic slide-level data for testing

```bash
python scripts/generate_synthetic_camelyon.py \
  --output-dir data/camelyon_synthetic \
  --num-slides 100
```

## Model Utilities

### Model Profiler
**Script**: `model_profiler.py`  
**Purpose**: Profile model inference time and memory usage

```bash
python scripts/model_profiler.py \
  --checkpoint checkpoints/pcam_real/best_model.pth \
  --profile-type time
```

### ONNX Export
**Script**: `export_onnx.py`  
**Purpose**: Export PyTorch model to ONNX format

```bash
python scripts/export_onnx.py \
  --checkpoint checkpoints/pcam_real/best_model.pth \
  --output models/pcam_model.onnx
```

## Data Download Scripts

### Manual PCam Download
**Script**: `download_pcam_manual.py`  
**Purpose**: Download PCam dataset from Zenodo (alternative to tensorflow-datasets)

```bash
python scripts/download_pcam_manual.py \
  --root-dir data/pcam_real
```

## Quick Reference

### Analysis Pipeline
```bash
# 1. Train model
python experiments/train_pcam.py --config experiments/configs/pcam_rtx4070_laptop.yaml

# 2. Evaluate with bootstrap CI
python experiments/evaluate_pcam.py \
  --checkpoint checkpoints/pcam_real/best_model.pth \
  --compute-bootstrap-ci

# 3. Analyze failures
python scripts/analyze_pcam_failures.py \
  --checkpoint checkpoints/pcam_real/best_model.pth \
  --data-root data/pcam_real

# 4. Optimize threshold
python scripts/optimize_threshold.py \
  --results results/pcam_real/metrics.json

# 5. Cross-validation (optional)
bash scripts/run_cv_full.sh
```

### Testing Pipeline
```bash
# 1. Generate synthetic data
python scripts/generate_synthetic_pcam.py

# 2. Quick training test
python experiments/train_pcam.py --config experiments/configs/pcam_synthetic.yaml

# 3. Quick CV test
bash scripts/run_cv_quick_test.sh
```

## Script Categories

### Analysis & Evaluation
- `analyze_pcam_failures.py` - Failure case analysis
- `optimize_threshold.py` - Threshold optimization
- `cross_validate_pcam.py` - K-fold cross-validation
- `model_profiler.py` - Performance profiling

### Data Generation
- `generate_synthetic_pcam.py` - Synthetic patch data
- `generate_synthetic_camelyon.py` - Synthetic slide data
- `download_pcam_manual.py` - Manual dataset download

### Model Utilities
- `export_onnx.py` - ONNX export
- Model quantization (planned)
- Model compression (planned)

## Common Options

Most analysis scripts support these common options:

```bash
--checkpoint PATH          # Path to model checkpoint
--data-root PATH          # Path to dataset
--output-dir PATH         # Output directory for results
--batch-size INT          # Batch size for inference
--num-workers INT         # DataLoader workers
--device cuda/cpu         # Device to use
--seed INT                # Random seed for reproducibility
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 32

# Reduce workers
--num-workers 2
```

### Slow Execution
```bash
# Use GPU
--device cuda

# Increase workers
--num-workers 8

# Use mixed precision
--use-amp
```

### Reproducibility Issues
```bash
# Set fixed seed
--seed 42

# Disable CUDNN benchmark
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Contributing

When adding new scripts:
1. Add comprehensive docstring
2. Support common CLI arguments
3. Include progress bars (tqdm)
4. Save results in JSON format
5. Add entry to this README
6. Create documentation in `docs/`

## See Also

- [experiments/README.md](../experiments/README.md) - Training and evaluation scripts
- [docs/DOCS_INDEX.md](../docs/DOCS_INDEX.md) - Complete documentation index
- [IMPROVEMENT_PLAN.md](../IMPROVEMENT_PLAN.md) - Research enhancement roadmap
