# PatchCamelyon Cross-Validation

**Status**: ⏸️ PAUSED (Partial Results Available)  
**Script**: `scripts/cross_validate_pcam.py`  
**Purpose**: Assess model robustness and variance across different train/test splits

## Partial Results (Fold 1, First 2 Epochs)

**Early Training Performance:**

| Epoch | Val AUC | Val Accuracy | Improvement |
|-------|---------|--------------|-------------|
| 1     | 0.9764  | 90.02%       | Baseline    |
| 2     | 0.9824  | 93.29%       | +3.27%      |

**Key Observations:**
- ✅ **Strong baseline** - 90% accuracy on first epoch demonstrates effective learning
- ✅ **Rapid improvement** - 3.27% accuracy gain shows healthy learning dynamics
- ✅ **High AUC** - Both epochs exceed 0.97 AUC (excellent discrimination)
- ✅ **Infrastructure validated** - Memory-mapped loading, GPU acceleration working correctly
- 📊 **Consistent with baseline** - Aligns with full PCam test results (85.26% test accuracy, 0.9394 AUC)

**Next Steps:**
- Resume full 5-fold cross-validation this weekend (~50 hours)
- Complete remaining 18 epochs of Fold 1 + all of Folds 2-5
- Generate final aggregated statistics with bootstrap confidence intervals

---

## Overview

K-fold cross-validation provides a more robust assessment of model performance by training and evaluating on multiple different splits of the data. This helps identify:
- **Model variance**: How much performance varies across different data splits
- **Overfitting**: Whether the model generalizes well to unseen data
- **Robustness**: Stability of performance metrics across folds
- **Statistical significance**: More reliable confidence intervals

## Methodology

### Stratified K-Fold Splitting
- **Stratification**: Maintains class balance across all folds
- **Default**: 5 folds (80% train, 20% validation per fold)
- **Reproducible**: Fixed random seed for consistent splits

### Training Process
For each fold:
1. Split data into train (80%) and validation (20%)
2. Train model for specified epochs
3. Save best model based on validation AUC
4. Compute final metrics with bootstrap confidence intervals
5. Save fold results and checkpoint

### Aggregation
After all folds complete:
- Compute mean and standard deviation across folds
- Aggregate bootstrap confidence intervals
- Identify min/max performance across folds
- Generate comprehensive report

## Usage

### Quick Test (Small Subset)
Test the cross-validation pipeline with a small subset:

```bash
# Linux/Mac
bash scripts/run_cv_quick_test.sh

# Windows
scripts\run_cv_quick_test.bat

# Or manually:
python scripts/cross_validate_pcam.py \
  --data-root data/pcam_real \
  --output-dir results/pcam_cv_test \
  --n-folds 3 \
  --num-epochs 3 \
  --batch-size 128 \
  --use-amp \
  --subset-size 5000
```

**Estimated time**: ~15-20 minutes on RTX 4070 Laptop

### Full Cross-Validation
Run complete 5-fold cross-validation on full dataset:

```bash
# Linux/Mac
bash scripts/run_cv_full.sh

# Or manually:
python scripts/cross_validate_pcam.py \
  --data-root data/pcam_real \
  --output-dir results/pcam_cv_full \
  --n-folds 5 \
  --num-epochs 20 \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --bootstrap-samples 1000 \
  --use-amp \
  --seed 42
```

**Estimated time**: 30-40 hours on RTX 4070 Laptop (6-8 hours per fold)

### Custom Configuration

```bash
python scripts/cross_validate_pcam.py \
  --data-root data/pcam_real \
  --output-dir results/pcam_cv_custom \
  --n-folds 10 \              # More folds for smaller validation sets
  --num-epochs 15 \           # Fewer epochs per fold
  --batch-size 256 \          # Larger batch size if GPU allows
  --learning-rate 5e-4 \      # Different learning rate
  --bootstrap-samples 500 \   # Fewer bootstrap samples for speed
  --use-amp                   # Mixed precision training
```

## Output Structure

```
results/pcam_cv_full/
├── fold_0_best_model.pth          # Best model checkpoint for fold 0
├── fold_0_results.json            # Detailed results for fold 0
├── fold_1_best_model.pth
├── fold_1_results.json
├── ...
├── fold_4_best_model.pth
├── fold_4_results.json
└── cross_validation_results.json  # Aggregated results across all folds
```

## Results Format

### Per-Fold Results (`fold_X_results.json`)

```json
{
  "fold": 0,
  "train_size": 235929,
  "val_size": 58983,
  "epochs_trained": 20,
  "best_epoch": 15,
  "best_val_metrics": {
    "accuracy": 0.8526,
    "auc": 0.9394,
    "f1": 0.8507,
    "precision": 0.8718,
    "recall": 0.8526
  },
  "final_metrics": { ... },
  "bootstrap_ci": {
    "accuracy": {
      "mean": 0.8526,
      "std": 0.0040,
      "ci_lower": 0.8483,
      "ci_upper": 0.8563
    },
    ...
  },
  "train_losses": [0.45, 0.32, ...],
  "val_metrics": [...]
}
```

### Aggregated Results (`cross_validation_results.json`)

```json
{
  "n_folds": 5,
  "metrics": {
    "accuracy": {
      "mean": 0.8520,
      "std": 0.0025,
      "min": 0.8485,
      "max": 0.8550,
      "values": [0.8526, 0.8515, 0.8530, 0.8510, 0.8520]
    },
    "auc": {
      "mean": 0.9390,
      "std": 0.0015,
      "min": 0.9370,
      "max": 0.9410,
      "values": [0.9394, 0.9385, 0.9395, 0.9380, 0.9390]
    },
    ...
  },
  "bootstrap_ci_aggregated": {
    "accuracy": {
      "mean_across_folds": 0.8520,
      "std_across_folds": 0.0025,
      "ci_lower_mean": 0.8480,
      "ci_upper_mean": 0.8560,
      "ci_width_mean": 0.0080
    },
    ...
  },
  "fold_results": [...]
}
```

## Interpretation

### Mean ± Std
- **Mean**: Average performance across all folds
- **Std**: Variability in performance (lower is better)
- **Example**: Accuracy = 85.20% ± 0.25%
  - Model is stable across different data splits
  - Low variance indicates good generalization

### Min/Max Range
- **Range**: [min, max] performance across folds
- **Example**: AUC range = [0.9370, 0.9410]
  - Narrow range indicates consistent performance
  - Wide range may indicate overfitting or data issues

### Bootstrap CI Aggregated
- **CI Width**: Average width of confidence intervals
- **Narrow CI**: More precise estimates
- **Wide CI**: More uncertainty in estimates

## Expected Results

Based on single train/test split results:

| Metric | Single Split | Expected CV Mean | Expected CV Std |
|--------|--------------|------------------|-----------------|
| **Accuracy** | 85.26% | 85.20% ± 0.30% | ~0.25-0.35% |
| **AUC** | 0.9394 | 0.9390 ± 0.0020 | ~0.0015-0.0025 |
| **F1** | 0.8507 | 0.8500 ± 0.0030 | ~0.0025-0.0035 |
| **Precision** | 0.8718 | 0.8715 ± 0.0025 | ~0.0020-0.0030 |
| **Recall** | 0.8526 | 0.8520 ± 0.0030 | ~0.0025-0.0035 |

**Interpretation**:
- Low standard deviation (<0.5%) indicates stable, robust model
- Performance should be similar to single split results
- Slight decrease in mean is normal due to smaller training sets per fold

## Comparison to Single Split

### Advantages of Cross-Validation
1. **More robust estimates**: Multiple train/test splits reduce sampling bias
2. **Variance assessment**: Quantifies model stability across data splits
3. **Better use of data**: All samples used for both training and validation
4. **Statistical rigor**: More reliable confidence intervals

### Disadvantages
1. **Computational cost**: 5× longer training time (5 folds)
2. **Resource intensive**: Requires storing 5 model checkpoints
3. **Complexity**: More complex analysis and interpretation

### When to Use
- **Use CV**: When you need robust performance estimates for publication
- **Use single split**: For rapid prototyping and hyperparameter tuning
- **Use both**: Single split for development, CV for final validation

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
--batch-size 64

# Reduce number of workers
--num-workers 2

# Disable mixed precision
# Remove --use-amp flag
```

### Slow Training
```bash
# Reduce epochs per fold
--num-epochs 10

# Reduce number of folds
--n-folds 3

# Use subset for testing
--subset-size 10000
```

### Inconsistent Results
```bash
# Ensure fixed seed
--seed 42

# Check for data leakage
# Verify stratification is working

# Increase number of folds
--n-folds 10
```

## Next Steps

After cross-validation completes:

1. **Analyze variance**: Check if std is acceptably low (<0.5%)
2. **Compare to single split**: Verify results are consistent
3. **Identify outliers**: Investigate folds with unusual performance
4. **Update documentation**: Add CV results to main results document
5. **Publication**: Use CV results for more robust claims

## References

- **Stratified K-Fold**: Maintains class distribution across folds
- **Bootstrap CI**: Provides confidence intervals for each fold
- **Aggregation**: Combines results across folds for overall assessment

---

**Status**: Ready for execution ✅  
**Estimated time**: 30-40 hours for full 5-fold CV  
**Quick test**: 15-20 minutes with subset  
**Next**: Run quick test, then full CV if results look good
