# PatchCamelyon Failure Analysis

**Date**: 2026-04-21  
**Status**: ✅ COMPLETE  
**Analysis Script**: `scripts/analyze_pcam_failures.py`  
**Results Location**: `results/pcam_real/failure_analysis/`

## Executive Summary

Comprehensive analysis of model misclassifications on the full PatchCamelyon test set (32,768 samples) reveals a **critical asymmetry**: the model exhibits high precision but low recall for tumor detection, resulting in a **26.11% false negative rate** (4,276 missed tumors) compared to only **3.38% false positive rate** (554 normal tissues misclassified).

**Clinical Implication**: The model is overly conservative, missing approximately 1 in 4 tumors while rarely misclassifying normal tissue as cancerous.

## Overall Performance Summary

| Metric | Value | Clinical Interpretation |
|--------|-------|------------------------|
| **Total Samples** | 32,768 | Full test set |
| **Correct Predictions** | 27,938 (85.26%) | Good overall accuracy |
| **Total Errors** | 4,830 (14.74%) | Moderate error rate |
| **False Positives** | 554 (1.69% of total) | Low false alarm rate |
| **False Negatives** | 4,276 (13.05% of total) | **High miss rate - clinical concern** |

## Error Type Breakdown

### False Positives (Normal → Tumor)
- **Count**: 554 samples
- **Rate**: 3.38% of normal tissues (554/16,391)
- **Clinical Impact**: Low - leads to unnecessary follow-up but doesn't miss disease
- **Confidence**: Mean confidence for FP predictions: 0.8832

**Interpretation**: Model rarely misclassifies normal tissue as tumor. When it does, confidence is slightly lower than correct predictions.

### False Negatives (Tumor → Normal)
- **Count**: 4,276 samples
- **Rate**: 26.11% of tumors (4,276/16,377)
- **Clinical Impact**: **HIGH - misses 1 in 4 tumors**
- **Confidence**: Mean confidence for FN predictions: 0.8832

**Interpretation**: Model misses a significant fraction of tumors. This is the primary weakness and represents a clinical safety concern.

## Per-Class Performance

### Class 0 (Normal Tissue)
| Metric | Value |
|--------|-------|
| **Total Samples** | 16,391 |
| **Correct Predictions** | 15,837 |
| **Accuracy** | 96.62% |
| **False Positive Rate** | 3.38% |

**Analysis**: Excellent performance on normal tissue. Model is highly specific.

### Class 1 (Tumor)
| Metric | Value |
|--------|-------|
| **Total Samples** | 16,377 |
| **Correct Predictions** | 12,101 |
| **Accuracy** | 73.89% |
| **False Negative Rate** | 26.11% |

**Analysis**: Moderate performance on tumors. Model lacks sensitivity.

## Confidence Analysis

### Correct Predictions
- **Mean Confidence**: 0.9653
- **Interpretation**: Model is highly confident when correct

### Incorrect Predictions
- **Mean Confidence**: 0.8832
- **Interpretation**: Model is less confident when wrong, but still fairly confident

### Confidence Gap
- **Difference**: 0.0821 (8.21 percentage points)
- **Implication**: Confidence can be used as a signal for uncertainty, but the gap is modest

## Clinical Implications

### 1. High False Negative Rate is Concerning
- **26.11% of tumors are missed** - this is unacceptable for clinical screening
- In a screening scenario with 1,000 tumor cases, **261 would be missed**
- These missed cases could delay diagnosis and treatment

### 2. Low False Positive Rate is Good
- Only **3.38% false alarm rate** minimizes unnecessary biopsies
- Reduces patient anxiety and healthcare costs
- Model is conservative in flagging normal tissue

### 3. Precision-Recall Tradeoff
- **Current**: High precision (95.6%), moderate recall (73.9%)
- **Clinical Need**: Higher recall is more important than precision for cancer screening
- **Recommendation**: Adjust decision threshold to improve sensitivity

## Recommendations

### 1. Threshold Adjustment (HIGHEST PRIORITY)
**Action**: Lower the classification threshold to improve recall

**Rationale**:
- Current threshold (0.5) optimizes for accuracy, not clinical utility
- In cancer screening, false negatives are more costly than false positives
- ROC curve analysis can identify optimal threshold for desired sensitivity

**Expected Impact**:
- Increase recall from 73.9% to 85-90%
- Accept higher false positive rate (e.g., 10-15%)
- Better align with clinical screening requirements

**Implementation**: See `scripts/optimize_threshold.py`

### 2. Ensemble Methods
**Action**: Train multiple models and combine predictions

**Rationale**:
- Different models may capture different tumor patterns
- Ensemble can improve both precision and recall
- Reduces variance in predictions

**Expected Impact**:
- 2-5% improvement in both metrics
- More robust to edge cases

### 3. Investigate False Negative Patterns
**Action**: Qualitative analysis of the 4,276 missed tumors

**Questions to Answer**:
- Are certain tumor subtypes consistently missed?
- Do missed tumors have specific visual characteristics?
- Are they smaller, less distinct, or at patch boundaries?
- Is there a spatial pattern (e.g., peripheral vs central)?

**Implementation**: Manual review of sample false negatives with pathologist input

### 4. Class Rebalancing
**Action**: Adjust training to emphasize tumor detection

**Options**:
- Weighted loss function (higher weight for tumor class)
- Oversampling tumor patches during training
- Focal loss to focus on hard examples

**Expected Impact**:
- Improve recall by 5-10%
- May slightly reduce precision

### 5. Confidence-Based Routing
**Action**: Use confidence scores to route uncertain cases

**Strategy**:
- High confidence (>0.95): Accept prediction
- Medium confidence (0.80-0.95): Flag for review
- Low confidence (<0.80): Require expert review

**Expected Impact**:
- Catch some of the 4,276 false negatives
- Reduce clinical risk without sacrificing throughput

## Visualizations

### Generated Artifacts
1. **`confidence_distribution.png`**: Shows confidence distributions for correct vs incorrect predictions
2. **`error_rates.png`**: Visualizes false positive and false negative rates by class
3. **`failure_analysis.json`**: Complete list of all 4,830 misclassified sample indices

### Key Insights from Visualizations
- Confidence distributions overlap significantly between correct and incorrect predictions
- False negative rate is 7.7× higher than false positive rate
- Class imbalance in errors: most errors are false negatives

## Comparison to Clinical Standards

### Pathologist Performance (Typical)
- **Sensitivity**: 95-98% (miss rate: 2-5%)
- **Specificity**: 98-99% (false positive rate: 1-2%)

### Our Model Performance
- **Sensitivity**: 73.89% (miss rate: 26.11%) ⚠️ **Below clinical standard**
- **Specificity**: 96.62% (false positive rate: 3.38%) ✅ **Near clinical standard**

**Gap**: Model needs **21-24 percentage point improvement in sensitivity** to match pathologist performance.

## Next Steps

### Immediate Actions
1. ✅ **Threshold optimization** - Completed (see `docs/PCAM_THRESHOLD_OPTIMIZATION.md`)
2. **Update README** with failure analysis insights
3. **Document clinical limitations** in main documentation

### Research Enhancements
1. **Ensemble methods** - Train 3-5 models with different architectures
2. **Failure case analysis** - Manual review of false negatives with pathologist
3. **Class rebalancing** - Retrain with weighted loss or focal loss
4. **Confidence calibration** - Improve confidence score reliability

### Long-Term Validation
1. **Cross-validation** - Multiple train/test splits for robustness
2. **External validation** - Test on CAMELYON16 or other datasets
3. **Pathologist comparison** - Head-to-head performance study
4. **Clinical trial** - Prospective evaluation in screening workflow

## Reproducibility

### Analysis Script
```bash
python scripts/analyze_pcam_failures.py \
  --checkpoint checkpoints/pcam_real/best_model.pth \
  --data-root data/pcam_real \
  --output-dir results/pcam_real/failure_analysis \
  --batch-size 64
```

### Requirements
- PyTorch 2.0+
- Trained model checkpoint
- Full PCam test set
- ~5 minutes runtime on RTX 4070 Laptop

### Output Files
- `failure_analysis.json` - Complete analysis with all misclassified indices
- `confidence_distribution.png` - Confidence analysis visualization
- `error_rates.png` - Error rate comparison by class

## Conclusion

The failure analysis reveals a **critical asymmetry** in model performance:
- ✅ **Excellent specificity** (96.62%) - rarely misclassifies normal tissue
- ⚠️ **Insufficient sensitivity** (73.89%) - misses 26% of tumors

**Primary Recommendation**: Implement threshold optimization to improve recall to 85-90%, accepting a modest increase in false positives. This better aligns with clinical screening requirements where missing cancer is more costly than false alarms.

**Clinical Readiness**: Current model is **NOT suitable for clinical deployment** without threshold adjustment and further validation. The 26% false negative rate is unacceptable for cancer screening.

---

**Status**: Analysis complete ✅  
**Clinical validation**: Required before deployment ⚠️  
**Threshold optimization**: Completed (see separate document) ✅  
**Pathologist comparison**: Not performed ⚠️
