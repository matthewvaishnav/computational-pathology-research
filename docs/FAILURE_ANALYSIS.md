# PatchCamelyon Failure Analysis

**Date**: 2026-04-21  
**Status**: ✅ COMPLETE  
**Script**: `scripts/analyze_pcam_failures.py`  
**Results**: `results/pcam_real/failure_analysis/`

## Executive Summary

Analyzed 4,830 misclassified samples from the PCam test set (32,768 total samples) to identify model weaknesses and patterns in errors. The analysis reveals a **conservative model** with high precision but concerning false negative rate for tumor detection.

## Key Findings

### Error Distribution
| Error Type | Count | Rate | Clinical Impact |
|------------|-------|------|-----------------|
| **False Positives** | 554 | 3.38% | Normal tissue misclassified as tumor (low concern) |
| **False Negatives** | 4,276 | 26.11% | **Tumors missed** (high concern) ⚠️ |
| **Total Errors** | 4,830 | 14.74% | Overall error rate |

### Class-Specific Performance
| Class | Total Samples | Correctly Classified | Accuracy |
|-------|---------------|---------------------|----------|
| **Class 0 (Normal)** | 16,391 | 15,837 | 96.62% |
| **Class 1 (Tumor)** | 16,377 | 12,101 | 73.89% |

### Confidence Analysis
- **Correct predictions**: Mean confidence = 0.9653
- **Incorrect predictions**: Mean confidence = 0.8832
- **Gap**: 0.0821 (8.21 percentage points)

The model is less confident when making mistakes, which is a positive indicator for uncertainty quantification.

## Clinical Implications

### ⚠️ High False Negative Rate (26.11%)
**Problem**: Model misses approximately 1 in 4 tumors

**Clinical Risk**: 
- Delayed diagnosis for cancer patients
- Potential for disease progression during missed detection
- Undermines trust in AI-assisted diagnosis

**Severity**: HIGH - False negatives in cancer detection are more dangerous than false positives

### ✅ Low False Positive Rate (3.38%)
**Benefit**: Model rarely misclassifies normal tissue as tumor

**Clinical Impact**:
- Reduces unnecessary biopsies and patient anxiety
- High specificity (96.62%) for normal tissue
- Conservative approach minimizes false alarms

## Model Behavior Analysis

### Conservative Classification Pattern
The model exhibits a **conservative bias toward normal classification**:
- High precision for tumor detection (95.6%) when it does predict tumor
- But low recall (73.9%) - misses many actual tumors
- Suggests the model requires high confidence before predicting tumor

### Confidence Distribution
![Confidence Distribution](../results/pcam_real/failure_analysis/confidence_distribution.png)

**Observations**:
- Correct predictions cluster at high confidence (>0.95)
- Incorrect predictions have broader distribution (0.7-0.95)
- Clear separation suggests confidence can be used for uncertainty estimation

### Error Rates by Type
![Error Rates](../results/pcam_real/failure_analysis/error_rates.png)

**Observations**:
- False negative rate (26.11%) is **7.7x higher** than false positive rate (3.38%)
- Asymmetric error distribution indicates model bias
- Suggests decision threshold optimization could improve balance

## Recommendations

### 1. Adjust Decision Threshold (HIGH PRIORITY)
**Current**: Model uses default 0.5 threshold for binary classification  
**Recommendation**: Lower threshold to increase sensitivity (recall) for tumor detection

**Expected Impact**:
- Reduce false negative rate (catch more tumors)
- Slightly increase false positive rate (acceptable tradeoff)
- Better balance between precision and recall

**Implementation**: Analyze ROC curve to find optimal threshold based on clinical requirements

### 2. Ensemble Methods (MEDIUM PRIORITY)
**Approach**: Train multiple models with different architectures/initializations  
**Benefit**: Combine predictions to reduce variance and improve robustness

**Expected Impact**:
- Improve overall accuracy by 2-5%
- Reduce both false positive and false negative rates
- More reliable predictions through consensus

### 3. Investigate False Negative Patterns (MEDIUM PRIORITY)
**Analysis**: Examine the 4,276 missed tumor samples for common characteristics

**Questions to Answer**:
- Are certain tumor types consistently missed?
- Do false negatives have lower tumor cell density?
- Are there spatial patterns in misclassifications?

**Benefit**: Identify specific weaknesses to address in model improvement

### 4. Confidence-Based Rejection (LOW PRIORITY)
**Approach**: Flag low-confidence predictions for human review

**Implementation**:
- Set confidence threshold (e.g., 0.85)
- Route low-confidence cases to pathologist
- Hybrid AI-human workflow

**Expected Impact**:
- Reduce errors on uncertain cases
- Maintain high confidence in automated predictions
- Practical deployment strategy

## Generated Artifacts

### 1. Confidence Distribution Plot
**File**: `results/pcam_real/failure_analysis/confidence_distribution.png`  
**Content**: Histogram comparing confidence for correct vs incorrect predictions

### 2. Error Rates Visualization
**File**: `results/pcam_real/failure_analysis/error_rates.png`  
**Content**: Bar chart showing false positive and false negative rates

### 3. Detailed Analysis Report
**File**: `results/pcam_real/failure_analysis/failure_analysis.json`  
**Content**: Complete analysis including:
- Indices of all 554 false positive samples
- Indices of all 4,276 false negative samples
- Class-specific statistics
- Error rate calculations
- Summary metrics

## Reproducibility

### Running the Analysis
```bash
python scripts/analyze_pcam_failures.py \
  --results results/pcam_real/metrics.json \
  --output-dir results/pcam_real/failure_analysis
```

### Requirements
- NumPy (array operations)
- Matplotlib (visualizations)
- Seaborn (plot styling)
- Input: `metrics.json` with predictions, labels, and probabilities

### Output Structure
```
results/pcam_real/failure_analysis/
├── confidence_distribution.png  # Confidence histogram
├── error_rates.png              # Error rate bar chart
└── failure_analysis.json        # Complete analysis report
```

## Comparison to Clinical Standards

### Pathologist Performance (Literature)
- **Sensitivity (Recall)**: 85-95% for tumor detection
- **Specificity**: 90-98% for normal tissue
- **Inter-observer agreement**: κ = 0.7-0.8 (substantial)

### Our Model Performance
- **Sensitivity (Recall)**: 73.89% ⚠️ (below pathologist range)
- **Specificity**: 96.62% ✅ (within pathologist range)
- **Overall Accuracy**: 85.26%

**Gap Analysis**: Model specificity matches human performance, but sensitivity needs improvement to reach clinical utility.

## Next Steps

### Immediate Actions
1. ✅ Complete failure analysis (DONE)
2. 🔄 Implement threshold optimization
3. 🔄 Analyze false negative patterns

### Future Research
1. Cross-validation for robustness assessment
2. Hyperparameter tuning focused on recall improvement
3. Ensemble methods for performance boost
4. Test on CAMELYON16 for generalization
5. Compare to pathologist performance with expert annotations

## Conclusion

The failure analysis reveals a **conservative model** with excellent specificity (96.62%) but concerning sensitivity (73.89%) for tumor detection. The high false negative rate (26.11%) is the primary limitation for clinical deployment.

**Key Takeaway**: Model requires threshold optimization and potentially ensemble methods to achieve clinical-grade performance for tumor detection. The current model is suitable for research but needs improvement before clinical use.

---

**Status**: Analysis complete ✅  
**Clinical readiness**: Not ready (high false negative rate) ⚠️  
**Research value**: High (identifies clear improvement path) ✅  
**Next priority**: Threshold optimization 🎯
