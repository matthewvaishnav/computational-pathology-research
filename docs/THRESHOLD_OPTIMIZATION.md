# PatchCamelyon threshold sensitivity analysis

**Status:** Retrospective benchmark analysis only  
**Dataset:** PatchCamelyon test split  
**Clinical status:** None

## Scientific boundary

The thresholds in this document were selected and evaluated using the same
PatchCamelyon test predictions. They are therefore **post-hoc operating-point
illustrations**, not independently validated thresholds.

The unit of analysis is a benchmark image patch, not a patient, diagnosis,
slide-level clinical decision, or pathology workflow. These results must not be
described as diagnoses saved, cancers detected, lives saved, clinical benefit,
clinical readiness, acceptable clinical workload, or deployment evidence.

A valid future threshold study must:

1. select the operating point using validation data only;
2. freeze the threshold before evaluating an untouched test set;
3. report calibration and uncertainty;
4. use an endpoint appropriate to the intended unit of analysis;
5. undergo external and workflow validation before any clinical interpretation.

## Retrospective test-set operating points

These numbers describe the trade-off observed on the same test predictions used
to choose the thresholds.

| Selection rule | Threshold | Accuracy | Sensitivity | Specificity | F1 | False negatives | False positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default | 0.500 | 85.3% | 73.9% | 96.6% | 0.834 | 4,276 | 554 |
| Maximum Youden J on test | 0.102 | 87.2% | 84.9% | 89.6% | 0.869 | 2,473 | 1,705 |
| Maximum F1 on test | 0.087 | 87.1% | 86.4% | 87.9% | 0.870 | 2,227 | 1,983 |
| Test point nearest 90% sensitivity | 0.051 | 85.2% | 90.0% | 80.3% | 0.858 | 1,639 | 3,226 |
| Test point nearest 95% sensitivity | 0.023 | 79.4% | 95.0% | 63.8% | 0.821 | 819 | 5,933 |

The 61.7% reduction in false-negative **patch predictions** at threshold 0.051
is a mathematical comparison within this test set. It is not an estimate of
patient benefit and must not be transported to a clinical population.

## Permitted interpretation

The model's ranking performance allows different sensitivity-specificity
trade-offs through threshold selection. Lower thresholds increase sensitivity
while increasing false positives. Because selection occurred on the test set,
the table is exploratory and requires confirmation under a validation-selected,
untouched-test protocol.

## Required rerun

Use validation predictions to select one or more preregistered operating points,
write the selected thresholds to a frozen configuration, and then evaluate them
once on an untouched test set. Until that rerun exists, no threshold in this
document is recommended for inference or deployment.

## Reproducibility record

Historical artifacts remain under:

- `results/pcam_real/threshold_optimization/`
- `scripts/optimize_threshold.py`

They are retained as a record of the retrospective analysis, not as clinical or
confirmatory evidence.
