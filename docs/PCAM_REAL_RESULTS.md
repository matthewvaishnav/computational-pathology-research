# PatchCamelyon Engineering Result Record

**Original evaluation date:** April 9, 2026  
**Claim boundary updated:** August 2, 2026

## Result

A binary patch classifier was evaluated on one official PatchCamelyon test split
containing 32,768 patches.

| Metric | Estimate | Historical patch-bootstrap 95% interval |
|---|---:|---:|
| Accuracy | `0.8526` | `[0.8483, 0.8563]` |
| ROC AUC | `0.9394` | `[0.9369, 0.9418]` |
| F1 | `0.8507` | `[0.8464, 0.8543]` |

The recorded confusion matrix was:

```text
              Predicted
              Normal  Tumor
Actual Normal  15,837    554
Actual Tumor    4,276 12,101
```

These are patch-level results on this model, checkpoint, preprocessing path, and
test split.

## Supported interpretation

The result demonstrates that the recorded implementation produced nontrivial
patch-level discrimination on the official PCam test set and that the evaluation
pipeline exported reproducible descriptive metrics.

It may be used as an engineering benchmark within this repository.

## Unsupported interpretation

This result does not establish:

- slide-level or patient-level performance;
- independent external validation;
- a clinically validated operating threshold;
- diagnostic sensitivity or specificity in practice;
- cancers, patients, or diagnoses saved;
- workflow or patient benefit;
- state-of-the-art performance;
- statistical superiority to published models evaluated under different
  protocols, hardware, preprocessing, or tuning budgets; or
- clinical or deployment readiness.

The 32,768 patches are not 32,768 independent patients. Patch-bootstrap intervals
must not be described as patient-level or clinical uncertainty.

## Dataset and model context

- Dataset: PatchCamelyon
- Task: binary patch classification
- Test split: 32,768 RGB patches
- Recorded model family: ResNet-18-based classifier with additional learned
  representation and classification components
- Hardware record: RTX 4070 Laptop GPU

Historical timing, throughput, parameter-count, and optimization figures should
be treated as environment-specific engineering notes unless independently
reproduced from the exact commit and configuration.

## Comparison policy

No cross-paper leaderboard is maintained. Published results from unrelated
systems are not directly comparable unless datasets, splits, preprocessing,
model-selection rules, tuning budgets, hardware, and statistical units are
controlled.

## Current authority

The repository-root [`CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md) overrides older
PCam threshold, clinical, superiority, and deployment language.
