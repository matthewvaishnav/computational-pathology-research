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

### Historical numerical position

The repository's historical comparison table collected 10 external PCam AUC
values. The recorded repository result of **0.9394 ROC AUC was numerically higher
than every value in that table**.

That descriptive fact is retained. It is stronger than saying only that the
model achieved nontrivial discrimination, but it is not the same as a matched
superiority experiment: the external values came from different studies,
training/tuning procedures, preprocessing paths, hardware, and reporting
conventions.

## Unsupported interpretation

This result does not establish:

- slide-level or patient-level performance;
- independent external validation;
- a clinically validated operating threshold;
- diagnostic sensitivity or specificity in practice;
- cancers, patients, or diagnoses saved;
- workflow or patient benefit;
- a protocol-controlled or statistically established state-of-the-art claim
  against unrelated published models;
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

The historical cross-paper table may be used for **descriptive numerical
context**: 0.9394 was the highest AUC value in that collected table. It must not
be treated as a controlled leaderboard or inferential superiority test.

A stronger comparative claim requires datasets, splits, preprocessing,
model-selection rules, tuning budgets, hardware, and statistical units to be
controlled.

## Current authority

The repository-root [`CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md) overrides older
PCam threshold, clinical, superiority, and deployment language.
