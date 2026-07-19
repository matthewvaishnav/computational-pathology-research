# Acquisition-Branch Audit Report

**Generated:** 2026-07-04 13:48:54
**Runtime:** 63.7 s
**Smoke test:** no

## What was audited

This audit loads existing projected features from completed pair-integrity falsification runs and evaluates per-branch metrics. **No training was rerun.** All embeddings are reused from prior runs.

## Datasets / Backbones included

| # | Dataset | Backbone | Runs loaded | Expected |
|---|---|---|---|
| SCORPION_DINOv2 | SCORPION_DINOv2 | — | 75 | 75 |
| canineSCC_DINOv2 | canineSCC_DINOv2 | — | 75 | 75 |
| SCORPION_Phikon | SCORPION_Phikon | — | 75 | 75 |
| SCORPION_ResNet50 | SCORPION_ResNet50 | — | 75 | 75 |

## Branch Separation Summary

Positive scanner_probe_delta = acquisition branch carries MORE scanner information than biological branch.
Negative tissue-retrieval delta = acquisition branch carries LESS tissue identity than biological branch.

### SCORPION_DINOv2

| Condition | Bio scanner probe | Acq scanner probe | Probe Δ | Bio retrieval | Acq retrieval | Retrieval Δ | Bio eff rank | Acq eff rank | Cross-cov |
|---|---|---|---|---|---|---|---|---|
| shuffled_region_pairs | 0.3736 | 0.8520 | +0.4784 | 0.9954 | 0.3911 | -0.6043 | 39.1 | 10.6 | 0.105443 |
| shuffled_sample_pairs | 0.3588 | 0.8558 | +0.4970 | 0.9792 | 0.4180 | -0.5613 | 35.2 | 11.1 | 0.102893 |
| true_pairs | 0.3998 | 0.8582 | +0.4585 | 0.9999 | 0.0944 | -0.9055 | 54.5 | 5.1 | 0.091709 |

### SCORPION_Phikon

| Condition | Bio scanner probe | Acq scanner probe | Probe Δ | Bio retrieval | Acq retrieval | Retrieval Δ | Bio eff rank | Acq eff rank | Cross-cov |
|---|---|---|---|---|---|---|---|---|
| shuffled_region_pairs | 0.4341 | 0.9673 | +0.5333 | 0.9603 | 0.4474 | -0.5129 | 44.4 | 15.4 | 0.111446 |
| shuffled_sample_pairs | 0.4085 | 0.9658 | +0.5572 | 0.8763 | 0.4735 | -0.4028 | 42.5 | 15.6 | 0.112640 |
| true_pairs | 0.5200 | 0.9711 | +0.4510 | 0.9997 | 0.0739 | -0.9258 | 46.8 | 8.6 | 0.088867 |

### SCORPION_ResNet50

| Condition | Bio scanner probe | Acq scanner probe | Probe Δ | Bio retrieval | Acq retrieval | Retrieval Δ | Bio eff rank | Acq eff rank | Cross-cov |
|---|---|---|---|---|---|---|---|---|
| shuffled_region_pairs | 0.3045 | 0.7806 | +0.4761 | 0.8839 | 0.3251 | -0.5588 | 81.7 | 14.3 | 0.085062 |
| shuffled_sample_pairs | 0.3067 | 0.7807 | +0.4740 | 0.8775 | 0.3448 | -0.5327 | 81.4 | 14.8 | 0.084652 |
| true_pairs | 0.3145 | 0.7845 | +0.4701 | 0.9726 | 0.1705 | -0.8021 | 85.1 | 9.2 | 0.089567 |

### canineSCC_DINOv2

| Condition | Bio scanner probe | Acq scanner probe | Probe Δ | Bio retrieval | Acq retrieval | Retrieval Δ | Bio eff rank | Acq eff rank | Cross-cov |
|---|---|---|---|---|---|---|---|---|
| shuffled_region_pairs | 0.3057 | 0.8647 | +0.5590 | 0.7293 | 0.4376 | -0.2917 | 54.5 | 20.3 | 0.087106 |
| shuffled_sample_pairs | 0.4093 | 0.8302 | +0.4209 | 0.7183 | 0.4383 | -0.2799 | 45.3 | 20.5 | 0.096097 |
| true_pairs | 0.3614 | 0.8651 | +0.5037 | 0.9334 | 0.1806 | -0.7528 | 74.0 | 13.8 | 0.089831 |

## Interpretation

- **Average scanner probe Δ (acq − bio):** +0.4708
  (bio=0.3989, acq=0.8697)
- **Average tissue retrieval Δ (acq − bio):** -0.8465
  (bio=0.9764, acq=0.1299)

**Finding: The acquisition-branch audit strongly supports branch separation across the tested datasets/backbones.**

Across the audited settings, the acquisition branch retained high scanner/acquisition recoverability while carrying much lower tissue-identity retrieval than the biological branch. This supports the interpretation that the model learned a useful acquisition/tissue branch separation, rather than only suppressing scanner signal in the biological branch.

## Claim boundaries

- This audit evaluates per-branch metrics on held-out test slides from existing trained models. No new training was performed.
- The factorization architecture was trained with a specific hyperparameter configuration (acquisition_dim=64, biological_dim=256, scanner_adversary_weight=0.5, scanner_acquisition_weight=0.5, scanner_dependence_weight=20.0). Results may not generalize to other configurations.
- Scanner probe accuracy is measured with a linear logistic regression classifier (balanced class weight). Non-linear scanner signatures may be underestimated.
- Tissue retrieval is measured by same-slide nearest-neighbor recall in the projected space. This captures same-tissue identity at the slide level, not finer-grained region matching.
- The audit covers SCORPION (human colorectal cancer, 5-scanner HTA v1.0 protocol) and external canine SCC (5-scanner veterinary oncology). Results are specific to these datasets and scanner ensembles.

## Output files

| File | Description |
|---|---|
| branch_audit_raw_metrics.csv | Per-run, per-branch metrics |
| branch_audit_summary.csv | Aggregated summary by dataset/condition |
| branch_separation_contrasts.csv | Per-run bio-vs-acq deltas |
| experiment_design.json | Audit configuration |
| run_log.txt | Timestamped log |
| acquisition_branch_audit_report.md | This report |
