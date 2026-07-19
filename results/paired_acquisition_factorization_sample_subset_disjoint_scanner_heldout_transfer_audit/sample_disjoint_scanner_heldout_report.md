# Sample-Disjoint Scanner-Heldout Transfer Audit Report

**Generated:** 2026-07-07 12:47:12
**Runtime:** 161.6 s
**Evidence tier:** full (5 held-out scanners, 5 seeds)

## Scientific question

Does the biological branch preserve tissue-category structure under BOTH scanner shift AND sample shift?

## Protocol

Sample-disjoint leave-one-scanner-out:
- Split sample IDs into disjoint train/test sets
- Train: 4 scanners, train samples only
- Test: held-out scanner, test samples only
- No sample appears in both train and test
- No held-out scanner patches appear in training
- Repeated across 5 random sample-split seeds
- Train fraction: 0.7

## Dataset

- Canine SCC DINOv2, fold 0 manifest
- 44 samples, all present in all 5 scanners
- 7 tissue categories
- Class imbalance: Cartilage (10 patches), Bone (195)

## Split diagnostics

- Mean train patches: 2155
- Mean test patches: 266
- Mean train samples: 30
- Mean test samples: 14
- Sample overlap violations: 0
- Held-out scanner in train violations: 0

## Key results

| Representation | Mean balanced acc | Mean macro F1 | Worst-scanner acc | Worst-scanner F1 |
|---|---:|---:|---:|---:|
| `linear_projection_k0` | 0.3590 | 0.3389 | 0.3452 | 0.3044 |
| `linear_projection_k1` | 0.3566 | 0.3374 | 0.3406 | 0.3034 |
| `linear_projection_k16` | 0.3619 | 0.3404 | 0.3514 | 0.3193 |
| `linear_projection_k2` | 0.3589 | 0.3307 | 0.3411 | 0.3158 |
| `linear_projection_k32` | 0.3619 | 0.3404 | 0.3514 | 0.3193 |
| `linear_projection_k4` | 0.3601 | 0.3358 | 0.3465 | 0.3193 |
| `linear_projection_k8` | 0.3619 | 0.3404 | 0.3514 | 0.3193 |
| `original_frozen_features` | 0.3582 | 0.3382 | 0.3451 | 0.3019 |
| `pca_removal_k1` | 0.3202 | 0.3031 | 0.3029 | 0.2737 |
| `pca_removal_k16` | 0.1777 | 0.1661 | 0.1351 | 0.1345 |
| `pca_removal_k2` | 0.3089 | 0.2946 | 0.2915 | 0.2680 |
| `pca_removal_k32` | 0.1587 | 0.1505 | 0.1395 | 0.1260 |
| `pca_removal_k4` | 0.2863 | 0.2727 | 0.2652 | 0.2479 |
| `pca_removal_k8` | 0.2107 | 0.1977 | 0.1807 | 0.1689 |
| `shuffled_sample_acquisition` | 0.3309 | 0.2778 | 0.3096 | 0.2349 |
| `shuffled_sample_biological` | 0.2258 | 0.2154 | 0.1829 | 0.1700 |
| `true_pair_acquisition` | 0.2625 | 0.2123 | 0.2173 | 0.1680 |
| `true_pair_biological` | 0.3192 | 0.3020 | 0.3109 | 0.2863 |

## Per-scanner breakdown (key representations)

### `original_frozen_features`

| Held-out scanner | Balanced acc | Macro F1 | Mean n_test |
|---|---:|---:|---:|
| cs2 | 0.3520 | 0.3368 | 266 |
| gt450 | 0.3493 | 0.3019 | 266 |
| nz20 | 0.3823 | 0.3546 | 266 |
| nz210 | 0.3625 | 0.3600 | 266 |
| p1000 | 0.3451 | 0.3376 | 266 |

### `true_pair_biological`

| Held-out scanner | Balanced acc | Macro F1 | Mean n_test |
|---|---:|---:|---:|
| cs2 | 0.3225 | 0.3013 | 266 |
| gt450 | 0.3302 | 0.3175 | 266 |
| nz20 | 0.3142 | 0.3060 | 266 |
| nz210 | 0.3180 | 0.2863 | 266 |
| p1000 | 0.3109 | 0.2991 | 266 |

### `true_pair_acquisition`

| Held-out scanner | Balanced acc | Macro F1 | Mean n_test |
|---|---:|---:|---:|
| cs2 | 0.2675 | 0.2156 | 266 |
| gt450 | 0.2173 | 0.1680 | 266 |
| nz20 | 0.2890 | 0.2423 | 266 |
| nz210 | 0.2881 | 0.2319 | 266 |
| p1000 | 0.2509 | 0.2039 | 266 |

### `shuffled_sample_biological`

| Held-out scanner | Balanced acc | Macro F1 | Mean n_test |
|---|---:|---:|---:|
| cs2 | 0.1829 | 0.1700 | 266 |
| gt450 | 0.2317 | 0.2172 | 266 |
| nz20 | 0.2544 | 0.2346 | 266 |
| nz210 | 0.2287 | 0.2251 | 266 |
| p1000 | 0.2311 | 0.2299 | 266 |

### `linear_projection_k4`

| Held-out scanner | Balanced acc | Macro F1 | Mean n_test |
|---|---:|---:|---:|
| cs2 | 0.3602 | 0.3452 | 266 |
| gt450 | 0.3649 | 0.3193 | 266 |
| nz20 | 0.3654 | 0.3351 | 266 |
| nz210 | 0.3636 | 0.3510 | 266 |
| p1000 | 0.3465 | 0.3285 | 266 |

## Per-class recall (key representations)

| Representation | Bone | Cartilage | Dermis | Epidermis | Inflamm/Necrosis | SCC | Subcutis |
|---|---:|---:|---:|---:|---:|---:|---:|
| `linear_projection_k4` | 0.2397 | 0.0000 | 0.1782 | 0.6540 | 0.3594 | 0.4009 | 0.3768 |
| `original_frozen_features` | 0.2194 | 0.0000 | 0.2005 | 0.6245 | 0.3705 | 0.3963 | 0.3839 |
| `shuffled_sample_biological` | 0.0948 | 0.0000 | 0.1654 | 0.4018 | 0.1755 | 0.3178 | 0.2299 |
| `true_pair_acquisition` | 0.2679 | 0.0000 | 0.1914 | 0.3798 | 0.2833 | 0.1582 | 0.3409 |
| `true_pair_biological` | 0.2491 | 0.0000 | 0.1751 | 0.5927 | 0.2820 | 0.4115 | 0.2488 |

## Interpretation

Sample-disjoint + scanner-heldout transfer is much harder than
scanner-heldout alone. All representations drop substantially from their
full-scanner-heldout performance. The key question is relative ranking
under this harder test.

### True-pair biological vs original frozen features

- Sample-disjoint balanced accuracy: 0.3582 -> 0.3192 (delta = -0.0391)
- Sample-disjoint macro F1: 0.3382 -> 0.3020 (delta = -0.0361)
- Biological preserves 89% of frozen accuracy.

### True-pair biological vs other representations

- vs true_pair_acquisition: +0.0567
- vs shuffled_sample_biological: +0.0934
- vs pca_removal_k32: +0.1605
- vs linear_projection_k4: -0.0409

### Summary

Under sample-disjoint held-out-scanner transfer, the biological branch
does not beat frozen features or linear projection in raw transfer
accuracy, but it preserves most frozen transfer (89%) while strongly
outperforming acquisition, shuffled-pair, and PCA controls. The
linear_projection_k4 baseline slightly exceeds both frozen and biological
in mean accuracy.

This directly weakens the sample-identity leakage objection: the biological
branch preserves category structure under scanner shift even when samples
are completely disjoint between train and test. The result supports
structured separation -- the biological branch retains transferable
category structure while the acquisition branch captures scanner-specific
features that do not transfer -- rather than best raw classifier
performance.

### True-pair acquisition branch

- Sample-disjoint balanced accuracy: 0.2625
- Biological beats acquisition by 0.0567. The acquisition branch encodes
  scanner-specific features that fail to transfer under combined scanner
  and sample shift.

### Shuffled-sample biological branch (control)

- Sample-disjoint balanced accuracy: 0.2258
- Biological beats shuffled by 0.0934. Breaking pair structure
  substantially degrades sample-disjoint transfer.

## Claim boundaries

- This audit tests whether the biological branch preserves category structure under both scanner shift and sample shift. It does not test clinical utility.
- Test sets are smaller than full scanner-heldout (subset of samples), so variance is higher.
- Cartilage (10 patches) is absent from many test splits. Cartilage-specific claims should not be drawn.
- Does not claim: clinical validation, diagnostic performance, patient-care utility, universal biological factorization, scanner bias solved, or deployment readiness.

## Output files

| File | Description |
|---|---|
| sample_disjoint_scanner_heldout_raw_metrics.csv | Per-run metrics |
| sample_disjoint_scanner_heldout_summary.csv | Aggregated by representation |
| sample_disjoint_scanner_heldout_per_scanner.csv | Per-scanner breakdown |
| sample_disjoint_scanner_heldout_per_class_recall.csv | Per-class recall |
| sample_disjoint_scanner_heldout_split_diagnostics.csv | Split diagnostics |
| experiment_design.json | Experiment configuration |
| run_log.txt | Timestamped run log |
| sample_disjoint_scanner_heldout_report.md | This report |
