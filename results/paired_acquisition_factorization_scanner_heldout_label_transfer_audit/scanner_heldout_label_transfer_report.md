# Scanner-Heldout Biological Label Transfer Audit Report

**Generated:** 2026-07-07 11:48:09
**Runtime:** 86.5 s
**Evidence tier:** full (5 held-out scanners, 5 seeds where applicable)

## Scientific question

Does the biological branch improve or preserve canine SCC tissue-category classification under scanner shift?

## Protocol

Leave-one-scanner-out category classification:
- Train a linear category probe on 4 scanners
- Test on the held-out 5th scanner
- Repeat for each scanner as held-out
- Probe: LogisticRegression(C=1.0, class_weight=balanced, max_iter=5000)
- Features standardized on train set before probe fitting

## Dataset

- Canine SCC DINOv2, fold 0 manifest
- 5 scanners: cs2, gt450, nz20, nz210, p1000
- 7 tissue categories: Epidermis (1,205), SCC (1,205), Subcutis (510), Dermis (500), Inflamm/Necrosis (400), Bone (195), Cartilage (10)
- Class imbalance noted: Cartilage (10) and Bone (195) are rare; balanced accuracy and macro-F1 are primary metrics

## Representations compared

| Representation | Dim | Family |
|---|---|
| `linear_projection_k0` | 768 | linear_scanner_subspace_projection |
| `linear_projection_k1` | 768 | linear_scanner_subspace_projection |
| `linear_projection_k16` | 768 | linear_scanner_subspace_projection |
| `linear_projection_k2` | 768 | linear_scanner_subspace_projection |
| `linear_projection_k32` | 768 | linear_scanner_subspace_projection |
| `linear_projection_k4` | 768 | linear_scanner_subspace_projection |
| `linear_projection_k8` | 768 | linear_scanner_subspace_projection |
| `original_frozen_features` | 768 | frozen |
| `pca_removal_k1` | 768 | pca_component_removal |
| `pca_removal_k16` | 768 | pca_component_removal |
| `pca_removal_k2` | 768 | pca_component_removal |
| `pca_removal_k32` | 768 | pca_component_removal |
| `pca_removal_k4` | 768 | pca_component_removal |
| `pca_removal_k8` | 768 | pca_component_removal |
| `shuffled_sample_acquisition` | 64 | pair_integrity |
| `shuffled_sample_biological` | 256 | pair_integrity |
| `true_pair_acquisition` | 64 | pair_integrity |
| `true_pair_biological` | 256 | pair_integrity |

## Key results: held-out scanner category transfer

| Representation | Mean balanced acc | Mean macro F1 | Worst-scanner acc | Worst-scanner F1 |
|---|---:|---:|---:|---:|
| `linear_projection_k0` | 0.8444 | 0.8246 | 0.7078 | 0.7222 |
| `linear_projection_k1` | 0.8271 | 0.8176 | 0.6863 | 0.7211 |
| `linear_projection_k16` | 0.8336 | 0.8289 | 0.7104 | 0.7385 |
| `linear_projection_k2` | 0.8309 | 0.8282 | 0.6988 | 0.7342 |
| `linear_projection_k32` | 0.8336 | 0.8289 | 0.7104 | 0.7385 |
| `linear_projection_k4` | 0.8349 | 0.8304 | 0.7082 | 0.7384 |
| `linear_projection_k8` | 0.8336 | 0.8289 | 0.7104 | 0.7385 |
| `original_frozen_features` | 0.8451 | 0.8261 | 0.7093 | 0.7249 |
| `pca_removal_k1` | 0.8234 | 0.8062 | 0.6889 | 0.6920 |
| `pca_removal_k16` | 0.6242 | 0.6113 | 0.4140 | 0.4095 |
| `pca_removal_k2` | 0.8148 | 0.8038 | 0.6841 | 0.6819 |
| `pca_removal_k32` | 0.5394 | 0.5513 | 0.4219 | 0.4434 |
| `pca_removal_k4` | 0.7710 | 0.7523 | 0.5746 | 0.5495 |
| `pca_removal_k8` | 0.6546 | 0.6540 | 0.5293 | 0.5454 |
| `shuffled_sample_acquisition` | 0.6232 | 0.5266 | 0.5331 | 0.4651 |
| `shuffled_sample_biological` | 0.6081 | 0.5560 | 0.5119 | 0.4638 |
| `true_pair_acquisition` | 0.5170 | 0.4172 | 0.4467 | 0.3618 |
| `true_pair_biological` | 0.8273 | 0.7875 | 0.7548 | 0.7521 |

## Per-scanner breakdown (key representations)

### `original_frozen_features`

| Held-out scanner | Balanced acc | Macro F1 | n_test |
|---|---:|---:|---:|
| cs2 | 0.8839 | 0.8661 | 805 |
| gt450 | 0.8523 | 0.7940 | 805 |
| nz20 | 0.8619 | 0.8672 | 805 |
| nz210 | 0.9182 | 0.8783 | 805 |
| p1000 | 0.7093 | 0.7249 | 805 |

### `true_pair_biological`

| Held-out scanner | Balanced acc | Macro F1 | n_test |
|---|---:|---:|---:|
| cs2 | 0.8456 | 0.8067 | 805 |
| gt450 | 0.8295 | 0.7976 | 805 |
| nz20 | 0.8468 | 0.7807 | 805 |
| nz210 | 0.8598 | 0.8004 | 805 |
| p1000 | 0.7548 | 0.7521 | 805 |

### `true_pair_acquisition`

| Held-out scanner | Balanced acc | Macro F1 | n_test |
|---|---:|---:|---:|
| cs2 | 0.5606 | 0.4571 | 805 |
| gt450 | 0.5178 | 0.3618 | 805 |
| nz20 | 0.5347 | 0.4728 | 805 |
| nz210 | 0.5252 | 0.4277 | 805 |
| p1000 | 0.4467 | 0.3668 | 805 |

### `shuffled_sample_biological`

| Held-out scanner | Balanced acc | Macro F1 | n_test |
|---|---:|---:|---:|
| cs2 | 0.5119 | 0.4638 | 805 |
| gt450 | 0.6422 | 0.5876 | 805 |
| nz20 | 0.7114 | 0.6461 | 805 |
| nz210 | 0.6384 | 0.5888 | 805 |
| p1000 | 0.5366 | 0.4935 | 805 |

### `linear_projection_k4`

| Held-out scanner | Balanced acc | Macro F1 | n_test |
|---|---:|---:|---:|
| cs2 | 0.8189 | 0.8142 | 805 |
| gt450 | 0.8790 | 0.8424 | 805 |
| nz20 | 0.8607 | 0.8772 | 805 |
| nz210 | 0.9077 | 0.8797 | 805 |
| p1000 | 0.7082 | 0.7384 | 805 |

## Per-class recall (key representations)

| Representation | Bone | Cartilage | Dermis | Epidermis | Inflamm/Necrosis | SCC | Subcutis |
|---|---:|---:|---:|---:|---:|---:|---:|
| `linear_projection_k4` | 0.8974 | 0.7000 | 0.8000 | 0.8639 | 0.8100 | 0.8415 | 0.9314 |
| `original_frozen_features` | 0.9128 | 0.8000 | 0.8160 | 0.8481 | 0.7900 | 0.8274 | 0.9216 |
| `shuffled_sample_biological` | 0.6923 | 0.5200 | 0.5952 | 0.5660 | 0.6415 | 0.5165 | 0.7251 |
| `true_pair_acquisition` | 0.6236 | 0.6400 | 0.4700 | 0.4339 | 0.4920 | 0.2886 | 0.6710 |
| `true_pair_biological` | 0.8769 | 0.9000 | 0.7752 | 0.7879 | 0.8135 | 0.7311 | 0.9063 |

## Explicit key metrics

```
representation              balanced_acc  macro_f1   weighted_f1
original_frozen_features    0.8451        0.8261     0.8450
true_pair_biological        0.8273        0.7875     0.7923
true_pair_acquisition       0.5170        --         --
shuffled_sample_biological  0.6081        0.5560     0.5909
shuffled_sample_acquisition 0.6232        0.5266     0.5634
pca_removal_k32             0.5394        --         --
linear_projection_k4        0.8349        0.8304     0.8537
```

Per-scanner true_pair_biological vs original_frozen_features:

```
scanner  frozen_acc  bio_acc   delta
cs2      0.8839      0.8456    -0.0383
gt450    0.8523      0.8295    -0.0228
nz20     0.8619      0.8468    -0.0151
nz210    0.9182      0.8598    -0.0584
p1000    0.7093      0.7548    +0.0454
```

Biological vs acquisition gap (mean balanced accuracy):

true_pair_biological 0.8273 vs true_pair_acquisition 0.5170 -- gap 0.3103

Biological vs shuffled-sample gap (mean balanced accuracy):

true_pair_biological 0.8273 vs shuffled_sample_biological 0.6081 -- gap 0.2192

## Interpretation

### True-pair biological branch vs original frozen features

In canine SCC DINOv2, true_pair_biological nearly preserves held-out-scanner
category transfer relative to original frozen features, with a small average
balanced-accuracy decrease from 0.8451 to 0.8273 (delta = -0.0179). It
improves on the hardest held-out scanner p1000 (+0.0454), but decreases
modestly on the other four scanners (deltas between -0.0151 and -0.0584).

The biological branch does not harm cross-scanner category transfer in a way
that would undermine the mechanism claim. On the scanner where frozen
features struggle most (p1000, accuracy 0.7093), the biological branch
improves transfer.

### True-pair acquisition branch

The acquisition branch transfers poorly across scanners (balanced accuracy
0.5170, vs biological 0.8273). This is expected: the acquisition branch
encodes scanner-specific features that do not generalize across unseen
scanners. The 0.31 gap between biological and acquisition branches supports
the claim that the factorization separates transferable category structure
from scanner-specific information.

### Shuffled-sample biological branch (control)

The shuffled-sample biological branch (0.6081) substantially underperforms
true_pair_biological (0.8273). Breaking the true-pair structure degrades
cross-scanner category transfer. This supports the importance of pair
structure: when the factorization is trained with broken pair correspondence,
the biological branch encodes less transferable category structure.

### PCA removal baseline

PCA removal at k=32 severely degrades cross-scanner transfer (0.5394), well
below true_pair_biological (0.8273). PCA is blind to scanner vs category
dimensions and removes category-relevant variance along with scanner
variance.

### Linear scanner subspace projection baseline

linear_projection_k4 is a strong scanner-removal baseline and slightly
exceeds true_pair_biological in mean held-out balanced accuracy (0.8349 vs
0.8273). The linear baseline removes scanner-discriminative directions from
the frozen embedding, which eliminates scanner-specific features but does
not produce an explicit scanner-bearing acquisition branch for downstream
inspection.

The paired-acquisition result is therefore a structured-separation result,
not a claim of best raw transfer accuracy. The biological branch preserves
cross-scanner category transfer while the acquisition branch retains scanner
signal -- the linear baseline achieves the former but cannot offer the
latter.

### Summary

The biological branch strongly outperforms the acquisition branch, shuffled-
sample biological branch, and PCA k32 under scanner-heldout transfer. It
nearly preserves cross-scanner category transfer relative to frozen features
and improves on the hardest scanner (p1000). This supports the interpretation
that true-pair factorization preserves transferable category structure better
than broken-pair controls and acquisition-dominated representations.

## Worst held-out scanner

The most challenging held-out scanner is p1000 (mean balanced accuracy
across all representations: 0.6237). On p1000, original frozen features
achieve 0.7093 and true_pair_biological achieves 0.7548 -- the only scanner
where the biological branch improves over frozen features.

## Rare class note

Cartilage (10 patches total) is absent from some held-out-scanner test
splits (nz20, p1000) and present with only 2 patches in others (cs2,
gt450, nz210). All metrics use balanced accuracy and macro-F1 with
zero_division=0. Cartilage-specific claims should not be drawn. Bone
(195 patches) is a minority class present in all test splits.

## Claim boundaries

- Tests cross-scanner category transfer. Does not test clinical utility
  or diagnostic performance.
- Pair-integrity representations use fold-0 projected features. The
  factorization was trained on fold-0 train slides which include all
  scanners. The held-out scanner evaluation tests whether scanner-suppressed
  features generalize across unseen scanners at probe time, not whether the
  factorization itself generalizes to unseen scanners at training time.
- Does not claim: clinical validation, diagnostic performance, patient-care
  utility, universal biological factorization, scanner bias solved, or
  deployment readiness.

## Output files

| File | Description |
|---|---|
| scanner_heldout_raw_metrics.csv | Per-run metrics |
| scanner_heldout_summary.csv | Aggregated by representation |
| scanner_heldout_per_scanner.csv | Per-scanner breakdown |
| scanner_heldout_per_class_recall.csv | Per-class recall |
| scanner_heldout_tradeoff_summary.csv | Tradeoff metrics |
| experiment_design.json | Experiment configuration |
| run_log.txt | Timestamped run log |
| scanner_heldout_label_transfer_report.md | This report |
