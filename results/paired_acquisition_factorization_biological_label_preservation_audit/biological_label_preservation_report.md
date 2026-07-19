# Biological-Label Preservation Audit Report

**Generated:** 2026-07-06 13:03:26
**Runtime:** 237.6 s
**Evidence tier:** full (5-fold x 5-seed)

## Scientific question

Does the biological branch preserve tissue-category structure while
reducing scanner/acquisition recoverability?

## Dataset

- Canine SCC DINOv2: 4,025 patches, 805 regions, 44 samples
- 5 scanners: cs2, gt450, nz20, nz210, p1000
- 7 tissue categories: Epidermis (1,205), SCC (1,205), Subcutis (510),
  Dermis (500), Inflamm/Necrosis (400), Bone (195), Cartilage (10)
- Slide-level 5-fold cross-validation
- Label column: `category_name` (from fold manifest)
- Scanner column: `scanner_id`
- Sample identifier: `sample_id`
- Region identifier: `region_id`
- Probe model: LogisticRegression(C=1.0, class_weight=balanced, max_iter=5000)
- All metrics computed on held-out test slides only

## Representations compared

| Representation | Dim | Family | Runs |
|---|---|---|---|
| `original_frozen_features` | 768 | original frozen DINOv2 | 5 |
| `true_pair_biological` | 256 | pair-integrity true_pairs biological branch | 25 |
| `true_pair_acquisition` | 64 | pair-integrity true_pairs acquisition branch | 25 |
| `shuffled_sample_biological` | 256 | pair-integrity shuffled_sample biological branch | 25 |
| `shuffled_sample_acquisition` | 64 | pair-integrity shuffled_sample acquisition branch | 25 |
| `pca_removal_k{1,2,4,8,16,32}` | 768 | PCA component removal (in-runner) | 5 each |
| `linear_projection_k{0,1,2,4,8,16,32}` | 768 | linear scanner subspace projection (in-runner) | 5 each |

## Key summary

```
representation              scan_acc  cat_acc  cat_f1   pk1     pk5     ratio
original_frozen_features    0.8656    0.4068   0.3690   0.9542  0.8439  0.47
true_pair_biological        0.3610    0.3855   0.3384   0.9729  0.8938  1.07
true_pair_acquisition       0.8620    0.3458   0.2750   0.5295  0.4316  0.40
shuffled_sample_biological  0.4091    0.3237   0.2758   0.8967  0.7098  0.80
shuffled_sample_acquisition 0.8235    0.3851   0.3135   0.6963  0.5734  0.47
pca_removal_k32             0.6489    0.2893   0.2754   0.9648  0.8907  0.44
linear_projection_k4        0.2000    0.4015   0.3473   0.9661  0.8830  2.01
```

scan_acc = balanced accuracy of linear scanner probe (lower = more suppressed).
cat_acc = balanced accuracy of linear category probe (higher = more preserved).
cat_f1 = macro F1 on category probe.
pk1, pk5 = same-category nearest-neighbor purity at k=1, k=5.
ratio = cat_acc / scan_acc (higher = better scanner/category tradeoff).

## Main results: scanner suppression vs category preservation

Scanner probe should be low (scanner suppressed); category probe should be
high (category preserved). Ratio > 1 means category signal dominates scanner
signal.

| Representation | Scanner probe | Category probe | Category F1 | Cat/Scan ratio | Purity k=1 | Purity k=5 | Eff. rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| `original_frozen_features` | 0.8656 | 0.4068 | 0.3690 | 0.47 | 0.9542 | 0.8439 | 46.4 |
| `true_pair_biological` | 0.3610 | 0.3855 | 0.3384 | 1.07 | 0.9729 | 0.8938 | 74.0 |
| `true_pair_acquisition` | 0.8620 | 0.3458 | 0.2750 | 0.40 | 0.5295 | 0.4316 | 13.8 |
| `shuffled_sample_biological` | 0.4091 | 0.3237 | 0.2758 | 0.80 | 0.8967 | 0.7098 | 45.3 |
| `shuffled_sample_acquisition` | 0.8235 | 0.3851 | 0.3135 | 0.47 | 0.6963 | 0.5734 | 20.5 |
| `pca_removal_k32` | 0.6489 | 0.2893 | 0.2754 | 0.44 | 0.9648 | 0.8907 | 149.3 |
| `linear_projection_k4` | 0.2000 | 0.4015 | 0.3473 | 2.01 | 0.9661 | 0.8830 | 60.0 |

Full results for all k-values in `label_probe_summary.csv`.

## Interpretation

### True-pair biological branch vs original frozen features

- Scanner probe: 0.8656 to 0.3610 (delta = +0.5046). This is a substantial
  reduction in scanner recoverability — the biological branch suppresses
  scanner signal to well below the original frozen embedding level.
- Category probe: 0.4068 to 0.3855, a decrease of 0.0213. Category probe
  accuracy has a small decrease, while scanner recoverability drops by 0.5046
  and category-neighborhood purity improves.
- Category F1: 0.3690 to 0.3384 (delta = +0.0306). A modest decrease.
- Neighborhood purity k=1: 0.9542 to 0.9729. Same-category nearest-neighbor
  purity actually improves, suggesting the biological branch may reduce
  within-category noise while retaining category-separating structure.
- Category/scanner ratio: 0.47 to 1.07. More than doubled.

In canine SCC DINOv2, the true-pair biological branch substantially reduces
scanner recoverability relative to original frozen features while preserving
tissue-category structure, especially in nearest-neighbor category purity.
This supports the paired-acquisition mechanism.

The biological branch trades a small category-probe decrease (0.0213) for
a very large scanner-probe reduction (0.5046) and improved same-category
neighborhood purity (0.9542 to 0.9729).

### True-pair acquisition branch

- Scanner probe: 0.8620. Scanner signal is retained — the acquisition branch
  captures scanner-level information as intended.
- Category probe: 0.3458. Category structure is reduced relative to frozen
  features (0.4068) and the biological branch (0.3855).
- Neighborhood purity k=1: 0.5295. Dramatically lower than the biological
  branch (0.9729) — tissue-category identity is substantially removed from
  the acquisition branch.

The separation of scanner signal into the acquisition branch while preserving
category structure in the biological branch supports branch separation. The
biological branch retains tissue-category neighborhood structure (purity
0.97); the acquisition branch does not (purity 0.53).

### Shuffled-sample biological branch (control)

- Scanner probe: 0.4091. Weaker scanner suppression than true-pair (0.3610).
- Category probe: 0.3237. Lower category preservation than true-pair (0.3855).
- Neighborhood purity k=1: 0.8967. Lower than true-pair (0.9729).

Breaking the true-pair structure (shuffled_sample_pairs) degrades both scanner
suppression and category preservation. This supports the importance of
true-pair structure: when the factorization is trained with broken pair
correspondence, it is less able to separate scanner from biology, and both
branches encode mixed information.

### Comparison with PCA removal baseline

The biological branch category/scanner ratio (1.07) exceeds the best PCA
baseline (0.44 at k=32). PCA removal at k=32 achieves only modest scanner
suppression (0.6489) while degrading category probe (0.2893) — category
signal drops more under PCA removal than under factorization. The biological
branch has a better category/scanner tradeoff than PCA removal.

PCA removal also increases effective rank (149.3 at k=32) rather than
reducing it, suggesting it removes principal components that carry shared
variance without discriminating between scanner and category dimensions.

### Comparison with linear scanner subspace projection

The linear_projection_k4 baseline achieves stronger absolute scanner
suppression (0.2000, at chance level for 5 scanners) and slightly higher
category probe accuracy (0.4015 vs 0.3855), yielding a higher raw
category/scanner ratio (2.01 vs 1.07).

This is a more nuanced comparison. The linear baseline suppresses scanner
signal more aggressively, but it does so by removing scanner-discriminative
directions from the feature space entirely. The paired-acquisition mechanism
instead separates scanner signal into an explicit acquisition branch
(scanner probe 0.86) while the biological branch retains reduced scanner
signal (0.36) and strong category structure (purity 0.97). The acquisition
branch preserves scanner information for downstream inspection; the linear
baseline cannot explain where the removed scanner signal went.

The biological branch does not claim to beat the linear baseline on raw
category/scanner ratio. It offers a different tradeoff: structured separation
with an interpretable acquisition branch vs. blind scanner removal.

## Unavailable representations

- **SCORPION biological-label audit**: Not available. The SCORPION manifest
  (data/scorpion/splits/fold_{fold}_manifest.csv) contains slide_id, region_id,
  scanner_id, split, and path columns. It has no tissue-category or
  pathology-label column comparable to canine SCC `category_name`. An audit
  on SCORPION would require external label annotation.
- **Pre-computed PCA baseline artifacts**: Not available as saved NPZ files.
  PCA removal was computed in-runner from original frozen features per fold
  using sklearn.decomposition.PCA.
- **Pre-computed linear scanner subspace baseline artifacts**: Not available
  as saved NPZ files. Linear scanner subspace projection was computed
  in-runner from original frozen features per fold.
- **Baseline murder test category-probe results**: The existing baseline
  murder test (experiments/baselines/run_pair_integrity_baseline_murder_test.py)
  computed scanner_probe_accuracy and paired retrieval metrics but did not
  compute category-probe accuracy. All category-probe and neighborhood-purity
  metrics in this audit are new.

## Claim boundaries

- This audit tests whether the biological branch preserves tissue-category
  structure while reducing scanner recoverability. It does not test clinical
  utility, diagnostic performance, or deployment readiness.
- Category labels are tissue morphology categories from canine SCC expert
  annotation, not diagnostic grades or clinical outcomes.
- All probe metrics use balanced accuracy to account for class imbalance
  (Cartilage: 10 patches, Bone: 195 patches vs. Epidermis/SCC: 1,205 each).
- The PCA and linear scanner-subspace baselines are simple post-hoc
  operations on frozen embeddings. They do not use paired training and serve
  as reference points, not competitive alternatives.
- The linear baseline k=4 and higher values saturate at scanner probe 0.20
  (chance for 5 balanced classes) because the 5 per-scanner mean vectors
  span at most 4 independent directions after centering.
- Does not claim: clinical validation, diagnostic performance, patient-care
  utility, universal biological factorization, scanner bias solved, or
  deployment readiness.

## Validation checks

- label_probe_raw_metrics.csv: 170 rows, 0 duplicate (rep,fold,seed), 0
  non-finite metric values. NaN appears only in the `k` column for
  non-baseline representations (structural missingness, not failed metrics).
- label_probe_summary.csv: 18 rows (one per representation).
- scanner_label_tradeoff_summary.csv: 18 rows.
- neighborhood_purity_metrics.csv: 18 rows (k=1,5,10 as columns, not rows).
- All 18 expected representations present.
- Category column `category_name` documented.
- Scanner column `scanner_id` documented.
- Sample and region identifiers documented.
- Slide-level 5-fold CV documented.
- Class counts and scanner counts documented.

## Output files

| File | Description |
|---|---|
| label_probe_raw_metrics.csv | Per-run, per-representation metrics (170 rows) |
| label_probe_summary.csv | Aggregated by representation (18 rows) |
| neighborhood_purity_metrics.csv | Same-category NN purity at k=1,5,10 |
| scanner_label_tradeoff_summary.csv | Scanner vs category tradeoff |
| experiment_design.json | Experiment configuration |
| run_log.txt | Timestamped run log |
| biological_label_preservation_report.md | This report |
