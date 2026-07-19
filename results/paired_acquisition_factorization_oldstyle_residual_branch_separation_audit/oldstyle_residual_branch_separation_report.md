# Old-Style Residual Branch-Separation Audit

## Branch

experiment/oldstyle-residual-branch-separation-audit

## Question

Does the old-style scanner-centroid/QR residual component act like a clean
scanner branch, or does it leak biological category structure?

## Dataset and Protocol

- Dataset: canine cutaneous SCC DINOv2 frozen features.
- Label column: category_name.
- Scanner column: scanner_id.
- Sample column: sample_id.
- Region column: region_id.
- Folds: 0, 1, 2, 3, 4.
- Pair-integrity seeds: 911, 912, 913, 914, 915.
- Old-style k values: 1, 2, 3, 4 only.

## Formulas

Old-style scanner-centroid directions are fit per fold on old-style standardized
features. The StandardScaler is fit on all aligned frozen features to reproduce
the old linear_projection_k4 convention used by the consistency audit.

For fit rows X_fit and scanner labels s:
- grand_mean = mean(X_fit)
- direction_s = mean(X_fit where scanner_id == s) - grand_mean
- Q_k = first k QR-orthonormalized direction rows
- oldstyle_removed_k = X @ Q_k.T @ Q_k
- oldstyle_keep_k = X - oldstyle_removed_k

There are five scanners, so the scanner-centroid rank is at most four.

## Row Counts

- Raw metric rows: 145.
- Summary rows: 13.
- Branch contrast rows: 95.
- Neighborhood rows: 145.

## Key Metrics

- original_frozen_features: n=5, scanner_acc=0.8638, scanner_f1=0.8640, category_acc=0.4003, category_macro_f1=0.3579, category_weighted_f1=0.4270, purity_k1=0.9602, purity_k5=0.8717, purity_k10=0.7306, same_sample_top1=0.9962
- true_pair_biological: n=25, scanner_acc=0.3614, scanner_f1=0.3570, category_acc=0.3860, category_macro_f1=0.3389, category_weighted_f1=0.4174, purity_k1=0.9729, purity_k5=0.8973, purity_k10=0.7509, same_sample_top1=0.9982
- true_pair_acquisition: n=25, scanner_acc=0.8651, scanner_f1=0.8650, category_acc=0.3456, category_macro_f1=0.2736, category_weighted_f1=0.3246, purity_k1=0.5736, purity_k5=0.4712, purity_k10=0.4183, same_sample_top1=0.6777
- shuffled_sample_biological: n=25, scanner_acc=0.4093, scanner_f1=0.4077, category_acc=0.3228, category_macro_f1=0.2752, category_weighted_f1=0.3386, purity_k1=0.9273, purity_k5=0.7801, purity_k10=0.6355, same_sample_top1=0.9673
- shuffled_sample_acquisition: n=25, scanner_acc=0.8302, scanner_f1=0.8295, category_acc=0.3871, category_macro_f1=0.3150, category_weighted_f1=0.3727, purity_k1=0.7309, purity_k5=0.6086, purity_k10=0.5338, same_sample_top1=0.8402
- oldstyle_keep_k1: n=5, scanner_acc=0.7672, scanner_f1=0.7692, category_acc=0.3997, category_macro_f1=0.3562, category_weighted_f1=0.4251, purity_k1=0.9623, purity_k5=0.8756, purity_k10=0.7358, same_sample_top1=0.9958
- oldstyle_removed_k1: n=5, scanner_acc=0.2822, scanner_f1=0.2315, category_acc=0.1646, category_macro_f1=0.1139, category_weighted_f1=0.1451, purity_k1=0.2481, purity_k5=0.2288, purity_k10=0.2272, same_sample_top1=0.1619
- oldstyle_keep_k2: n=5, scanner_acc=0.6035, scanner_f1=0.5948, category_acc=0.3968, category_macro_f1=0.3526, category_weighted_f1=0.4209, purity_k1=0.9642, purity_k5=0.8804, purity_k10=0.7393, same_sample_top1=0.9972
- oldstyle_removed_k2: n=5, scanner_acc=0.3874, scanner_f1=0.3782, category_acc=0.2090, category_macro_f1=0.1245, category_weighted_f1=0.1384, purity_k1=0.2848, purity_k5=0.2896, purity_k10=0.2895, same_sample_top1=0.2347
- oldstyle_keep_k3: n=5, scanner_acc=0.4672, scanner_f1=0.4471, category_acc=0.3986, category_macro_f1=0.3543, category_weighted_f1=0.4218, purity_k1=0.9658, purity_k5=0.8861, purity_k10=0.7433, same_sample_top1=0.9979
- oldstyle_removed_k3: n=5, scanner_acc=0.4788, scanner_f1=0.4674, category_acc=0.2398, category_macro_f1=0.1651, category_weighted_f1=0.2192, purity_k1=0.3073, purity_k5=0.3215, purity_k10=0.3152, same_sample_top1=0.2826
- oldstyle_keep_k4: n=5, scanner_acc=0.2000, scanner_f1=0.0667, category_acc=0.4004, category_macro_f1=0.3485, category_weighted_f1=0.4250, purity_k1=0.9678, purity_k5=0.8895, purity_k10=0.7456, same_sample_top1=0.9982
- oldstyle_removed_k4: n=5, scanner_acc=0.5384, scanner_f1=0.5361, category_acc=0.2421, category_macro_f1=0.1746, category_weighted_f1=0.2313, purity_k1=0.3464, purity_k5=0.3387, purity_k10=0.3341, same_sample_top1=0.3139

## Paired vs Oldstyle Branch Contrasts

- paired_category_contrast = 0.0404 (true_pair_biological category_acc - true_pair_acquisition category_acc).
- paired_scanner_contrast = 0.5037 (true_pair_acquisition scanner_acc - true_pair_biological scanner_acc).
- oldstyle_category_contrast_k4 = 0.1583 (oldstyle_keep_k4 category_acc - oldstyle_removed_k4 category_acc).
- oldstyle_scanner_contrast_k4 = 0.3384 (oldstyle_removed_k4 scanner_acc - oldstyle_keep_k4 scanner_acc).

## Leakage Findings

- paired_acquisition_category_leakage = 0.3456.
- oldstyle_removed_category_leakage_k4 = 0.2421.
- paired_bio_scanner_leakage = 0.3614.
- oldstyle_keep_scanner_leakage_k4 = 0.2000.

## Key Questions

1. Does oldstyle_keep_k4 suppress scanner more strongly than true_pair_biological?
   Yes: oldstyle_keep_k4 scanner_acc=0.2000, true_pair_biological scanner_acc=0.3614.
2. Does oldstyle_keep_k4 preserve category signal better than true_pair_biological?
   Yes: oldstyle_keep_k4 category_acc=0.4004, true_pair_biological category_acc=0.3860.
3. Does oldstyle_removed_k4 carry scanner signal?
   Scanner_acc=0.5384.
4. Does oldstyle_removed_k4 leak category signal?
   Category_acc=0.2421.
5. Is oldstyle_removed_k4 cleaner than true_pair_acquisition at keeping category signal out?
   Yes: oldstyle_removed_k4 category_acc=0.2421, true_pair_acquisition category_acc=0.3456.
6. Does old-style linear residual decomposition fully explain paired-acquisition branch separation?
   This audit should be read through the leakage and contrast values above.
7. Or do the two methods occupy different separation-frontier points?
   This is suggested when scanner/category tradeoffs differ across keep/removed and
   biological/acquisition branches.

## Bounded Interpretation

The oldstyle_keep_k4 result should be treated as the stronger raw scanner-removal
linear baseline when compared with true_pair_biological. A paired-acquisition
claim should therefore focus on structured separation, not on beating this
old-style baseline on raw scanner suppression.

The decisive check is whether oldstyle_removed_k4 carries scanner signal while
keeping category signal out as well as, or better than, true_pair_acquisition.
If it does, that supports a stronger linear residual baseline. If it does not,
that suggests the paired and old-style decompositions occupy different
separation frontier points.

## Previous Interpretation

The linear baseline consistency correction remains in force: the logistic-SVD
split from the earlier residual audit was weaker for scanner removal than the
old-style centroid/QR projection. Any prior statement comparing paired
acquisition to the strongest simple linear scanner removal should use the
old-style baseline, not the logistic-SVD split.

## Validation Checks

- Expected representations present: True.
- k values present: [1, 2, 3, 4].
- oldstyle_keep_k4 scanner reference target: 0.2000.
- oldstyle_keep_k4 scanner observed: 0.2000.
- Validation issue count: 0.
  - No validation issues found.

## Files Created

- oldstyle_residual_raw_metrics.csv
- oldstyle_residual_summary.csv
- oldstyle_residual_branch_contrasts.csv
- oldstyle_residual_neighborhood_purity.csv
- oldstyle_residual_branch_separation_report.md
- experiment_design.json
- run_log.txt

Runtime seconds: 159.6

## Readiness

Ready to commit after external diff hygiene checks pass; no staging or commit performed.
