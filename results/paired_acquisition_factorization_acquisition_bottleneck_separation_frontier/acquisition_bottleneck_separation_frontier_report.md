# Acquisition Bottleneck Separation-Frontier Sweep

## Branch

experiment/acquisition-bottleneck-separation-frontier

## Controllable Parameters Found

- acquisition_dim: controllable through ProjectionConfig.
- biological_dim: controllable through ProjectionConfig; held fixed at 256.
- hidden_dim: controllable through ProjectionConfig; held fixed at 512.
- reconstruction_weight, variance_weight, covariance_weight: controllable; held at current values.
- scanner_adversary_weight, scanner_acquisition_weight, scanner_dependence_weight: controllable; held at current values.
- cross_covariance_weight: controllable; swept at current 0.05 and stronger 0.20 for selected dimensions.
- gradient_reversal_strength: controllable; held at current 1.0.
- fold and seed controls: available and used.

## Fixed References

- true_pair_biological: scanner_acc 0.3614, category_acc 0.3860.
- true_pair_acquisition: scanner_acc 0.8651, category_acc 0.3456.
- oldstyle_keep_k4: scanner_acc 0.2000, category_acc 0.4004.
- oldstyle_removed_k4: scanner_acc 0.5384, category_acc 0.2421.

## Smoke Variants

- acq_dim8_default: acq_dim=8, cross_covariance_weight=0.05.
- acq_dim16_default: acq_dim=16, cross_covariance_weight=0.05.
- acq_dim32_default: acq_dim=32, cross_covariance_weight=0.05.
- acq_dim64_current: acq_dim=64, cross_covariance_weight=0.05.
- acq_dim16_stronger_xcov: acq_dim=16, cross_covariance_weight=0.20.
- acq_dim32_stronger_xcov: acq_dim=32, cross_covariance_weight=0.20.

## Selected Full Variants

- acq_dim8_default: selected from smoke frontier_score=0.8812.
- acq_dim16_stronger_xcov: selected from smoke frontier_score=0.8702.

## Row Counts

- Smoke raw rows: 12.
- Full raw rows: 100.
- Variant summary rows: 16.
- Branch contrast rows: 56.
- Selection log rows: 6.

## Key Metrics

### smoke
- acq_dim16_default: bio scanner=0.3722, bio category=0.2757, acq scanner=0.8500, acq category=0.2340.
- acq_dim16_stronger_xcov: bio scanner=0.3556, bio category=0.2835, acq scanner=0.8556, acq category=0.1665.
- acq_dim32_default: bio scanner=0.3528, bio category=0.2987, acq scanner=0.8278, acq category=0.2857.
- acq_dim32_stronger_xcov: bio scanner=0.3486, bio category=0.3052, acq scanner=0.8222, acq category=0.2525.
- acq_dim64_current: bio scanner=0.3375, bio category=0.2999, acq scanner=0.8389, acq category=0.3116.
- acq_dim8_default: bio scanner=0.3583, bio category=0.3069, acq scanner=0.8389, acq category=0.1343.

### full
- acq_dim16_stronger_xcov: bio scanner=0.3593, bio category=0.3824, acq scanner=0.8638, acq category=0.1689.
- acq_dim8_default: bio scanner=0.3691, bio category=0.3852, acq scanner=0.8643, acq category=0.1598.

## Frontier Comparison

Lower acquisition category accuracy means less category leakage in the
scanner/acquisition branch. Higher acquisition scanner accuracy means stronger
scanner capture. Lower biological scanner accuracy means less scanner leakage
in the biological branch.

- acq_dim8_default: acq_category_delta_vs_true_pair=-0.1858, acq_scanner_delta_vs_oldstyle_removed=0.3259, bio_scanner_delta_vs_true_pair=0.0077, bio_category_delta_vs_true_pair=-0.0008.
- acq_dim16_stronger_xcov: acq_category_delta_vs_true_pair=-0.1767, acq_scanner_delta_vs_oldstyle_removed=0.3254, bio_scanner_delta_vs_true_pair=-0.0021, bio_category_delta_vs_true_pair=-0.0036.

## Validation Checks

- Duplicate checks passed: True.
- Nonfinite metric checks passed: True.
- Smoke variants documented: 6 / 6.
- Selected full variants documented: acq_dim8_default, acq_dim16_stronger_xcov.
- Baseline references included: True.
- Validation issue count: 0.
  - No validation issues found.

## Bounded Interpretation

This sweep tests whether acquisition bottleneck capacity moves the separation
frontier in this audit. It is not a use-context or downstream-care claim.
The oldstyle centroid/QR reference remains the strongest raw scanner-removal
baseline. The paired-acquisition target here is structured separation: keeping
an explicit scanner-bearing acquisition branch while reducing category leakage.

## Key Questions

1. Does reducing acquisition capacity reduce acquisition category leakage?
   See acquisition category metrics and deltas above.
2. Does scanner capture survive the bottleneck?
   See acquisition scanner metrics above.
3. Does biological branch scanner leakage improve?
   See biological scanner metrics above.
4. Does category preservation degrade?
   See biological category metrics above.
5. Does any variant move the separation frontier?
   A move is supported when leakage falls while scanner capture remains above oldstyle_removed_k4.
6. Does this weaken or strengthen the paired-acquisition mechanism story?
   It strengthens the mechanism story only if structured separation improves without hiding the oldstyle raw-removal boundary.
7. Does oldstyle centroid/QR remain the best raw scanner-removal baseline?
   Yes unless a full variant moves biological scanner below 0.2000 with comparable category preservation.

## Files Created

- frontier_smoke_raw_metrics.csv
- frontier_full_raw_metrics.csv
- frontier_variant_summary.csv
- frontier_branch_contrasts.csv
- frontier_variant_selection_log.csv
- acquisition_bottleneck_separation_frontier_report.md
- experiment_design.json
- run_log.txt

Runtime seconds: 1443.6
Epochs: 75
Device: cuda

## Readiness

Ready to commit after external diff hygiene checks pass; no staging or commit performed.
