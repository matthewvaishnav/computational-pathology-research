# Frontier-Selected Downstream Validation

## Branch

experiment/frontier-selected-downstream-validation

## Question

Did reducing acquisition-branch category leakage preserve or improve downstream transfer robustness, or did it only make the branch audit cleaner?

## Frontier Variants

- acq_dim8_default
- acq_dim16_stronger_xcov

## Reference Boundary

- true_pair_acquisition branch audit: scanner 0.8651, category leakage 0.3456.
- acq_dim8_default acquisition branch audit: scanner 0.8643, category leakage 0.1598.
- acq_dim16_stronger_xcov acquisition branch audit: scanner 0.8638, category leakage 0.1689.
- oldstyle_keep_k4 remains best raw scanner removal: scanner 0.2000, category 0.4004.

## Protocol

- Scanner-heldout label transfer: train category probe on four scanners and test on the held-out scanner.
- Sample-disjoint scanner-heldout transfer: train on four scanners and train-sample subset; test on held-out scanner and disjoint samples.
- Scanner-confounded label robustness: train on scanner/category-confounded fit subsets and test on the unmodified held-out fold.
- Probe model: standardized LogisticRegression with balanced class weights.

## Row Counts

- Raw metric rows: 3750.
- Summary rows: 18.
- Contrast rows: 12.
- Smoke mode: False.
- Runtime seconds: 423.3.

## Key Metrics

### scanner_heldout_label_transfer

- true_pair_biological: balanced_acc=0.8273, macro_f1=0.7887, min_acc=0.6969.
- acq_dim8_default_biological: balanced_acc=0.8221, macro_f1=0.7852, min_acc=0.7159.
- acq_dim16_stronger_xcov_biological: balanced_acc=0.8292, macro_f1=0.7911, min_acc=0.7093.
- true_pair_acquisition: balanced_acc=0.5153, macro_f1=0.4147, min_acc=0.3479.
- acq_dim8_default_acquisition: balanced_acc=0.1751, macro_f1=0.0565, min_acc=0.0840.
- acq_dim16_stronger_xcov_acquisition: balanced_acc=0.2037, macro_f1=0.0869, min_acc=0.0928.

### sample_disjoint_scanner_heldout_transfer

- true_pair_biological: balanced_acc=0.3260, macro_f1=0.3013, min_acc=0.1754.
- acq_dim8_default_biological: balanced_acc=0.3295, macro_f1=0.2960, min_acc=0.1872.
- acq_dim16_stronger_xcov_biological: balanced_acc=0.3371, macro_f1=0.3108, min_acc=0.1968.
- true_pair_acquisition: balanced_acc=0.2654, macro_f1=0.2165, min_acc=0.1327.
- acq_dim8_default_acquisition: balanced_acc=0.1304, macro_f1=0.0494, min_acc=0.0000.
- acq_dim16_stronger_xcov_acquisition: balanced_acc=0.1404, macro_f1=0.0676, min_acc=0.0000.

### scanner_confounded_label_robustness

- true_pair_biological: balanced_acc=0.3791, macro_f1=0.3475, min_acc=0.2666.
- acq_dim8_default_biological: balanced_acc=0.3794, macro_f1=0.3480, min_acc=0.2623.
- acq_dim16_stronger_xcov_biological: balanced_acc=0.3761, macro_f1=0.3454, min_acc=0.2655.
- true_pair_acquisition: balanced_acc=0.3072, macro_f1=0.2489, min_acc=0.1782.
- acq_dim8_default_acquisition: balanced_acc=0.1623, macro_f1=0.1251, min_acc=0.0996.
- acq_dim16_stronger_xcov_acquisition: balanced_acc=0.1784, macro_f1=0.1351, min_acc=0.1099.

## Frontier Contrasts

For biological branches, positive category-accuracy deltas indicate stronger downstream category transfer.
For acquisition branches, negative category-accuracy deltas indicate lower downstream category leakage.

- sample_disjoint_scanner_heldout_transfer acq_dim16_stronger_xcov_acquisition vs true_pair_acquisition: delta_balanced_acc=-0.1250.
- sample_disjoint_scanner_heldout_transfer acq_dim16_stronger_xcov_biological vs true_pair_biological: delta_balanced_acc=0.0111.
- sample_disjoint_scanner_heldout_transfer acq_dim8_default_acquisition vs true_pair_acquisition: delta_balanced_acc=-0.1349.
- sample_disjoint_scanner_heldout_transfer acq_dim8_default_biological vs true_pair_biological: delta_balanced_acc=0.0035.
- scanner_confounded_label_robustness acq_dim16_stronger_xcov_acquisition vs true_pair_acquisition: delta_balanced_acc=-0.1289.
- scanner_confounded_label_robustness acq_dim16_stronger_xcov_biological vs true_pair_biological: delta_balanced_acc=-0.0030.
- scanner_confounded_label_robustness acq_dim8_default_acquisition vs true_pair_acquisition: delta_balanced_acc=-0.1450.
- scanner_confounded_label_robustness acq_dim8_default_biological vs true_pair_biological: delta_balanced_acc=0.0003.
- scanner_heldout_label_transfer acq_dim16_stronger_xcov_acquisition vs true_pair_acquisition: delta_balanced_acc=-0.3117.
- scanner_heldout_label_transfer acq_dim16_stronger_xcov_biological vs true_pair_biological: delta_balanced_acc=0.0018.
- scanner_heldout_label_transfer acq_dim8_default_acquisition vs true_pair_acquisition: delta_balanced_acc=-0.3402.
- scanner_heldout_label_transfer acq_dim8_default_biological vs true_pair_biological: delta_balanced_acc=-0.0053.

## Validation

- Validation issues: 0.
- No validation issues found.

## Bounded Interpretation

This is a downstream stress check for the frontier-selected variants, not a broad generalization claim.
The central question is whether the cleaner acquisition branch remains compatible with downstream category transfer in the biological branch.
The oldstyle centroid/QR result remains the strongest raw scanner-removal boundary.

## Files Created

- frontier_downstream_raw_metrics.csv
- frontier_downstream_summary.csv
- frontier_downstream_contrasts.csv
- frontier_downstream_split_diagnostics.csv
- frontier_downstream_per_class_recall.csv
- frontier_downstream_per_scanner_errors.csv
- frontier_selected_downstream_validation_report.md
- experiment_design.json
- run_log.txt
