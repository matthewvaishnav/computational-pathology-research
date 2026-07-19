# Frontier-Selected Cross-Backbone Validation

## Branch

experiment/frontier-selected-crossbackbone-validation

## Question

Do the selected acquisition-bottleneck variants preserve the separation-frontier improvement outside canine SCC DINOv2?

## Tested Settings

- SCORPION DINOv2: pair/tissue retrieval metrics, no biological labels in archive.
- SCORPION Phikon: pair/tissue retrieval metrics, no biological labels in archive.
- SCORPION ResNet50: pair/tissue retrieval metrics, no biological labels in archive.
- Canine SCC DINOv2: fixed reference from the preceding branch-separation audit.

## Metric Definitions

- scanner_probe_accuracy: balanced scanner accuracy from a standardized logistic probe.
- mean_top1_retrieval: mean same-region top-1 retrieval across scanner pairs on the test split.
- mean_paired_cosine: mean cosine similarity for same-region cross-scanner pairs on the test split.
- effective_rank: entropy effective rank of centered test-branch features.
- biological_acquisition_cross_covariance: RMS cross-covariance after standardizing biological and acquisition test features.
- SCORPION acquisition-branch mean_top1_retrieval is treated as paired-region/tissue leakage, because labels are unavailable.

## Row Counts

- Raw SCORPION metric rows: 450.
- Summary rows: 18.
- Contrast rows: 12.
- Canine reference rows: 7.
- Folds: 0, 1, 2, 3, 4.
- Seeds: 701, 702, 703, 704, 705.
- Epochs: 75.
- Device: cuda.
- Runtime seconds: 2046.0.

## SCORPION Key Metrics

### dinov2
- true_pair_current: bio scanner=0.3998, bio retrieval=0.9999, acq scanner=0.8582, acq retrieval leakage=0.0944, xcov=0.0917.
- acq_dim8_default: bio scanner=0.4082, bio retrieval=0.9999, acq scanner=0.8508, acq retrieval leakage=0.0231, xcov=0.0715.
- acq_dim16_stronger_xcov: bio scanner=0.4052, bio retrieval=0.9999, acq scanner=0.8565, acq retrieval leakage=0.0253, xcov=0.0718.

### phikon
- true_pair_current: bio scanner=0.5200, bio retrieval=0.9997, acq scanner=0.9711, acq retrieval leakage=0.0739, xcov=0.0889.
- acq_dim8_default: bio scanner=0.5066, bio retrieval=0.9996, acq scanner=0.9733, acq retrieval leakage=0.0204, xcov=0.0597.
- acq_dim16_stronger_xcov: bio scanner=0.5151, bio retrieval=0.9997, acq scanner=0.9772, acq retrieval leakage=0.0219, xcov=0.0596.

### resnet50
- true_pair_current: bio scanner=0.3145, bio retrieval=0.9726, acq scanner=0.7845, acq retrieval leakage=0.1705, xcov=0.0896.
- acq_dim8_default: bio scanner=0.3034, bio retrieval=0.9725, acq scanner=0.7701, acq retrieval leakage=0.0505, xcov=0.0749.
- acq_dim16_stronger_xcov: bio scanner=0.3005, bio retrieval=0.9719, acq scanner=0.7767, acq retrieval leakage=0.0646, xcov=0.0750.

## Canine SCC DINOv2 Reference

- true_pair_biological: scanner=0.3614, category=0.3860.
- true_pair_acquisition: scanner=0.8651, category=0.3456.
- acq_dim8_default_biological: scanner=0.3691, category=0.3852.
- acq_dim8_default_acquisition: scanner=0.8643, category=0.1598.
- acq_dim16_stronger_xcov_biological: scanner=0.3593, category=0.3824.
- acq_dim16_stronger_xcov_acquisition: scanner=0.8638, category=0.1689.
- oldstyle_keep_k4: scanner=0.2000, category=0.4004.

## Interpretation Boundary

The canine SCC DINOv2 reference remains the labeled frontier anchor: bottleneck variants sharply reduce acquisition-branch category leakage while preserving scanner capture, but oldstyle centroid/QR remains the strongest raw scanner-removal baseline.
For SCORPION, this audit cannot make category-leakage claims because labels are unavailable in the frozen archives. It instead tests whether acquisition branches keep scanner capture while reducing paired-region/tissue retrieval relative to the true-pair current split.

## Validation

- Validation issues: 0.
- No validation issues found.

## Files Created

- frontier_crossbackbone_raw_metrics.csv
- frontier_crossbackbone_summary.csv
- frontier_crossbackbone_contrasts.csv
- frontier_crossbackbone_canine_reference.csv
- frontier_selected_crossbackbone_validation_report.md
- experiment_design.json
- run_log.txt

## Readiness

No staging or commit performed by this runner.
