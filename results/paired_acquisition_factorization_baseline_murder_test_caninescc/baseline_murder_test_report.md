# Pair-Integrity Baseline Murder Test

## Question

Can linear scanner-subspace projection or PCA component removal achieve the same scanner suppression as Paired-Acquisition Neural Factorization without damaging tissue preservation?

## Datasets

- External canine SCC DINOv2: folds 0, 1, 2, 3, 4.

## Baselines run

- `original_frozen_features`: frozen DINOv2 embeddings evaluated directly.
- `paired_consistency_reference`: existing locked paired-consistency projected features.
- `linear_scanner_subspace_projection_k*`: top-k logistic scanner-discriminative directions removed after fold-fit standardization.
- `pca_component_removal_k*`: top-k PCA directions removed after fold-fit standardization.
- `paired_acquisition_neural_factorization_reference`: existing locked Paired-Acquisition Neural Factorization dep20 projected features used as the method reference.

## Baselines skipped

- random_pair_training skipped for caninescc; no safe existing baseline mode was available without new pair-training semantics beyond the completed shuffled-pair falsification controls.
- optional ablations skipped for caninescc; adversarial_only/no_acquisition_branch/no_covariance_penalty were not available as clean locked runners.

## Result summary

| dataset   | baseline                                          |   n_runs |   scanner_probe |   mean_cosine |   worst_cosine |   mean_top1 |   worst_top1 |   effective_rank |   runtime_s |
|:----------|:--------------------------------------------------|---------:|----------------:|--------------:|---------------:|------------:|-------------:|-----------------:|------------:|
| caninescc | paired_acquisition_neural_factorization_reference |       25 |        0.361408 |      0.729961 |       0.656736 |    0.933392 |     0.884431 |          74.0444 |    0.342672 |
| caninescc | paired_consistency_reference                      |       25 |        0.752868 |      0.696022 |       0.6273   |    0.930637 |     0.881242 |          79.7788 |    0.215489 |
| caninescc | linear_scanner_subspace_projection_k0             |        5 |        0.864095 |      0.733393 |       0.656856 |    0.872764 |     0.794652 |          53.7262 |    1.31521  |
| caninescc | linear_scanner_subspace_projection_k1             |        5 |        0.80742  |      0.733874 |       0.657453 |    0.872392 |     0.792734 |          53.6395 |    1.95891  |
| caninescc | linear_scanner_subspace_projection_k16            |        5 |        0.706582 |      0.739146 |       0.664931 |    0.873753 |     0.795921 |          53.2866 |    2.52249  |
| caninescc | linear_scanner_subspace_projection_k2             |        5 |        0.765719 |      0.735307 |       0.658889 |    0.873278 |     0.794166 |          53.5358 |    1.62187  |
| caninescc | linear_scanner_subspace_projection_k32            |        5 |        0.706582 |      0.739146 |       0.664931 |    0.873753 |     0.795921 |          53.2866 |    2.11906  |
| caninescc | linear_scanner_subspace_projection_k4             |        5 |        0.706582 |      0.739146 |       0.664931 |    0.873753 |     0.795921 |          53.2866 |    1.69612  |
| caninescc | linear_scanner_subspace_projection_k8             |        5 |        0.706582 |      0.739146 |       0.664931 |    0.873753 |     0.795921 |          53.2866 |    2.03975  |
| caninescc | original_frozen_features                          |        5 |        0.862818 |      0.919298 |       0.890698 |    0.832942 |     0.726894 |          46.441  |    1.06987  |
| caninescc | pca_component_removal_k0                          |        5 |        0.864095 |      0.733393 |       0.656856 |    0.872764 |     0.794652 |          53.7262 |    1.86804  |
| caninescc | pca_component_removal_k1                          |        5 |        0.864761 |      0.701313 |       0.615579 |    0.883891 |     0.813656 |          70.9724 |    1.38299  |
| caninescc | pca_component_removal_k16                         |        5 |        0.727899 |      0.629379 |       0.526358 |    0.931073 |     0.881695 |         126.792  |    1.59389  |
| caninescc | pca_component_removal_k2                          |        5 |        0.852635 |      0.69325  |       0.600969 |    0.900791 |     0.826102 |          80.292  |    1.91162  |
| caninescc | pca_component_removal_k32                         |        5 |        0.640836 |      0.597578 |       0.480503 |    0.934709 |     0.886203 |         149.298  |    1.61675  |
| caninescc | pca_component_removal_k4                          |        5 |        0.842416 |      0.677073 |       0.581192 |    0.908445 |     0.848165 |          88.7639 |    1.55933  |
| caninescc | pca_component_removal_k8                          |        5 |        0.817631 |      0.649627 |       0.54004  |    0.916871 |     0.852881 |         106.989  |    1.71433  |

## Best simple baseline and decision

### External canine SCC DINOv2

- Best simple baseline: `linear_scanner_subspace_projection_k16`.
- Classification: scanner suppression alone is insufficient; tissue-preserving factorization remains valuable.
- Best simple metrics: scanner_probe=0.706582, mean_paired_cosine=0.739146, mean_top1_retrieval=0.873753.

## Failure cases

- k values above the learned scanner-subspace rank collapse to the maximum available scanner rank; this is reported as `effective_k`.
- If a simple baseline lowers scanner probe but lowers paired cosine or retrieval relative to the neural reference, it is counted as scanner suppression with tissue damage.

## Validation

- Raw metric rows: 125.
- Required metric columns are present with no missing or nonfinite values.
- Duplicate dataset/fold/seed/baseline rows were rejected during validation.
- Expected folds and reference seeds were validated where applicable.

## Reproduction

```powershell
python experiments/baselines/run_pair_integrity_baseline_murder_test.py --out-dir results/paired_acquisition_factorization_baseline_murder_test_caninescc --datasets caninescc --folds 0 1 2 3 4
```

## Output files

- `results/paired_acquisition_factorization_baseline_murder_test_caninescc/raw_baseline_metrics.csv`
- `results/paired_acquisition_factorization_baseline_murder_test_caninescc/baseline_summary.csv`
- `results/paired_acquisition_factorization_baseline_murder_test_caninescc/baseline_contrasts.csv`
- `results/paired_acquisition_factorization_baseline_murder_test_caninescc/baseline_murder_test_report.md`
- `results/paired_acquisition_factorization_baseline_murder_test_caninescc/run_log.txt`

## Manifests used

- caninescc fold 0: C:\Users\matth\computational-pathology-research\data\external_multiscanner_caninescc\patch_manifests\splits\fold_0_patch_manifest.csv
- caninescc fold 1: C:\Users\matth\computational-pathology-research\data\external_multiscanner_caninescc\patch_manifests\splits\fold_1_patch_manifest.csv
- caninescc fold 2: C:\Users\matth\computational-pathology-research\data\external_multiscanner_caninescc\patch_manifests\splits\fold_2_patch_manifest.csv
- caninescc fold 3: C:\Users\matth\computational-pathology-research\data\external_multiscanner_caninescc\patch_manifests\splits\fold_3_patch_manifest.csv
- caninescc fold 4: C:\Users\matth\computational-pathology-research\data\external_multiscanner_caninescc\patch_manifests\splits\fold_4_patch_manifest.csv

## Claim boundary

This is a peer-review-hardening baseline test. It does not claim clinical validation, diagnostic performance, disease biology discovery, complete scanner invariance, or deployment readiness.
