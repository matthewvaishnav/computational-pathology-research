# Pair-Integrity Baseline Murder Test

## Question

Can linear scanner-subspace projection or PCA component removal achieve the same scanner suppression as Paired-Acquisition Neural Factorization without damaging tissue preservation?

## Datasets

- SCORPION DINOv2: folds 0, 1, 2, 3, 4.

## Baselines run

- `original_frozen_features`: frozen DINOv2 embeddings evaluated directly.
- `paired_consistency_reference`: existing locked paired-consistency projected features.
- `linear_scanner_subspace_projection_k*`: top-k logistic scanner-discriminative directions removed after fold-fit standardization.
- `pca_component_removal_k*`: top-k PCA directions removed after fold-fit standardization.
- `paired_acquisition_neural_factorization_reference`: existing locked Paired-Acquisition Neural Factorization dep20 projected features used as the method reference.

## Baselines skipped

- random_pair_training skipped for scorpion; no safe existing baseline mode was available without new pair-training semantics beyond the completed shuffled-pair falsification controls.
- optional ablations skipped for scorpion; adversarial_only/no_acquisition_branch/no_covariance_penalty were not available as clean locked runners.

## Result summary

| dataset   | baseline                                          |   n_runs |   scanner_probe |   mean_cosine |   worst_cosine |   mean_top1 |   worst_top1 |   effective_rank |   runtime_s |
|:----------|:--------------------------------------------------|---------:|----------------:|--------------:|---------------:|------------:|-------------:|-----------------:|------------:|
| scorpion  | paired_acquisition_neural_factorization_reference |       25 |        0.398907 |      0.878856 |       0.850166 |    0.999787 |     0.998733 |          54.5017 |    0.160027 |
| scorpion  | paired_consistency_reference                      |       25 |        0.782489 |      0.847591 |       0.820211 |    0.999867 |     0.999111 |          56.9326 |    0.127447 |
| scorpion  | linear_scanner_subspace_projection_k0             |        5 |        0.866489 |      0.867213 |       0.820676 |    0.999889 |     0.998889 |          33.9743 |    1.22092  |
| scorpion  | linear_scanner_subspace_projection_k1             |        5 |        0.815733 |      0.869364 |       0.828988 |    0.999889 |     0.998889 |          33.8968 |    1.3527   |
| scorpion  | linear_scanner_subspace_projection_k16            |        5 |        0.724267 |      0.8813   |       0.849974 |    0.999889 |     0.998889 |          33.4342 |    0.952438 |
| scorpion  | linear_scanner_subspace_projection_k2             |        5 |        0.786    |      0.872349 |       0.834445 |    0.999889 |     0.998889 |          33.8022 |    1.35172  |
| scorpion  | linear_scanner_subspace_projection_k32            |        5 |        0.724267 |      0.8813   |       0.849974 |    0.999889 |     0.998889 |          33.4342 |    1.05685  |
| scorpion  | linear_scanner_subspace_projection_k4             |        5 |        0.724267 |      0.8813   |       0.849974 |    0.999889 |     0.998889 |          33.4342 |    1.01586  |
| scorpion  | linear_scanner_subspace_projection_k8             |        5 |        0.724267 |      0.8813   |       0.849974 |    0.999889 |     0.998889 |          33.4342 |    1.11121  |
| scorpion  | original_frozen_features                          |        5 |        0.865289 |      0.986799 |       0.982086 |    0.998622 |     0.993667 |          29.9706 |    0.241105 |
| scorpion  | pca_component_removal_k0                          |        5 |        0.866489 |      0.867213 |       0.820676 |    0.999889 |     0.998889 |          33.9743 |    0.976509 |
| scorpion  | pca_component_removal_k1                          |        5 |        0.8684   |      0.849063 |       0.795508 |    0.999233 |     0.995667 |          41.2708 |    0.81974  |
| scorpion  | pca_component_removal_k16                         |        5 |        0.634489 |      0.817958 |       0.765196 |    1        |     1        |          74.4777 |    0.830299 |
| scorpion  | pca_component_removal_k2                          |        5 |        0.869022 |      0.835997 |       0.779659 |    0.999356 |     0.994667 |          46.0352 |    0.953656 |
| scorpion  | pca_component_removal_k32                         |        5 |        0.559911 |      0.806356 |       0.764934 |    1        |     1        |          84.2571 |    0.847406 |
| scorpion  | pca_component_removal_k4                          |        5 |        0.862267 |      0.826094 |       0.756193 |    0.999889 |     0.998889 |          51.2488 |    1.00523  |
| scorpion  | pca_component_removal_k8                          |        5 |        0.8164   |      0.806906 |       0.746733 |    0.999778 |     0.997778 |          63.7255 |    0.757376 |

## Best simple baseline and decision

### SCORPION DINOv2

- Best simple baseline: `linear_scanner_subspace_projection_k16`.
- Classification: scanner suppression alone is insufficient; tissue-preserving factorization remains valuable.
- Best simple metrics: scanner_probe=0.724267, mean_paired_cosine=0.881300, mean_top1_retrieval=0.999889.

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
python experiments/baselines/run_pair_integrity_baseline_murder_test.py --out-dir results/paired_acquisition_factorization_baseline_murder_test --datasets scorpion --folds 0 1 2 3 4
```

## Output files

- `results/paired_acquisition_factorization_baseline_murder_test/raw_baseline_metrics.csv`
- `results/paired_acquisition_factorization_baseline_murder_test/baseline_summary.csv`
- `results/paired_acquisition_factorization_baseline_murder_test/baseline_contrasts.csv`
- `results/paired_acquisition_factorization_baseline_murder_test/baseline_murder_test_report.md`
- `results/paired_acquisition_factorization_baseline_murder_test/run_log.txt`

## Manifests used

- scorpion fold 0: C:\Users\matth\computational-pathology-research\data\scorpion\splits\fold_0_manifest.csv
- scorpion fold 1: C:\Users\matth\computational-pathology-research\data\scorpion\splits\fold_1_manifest.csv
- scorpion fold 2: C:\Users\matth\computational-pathology-research\data\scorpion\splits\fold_2_manifest.csv
- scorpion fold 3: C:\Users\matth\computational-pathology-research\data\scorpion\splits\fold_3_manifest.csv
- scorpion fold 4: C:\Users\matth\computational-pathology-research\data\scorpion\splits\fold_4_manifest.csv

## Claim boundary

This is a peer-review-hardening baseline test. It does not claim clinical validation, diagnostic performance, disease biology discovery, complete scanner invariance, or deployment readiness.
