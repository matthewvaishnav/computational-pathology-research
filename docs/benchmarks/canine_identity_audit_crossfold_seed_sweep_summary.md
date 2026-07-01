# Canine Paired-Acquisition Neural Factorization crossfold identity-audit seed sweep

## Purpose

This benchmark audit tests whether the Paired-Acquisition Neural Factorization representation-separation pattern holds across folds 0--4 and seeds 911--915 on the external multi-scanner canine SCC paired-acquisition dataset.

The audit compares:

- raw frozen DINOv2 features;
- paired-reference projected features;
- Paired-Acquisition Neural Factorization biological features;
- Paired-Acquisition Neural Factorization acquisition features.

The expected separation pattern is:

- biological features should suppress scanner identity while preserving same-region and same-sample biological recoverability;
- acquisition features should retain scanner identity while no longer behaving like a biological retrieval space;
- random-label probes should remain near chance;
- collapse controls should remain non-degenerate.

## Inputs

| Representation | Source |
|---|---|
| Raw DINOv2 features | `results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz` |
| Paired-reference features | `results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold/fold_{0..4}/runs/paired_reference_seed_{911..915}/projected_features.npz` |
| Paired-Acquisition Neural Factorization biological features | `results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold/fold_{0..4}/runs/pathoalign_dep20_seed_{911..915}/projected_features.npz`, key `features` |
| Paired-Acquisition Neural Factorization acquisition features | `results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold/fold_{0..4}/runs/pathoalign_dep20_seed_{911..915}/projected_features.npz`, key `acquisition_features` |
| Metadata manifest | `results/external_multiscanner_caninescc/geometry_qualified/geometry_qualified_manifest.csv` |

All audits used `sample_id` as the blocking unit.

## Group summary

| Representation | Runs | Scanner probe accuracy | Scanner probe std | Random-label probe | Region R@1 | Region R@1 std | Sample R@1 | Sample R@1 std | Region cross-scanner cosine | Cross-scanner cosine std | Effective rank | Effective rank std | Zero-var fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw DINOv2 features | 1 | 0.878261 | NA | 0.207950 | 0.872547 | NA | 0.985839 | NA | 0.918855 | NA | 442.802 | NA | 0.000000 |
| Paired-reference features | 25 | 0.745660 | 0.010262 | 0.206301 | 0.939886 | 0.003742 | 0.998847 | 0.000424 | 0.781365 | 0.003435 | 186.481 | 1.114 | 0.000000 |
| Paired-Acquisition Neural Factorization biological features | 25 | 0.299955 | 0.014763 | 0.203190 | 0.959463 | 0.003424 | 0.998966 | 0.000391 | 0.835016 | 0.003196 | 180.615 | 1.185 | 0.000000 |
| Paired-Acquisition Neural Factorization acquisition features | 25 | 0.968407 | 0.005381 | 0.224040 | 0.031155 | 0.003593 | 0.420661 | 0.007702 | 0.308065 | 0.017155 | 47.355 | 0.513 | 0.000000 |

## Interpretation

Across 25 Paired-Acquisition Neural Factorization biological-branch runs, scanner-probe accuracy is 0.299955, far below raw DINOv2 at 0.878261 and far below the paired-reference projection at 0.745660. At the same time, same-region retrieval improves from 0.872547 in raw DINOv2 to 0.959463, and same-sample retrieval remains essentially saturated at 0.998966.

Across 25 Paired-Acquisition Neural Factorization acquisition-branch runs, scanner-probe accuracy is 0.968407 while region-level retrieval falls to 0.031155. This is the expected opposite pattern if Paired-Acquisition Neural Factorization separates scanner/acquisition information into the acquisition branch rather than merely deleting it.

Random-label probes remain near the five-class chance baseline across the biological and reference representations. The acquisition branch random-label probe is slightly above 0.20 but remains far below its true scanner-probe accuracy. Zero-variance dimension fraction is 0.000000 for every representation group, so the scanner reduction in the biological branch is not explained by trivial zero-variance collapse.

The paired-reference projection improves biological retrieval relative to raw DINOv2, but it remains highly scanner-decodable. Paired-Acquisition Neural Factorization biological features retain or improve the biological retrieval gains while sharply reducing scanner decodability.

## Clean result statement

Across five folds and five seeds on an external paired-scanner canine SCC benchmark, Paired-Acquisition Neural Factorization suppresses scanner decodability in the biological branch while preserving biological retrieval, and retains scanner decodability in the acquisition branch.

Numerically, scanner-probe accuracy falls from 0.878261 in raw DINOv2 to 0.299955 in Paired-Acquisition Neural Factorization biological features, while region retrieval rises from 0.872547 to 0.959463. Conversely, Paired-Acquisition Neural Factorization acquisition features retain scanner-probe accuracy at 0.968407 while region retrieval falls to 0.031155.

## Claim boundary

This is a representation-identifiability and branch-separation benchmark result on an external paired-scanner canine SCC dataset. It is not clinical diagnostic validation, regulatory validation, or evidence of prospective patient benefit.
