# Canine Paired-Acquisition Neural Factorization biological versus acquisition branch audit

## Purpose

This benchmark audit tests the stronger Paired-Acquisition Neural Factorization separation claim on the external multi-scanner canine SCC paired-acquisition feature set:

- the Paired-Acquisition Neural Factorization biological branch should suppress scanner identity while preserving biological recoverability;
- the Paired-Acquisition Neural Factorization acquisition branch should preserve scanner identity;
- neither branch should collapse trivially.

## Inputs

| Representation | Source |
|---|---|
| Raw DINOv2 features | `results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz` |
| Paired-Acquisition Neural Factorization biological features | `results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold/fold_0/runs/pathoalign_dep20_seed_{911..915}/projected_features.npz`, key `features` |
| Paired-Acquisition Neural Factorization acquisition features | `results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold/fold_0/runs/pathoalign_dep20_seed_{911..915}/projected_features.npz`, key `acquisition_features` |
| Metadata manifest | `results/external_multiscanner_caninescc/geometry_qualified/geometry_qualified_manifest.csv` |

All audits used `sample_id` as the blocking unit.

## Group summary

| Representation | Runs | Scanner probe accuracy | Random-label probe | Region R@1 | Sample R@1 | Region cross-scanner cosine | Effective rank | Zero-var fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw DINOv2 features | 1 | 0.878261 | 0.207950 | 0.872547 | 0.985839 | 0.918855 | 442.802 | 0.000000 |
| Paired-Acquisition Neural Factorization biological features | 5 | 0.296845 | 0.204919 | 0.958211 | 0.999354 | 0.839324 | 181.210 | 0.000000 |
| Paired-Acquisition Neural Factorization acquisition features | 5 | 0.961391 | 0.227230 | 0.035180 | 0.426683 | 0.322286 | 47.987 | 0.000000 |

## Interpretation

The biological branch sharply suppresses scanner decodability relative to raw DINOv2 while preserving or improving biological retrieval. Scanner-probe accuracy falls from 0.878261 in raw DINOv2 to 0.296845 in the Paired-Acquisition Neural Factorization biological branch, while region-level retrieval increases from 0.872547 to 0.958211 and sample-level retrieval increases from 0.985839 to 0.999354.

The acquisition branch shows the opposite pattern. Scanner-probe accuracy is 0.961391, while region-level retrieval falls to 0.035180. This is the expected result if scanner/acquisition information has been separated into the acquisition branch rather than erased from the model.

Random-label probes remain near the five-class chance baseline, and the zero-variance dimension fraction is 0.000000 across all representations. The acquisition branch has lower effective rank, but it is not collapsed by the zero-variance diagnostic.

## Clean separation claim

Paired-Acquisition Neural Factorization separates scanner/acquisition identity from biological identity: the biological branch suppresses scanner decodability while preserving same-region and same-sample recoverability, and the acquisition branch retains scanner decodability while no longer behaving like a biological retrieval space.

## Claim boundary

This is a representation-identifiability and branch-separation benchmark result on an external paired-scanner canine SCC dataset. It is not clinical diagnostic validation, regulatory validation, or evidence of prospective patient benefit.
