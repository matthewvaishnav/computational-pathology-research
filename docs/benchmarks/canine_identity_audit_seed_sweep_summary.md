# Canine paired-scanner identity-audit seed sweep

## Purpose

This note records the five-seed Paired-Acquisition Neural Factorization Oncology Identity Benchmark audit on the external multi-scanner canine SCC paired-acquisition feature set.

The sweep compares raw frozen DINOv2 features, paired-reference projected representations, and Paired-Acquisition Neural Factorization projected biological representations from fold 0 across seeds 911--915.

## Inputs

| Run | Source |
|---|---|
| Raw DINOv2 | `results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz` |
| Paired reference | `results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold/fold_0/runs/paired_reference_seed_{911..915}/projected_features.npz` |
| Paired-Acquisition Neural Factorization | `results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold/fold_0/runs/pathoalign_dep20_seed_{911..915}/projected_features.npz` |
| Metadata manifest | `results/external_multiscanner_caninescc/geometry_qualified/geometry_qualified_manifest.csv` |

All audits used `sample_id` as the blocking unit. Raw DINOv2 is a single frozen-feature baseline. Paired reference and Paired-Acquisition Neural Factorization are five-seed projected-representation sweeps.

## Group summary

| Representation | Runs | Scanner probe accuracy ↓ | Scanner probe std | Random-label probe | Region R@1 ↑ | Region R@1 std | Sample R@1 ↑ | Region cross-scanner cosine ↑/≈ | Cross-scanner cosine std | Effective rank | Effective rank std | Zero-var fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw DINOv2 | 1 | 0.878261 | NA | 0.207950 | 0.872547 | NA | 0.985839 | 0.918855 | NA | 442.802 | NA | 0.000000 |
| Paired reference | 5 | 0.748820 | 0.009568 | 0.204870 | 0.938236 | 0.002053 | 0.999205 | 0.786243 | 0.000869 | 187.286 | 0.327 | 0.000000 |
| Paired-Acquisition Neural Factorization | 5 | 0.296845 | 0.010175 | 0.204919 | 0.958211 | 0.000969 | 0.999354 | 0.839324 | 0.000990 | 181.210 | 0.325 | 0.000000 |

## Interpretation

Raw DINOv2 strongly preserves biological identity but also strongly encodes scanner identity. Its scanner-probe accuracy is 0.878261 against a five-class chance baseline near 0.20.

Across five Paired-Acquisition Neural Factorization seeds, scanner-probe accuracy falls to 0.296845 while region-level retrieval improves to 0.958211 and sample-level retrieval remains essentially saturated at 0.999354. Random-label probe accuracy remains near chance, and collapse checks remain non-degenerate with zero zero-variance dimensions.

The paired-reference projection reduces scanner identity relative to raw DINOv2 but remains much more scanner-decodable than Paired-Acquisition Neural Factorization. Paired-Acquisition Neural Factorization also improves region retrieval relative to the paired reference and improves region cross-scanner cosine relative to the paired reference.

This supports the benchmark-level claim that Paired-Acquisition Neural Factorization reduces acquisition identity in the biological representation while preserving or improving biological recoverability across this external paired-scanner canine SCC fold and five-seed sweep.

## Clean claim

Paired-Acquisition Neural Factorization reduces scanner decodability from 0.878261 in raw DINOv2 to 0.296845 across five seeds, while improving region retrieval from 0.872547 to 0.958211, preserving sample retrieval near 1.0, keeping random-label controls near chance, and avoiding representation collapse.

## Claim boundary

This is a representation-identifiability benchmark result on an external paired-scanner canine SCC dataset. It is not clinical diagnostic validation, regulatory validation, or evidence of prospective patient benefit.
