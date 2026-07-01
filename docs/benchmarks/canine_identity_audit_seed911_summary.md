# Canine paired-scanner identity audit: DINOv2 raw vs Paired-Acquisition Neural Factorization seed 911

## Purpose

This note records the first Paired-Acquisition Neural Factorization Oncology Identity Benchmark run on the external multi-scanner canine SCC paired-acquisition feature set.

The audit compares frozen raw DINOv2 features, a paired-reference projected representation, and a Paired-Acquisition Neural Factorization projected biological representation from fold 0 / seed 911.

## Inputs

| Run | Source |
|---|---|
| Raw DINOv2 | `results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz` |
| Paired reference | `results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold/fold_0/runs/paired_reference_seed_911/projected_features.npz` |
| Paired-Acquisition Neural Factorization | `results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold/fold_0/runs/pathoalign_dep20_seed_911/projected_features.npz` |
| Metadata manifest | `results/external_multiscanner_caninescc/geometry_qualified/geometry_qualified_manifest.csv` |

All audits used `sample_id` as the blocking unit.

## Benchmark table

| Representation | Scanner probe accuracy ↓ | Random-label probe | Region R@1 ↑ | Sample R@1 ↑ | Region cross-scanner cosine ↑/≈ | Effective rank | Zero-var fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw DINOv2 | 0.878261 | 0.207950 | 0.872547 | 0.985839 | 0.918855 | 442.802 | 0.000000 |
| Paired reference | 0.753789 | 0.198261 | 0.939130 | 0.999255 | 0.785956 | 187.667 | 0.000000 |
| Paired-Acquisition Neural Factorization | 0.298634 | 0.201242 | 0.959255 | 0.999255 | 0.839281 | 181.359 | 0.000000 |

## Interpretation

Raw DINOv2 preserves biological identity but also strongly encodes scanner identity. Its scanner probe accuracy is 0.878261 against a five-class chance baseline near 0.20.

The Paired-Acquisition Neural Factorization biological representation reduces scanner-probe accuracy to 0.298634 while preserving or improving region-level and sample-level retrieval. The random-label probe remains near chance, and collapse checks remain non-degenerate.

This supports the benchmark-level claim that Paired-Acquisition Neural Factorization reduces acquisition identity in the biological representation without destroying same-region or same-sample biological recoverability for this fold and seed.

## Claim boundary

This is a representation-identifiability benchmark result on an external paired-scanner canine SCC dataset. It is not a clinical diagnostic validation.
