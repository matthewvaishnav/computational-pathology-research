# SCORPION Pair-Integrity Falsification Summary

## Run Status

- Dataset: SCORPION
- Backbone: DINOv2-Base
- Seeds: 701, 702, 703, 704, 705
- Folds: 0, 1, 2, 3, 4
- Conditions: true_pairs, shuffled_region_pairs, shuffled_sample_pairs
- Runtime seconds: 1064.2
- Completed runs: 75 / 75
- Smoke/full pass status for this command: True
- Scanner-adversary-only condition: unavailable; no clean existing SCORPION implementation was found, so it was not added.

## Main Result Table

| Condition | Scanner probe | Mean paired cosine | Worst paired cosine | Mean top-1 retrieval | Worst top-1 retrieval | Effective rank | Bio/acq cross-cov RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| true_pairs | 0.399778 | 0.879577 | 0.850230 | 0.999913 | 0.999133 | 54.451412 | 0.091709 |
| shuffled_region_pairs | 0.373573 | 0.808929 | 0.763182 | 0.995358 | 0.983933 | 39.076858 | 0.105443 |
| shuffled_sample_pairs | 0.358836 | 0.766820 | 0.716845 | 0.979242 | 0.949400 | 35.225852 | 0.102893 |

## Pair-Integrity Falsification Logic

Expected result: true pairs should reduce scanner probe while preserving or improving tissue identity metrics.

Falsification logic: if shuffled pairs perform similarly to true pairs, the paired-acquisition claim is weakened. If shuffled pairs reduce scanner signal but damage tissue identity metrics, the claim is strengthened. If true pairs fail, that failure should be reported honestly.

## True-Pair Comparison

- `shuffled_region_pairs`: scanner_similar_to_true=True; tissue_damage_vs_true=True; true_better_tissue_metrics=True.
- `shuffled_sample_pairs`: scanner_similar_to_true=False; tissue_damage_vs_true=True; true_better_tissue_metrics=True.

## Claim Boundary

This experiment does not prove clinical robustness, diagnosis, disease biology, deployment readiness, complete scanner invariance, or perfect disentanglement. It only tests whether the factorization effect depends on true pair integrity.

## Metric Availability

- scanner_probe_accuracy: available.
- mean_paired_cosine: available.
- worst_paired_cosine: available.
- mean_top1_retrieval: available.
- worst_top1_retrieval: available.
- effective_rank: available.
- biological_acquisition_cross_covariance: available as normalized biological/acquisition cross-covariance RMS on held-out test slides.

## Readiness

Current classification: peer-review-hardening.

## Artifacts

- raw_run_metrics.csv
- condition_summary.csv
- slide_blocked_contrasts.csv
- fold_blocked_contrasts.csv
- pair_integrity_falsification_summary.md
- run_log.txt
- pair_construction_audit.csv
- experiment_design.json

## Exact Retry Command

```powershell
python experiments/scorpion/run_pair_integrity_falsification.py --base-features results/scorpion/features/fold_0_dinov2_base.npz --manifests-dir data/scorpion/splits --out-dir results/paired_acquisition_factorization_pair_integrity_scorpion --seeds 701 702 703 704 705 --folds 0 1 2 3 4 --conditions true_pairs shuffled_region_pairs shuffled_sample_pairs --epochs 75 --region-batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --device cuda
```
