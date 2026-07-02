# Canine SCC Pair-Integrity Falsification Summary

## Run Status

- Dataset: external multi-scanner canine cutaneous SCC
- Backbone: DINOv2-Base
- Seeds: 911, 912, 913, 914, 915
- Folds: 0, 1, 2, 3, 4
- Conditions: true_pairs, shuffled_region_pairs, shuffled_sample_pairs
- Runtime seconds: 1827.6
- Completed runs: 75 / 75
- Smoke/full pass status for this command: True
- Scanner-adversary-only condition: unavailable; no clean existing canine implementation was found, so it was not added.

## Main Result Table

| Condition | Scanner probe | Mean paired cosine | Worst paired cosine | Mean top-1 retrieval | Worst top-1 retrieval | Effective rank | Bio/acq cross-cov RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| true_pairs | 0.361408 | 0.729961 | 0.656736 | 0.933392 | 0.884431 | 74.044385 | 0.089831 |
| shuffled_region_pairs | 0.305673 | 0.542164 | 0.421063 | 0.729274 | 0.515828 | 54.513186 | 0.087106 |
| shuffled_sample_pairs | 0.409302 | 0.584855 | 0.497105 | 0.718254 | 0.565006 | 45.327860 | 0.096097 |

## Pair-Integrity Falsification Logic

Expected result: true pairs should preserve tissue identity metrics better than shuffled-pair controls.

Falsification logic: if shuffled pairs suppress scanner signal but damage paired-tissue consistency/retrieval, that supports the interpretation that true same-tissue pairing matters. If shuffled pairs perform similarly to true pairs on tissue preservation, the paired-acquisition claim is weakened and must be reported honestly. If true pairs fail, report failure honestly.

## True-Pair Comparison

- `shuffled_region_pairs`: scanner_similar_to_true=False; tissue_damage_vs_true=True; true_better_tissue_metrics=True.
- `shuffled_sample_pairs`: scanner_similar_to_true=False; tissue_damage_vs_true=True; true_better_tissue_metrics=True.

## Claim Boundary

This does not prove clinical robustness, diagnosis, disease biology discovery, human clinical generalization from canine SCC, deployment readiness, complete scanner invariance, or perfect disentanglement. It is an external pair-structure falsification control.

## Metric Availability

- scanner_probe_accuracy: available.
- mean_paired_cosine: available.
- worst_paired_cosine: available.
- mean_top1_retrieval: available.
- worst_top1_retrieval: available.
- effective_rank: available.
- biological_acquisition_cross_covariance: available as normalized biological/acquisition cross-covariance RMS on held-out test samples.

## Readiness

Current classification: peer-review-hardening.

## Artifacts

- raw_run_metrics.csv
- condition_summary.csv
- sample_blocked_contrasts.csv
- pair_integrity_falsification_summary.md
- run_log.txt
- pair_construction_audit.csv
- experiment_design.json

## Exact Retry Command

```powershell
python experiments/canine/run_pair_integrity_falsification_caninescc.py --base-features results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz --manifests-dir data/external_multiscanner_caninescc/patch_manifests/splits --out-dir results/paired_acquisition_factorization_pair_integrity_caninescc --seeds 911 912 913 914 915 --folds 0 1 2 3 4 --conditions true_pairs shuffled_region_pairs shuffled_sample_pairs --epochs 75 --region-batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --device cuda
```
