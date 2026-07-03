# SCORPION phikon Pair-Integrity Falsification Summary

## Run Status

- Dataset: SCORPION
- Backbone: phikon
- Feature archive: `results/scorpion/features/fold_0_phikon.npz`
- Seeds: 701, 702, 703, 704, 705
- Folds: 0, 1, 2, 3, 4
- Conditions: true_pairs, shuffled_region_pairs, shuffled_sample_pairs
- Runtime seconds: 1068.0
- Completed runs: 75 / 75
- Smoke/full pass status for this command: True
- Scanner-adversary-only condition: unavailable; no clean existing SCORPION implementation was found, so it was not added.

## Main Result Table

| Condition | Scanner probe | Mean paired cosine | Worst paired cosine | Mean top-1 retrieval | Worst top-1 retrieval | Effective rank | Bio/acq cross-cov RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| true_pairs | 0.520044 | 0.864493 | 0.830021 | 0.999680 | 0.997444 | 46.750716 | 0.088867 |
| shuffled_region_pairs | 0.434071 | 0.691315 | 0.602224 | 0.960316 | 0.899178 | 44.390345 | 0.111446 |
| shuffled_sample_pairs | 0.408533 | 0.615847 | 0.474120 | 0.876287 | 0.675667 | 42.545296 | 0.112640 |

## Pair-Integrity Falsification Logic

Expected result: true pairs should preserve tissue identity metrics better than shuffled-pair controls.

Falsification logic: if shuffled pairs suppress scanner signal but damage paired-tissue consistency/retrieval, that supports the interpretation that true same-tissue pairing matters. If shuffled pairs perform similarly to true pairs on tissue preservation, the paired-acquisition claim is weakened and should be reported honestly.

## True-Pair Comparison

- `shuffled_region_pairs`: true_better_all_tissue_metrics=True; scanner_probe_lower_than_true=True; tissue_damage_vs_true=True.
- `shuffled_sample_pairs`: true_better_all_tissue_metrics=True; scanner_probe_lower_than_true=True; tissue_damage_vs_true=True.

## Classification

scanner suppression separated from useful tissue preservation.

## Claim Boundary

This is peer-review hardening only. It does not establish clinical validation, diagnostic performance, disease biology discovery, human clinical generalization, deployment readiness, complete scanner invariance, or perfect disentanglement.

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
python experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py --base-features results/scorpion/features/fold_0_phikon.npz --manifests-dir data/scorpion/splits --out-dir results/paired_acquisition_factorization_pair_integrity_scorpion_phikon --backbone phikon --seeds 701 702 703 704 705 --folds 0 1 2 3 4 --conditions true_pairs shuffled_region_pairs shuffled_sample_pairs --epochs 75 --region-batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --device cuda
```
