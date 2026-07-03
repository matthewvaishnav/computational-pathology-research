# SCORPION resnet50 Pair-Integrity Falsification Summary

## Run Status

- Dataset: SCORPION
- Backbone: resnet50
- Feature archive: `results/scorpion/features/fold_0_resnet50_imagenet.npz`
- Seeds: 701, 702, 703, 704, 705
- Folds: 0, 1, 2, 3, 4
- Conditions: true_pairs, shuffled_region_pairs, shuffled_sample_pairs
- Runtime seconds: 1138.5
- Completed runs: 75 / 75
- Smoke/full pass status for this command: True
- Scanner-adversary-only condition: unavailable; no clean existing SCORPION implementation was found, so it was not added.

## Main Result Table

| Condition | Scanner probe | Mean paired cosine | Worst paired cosine | Mean top-1 retrieval | Worst top-1 retrieval | Effective rank | Bio/acq cross-cov RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| true_pairs | 0.314462 | 0.654441 | 0.597813 | 0.972620 | 0.945489 | 85.087467 | 0.089567 |
| shuffled_region_pairs | 0.304507 | 0.518814 | 0.448693 | 0.883944 | 0.798044 | 81.669536 | 0.085062 |
| shuffled_sample_pairs | 0.306658 | 0.508324 | 0.443892 | 0.877482 | 0.800311 | 81.437735 | 0.084652 |

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
python experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py --base-features results/scorpion/features/fold_0_resnet50_imagenet.npz --manifests-dir data/scorpion/splits --out-dir results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50 --backbone resnet50 --seeds 701 702 703 704 705 --folds 0 1 2 3 4 --conditions true_pairs shuffled_region_pairs shuffled_sample_pairs --epochs 75 --region-batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --device cuda
```
