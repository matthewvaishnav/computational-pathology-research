# SCORPION resnet50 Pair-Integrity Falsification Summary

## Run Status

- Dataset: SCORPION
- Backbone: resnet50
- Feature archive: `results/scorpion/features/fold_0_resnet50_imagenet.npz`
- Seeds: 701
- Folds: 0
- Conditions: true_pairs, shuffled_region_pairs, shuffled_sample_pairs
- Runtime seconds: 5.3
- Completed runs: 3 / 3
- Smoke/full pass status for this command: True
- Scanner-adversary-only condition: unavailable; no clean existing SCORPION implementation was found, so it was not added.

## Main Result Table

| Condition | Scanner probe | Mean paired cosine | Worst paired cosine | Mean top-1 retrieval | Worst top-1 retrieval | Effective rank | Bio/acq cross-cov RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| true_pairs | 0.500000 | 0.711655 | 0.671643 | 0.935000 | 0.890000 | 52.787655 | 0.199868 |
| shuffled_region_pairs | 0.506000 | 0.630372 | 0.567226 | 0.909000 | 0.845000 | 75.018283 | 0.167442 |
| shuffled_sample_pairs | 0.522000 | 0.592270 | 0.529732 | 0.919500 | 0.865000 | 90.590606 | 0.153378 |

## Pair-Integrity Falsification Logic

Expected result: true pairs should preserve tissue identity metrics better than shuffled-pair controls.

Falsification logic: if shuffled pairs suppress scanner signal but damage paired-tissue consistency/retrieval, that supports the interpretation that true same-tissue pairing matters. If shuffled pairs perform similarly to true pairs on tissue preservation, the paired-acquisition claim is weakened and should be reported honestly.

## True-Pair Comparison

- `shuffled_region_pairs`: true_better_all_tissue_metrics=True; scanner_probe_lower_than_true=False; tissue_damage_vs_true=True.
- `shuffled_sample_pairs`: true_better_all_tissue_metrics=True; scanner_probe_lower_than_true=False; tissue_damage_vs_true=True.

## Classification

supports pair-integrity mechanism.

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
python experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py --base-features results/scorpion/features/fold_0_resnet50_imagenet.npz --manifests-dir data/scorpion/splits --out-dir results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50_smoke --backbone resnet50 --seeds 701 --folds 0 --conditions true_pairs shuffled_region_pairs shuffled_sample_pairs --epochs 1 --region-batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --device cuda
```
