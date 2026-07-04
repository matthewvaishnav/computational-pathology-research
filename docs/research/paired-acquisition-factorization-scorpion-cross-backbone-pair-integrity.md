# SCORPION Cross-Backbone Pair-Integrity Falsification

Status: completed on 2026-07-03

Evidence commit: `14726b13e7c0f23f9fe494399bab9fd902fecd7a`

## Question

Does the SCORPION pair-integrity result reproduce outside the original DINOv2 feature family, or can broken pair construction still look useful after changing the frozen representation backbone?

This is a peer-review-hardening falsification control for Paired-Acquisition Neural Factorization. It separates scanner-probe suppression from useful tissue preservation on Phikon and ImageNet ResNet50 features.

## Design

The experiment used the SCORPION paired-scanner setup with the frozen Paired-Acquisition Neural Factorization objective and schedule. The changed factors were the frozen feature backbone and the construction of positive pair groups during training.

Held-out evaluation remained on the real, unshuffled SCORPION test folds. Metrics were computed over five original-slide-blocked folds and five optimization seeds for each backbone.

## Conditions

| Condition | Pair construction |
|---|---|
| `true_pairs` | All five scanner views in a positive group came from the same tissue region. |
| `shuffled_region_pairs` | Non-anchor scanner views were deranged within the same slide, breaking exact region identity while preserving slide context. |
| `shuffled_sample_pairs` | Non-anchor scanner views were deranged across different slides, breaking same-region and same-slide identity. |

## Run Configuration

| Field | Phikon | ResNet50 |
|---|---|---|
| Dataset | SCORPION | SCORPION |
| Feature archive | `results/scorpion/features/fold_0_phikon.npz` | `results/scorpion/features/fold_0_resnet50_imagenet.npz` |
| Fold manifests | `data/scorpion/splits/` | `data/scorpion/splits/` |
| Folds | `0, 1, 2, 3, 4` | `0, 1, 2, 3, 4` |
| Seeds | `701, 702, 703, 704, 705` | `701, 702, 703, 704, 705` |
| Conditions | `true_pairs`, `shuffled_region_pairs`, `shuffled_sample_pairs` | `true_pairs`, `shuffled_region_pairs`, `shuffled_sample_pairs` |
| Smoke runs | `3 / 3` | `3 / 3` |
| Full runs | `75 / 75` | `75 / 75` |
| Epochs | `75` | `75` |
| Region batch size | `32` | `32` |
| Learning rate | `3e-4` | `3e-4` |
| Weight decay | `1e-4` | `1e-4` |

## Results

| Backbone | Condition | Scanner probe accuracy | Mean paired cosine | Worst paired cosine | Mean top-1 retrieval | Worst top-1 retrieval |
|---|---|---:|---:|---:|---:|---:|
| Phikon | `true_pairs` | 0.520044 | 0.864493 | 0.830021 | 0.999680 | 0.997444 |
| Phikon | `shuffled_region_pairs` | 0.434071 | 0.691315 | 0.602224 | 0.960316 | 0.899178 |
| Phikon | `shuffled_sample_pairs` | 0.408533 | 0.615847 | 0.474120 | 0.876287 | 0.675667 |
| ResNet50 | `true_pairs` | 0.314462 | 0.654441 | 0.597813 | 0.972620 | 0.945489 |
| ResNet50 | `shuffled_region_pairs` | 0.304507 | 0.518814 | 0.448693 | 0.883944 | 0.798044 |
| ResNet50 | `shuffled_sample_pairs` | 0.306658 | 0.508324 | 0.443892 | 0.877482 | 0.800311 |

## True-Pair Contrasts

Difference definition:

```text
true_pairs minus shuffled control
```

| Backbone | Comparison | Scanner probe | Mean cosine | Worst cosine | Mean retrieval | Worst retrieval |
|---|---|---:|---:|---:|---:|---:|
| Phikon | true minus region-shuffled | +0.085973 | +0.173178 | +0.227798 | +0.039364 | +0.098267 |
| Phikon | true minus sample-shuffled | +0.111511 | +0.248646 | +0.355902 | +0.123393 | +0.321778 |
| ResNet50 | true minus region-shuffled | +0.009956 | +0.135626 | +0.149121 | +0.088676 | +0.147444 |
| ResNet50 | true minus sample-shuffled | +0.007804 | +0.146117 | +0.153921 | +0.095138 | +0.145178 |

Scanner-probe values are lower-is-better for scanner suppression, so the positive scanner deltas mean the shuffled controls sometimes suppressed scanner identity as much as or more than true pairs. The tissue-preservation metrics are higher-is-better, and all true-pair tissue-preservation deltas were positive.

## Interpretation

Cross-backbone pair-integrity falsification on Phikon and ResNet50 reproduced the central DINOv2 pattern: true same-tissue pairs preserved tissue identity substantially better than shuffled-pair controls, even when shuffled controls achieved comparable or stronger scanner-probe suppression.

The result supports the narrow pair-integrity mechanism. Useful factorization is not explained by scanner suppression alone; it depends on preserving true same-tissue structure while scanner-associated information becomes less recoverable from the biological branch.

## Claim Boundary

This is peer-review-hardening evidence. It does not establish clinical validation, diagnostic performance, disease biology discovery, human clinical generalization, complete scanner invariance, perfect disentanglement, deployment readiness, prospective workflow safety, or regulatory readiness.

## Reproduced Commands

```powershell
python experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py --base-features results/scorpion/features/fold_0_phikon.npz --manifests-dir data/scorpion/splits --out-dir results/paired_acquisition_factorization_pair_integrity_scorpion_phikon --backbone phikon --seeds 701 702 703 704 705 --folds 0 1 2 3 4 --conditions true_pairs shuffled_region_pairs shuffled_sample_pairs --epochs 75 --region-batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --device cuda
```

```powershell
python experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py --base-features results/scorpion/features/fold_0_resnet50_imagenet.npz --manifests-dir data/scorpion/splits --out-dir results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50 --backbone resnet50 --seeds 701 702 703 704 705 --folds 0 1 2 3 4 --conditions true_pairs shuffled_region_pairs shuffled_sample_pairs --epochs 75 --region-batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --device cuda
```

## Output Files

- `experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py`
- `results/paired_acquisition_factorization_autonomous_research_run/autonomous_research_report.md`
- `results/paired_acquisition_factorization_autonomous_research_run/pair_integrity_contrast_summary.csv`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_phikon/raw_run_metrics.csv`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_phikon/condition_summary.csv`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_phikon/slide_blocked_contrasts.csv`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_phikon/pair_integrity_falsification_summary.md`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50/raw_run_metrics.csv`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50/condition_summary.csv`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50/slide_blocked_contrasts.csv`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50/pair_integrity_falsification_summary.md`
