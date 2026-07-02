# SCORPION Pair-Integrity Falsification

Status: completed on 2026-07-02

## Question

Does the observed scanner-suppression and tissue-preservation pattern depend on true same-tissue paired acquisitions, or can broken pair construction produce the same useful factorization effect?

This is a falsification control for the paired-acquisition claim. It separates scanner suppression from tissue preservation.

## Design

The experiment used the existing SCORPION DINOv2-Base paired-acquisition setup with the frozen Paired-Acquisition Neural Factorization objective and schedule. The only changed factor was the construction of positive pair groups during training.

The held-out evaluation remained on the real, unshuffled SCORPION test folds. Metrics were computed over five original-slide-blocked folds and five optimization seeds.

## Conditions

| Condition | Pair construction |
|---|---|
| `true_pairs` | All five scanner views in a positive group came from the same tissue region. |
| `shuffled_region_pairs` | Non-anchor scanner views were deranged within the same slide, breaking exact region identity while preserving slide context. |
| `shuffled_sample_pairs` | Non-anchor scanner views were deranged across different slides, breaking same-region and same-slide identity. |

`scanner_adversary_only` was not included because no clean existing SCORPION implementation was found.

## Run Configuration

| Field | Value |
|---|---|
| Dataset | SCORPION |
| Backbone | DINOv2-Base |
| Feature archive | `results/scorpion/features/fold_0_dinov2_base.npz` |
| Fold manifests | `data/scorpion/splits/` |
| Folds | `0, 1, 2, 3, 4` |
| Seeds | `701, 702, 703, 704, 705` |
| Conditions | `true_pairs`, `shuffled_region_pairs`, `shuffled_sample_pairs` |
| Completed runs | `75 / 75` |
| Epochs | `75` |
| Region batch size | `32` |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Runtime | `1064.2` seconds |

## Results

| Condition | Scanner probe accuracy | Mean paired cosine | Worst paired cosine | Mean top-1 retrieval | Worst top-1 retrieval |
|---|---:|---:|---:|---:|---:|
| `true_pairs` | 0.399778 | 0.879577 | 0.850230 | 0.999913 | 0.999133 |
| `shuffled_region_pairs` | 0.373573 | 0.808929 | 0.763182 | 0.995358 | 0.983933 |
| `shuffled_sample_pairs` | 0.358836 | 0.766820 | 0.716845 | 0.979242 | 0.949400 |

Additional available metrics:

| Condition | Effective rank | Biological/acquisition cross-covariance RMS |
|---|---:|---:|
| `true_pairs` | 54.451412 | 0.091709 |
| `shuffled_region_pairs` | 39.076858 | 0.105443 |
| `shuffled_sample_pairs` | 35.225852 | 0.102893 |

## Interpretation

Broken-pair controls suppressed scanner signal similarly or more than true pairs, but damaged tissue-preservation metrics. Mean paired cosine fell from 0.879577 under true pairs to 0.808929 under region-shuffled pairs and 0.766820 under sample-shuffled pairs. Worst paired cosine and same-region retrieval also declined, with the strongest degradation under sample-shuffled pairing.

This supports the narrower interpretation that the useful scanner-suppression/tissue-preservation tradeoff depends on true same-tissue pair integrity, rather than scanner suppression alone.

## Claim Boundary

This is peer-review hardening only. It does not establish clinical validation, diagnostic performance, disease biology discovery, complete scanner invariance, perfect disentanglement, deployment readiness, prospective workflow safety, or regulatory readiness.

## Reproduced Command

```powershell
python experiments/scorpion/run_pair_integrity_falsification.py --base-features results/scorpion/features/fold_0_dinov2_base.npz --manifests-dir data/scorpion/splits --out-dir results/paired_acquisition_factorization_pair_integrity_scorpion --seeds 701 702 703 704 705 --folds 0 1 2 3 4 --conditions true_pairs shuffled_region_pairs shuffled_sample_pairs --epochs 75 --region-batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --device cuda
```

## Output Files

- `experiments/scorpion/run_pair_integrity_falsification.py`
- `results/paired_acquisition_factorization_pair_integrity_scorpion/raw_run_metrics.csv`
- `results/paired_acquisition_factorization_pair_integrity_scorpion/condition_summary.csv`
- `results/paired_acquisition_factorization_pair_integrity_scorpion/slide_blocked_contrasts.csv`
- `results/paired_acquisition_factorization_pair_integrity_scorpion/fold_blocked_contrasts.csv`
- `results/paired_acquisition_factorization_pair_integrity_scorpion/pair_integrity_falsification_summary.md`
- `results/paired_acquisition_factorization_pair_integrity_scorpion/run_log.txt`
- `results/paired_acquisition_factorization_pair_integrity_scorpion/pair_construction_audit.csv`
