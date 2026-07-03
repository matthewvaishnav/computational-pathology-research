# Canine SCC Pair-Integrity Falsification

Status: completed on 2026-07-02

## Question

Does the external canine SCC result depend on true same-tissue paired acquisitions, or can broken pair construction produce scanner suppression without preserving tissue identity?

This is an external falsification control for Paired-Acquisition Neural Factorization. It separates scanner suppression from useful tissue preservation on the independent five-scanner canine SCC benchmark.

## Design

The experiment used the locked external canine SCC DINOv2-Base paired-acquisition setup. The frozen Paired-Acquisition Neural Factorization objective and schedule were held fixed. The only changed factor was positive pair construction during training.

Held-out evaluation remained on the real, unshuffled sample-blocked test folds. Metrics were computed across five folds and five optimization seeds.

## Conditions

| Condition | Pair construction |
|---|---|
| `true_pairs` | All five scanner views in a positive group came from the same tissue region. |
| `shuffled_region_pairs` | Non-anchor scanner views were deranged to wrong tissue regions while preserving scanner/view structure. |
| `shuffled_sample_pairs` | Non-anchor scanner views were paired across different biological samples where possible. |

`scanner_adversary_only` was not included because no clean existing canine implementation was available.

## Run Configuration

| Field | Value |
|---|---|
| Dataset | Multi-scanner canine cutaneous SCC |
| Backbone | DINOv2-Base |
| Feature archive | `results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz` |
| Fold manifests | `data/external_multiscanner_caninescc/patch_manifests/splits/` |
| Folds | `0, 1, 2, 3, 4` |
| Seeds | `911, 912, 913, 914, 915` |
| Conditions | `true_pairs`, `shuffled_region_pairs`, `shuffled_sample_pairs` |
| Completed runs | `75 / 75` |
| Epochs | `75` |
| Region batch size | `32` |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Runtime | `1827.6` seconds |

## Results

| Condition | Scanner probe accuracy | Mean paired cosine | Worst paired cosine | Mean top-1 retrieval | Worst top-1 retrieval | Effective rank | Biological/acquisition cross-covariance |
|---|---:|---:|---:|---:|---:|---:|---:|
| `true_pairs` | 0.361408 | 0.729961 | 0.656736 | 0.933392 | 0.884431 | 74.044385 | 0.089831 |
| `shuffled_region_pairs` | 0.305673 | 0.542164 | 0.421063 | 0.729274 | 0.515828 | 54.513186 | 0.087106 |
| `shuffled_sample_pairs` | 0.409302 | 0.584855 | 0.497105 | 0.718254 | 0.565006 | 45.327860 | 0.096097 |

## Interpretation

True pairs beat both shuffled controls on tissue-preservation metrics. Region-shuffled pairs suppressed scanner signal more than true pairs, but damaged paired cosine and same-region retrieval. Sample-shuffled pairs also degraded paired cosine and retrieval.

This externally strengthens the pair-integrity interpretation: useful scanner suppression depends on true same-tissue pairing, not scanner suppression alone.

## Claim Boundary

This is peer-review-hardening evidence. It does not establish clinical validation, diagnostic performance, disease biology discovery, human clinical generalization, complete scanner invariance, perfect disentanglement, deployment readiness, prospective workflow safety, or regulatory readiness.

## Reproduced Command

```powershell
python experiments/canine/run_pair_integrity_falsification_caninescc.py --base-features results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz --manifests-dir data/external_multiscanner_caninescc/patch_manifests/splits --out-dir results/paired_acquisition_factorization_pair_integrity_caninescc --seeds 911 912 913 914 915 --folds 0 1 2 3 4 --conditions true_pairs shuffled_region_pairs shuffled_sample_pairs --epochs 75 --region-batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --device cuda
```

## Output Files

- `experiments/canine/run_pair_integrity_falsification_caninescc.py`
- `results/paired_acquisition_factorization_pair_integrity_caninescc/raw_run_metrics.csv`
- `results/paired_acquisition_factorization_pair_integrity_caninescc/condition_summary.csv`
- `results/paired_acquisition_factorization_pair_integrity_caninescc/sample_blocked_contrasts.csv`
- `results/paired_acquisition_factorization_pair_integrity_caninescc/pair_integrity_falsification_summary.md`
- `results/paired_acquisition_factorization_pair_integrity_caninescc/run_log.txt`
- `results/paired_acquisition_factorization_pair_integrity_caninescc/pair_construction_audit.csv`
