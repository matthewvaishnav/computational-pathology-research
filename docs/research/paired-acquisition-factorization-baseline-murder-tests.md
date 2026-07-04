# Paired-Acquisition Neural Factorization Baseline Murder Tests

Status: completed on 2026-07-03

## Question

Can simple post-hoc scanner-removal operations on frozen DINOv2 embeddings match
the scanner-suppression/tissue-preservation tradeoff achieved by Paired-Acquisition
Neural Factorization, or does the learned factorization add value beyond what linear
scanner-subspace projection and PCA component removal can achieve?

This is a peer-review-hardening baseline test. It strengthens the claim that
paired-acquisition supervision is necessary for the observed tradeoff.

## Design

All baselines operate on frozen DINOv2-base features without trainable parameters,
using only fold-fit standardization and linear algebraic operations computed from
the fit split:

| Baseline | Description |
|---|---|
| `original_frozen_features` | Frozen DINOv2 embeddings evaluated directly after fold alignment. |
| `linear_scanner_subspace_projection_k*` | Top-k logistic scanner-discriminative directions removed after standardization. Directions learned via balanced L2-penalized logistic regression on fit-scanner labels, then SVD-decomposed. |
| `pca_component_removal_k*` | Top-k PCA directions removed after fit-mean centering and standardization. |
| `paired_consistency_reference` | Existing locked paired-consistency projected features (five seeds per fold). |
| `paired_acquisition_neural_factorization_reference` | Existing locked PathoAlign dep20 projected features (five seeds per fold) used as the method reference. |

k values tested: 0, 1, 2, 4, 8, 16, 32.

Both linear scanner-subspace projection and PCA were computed per fold using only
the fit split. Evaluation used the held-out test split with original-slide blocking
(SCORPION) or sample blocking (canine SCC).

Three baselines were skipped:

- **random_pair_training**: no safe existing baseline mode was available without new
  pair-training semantics beyond the completed shuffled-pair falsification controls.
- **optional ablations** (adversarial_only, no_acquisition_branch, no_covariance_penalty):
  not available as clean locked runners.

## SCORPION DINOv2 Results

125 rows across 5 folds (0--4), 17 baselines, 5 reference seeds.
Runtime: 51 seconds.

| Baseline | Scanner Probe ↓ | Mean Cosine ↑ | Worst Cosine ↑ | Mean Top1 ↑ | Eff. Rank |
|---|---|---|---|---|---|
| **Neural Factorization** | **0.399** | **0.879** | **0.850** | **0.9998** | 54.5 |
| Paired Consistency Ref | 0.782 | 0.848 | 0.820 | 0.9999 | 56.9 |
| Original Frozen Features | 0.865 | 0.987 | 0.982 | 0.9986 | 30.0 |
| Linear Scanner k=0 | 0.866 | 0.867 | 0.821 | 0.9999 | 34.0 |
| Linear Scanner k=1 | 0.816 | 0.869 | 0.829 | 0.9999 | 33.9 |
| Linear Scanner k=2 | 0.786 | 0.872 | 0.834 | 0.9999 | 33.8 |
| **Linear Scanner k≥4 (best)** | **0.724** | **0.881** | **0.850** | **0.9999** | 33.4 |
| PCA k=1 | 0.868 | 0.849 | 0.796 | 0.9992 | 41.3 |
| PCA k=8 | 0.816 | 0.807 | 0.747 | 0.9998 | 63.7 |
| PCA k=16 | 0.634 | 0.818 | 0.765 | 1.0000 | 74.5 |
| **PCA k=32** | **0.560** | **0.806** | **0.765** | **1.0000** | 84.3 |

## External Canine SCC DINOv2 Results

125 rows across 5 folds (0--4), 17 baselines, 5 reference seeds.
Runtime: 92 seconds.

| Baseline | Scanner Probe ↓ | Mean Cosine ↑ | Worst Cosine ↑ | Mean Top1 ↑ | Eff. Rank |
|---|---|---|---|---|---|
| **Neural Factorization** | **0.361** | **0.730** | **0.657** | **0.933** | 74.0 |
| Paired Consistency Ref | 0.753 | 0.696 | 0.627 | 0.931 | 79.8 |
| Original Frozen Features | 0.863 | 0.919 | 0.891 | 0.833 | 46.4 |
| Linear Scanner k=0 | 0.864 | 0.733 | 0.657 | 0.873 | 53.7 |
| Linear Scanner k=1 | 0.807 | 0.734 | 0.657 | 0.872 | 53.6 |
| Linear Scanner k=2 | 0.766 | 0.735 | 0.659 | 0.873 | 53.5 |
| **Linear Scanner k≥4 (best)** | **0.707** | **0.739** | **0.665** | **0.874** | 53.3 |
| PCA k=1 | 0.865 | 0.701 | 0.616 | 0.884 | 71.0 |
| PCA k=8 | 0.818 | 0.650 | 0.540 | 0.917 | 107.0 |
| PCA k=16 | 0.728 | 0.629 | 0.526 | 0.931 | 126.8 |
| **PCA k=32** | **0.641** | **0.598** | **0.481** | **0.935** | 149.3 |

## Cross-Dataset Pattern

| Metric | SCORPION | Canine SCC |
|---|---|---|
| Neural scanner probe | 0.399 | 0.361 |
| Neural mean cosine | 0.879 | 0.730 |
| Best linear scanner probe | 0.724 | 0.707 |
| Best linear mean cosine | 0.881 | 0.739 |
| Best PCA scanner probe | 0.560 | 0.641 |
| Best PCA mean cosine | 0.806 | 0.598 |
| Original scanner probe | 0.865 | 0.863 |
| Original mean cosine | 0.987 | 0.919 |

## Interpretation

Linear scanner-subspace projection preserves tissue structure (cosine 0.881
SCORPION, 0.739 canine) but saturates quickly at the learned scanner-subspace
rank (k≥4), leaving substantial recoverable scanner signal (probe 0.724 SCORPION,
0.707 canine). The scanner subspace appears to capture only a fraction of the
acquisition variation that neural factorization can separate.

PCA component removal suppresses scanner further (probe 0.560 SCORPION,
0.641 canine) but causes substantial tissue damage: mean cosine drops from 0.987
to 0.806 (SCORPION) and from 0.919 to 0.598 (canine SCC). The dominant-variance
directions carry both scanner and tissue information, so removing them deletes
useful biology.

Paired-Acquisition Neural Factorization achieves scanner probe 0.399
(SCORPION) and 0.361 (canine SCC) while preserving mean cosine at 0.879 and
0.730 respectively. No simple baseline comes within 0.03 scanner probe of the
neural reference on either dataset.

## Decision

**Scanner suppression alone is insufficient; tissue-preserving factorization
remains valuable.** Across both SCORPION and external canine SCC, simple
scanner-removal baselines cannot reproduce the scanner-suppression/tissue-preservation
tradeoff achieved by Paired-Acquisition Neural Factorization.

Linear projection preserves tissue but barely suppresses scanner. PCA suppresses
scanner more but damages tissue. Neural factorization achieves substantially
stronger scanner suppression while preserving tissue structure.

The baseline objection is externally weakened.

## Claim Boundary

This is peer-review-hardening evidence only. It does not claim clinical validation,
diagnostic performance, disease biology discovery, human clinical generalization,
complete scanner invariance, or deployment readiness.

## Reproduction

```powershell
# SCORPION:
python -u experiments/baselines/run_pair_integrity_baseline_murder_test.py --datasets scorpion --folds 0 1 2 3 4

# Canine SCC:
python -u experiments/baselines/run_pair_integrity_baseline_murder_test.py --out-dir results/paired_acquisition_factorization_baseline_murder_test_caninescc --datasets caninescc --folds 0 1 2 3 4
```

## Output Files

- `results/paired_acquisition_factorization_baseline_murder_test/` — SCORPION results
- `results/paired_acquisition_factorization_baseline_murder_test_caninescc/` — Canine SCC results
- `experiments/baselines/run_pair_integrity_baseline_murder_test.py` — Runner script
