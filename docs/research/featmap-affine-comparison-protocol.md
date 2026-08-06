# Prospective FEATMAP-style affine comparison

## Purpose

This protocol compares **Paired-Acquisition Neural Factorization** with simple
paired scanner harmonization on the exact frozen SCORPION DINOv2 inputs and
original-slide-blocked folds.

The comparison separates two questions:

1. **Harmonization:** which method most strongly reduces linearly recoverable
   scanner identity while retaining same-region geometry?
2. **Factorization:** which method creates an explicit acquisition
   representation that can be decoded, capacity-controlled, reconstructed, and
   audited separately from the tissue-oriented representation?

FEATMAP is relevant to the first question. The neural model was designed
primarily for the second. The analysis must not merge these into one
leaderboard or imply that a harmonizer and a factorization model provide the
same object.

## Frozen data and split contract

- Dataset: SCORPION.
- Biological unit: original H&E slide.
- Observations: 2,400 patches from 480 aligned tissue regions on 48 slides.
- Acquisition conditions: AT2, B300, DP200, GT450, and P1000.
- Frozen representation: 768-dimensional DINOv2-Base embedding archive already
  bound by SHA-256 in the capacity-matched campaign runner.
- Evaluation: the same five original-slide-blocked folds used by the promoted
  capacity-matched evidence.
- Fit pool: train and validation slides only.
- Test pool: the held-out slides for the current fold.
- Test rows may be projected and evaluated, but may never estimate a transform,
  select ridge regularization, choose a reference scanner, or select a method.

## Registered methods

| Method | Learned object | Role |
|---|---|---|
| Identity standardized | Fit-only global z-score | No-harmonization reference |
| Centroid translation | Source-to-reference mean shift | Minimal paired correction |
| Orthogonal Procrustes | Translation plus orthogonal rotation/reflection | Rigid geometric map |
| Affine least squares | Unregularized global affine map | Direct high-dimensional affine baseline |
| Ridge affine | Regularized global affine map | Primary FEATMAP-style baseline |
| Paired-Acquisition Neural Factorization | Biological and acquisition neural branches with joint reconstruction | Explicit factorization method |

Every scanner is used as the reference in turn. This avoids choosing the most
favourable scanner after seeing test results. The five reference-scanner
outcomes are averaged **within each original slide** before inference. They are
sensitivity conditions, not five independent samples.

## Ridge selection

For every fold and every source/reference scanner pair:

1. fit candidate maps on train-slide pairs;
2. evaluate paired embedding mean-squared error on validation-slide pairs;
3. select from
   \(\{10^{-3},10^{-2},10^{-1},1,10,100,1000\}\);
4. break exact ties toward the smaller value;
5. refit the chosen map on train and validation pairs;
6. apply the frozen map to all rows for evaluation.

The unregularized affine result remains reportable even if it performs poorly.
It is particularly important because each fold contains fewer paired fit
regions than embedding dimensions, making the unrestricted map underdetermined.

## Metrics

### Harmonization endpoints

- held-out linear scanner balanced accuracy;
- average and worst scanner-pair cosine agreement;
- average and worst bidirectional same-region top-1 retrieval.

The direct registered contrast is:

> Paired-Acquisition Neural Factorization minus ridge affine.

All affine variants are additionally compared with the identity-standardized
representation.

### Factorization endpoints

The affine methods do not have these outputs and must be marked not applicable:

- acquisition-branch scanner recoverability;
- biological leakage into the acquisition branch;
- joint reconstruction;
- acquisition bottleneck sensitivity;
- acquisition-component swapping or other latent interventions.

Absence of an output is not a loss on a harmonization leaderboard. It means the
method does not answer the factorization question.

## Inference

- Average the five neural optimization seeds within fold, method, and original
  slide before comparison.
- Average the five affine reference-scanner conditions within method and
  original slide before comparison.
- Form paired slide-level differences.
- Resample the five trained folds, then slides within each sampled fold.
- Use 100,000 deterministic bootstrap draws and report 95% intervals.
- Do not report slide-independent sign-flip p-values.
- Treat retrieval as preserved only when the interval stays above the
  preregistered \(-0.02\) noninferiority margin.

## Claim boundary

The comparison can establish which tested representation has lower linear
scanner recoverability under this protocol. It cannot establish:

- pure biological factors;
- complete scanner invariance or information-theoretic independence;
- biological preservation from cosine or retrieval alone;
- diagnostic, clinical, workflow, or patient benefit;
- general superiority outside SCORPION and the frozen DINOv2 representation;
- that conceptual overlap with FEATMAP implies copying.

If an affine method removes scanner information more aggressively, that result
must be reported directly. The remaining neural contribution must then be
positioned as an explicit, inspectable acquisition representation rather than
as superior raw scanner removal.

## Execution

```bash
python experiments/scorpion/run_paired_affine_baselines.py \
  --base-features <frozen-dinov2-archive.npz> \
  --manifests-dir <frozen-fold-manifests> \
  --out-dir <paired-affine-campaign>

python scripts/scorpion/analyze_paired_affine_baselines.py \
  --experiment-dir <paired-affine-campaign> \
  --factorization-slide-metrics \
    <capacity-campaign>/analysis/seed_averaged_slide_metrics.csv \
  --out-dir <paired-affine-analysis> \
  --bootstrap-draws 100000
```

No numerical result becomes public claim evidence until completeness,
provenance, source hashes, and output hashes are validated and promoted in a
separate forward-valid evidence package.
