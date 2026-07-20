# Paired-acquisition bottleneck × cross-covariance factorial

## Status and claim boundary

This document freezes the design for Issue #51 before any full factorial is
executed. It is an execution and analysis plan, not a result. Passing the smoke
gate authorizes the locked full run; it does not support a positive scientific
claim. The 350 unresolved historical artifacts remain outside this provenance
boundary.

## Question

Does paired-acquisition factorization improve biological/acquisition separation
across acquisition-branch bottleneck capacity, and how does that interact with
cross-covariance regularization?

## Frozen factors

The acquisition-branch bottleneck dimensions are:

```text
[2, 4, 8, 16, 32, 64]
```

The cross-covariance weights are independently crossed with every dimension:

```text
[0.0, 0.05, 0.20]
```

The weight grid is fixed across dimensions. `0.0` is the no-cross-covariance
control, `0.05` is the existing frontier default, and `0.20` is the existing
stronger-separation setting. No dimension-specific weight tuning is permitted.

## Fixed producer configuration

Every cell uses the real canine SCC DINOv2 paired-acquisition producer with:

- biological dimension `256` and hidden dimension `512`;
- scanner adversary, acquisition, dependence, gradient-reversal,
  reconstruction, variance, covariance, and temperature settings inherited
  unchanged from the existing frontier producer;
- true-pair assignments generated from one fold and seed;
- region batch size `32`, learning rate `3e-4`, and weight decay `1e-4`;
- no reuse of historical projections or checkpoints;
- one clean, stable Git commit and one environment binding per complete grid.

All 18 smoke cells must share the same source feature SHA-256, split-manifest
SHA-256, generated pair-assignment SHA-256, environment SHA-256, and producing
commit. The output directory is excluded from the semantic producer identity;
the exact shell invocation remains recorded in the run log.

## Gate 1 — complete deterministic smoke grid

Gate 1 uses fold `0`, seed `911`, and one epoch for all 18 cells:

```text
6 dimensions × 3 cross-covariance weights = 18 runs
```

A passing aggregate release must contain exactly one provenance-valid run per
cell. The gate fails closed on any missing or duplicate cell, colliding run ID,
non-finite required metric, changed hyperparameter, mixed commit/environment,
dataset or split mismatch, pair-assignment mismatch, missing checkpoint or
training history, or invalid checksum.

Run from a clean checkout containing the canonical canine feature archive and
fold manifests:

```powershell
python experiments/paired_acquisition/run_provenance_bound_factorial_smoke.py `
  --release-dir results/paired_acquisition_factorial/smoke-gate-v1 `
  --device cuda
```

Validate the resulting aggregate release again with:

```powershell
python scripts/provenance/validate_paired_acquisition_factorial_release.py `
  results/paired_acquisition_factorial/smoke-gate-v1/release_manifest.json
```

Partial cell releases are temporary and are not promoted. The aggregate output
is exposed only after all 18 cells and the final release-level artifacts pass
validation.

## Gate 2 — locked full run

The full factorial may start only after Gate 1 passes. It retains the same 18
factor combinations and expands across folds `[0, 1, 2, 3, 4]`, seeds
`[911, 912, 913, 914, 915]`, and the existing `75`-epoch training budget:

```text
18 factor cells × 5 folds × 5 seeds = 450 runs
```

Every attempted cell must be retained, including failures. A failed cell is
reported as a failure; it is not silently dropped, rerun under changed settings,
or replaced by a tuned neighboring condition.

## Frozen outcomes

The cell table must preserve the complete branch-level metrics already emitted
by the producer:

- scanner balanced accuracy and macro F1;
- biological-category balanced accuracy, macro F1, and weighted F1;
- same-category neighborhood purity at `k = 1, 5, 10`.

The locked full analysis must additionally report cross-scanner pair
consistency, same-region retrieval, biological-sample-level uncertainty, the
complete per-cell table, and an aggregate factorial analysis that separates:

1. bottleneck-capacity effects;
2. cross-covariance regularization effects;
3. their interaction.

No favorable cell may be selected post hoc as a replacement for the complete
factorial analysis.

## Stop conditions

Stop before Gate 2 if Gate 1 is incomplete, non-finite, non-reproducible, or
provenance-incomplete. Dataset access or compute shortage is a reported
blocker, not permission to reduce the grid, change the seed/fold structure, or
retune individual cells.
