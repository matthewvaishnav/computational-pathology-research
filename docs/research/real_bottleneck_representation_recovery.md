# Real Bottleneck Representation Recovery

**Branch:** `research/real-bottleneck-artifact-recovery-and-adjudication-v2-20260804`
**Runner:** `experiments/paired_acquisition/run_real_bottleneck_representation_recovery.py`
**Tests:** `tests/test_real_bottleneck_representation_recovery.py`

## Why this exists

The frozen real paired-scanner run
(`complete_mixed_real_paired_scanner_allocation_effects`) stored metrics but no
per-cell projected representations or checkpoints. The no-training fixed-estimand
adjudication therefore correctly stopped at
`fixed_estimand_adjudication_not_ready` with all 50 neural cells unrecoverable.

This runner performs an **exact deterministic replay** of those 50 cells solely
to recover and archive the missing immutable representations and checkpoints. It
is provenance remediation and completion of a blocked audit, not a new
architecture experiment.

## Exact replay requirement

The replay is not a new experiment. It uses the exact frozen training
implementation from commit
`e95d8526958ac781748f92b4ebb617b75a52fce0` — the same implementation that
produced the frozen result — together with the exact frozen feature arrays,
metadata, folds, preprocessing, architecture definitions, parameter counts,
losses, optimizer (AdamW), learning rate (`3e-4`), weight decay (`1e-4`), epochs
(`75`), checkpoint-selection policy (minimum validation objective only),
deterministic settings, seeds, and probe configurations.

Before any fit, the runner verifies that the working training implementation is
source-equivalent to the frozen commit for the three modules that participate in
training:

- `run_real_paired_scanner_bottleneck_allocation_validation.py`
- `run_crossed_target_scanner_prototype_factorization.py`
- `run_synthetic_crossed_factor_identifiability.py`

## Replay scope

Exactly the canine primary neural grid:

- dataset: canine SCC;
- backbone: exact frozen `dinov2_base`;
- folds: `0, 1, 2, 3, 4`;
- model seeds: `2201, 2202, 2203, 2204, 2205`;
- families: `real_b32_reference` and `real_b64_parameter_matched`;
- total: `5 x 5 x 2 = 50` fits on CUDA.

Not replayed: SCORPION, broken-pair controls, routed consensus, synthetic
models, alternative widths, alternative hidden sizes, new seeds, and any pixel
or WSI model.

## Replay acceptance

For every cell the replayed metrics are compared against the frozen per-run
record. All frozen fields are required to replicate:

- biological scanner linear and nonlinear probe metrics;
- acquisition scanner probe metrics;
- seven-category exploratory category metrics;
- acquisition-category leakage;
- overall and worst-pair retrieval;
- same-region and different-region similarity and margin;
- spectral accessibility diagnostics;
- reconstruction losses and training history checkpoints;
- operational flags;
- parameter counts;
- selected checkpoint epoch.

The strict deterministic numeric tolerance is fixed before execution at
`1e-6` (absolute). A genuine replay on the same device and implementation must
agree far below that bound; the tolerance is never widened after a mismatch.
Missing frozen fields are reported as unavailable, never fabricated. Aggregate
agreement alone is insufficient — every one of the 50 cells must replicate.

If any required cell fails deterministic replication, partial outputs are
preserved, the status is `real_bottleneck_representation_recovery_failed`, and
Phase C is not run.

## Persisted per-cell artifacts

For each accepted cell:

```
canine_scc/fold_<fold>/<family>/seed_<seed>/projected_features.npz
canine_scc/fold_<fold>/<family>/seed_<seed>/checkpoint.pt
```

`projected_features.npz` contains biological, acquisition, and combined
representations plus immutable row index, region/slide/scanner identifiers,
category metadata, train/validation/test membership, fold, family, seed, and the
feature-input / row-order / region-order / slide-order / scanner-order /
category-order SHA-256 values. Category metadata is stored only after training
and never enters factorizer optimization.

`checkpoint.pt` contains the exact model state, architecture configuration,
optimizer-independent inference configuration (including the training-fit
standard scaler), selected epoch, source commit, input hashes, and
family/fold/seed identity.

## Recovery outputs

Written atomically:

- `real_bottleneck_representation_recovery_result.json`
- `real_bottleneck_representation_recovery_summary.csv`
- `real_bottleneck_representation_recovery_manifest.json`

Statuses:

- `complete_exact_real_bottleneck_representation_recovery`
- `real_bottleneck_representation_recovery_failed`

No scientific success status is assigned; this is an artifact-recovery status.
