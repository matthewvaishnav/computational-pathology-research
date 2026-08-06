# Fixed-Estimand Real Feature-Space Adjudication v2

**Branch:** `research/real-bottleneck-artifact-recovery-and-adjudication-v2-20260804`
**Runner:** `experiments/paired_acquisition/run_fixed_estimand_real_feature_space_adjudication_v2.py`
**Tests:** `tests/test_fixed_estimand_real_feature_space_adjudication_v2.py`

## Purpose

The original fixed-estimand adjudication stopped correctly at
`fixed_estimand_adjudication_not_ready` because the frozen real paired-scanner
run persisted no per-cell B32/B64 representations. Phase B recovered exactly
those 50 canine cells by exact deterministic replay and archived
`projected_features.npz` plus `checkpoint.pt` per cell. This runner is the
**versioned, no-training adjudication** that consumes the recovered artifacts and
completes the previously blocked fixed-estimand audit.

The v2 result is a new forward-valid artifact. The prior
`fixed_estimand_adjudication_not_ready` result is neither overwritten nor
reinterpreted.

## No-training execution

The v2 runner imports and reuses the original adjudication implementation:

- accepts an explicit `--recovery-manifest`;
- requires recovery status `complete_exact_real_bottleneck_representation_recovery`;
- verifies all 50 projected-feature and checkpoint hashes;
- initializes zero optimizers, executes zero backward passes, and trains zero
  models;
- loads the recovered arrays only (never retrains);
- reproduces the frozen seven-category neural metrics from the loaded arrays
  before fixed-estimand scoring (a mismatch fails closed);
- uses the exact corrected five-category implementation;
- retains the seven-category endpoint as exploratory.

## Fixed estimand

Reused unchanged from the corrected fixed-estimand implementation:

- retained: `Dermis`, `Epidermis`, `Inflamm/Necrosis`, `SCC`, `Subcutis`;
- excluded: `Bone`, `Cartilage`;
- at least two fit and two held-out biological samples per category per fold;
- vocabulary from metadata only; probe fitting uses training rows and labels
  only; same-region and same-sample NN exclusions match the corrected audit.

## Evaluations

- Neural B32/B64 biological branches: corrected five-category balanced accuracy,
  macro F1, per-category recall, fit-pool purity, scanner metrics with paired
  permutation null, and cross-scanner retrieval over the identical candidate
  pool. Model seeds are averaged within fold before cross-fold inference, and
  the per-seed distribution is retained.
- Acquisition branches and combined biological+acquisition representations are
  reported but not placed on the biological Pareto frontier.
- Simple scanner-removal baselines (original frozen features, centroid/QR,
  paired linear, PCA removal, scanner-balanced random control) are recomputed
  with the identical candidate pool and probe family.

## Adjudication

- Fold-level and aggregate Pareto fronts (canine three-axis: lower scanner BA,
  higher corrected five-category BA, higher worst-pair retrieval; SCORPION
  two-axis scanner-retrieval frontier).
- Fixed 0.02 dominance margins; weak, strictly material, and cross-fold material
  dominance (material in at least four of five folds, no larger-than-margin
  reversal). No weighted composite score.
- The eight required canine fold-aware contrasts (all available once the neural
  cells are present), with fold effects, mean/median/min/max, positive-fold
  counts, and fold-then-unit bootstrap intervals.
- `neural_feature_space_increment_supported`: a neural family improves corrected
  category BA by at least 0.02 over every simple scanner-removal baseline in at
  least four folds without increasing scanner BA by more than 0.02 or reducing
  worst-pair retrieval by more than 0.02.
- `simple_baseline_pareto_dominance_supported`: a simple baseline materially
  dominates both B32 and B64 under the canine three-axis frontier in at least
  four folds and is non-inferior on the SCORPION scanner-retrieval frontier.
- Synthetic-to-real transport requires corrected five-category neural category
  gain; retrieval gain alone is never counted as biological accessibility.
- Layer-2 and pixel-space work remain prohibited; the Layer-2 gap schema is a
  future data-remediation specification only.

## Statuses

The existing scientific status vocabulary is used:

- `complete_simple_baseline_pareto_dominance_supported`
- `complete_neural_feature_space_increment_supported`
- `complete_mixed_fixed_estimand_real_feature_space_evidence`
- `complete_no_neural_feature_space_increment_supported`
- `fixed_estimand_adjudication_not_ready`
- `fixed_estimand_adjudication_failed`

Poor scientific results receive a scientific status; execution/integrity
failures receive the failure status.

## Outputs

Written atomically into a single new output directory:

- `fixed_estimand_real_feature_space_adjudication_v2_result.json`
- `fixed_estimand_real_feature_space_adjudication_v2_summary.csv`
- `fixed_estimand_real_feature_space_adjudication_v2_manifest.json`
- `fixed_estimand_layer2_missing_metadata_schema_v2.json`

An existing output directory is never overwritten. Frozen evidence and the
prior adjudication result are never modified.
