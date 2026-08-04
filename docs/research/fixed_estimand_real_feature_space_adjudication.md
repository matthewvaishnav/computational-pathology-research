# Fixed-Estimand Real Feature-Space Adjudication

**Branch:** `research/fixed-estimand-real-feature-space-adjudication-20260804`
**Base:** `research/real-paired-scanner-bottleneck-allocation-validation-20260803`
**Runner:** `experiments/paired_acquisition/run_fixed_estimand_real_feature_space_adjudication.py`
**Tests:** `tests/test_fixed_estimand_real_feature_space_adjudication.py`
**Status vocabulary:** `fixed_estimand_adjudication_not_ready`,
`fixed_estimand_adjudication_failed`, and the four scientific statuses.

## Purpose

This is a **no-training, fixed-estimand, fold-compatible final adjudication** of
the real paired-scanner feature-space evidence produced by the frozen
`complete_mixed_real_paired_scanner_allocation_effects` result. It re-asks the
B64-versus-B32 questions under the repository's authoritative corrected
fixed-five-category estimand rather than the exploratory seven-category endpoint.

The adjudication never trains: no optimizer is constructed, no backward pass is
executed, no factorizer or feature encoder is trained, no synthetic data is
generated, and no WSI or pixel model is built. Only frozen result artifacts,
immutable saved arrays, and deterministic simple-baseline implementations are
used. If the required neural representations cannot be recovered from immutable
saved arrays or frozen checkpoints, the runner **fails closed** with
`fixed_estimand_adjudication_not_ready` and enumerates the exact missing
artifacts instead of retraining.

## Frozen inputs

The runner verifies the complete inherited chain before any evaluation:

| Artifact | File SHA-256 | Internal SHA-256 |
| --- | --- | --- |
| `real_paired_scanner_bottleneck_allocation_validation_result.json` | `7293ddcd…57762` | `fbb8be9a…960f7` |
| `real_paired_scanner_bottleneck_allocation_readiness.json` | `31360bbd…c86ed` | `b092a70a…b13e` |
| `real_paired_scanner_bottleneck_allocation_validation_manifest.json` | `9c61429e…d136037` | `bf305048…6a0066` |
| copied result (Documents/Codex path) | `7293ddcd…57762` | n/a |

All 225 immutable input artifacts registered in the frozen result are re-hashed
before and after execution. The frozen synthetic factorial result and its 11
inherited artifacts are verified through the existing
`verify_synthetic_factorial` implementation.

## Fixed estimand

The corrected fixed-five-category estimand is **reused**, not reimplemented:

- Retained: `Dermis`, `Epidermis`, `Inflamm/Necrosis`, `SCC`, `Subcutis`.
- Excluded: `Bone`, `Cartilage`.
- Support: at least two fit and two held-out biological samples per category per
  fold.
- Vocabulary derived from fold-manifest metadata only.
- Probe fitting uses training rows and training labels only; preprocessing uses
  training data only.
- Held-out categories absent from a training split receive zero recall rather
  than being silently removed.
- Same-region and same-sample nearest-neighbour exclusions match the corrected
  audit.
- The frozen seven-category endpoint is retained separately as
  `exploratory_seven_category_endpoint`.

## Methods

Adjudicated on the biological Pareto frontier (same canine DINOv2 array and
folds):

1. original frozen features;
2. centroid/QR scanner-subspace projection;
3. paired linear scanner transform;
4. PCA scanner-component removal;
5. neural B32 biological branch;
6. neural B64 biological branch.

Reported but **not** placed on the biological Pareto frontier: neural B32/B64
acquisition branches, combined biological-and-acquisition branches, broken-pair
neural controls, and the scanner-balanced random control.

## Neural representation recovery

The expected neural grid is `5 folds x 5 seeds x 2 families = 50` cells:

- dataset `canine_scc`;
- folds `0..4`;
- model seeds `2201..2205`;
- `real_b32_reference` and `real_b64_parameter_matched`.

For every cell the runner records dataset, fold, seed, family, artifact or
representation SHA-256, row/region/slide/scanner/category order hashes, exact
parameter count, and zero additional training. Saved representation arrays are
preferred over checkpoint inference. If a cell cannot be recovered from an
immutable saved array or a frozen checkpoint, the runner does **not** retrain; it
records the cell as missing, enumerates the exact missing artifacts
(`projected_features.npz` biological + acquisition arrays and `checkpoint.pt`),
and returns `fixed_estimand_adjudication_not_ready`.

The frozen real paired-scanner runner persisted no per-cell projected features or
checkpoints, so in the current repository state all 50 neural cells are
unrecoverable. Deterministic simple-baseline evidence is still computed and
reported in full.

## Evaluations

- **Corrected category**: balanced accuracy, macro F1, per-category recall,
  fit-pool category purity at k=1/5/10 with same-region and same-sample
  exclusions, worst-fold BA, scanner-stratified BA, slide-averaged BA.
- **Scanner**: linear balanced accuracy and macro F1, paired permutation null
  (region-block-preserving), per-scanner recall, fold-level values; nonlinear
  probe reported only when a calibrated probe exists in the corrected evidence
  family.
- **Retrieval**: overall top-1/top-5, mean reciprocal rank, every ordered
  scanner-pair top-1, worst ordered pair top-1, same/different-region cosine
  similarity, and margin over the identical fixed held-out candidate pool for
  every method.

## Adjudication outputs

- Fold-level and aggregate Pareto fronts (canine three-axis: lower scanner BA,
  higher corrected five-category BA, higher worst-pair retrieval; SCORPION
  two-axis scanner-retrieval frontier).
- Fixed 0.02 dominance margins; weak, strictly material, and cross-fold material
  dominance (material in at least four of five folds with no larger-than-margin
  reversal). No weighted composite score is created.
- The eight required canine fold-aware contrasts plus SCORPION contrasts
  (category axes excluded), with fold effects, mean/median/min/max,
  positive-fold counts, and deterministic fold-then-unit bootstrap intervals.
- Pair-supervision attribution separated from neural-architecture attribution.
- Synthetic-to-real transport decision requiring corrected five-category neural
  category gain (retrieval gain alone is never transport).
- Layer-2 missing-metadata schema as a future data-remediation specification
  only; swap assignments are never inferred from filenames, row order, or
  scanner labels.
- Claim adjudication table with `supported`, `unsupported`, `unresolved`, and
  `prohibited by evidence scope` verdicts.
- Dataset conclusions and a single top-level status.

## Claim boundaries

- True-pair superiority over broken pairs supports paired supervision, not
  neural superiority.
- Retrieval improvement does not establish biological-label accessibility.
- Increased scanner recoverability can explain part of a retrieval gain.
- An explicit acquisition branch is structurally different from a projection
  baseline; structural difference is not empirical superiority.
- Fixed-feature evidence does not establish pixel behavior.
- Canine tissue categories are descriptive labels, not clinical endpoints.
- SCORPION cannot support a category-accessibility claim without validated
  labels.

## Outputs

Written atomically into a single new timestamped output directory:

- `fixed_estimand_real_feature_space_adjudication_result.json`
- `fixed_estimand_real_feature_space_adjudication_summary.csv`
- `fixed_estimand_real_feature_space_adjudication_manifest.json`
- `fixed_estimand_layer2_missing_metadata_schema.json`

An existing output directory is never overwritten. Frozen evidence is never
modified, and no public claim or manuscript file is updated automatically.
