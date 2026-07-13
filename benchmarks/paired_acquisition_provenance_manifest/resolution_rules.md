# Canonical resolution rules

## Allowed archive-level statuses

Only four values are allowed.

### confirmed

All observed required fields agree with applicable family/path lineage and no
contradictory metadata is present. This status describes current-state
metadata-lineage consistency, not historical byte origin or a verified
producing invocation. Path agreement alone cannot receive high confidence.

### corrected

A metadata-lineage conflict exists and archive-specific evidence uniquely binds
the present archive to the adjudicating historical invocation. Qualifying
evidence is limited to:

- an exact historical content hash linked to a verified run manifest;
- an archive-specific run record identifying output path, fold, seed,
  condition, variant, evaluation split, model, backbone, and run ID;
- a deterministic generator/config record uniquely binding the archive to one
  invocation;
- a source commit combined with archive-specific invocation evidence; or
- exact deterministic internal proof uniquely identifying the generating
  metadata.

The observed value remains preserved and the original archive is not modified.
No current row satisfies this status.

### unresolved

An unresolved metadata-lineage conflict exists because archive-specific
adjudication evidence is absent. A `canonical_*` field may retain a proposed
canonical value supported by family-level lineage context, but
`resolution_notes` must use the phrase "proposed canonical value" and must not
present that value as adjudicated.

### legacy-optional

An archive predates an optional field, lacks that field without contradiction,
and has no metadata-lineage conflict. Optional absence is not itself a conflict
and does not assert a canonical backbone. A duplicate-content equivalence, if
present, is reported separately and does not by itself override this status.

## Confidence values

Allowed values are `high`, `medium`, `low`, and `not_applicable`.

- Current unresolved proposals and confirmations use `medium`: multiple
  lineage sources agree, but historical producing invocation and byte binding
  remain unverified.
- A weaker path-only unresolved proposal would use `low`.
- Optional-only legacy absence uses `not_applicable`.
- A corrected row may never use `not_applicable`.
- No current row uses `high`.

For an unresolved row, confidence describes support for the proposed value; it
does not turn the proposal into an adjudicated change.

## Status precedence

One row has one archive-level status. Apply this precedence:

1. A conflict lacking archive-specific adjudication evidence is `unresolved`.
2. Otherwise, a directly adjudicated canonical change is `corrected`.
3. Otherwise, optional-only absence is `legacy-optional`.
4. Otherwise, complete current metadata-lineage agreement is `confirmed`.

This matters for the 226 archives without optional explicit backbone metadata.
Two hundred also contain a source-label conflict and therefore receive
`unresolved`; only the remaining 26 have no contradiction and receive
`legacy-optional`.

## Current unresolved lineage proposals

The shared generator implementation at
`experiments/scorpion/run_pathoalign_projection.py` writes generic model and
source labels. Dataset- and backbone-specific callers use canine manifests or
explicit DINOv2, Phikon, and ResNet50 source archives, then retain those generic
helper labels.

Reachable generator/result history contains the caller, shared helper, and
experiment-level run context. These are reconstructed family-level lineage
objects: they do not bind present NPZ bytes to exact historical invocations.
They support, but do not adjudicate, these proposed canonical values:

- 200 canine source proposals: `external_multiscanner_caninescc`.
- 150 Phikon/ResNet50 model proposals: `<backbone>_pathoalign`.

All 350 rows remain unresolved. These proposals do not demonstrate that a
different dataset or backbone generated the features, make a
scientific-validity determination, or add a scientific conclusion.

## Fail-closed correction gate

A `corrected` row must have all of:

- a conflict with a changed canonical value;
- an archive-specific adjudication evidence type;
- a reference identifying the archive, exact output identity, producing
  invocation or run ID, and concrete evidence object;
- exact fold, seed, condition, variant, evaluation-split, model, and backbone
  binding, with explicit treatment of fields that are not applicable;
- programmatic verification that the referenced invocation is unique rather
  than one of multiple plausible configurations;
- a historical output binding or exact internal deterministic proof, never a
  current-state hash represented as historical evidence; and
- confidence no stronger than the evidence supports and never
  `not_applicable`.

Path inference, family membership, filename inference, reachable code, an
aggregate experiment log, code showing typical family behavior, a current-state
hash, a generic generator reference, or a family-level reference bundle is
insufficient by itself. An allowed evidence-type string or internal verification
flag cannot substitute for archive-specific evidence validation.

The builder resolves a cited commit and its generator, configuration, run-log,
and verified-manifest blobs directly through Git. It derives candidate
uniqueness and exact archive/run binding from those committed records;
caller-supplied availability or verification booleans are not evidence.
