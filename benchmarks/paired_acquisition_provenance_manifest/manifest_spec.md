# Provenance manifest specification

## Scope

`provenance_manifest.csv` contains exactly one row for each archive discovered
by the allowlisted archive families in `build_provenance_manifest.py`. The
current scope is 426 archives: one singleton and seventeen complete five-fold
by five-seed grids. The absent oldstyle row-level family is reported with count
zero but does not create a manifest row.

Paths are sorted lexicographically after repository-relative POSIX
normalization. CSVs use UTF-8 without a byte-order mark, an explicit stable
column order, and `\n` line endings.

## Stable identity

Both `relative_path` and `canonical_path` are repository-relative,
case-preserving, forward-slash paths. `canonical_path` is formed after resolving
the file and proving that it remains under the repository root. No absolute
machine-specific prefix is emitted.

Archive identity version 1 is:

```text
SHA-256(
  canonical_path encoded as UTF-8
  + one NUL byte
  + lowercase raw-file content_sha256 encoded as ASCII
)
```

`content_sha256` is computed over the complete raw NPZ byte stream. Each file
is hashed twice around metadata inspection. Size, modification timestamp, and
digest must remain stable across both passes.

## Column definitions

### Identity

- `archive_id`: identity version 1 digest.
- `canonical_path`: resolved, repository-contained logical path without an
  absolute prefix.
- `relative_path`: lexical repository-relative POSIX path.
- `content_sha256`: SHA-256 of raw archive bytes.
- `file_size_bytes`: raw archive size.
- `archive_family`: allowlisted family identifier.
- `dataset`: validated family-level dataset domain.
- `fold`, `seed`: exact integer path/metadata identifiers where applicable.
- `condition`, `variant`: validated observed or family-defined run labels.
- `evaluation_split`: observed evaluation split; projected archives require
  `test`.

### Observed metadata

- `observed_source`, `observed_model`, `observed_backbone`: unmodified scalar
  values from `metadata_json`.
- `observed_metadata_json_sha256`: SHA-256 of the exact decoded JSON string
  re-encoded as UTF-8, before normalization.
- `observed_metadata_keys`: compact sorted JSON array of top-level keys.
- `metadata_json_present`: lowercase CSV boolean.
- `metadata_json_record_count`: must equal one.

The builder reads only the scalar `metadata_json` NPY member. It does not load
feature arrays.

### Path/family expectations

- `expected_source_from_path`
- `expected_model_family`
- `expected_backbone_from_path`
- `expected_dataset_from_path`
- `expected_condition_from_path`
- `expected_variant_from_path`

These fields are lineage expectations. They are not scientific ground truth.

### Conflict classification

- `source_label_conflict`
- `model_backbone_label_conflict`
- `backbone_path_conflict`
- `dataset_path_conflict`
- `duplicate_path_conflict`
- `metadata_missing`
- `metadata_malformed`
- `conflict_class`
- `conflict_evidence_basis`

`model_backbone_label_conflict` is gated: an explicit metadata backbone must be
present, must match the family/path-expected backbone, and must not occur in the
observed model label. Missing optional backbone metadata is never promoted to
that conflict. Multiple classes are semicolon-delimited in fixed order.

Duplicate canonical paths, missing files, malformed required metadata, and
multiple metadata records fail before output. Identical content under distinct
paths is recorded separately, does not override the archive-level metadata
resolution by itself, and is not automatically treated as a scientific defect.

### Resolution

- `canonical_source`
- `canonical_model`
- `canonical_backbone`
- `canonical_resolution_status`
- `resolution_confidence`
- `resolution_evidence_type`
- `resolution_evidence_reference`
- `resolution_notes`

The resolution fields separate five concepts:

1. `observed_*` fields preserve archive metadata verbatim.
2. `expected_*` fields record path/family lineage expectations.
3. `canonical_*` fields may hold proposed canonical metadata on an unresolved
   row, but those values are not adjudicated merely because they are populated.
4. `canonical_resolution_status=corrected` is reserved for an archive-specific
   adjudication satisfying `resolution_rules.md`.
5. `canonical_resolution_status=unresolved` means provenance remains open;
   `resolution_notes` must explicitly identify any populated canonical field as
   a "proposed canonical value".

The status, evidence type, and notes determine whether canonical fields are
proposed or adjudicated. Nonconflicting canonical fields preserve the exact
observed value rather than being silently normalized from family/path context.
The source archive is never rewritten.

### Adjudication needs

- `evidence_needed_for_adjudication`
- `generator_config_needed`
- `run_log_needed`
- `source_commit_needed`
- `archive_hash_comparison_needed`
- `human_review_needed`

Every current row records that historical archive-hash comparison remains
needed because no producing run supplied a trusted output checksum. Every
unresolved conflict row also records the absent archive-specific run ID,
historical output hash, exact output-path binding, exact fold/seed/run-label
record, producing invocation, and verified run manifest.

## Cross-table invariant

The exact issue set keyed by `(archive_id, canonical_path, conflict_class)` must
match between the manifest flags and `provenance_conflicts.csv`. Duplicate
issues, absent conflict rows, and conflict rows without a manifest issue fail
closed.
