# Paired-acquisition provenance validation report

## Scope and status

Status: **PASS** for deterministic inventory and validation in the present local artifact workspace; all metadata-lineage conflicts remain unresolved.

This is a read-only metadata-lineage and evidence-reconciliation audit. It does not run training, rewrite source archives, modify experiment outputs, or change scientific claims.

## Inventory

- Total archives: 426
- Total raw bytes: 1511442726
- Archives with metadata_json: 426
- Archives without optional explicit backbone metadata: 226
- Malformed metadata count: 0
- Duplicate canonical path count: 0
- Duplicate content group count: 0
- Unique content hashes: 426

### Family counts

| Archive family | Count |
|---|---:|
| `canine_original_dinov2` | 1 |
| `canine_true_pair_biological` | 25 |
| `canine_acq_dim8_biological` | 25 |
| `canine_acq_dim16_biological` | 25 |
| `canine_shuffled_region_control` | 25 |
| `canine_shuffled_sample_control` | 25 |
| `canine_same_category_different_sample_control` | 25 |
| `canine_scanner_balanced_random_control` | 25 |
| `canine_fully_random_control` | 25 |
| `scorpion_true_pair_biological` | 25 |
| `scorpion_phikon_true_pair_biological` | 25 |
| `scorpion_resnet50_true_pair_biological` | 25 |
| `scorpion_dinov2_acq_dim8_biological` | 25 |
| `scorpion_dinov2_acq_dim16_biological` | 25 |
| `scorpion_phikon_acq_dim8_biological` | 25 |
| `scorpion_phikon_acq_dim16_biological` | 25 |
| `scorpion_resnet50_acq_dim8_biological` | 25 |
| `scorpion_resnet50_acq_dim16_biological` | 25 |
| `oldstyle_keep_k4_row_level` | 0 |

### Dataset counts

| Dataset | Count |
|---|---:|
| `canine_scc` | 201 |
| `scorpion` | 225 |

## Conflict counts

- Source-label conflicts: 200
- Explicit gated backbone/model-label conflicts: 150
- Backbone/path conflicts: 0
- Dataset/path conflicts: 0
- Conflict-set overlap: 0
- Optional-backbone fallback conflicts: 0

These are metadata-lineage findings. They do not establish that a different dataset or backbone generated the features and do not determine scientific validity.

## Resolution counts

- confirmed: 50
- corrected: 0
- unresolved: 350
- legacy-optional: 26

The 350 conflict rows remain unresolved. Their conflicting canonical field carries a medium-confidence proposed value derived from path/family expectations, internal metadata, and compatible group-level run evidence; it is not an adjudicated correction.
The 226 optional-backbone absences are counted separately: 200 occur on rows with an unresolved source-label conflict, while 26 are optional-only and therefore receive the archive-level legacy-optional status.
The 50 confirmed rows have complete current observed-metadata agreement with applicable lineage expectations; confirmed does not assert historical byte origin.

## Evidence availability

- Archives with structured embedded configuration: 425
- Archives with group/family evidence references: 425
- Archives with associated family-level run logs: 425
- Archives with family-level source/result commit associations: 425
- Conflict archives with compatible per-run text records: 275
- Conflict archives with aggregate-only text records: 75
- Corrected rows with archive-specific adjudication evidence: 0
- Conflict archives lacking adjudication evidence: 350
- Unresolved rows with a proposed canonical conflict value: 350
- Archives with current-state content hashes: 426
- Archives with historical cryptographic output binding: 0

None of the 426 source NPZ paths is Git-tracked. Their public availability is not established by this package. Deterministic checking requires a workspace where the same archives and referenced Git objects are separately available.

## Duplicate-content assessment

No duplicate-content groups were found.

## Deterministic fingerprints

- Archive inventory fingerprint: `e68c1cf536b2ef8a843aea11f80547f05fb42891e1c2f7e15fbbd8472be1e7f5`
- provenance_manifest.csv SHA-256: `7c2207fd2192976aaa4ba72b3d8e6d03844d317b4b3c9a6ec3498bc1628ab78e`
- provenance_conflicts.csv SHA-256: `8a51f35836cd9eb16b227ba6debc5a772098fd97d9d15f924fa6b23a45e52a1e`

## Limitations

1. Path and family expectations are lineage-derived proposals, not adjudicated canonical values.
2. Family-level code and run records do not uniquely bind present bytes to historical invocations.
3. Present hashes fingerprint current local files only.
4. Public availability of the 426 source archives is not established.
5. Metadata reconciliation does not modify metrics, source archives, or prior scientific claims.
6. Conflict findings do not establish scientific invalidity, causal provenance, clinical relevance, or an incorrect historical model or dataset.
7. Crossed-preparation and crossed-site attribution remains future work.
