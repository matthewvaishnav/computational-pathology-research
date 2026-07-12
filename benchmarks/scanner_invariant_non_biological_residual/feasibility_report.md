# Scanner-Invariant Residual Provenance Feasibility Report

## Verdict

**PARTIAL: candidate residual-structure detection is feasible; non-biological attribution is not.**

Existing artifacts support a no-training, category-conditioned residual-structure audit for canine scanner-suppressed embeddings after lineage review. They do not establish scanner invariance or non-biological origin. SCORPION supports identity controls but lacks category labels. Site- and preparation-level attribution remains blocked.

## Execution boundary

- No new representation training was run.
- No probe, classifier, residualizer, projection, or metric was fit.
- No oldstyle representation was reconstructed.
- No existing data, result, checkpoint, or experiment output was modified.
- This report records artifact and metadata feasibility only.

## Scientific premise

Paired scanner acquisitions vary scanner while holding both tissue biology and pre-scanner slide/preparation factors fixed. Scanner agreement can therefore preserve both. Paired data can support detection of scanner-shared residual structure, but cannot identify its biological versus preparation origin without independent provenance variation.

The general invariance blind spot is not itself a new result. The defensible future novelty would be a literature-positioned paired-scanner provenance audit with crossed preparation data and explicit identifiability gates. The current sample-link path is an exploratory gate.

## Dataset capability

| Dataset | Rows | Regions | Samples/slides | Scanners | Categories | Category-adjusted audit | Provenance attribution |
|---|---:|---:|---:|---:|---:|---|---|
| Canine SCC | 4025 | 805 | 44 | 5 | 7 | available | blocked |
| SCORPION | 2400 | 480 | 48 | 5 | 0 | blocked | blocked |

Canine composite keys are unique and its geometry-qualified manifest contains rows for 805 regions on 5 scanners. The proposed different-region, same-category sample audit has 112 held-out-fold-eligible sample-category cells, 759 eligible regions, and 3795 eligible scanner observations across 6 categories. Excluded category: Cartilage.

| Held-out fold | Eligible cells | Eligible regions | Eligible observations | Non-estimable categories |
|---:|---:|---:|---:|---|
| 0 | 25 | 141 | 705 | none |
| 1 | 26 | 183 | 915 | none |
| 2 | 21 | 130 | 650 | Bone |
| 3 | 20 | 148 | 740 | none |
| 4 | 20 | 157 | 785 | Cartilage |

## Representation artifact inventory

| Family | Archives | Expected | Checked | Integrity/status | Representative join |
|---|---:|---:|---:|---|---|
| `canine_original_dinov2` | 1 | 1 | 1 | available | 4025/4025 |
| `canine_true_pair_biological` | 25 | 25 | 25 | manual_review | 4025/4025 |
| `canine_acq_dim8_biological` | 25 | 25 | 25 | manual_review | 4025/4025 |
| `canine_acq_dim16_biological` | 25 | 25 | 25 | manual_review | 4025/4025 |
| `canine_shuffled_region_control` | 25 | 25 | 25 | manual_review | 4025/4025 |
| `canine_shuffled_sample_control` | 25 | 25 | 25 | manual_review | 4025/4025 |
| `canine_same_category_different_sample_control` | 25 | 25 | 25 | manual_review | 4025/4025 |
| `canine_scanner_balanced_random_control` | 25 | 25 | 25 | manual_review | 4025/4025 |
| `canine_fully_random_control` | 25 | 25 | 25 | manual_review | 4025/4025 |
| `scorpion_true_pair_biological` | 25 | 25 | 25 | available | 2400/2400 |
| `scorpion_phikon_true_pair_biological` | 25 | 25 | 25 | manual_review | 2400/2400 |
| `scorpion_resnet50_true_pair_biological` | 25 | 25 | 25 | manual_review | 2400/2400 |
| `scorpion_dinov2_acq_dim8_biological` | 25 | 25 | 25 | available | 2400/2400 |
| `scorpion_dinov2_acq_dim16_biological` | 25 | 25 | 25 | available | 2400/2400 |
| `scorpion_phikon_acq_dim8_biological` | 25 | 25 | 25 | manual_review | 2400/2400 |
| `scorpion_phikon_acq_dim16_biological` | 25 | 25 | 25 | manual_review | 2400/2400 |
| `scorpion_resnet50_acq_dim8_biological` | 25 | 25 | 25 | manual_review | 2400/2400 |
| `scorpion_resnet50_acq_dim16_biological` | 25 | 25 | 25 | manual_review | 2400/2400 |
| `oldstyle_keep_k4_row_level` | 0 | n/a | 0 | blocked | not available |

All 17 expected five-fold by five-seed archive grids are complete. The audit checked all 426 discovered archives; 0 failed schema, alignment, fold, split, or training-metadata checks.

Lineage review remains required: 200 archives have an internal dataset/source conflict and 150 have an internal backbone/model conflict. All 200 source conflicts occur in the audited canine projected archives.

The primary canine archive is `manual_review` for confirmatory use because its internal source string is inconsistent with its path, rows, fold, seed, and split evidence:

- `metadata_json.source`: `SCORPION DINOv2 projection experiment`

A frozen representation manifest and checksums are required before metric execution.

SCORPION has DINOv2, Phikon, and ResNet50 true-pair and bottleneck archives for cross-backbone sensitivity. It still cannot support category-adjusted testing because category labels are absent, and Phikon/ResNet50 internal model strings require lineage correction.

## Scanner premise

- Five-scanner chance: 0.200.
- Original frozen scanner probe: 0.866.
- True-pair biological scanner probe: 0.361.
- Acquisition-dim-8 biological scanner probe: 0.369.
- Acquisition-dim-16 biological scanner probe: 0.359.
- Oldstyle `keep_k4` scanner probe: 0.200 (summary evidence only).

The neural biological branches are scanner-suppressed, not established scanner-invariant. The only current chance-level summary candidate is oldstyle `keep_k4`, but no row-level oldstyle embedding archive is present. The strict scanner-invariant residual metric is therefore blocked in this read-only phase.

## Candidate-metric feasibility

| Metric | Status | Interpretation ceiling |
|---|---|---|
| `M1_canine_cross_scanner_sample_link_auc` | partial | coarse_category_adjusted_sample_structure_only |
| `M1_oldstyle_cross_scanner_sample_link_auc` | blocked | scanner_removal_residual_structure_only |
| `M1_scorpion_category_adjusted_sample_link_auc` | blocked | unadjusted_slide_region_structure_only |
| `M2_cross_sample_site_preparation_link_auc` | blocked | non_biological_attribution_blocked |
| `M7_canine_technical_proxy_association` | partial | measured_proxy_association_in_scanner_suppressed_representation_only |

## Metadata boundary

- Site/laboratory fields: none found in allowlisted fields.
- Preparation/batch/stain fields: none found in allowlisted fields.
- Technical proxy fields available: adaptive_crop_side_level0, annotation_id, area, bbox_area_pixels, bbox_height, bbox_width, bbox_x, bbox_y, image_height, image_width, inside_image_fraction, lastmodified.
- Successfully read allowlisted metadata files: 9; unreadable/empty files: 0.

Sample, slide, and region IDs are biological identities as well as possible process carriers. Geometry, crop, padding, registration, orientation, file-size, and annotation fields are proxy controls and may reflect tissue or scanner. None is a validated preparation label.

SCORPION can support cross-scanner slide/region and cross-backbone controls, but it cannot establish that structure is unexplained by category because category labels are absent.

| Technical proxy source | Level | Status | Join coverage |
|---|---|---|---:|
| `geometry_crop_qc` | scanner_observation_and_region | available | 4025/4025 |
| `registration_affine` | sample_by_scanner | available | 4025/4025 |
| `tiff_metadata` | sample_by_scanner_file | available | 4025/4025 |
| `tiff_file_size` | sample_by_scanner_file | available | 4025/4025 |
| `annotation_history_geometry_delta` | scanner_observation_region_annotation | available | 4025/4025 |

All listed proxy sources currently join completely, but M7 remains partial until each proxy has a pre-declared level, independent unit, missingness rule, aliasing audit, and estimable contrast. A generic region/sample block is not valid for every proxy level.

## Existing metric pitfall

Existing same-sample top-1 retrieval near 1.0 is not evidence for sample-level residual artifact because another scanner view of the exact same region was eligible as the nearest neighbor. The proposed primary metric must use a different region from the same sample, exact category matching, a different scanner, and a same-target-scanner negative from another sample.

## Blockers before metric execution

1. Resolve archive lineage conflicts and freeze a representation manifest.
2. Keep neural candidates labeled scanner-suppressed unless they pass a stronger operational gate.
3. Materialize oldstyle row-level output only under a separately authorized derived-artifact step if the strict removal audit is desired.
4. Pre-register per-fold/category/scanner-direction eligibility and rare-stratum exclusions.
5. Implement exact-region exclusion and atomic region-bundle permutations.

## Blockers before non-biological attribution

1. Add explicit site/preparation/processing/stain metadata with definitions and lineage.
2. Demonstrate that provenance levels repeat across independent biological units.
3. Demonstrate that provenance is not aliased with sample, scanner, category, or fold.
4. Test provenance across different samples or blocks so fine-grained biology cannot trivially supply the match.

## Claim boundary

A later positive canine sample-link result could support: same-category cross-scanner sample association is detectable in fixed scanner-suppressed embeddings. It could not establish that the structure is non-biological. A technical-proxy association would remain association-only. A valid crossed site/preparation result could support association with a measured non-scanner provenance variable; use `operationally scanner-invariant` only if the separate G2 gate also passes.

This feasibility audit supports no clinical, diagnostic, deployment, patient-care, scanner-bias-solved, universal disentanglement, or causal artifact claim.

## Bottom line

Do not train a new representation. First fix representation provenance and acquire or recover crossed preparation/site metadata. If the immediate goal is only candidate discovery, the next authorized artifact should be a no-training implementation of matched cross-scanner, different-region, same-category sample-link AUC on existing canine test embeddings with the listed controls.

## Deterministic evidence fingerprint

`990d8e7c7a869e6a5fcaf17a85da97427fedd715438890bf7562f8dab126e290`

This fingerprint covers the audit's deterministic schema/count/status payload. It is not a checksum of the full feature payloads.
