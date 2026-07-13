# Required Metadata for Residual Provenance Auditing

## Purpose

This schema separates three questions that are otherwise easy to conflate:

1. Can existing embeddings be aligned to paired observations?
2. Is residual structure unexplained by recorded tissue/category labels?
3. Is that structure associated with a measured non-scanner provenance variable?

Only the first two are partly answerable from current artifacts. The third is blocked by missing provenance metadata and insufficient crossing.

## Required row hierarchy

Every observation must map to a documented hierarchy:

`specimen_id -> block_id -> section_id -> slide_id -> region_id -> scanner_observation_id`

Identifiers may be unavailable at some levels, but aliases must be declared. In the current canine artifacts, `slide_id` behaves as the sample identifier and is not an independent preparation label. In SCORPION, `sample_number` repeats across slides and is not globally unique.

## Minimum computational schema

| Field | Level | Required for | Current canine | Current SCORPION |
|---|---|---|---|---|
| `dataset_id` | dataset | lineage and joins | inferable; should be explicit | inferable; should be explicit |
| `observation_id` | scanner observation | uniqueness | derivable from composite key | derivable from composite key |
| `scanner_id` | scanner observation | scanner gate and matching | available | available |
| `sample_id` or documented equivalent | biological unit | split and sample association | available | `slide_id` available |
| `region_id` | tissue region | exact-region exclusion and bundles | available | available |
| `category_id` and label definition | region | category control | available, seven categories | absent |
| `fold` and `split` | observation/group | leakage-safe evaluation | available | available |
| `representation_id` | archive/vector | candidate identity | path-derived; manifest needed | path-derived; manifest needed |
| `feature_row_key` | observation/vector | one-to-one join | composite join available | composite join available |

The join key must include scanner observation identity. `region_id` alone is invalid because each region has five scanner views.

## Representation provenance manifest

Each fixed representation family needs one machine-readable record with:

- dataset and dataset version;
- backbone and backbone revision;
- representation family and branch;
- feature vector key and dimensionality;
- fold and seed;
- training/fit split IDs;
- projection or removal method;
- configuration identifier;
- producing code path and commit;
- source feature archive;
- row-key definition;
- archive checksum; and
- known deviations or lineage warnings.

This is a confirmatory gate, not bookkeeping. Several local canine archives contain `metadata_json.source` text naming SCORPION, and some cross-backbone paths retain DINOv2 model/source strings. Path-derived dataset or backbone labels must therefore remain `manual_review` until a corrected manifest is frozen.

## Provenance metadata needed for attribution

At least one explicit upstream variable is required. Candidate fields include:

| Field family | Examples | Why it matters |
|---|---|---|
| Collection/site | `site_id`, `laboratory_id`, `collection_protocol_id` | Tests scanner-shared institutional or collection structure |
| Tissue processing | `fixation_protocol_id`, `processing_batch_id`, `block_id` | Separates pre-scanner preparation from scanner acquisition |
| Section/slide preparation | `section_id`, `section_thickness`, `microtome_id`, `slide_prep_batch_id` | Identifies variation introduced before scanning |
| Staining | `stain_protocol_id`, `stain_batch_id`, `reagent_lot_id`, `stain_date_bin` | Tests scanner-constant stain/process factors |
| Handling | `operator_id`, `coverslip_batch_id`, `mounting_medium_id` | Tests preparation/handling structure |
| Acquisition chronology | `scan_session_id`, `scan_date_bin` | Separates scanner hardware from session effects |

Free-text filenames or directory paths are not substitutes for defined provenance variables.

## Crossed-design requirements

Metadata presence alone is insufficient. A provenance factor is analyzable only when:

- each positive `provenance x category` cell contains multiple independent biological units and the category contains an eligible different-provenance negative level;
- each target category includes at least two provenance levels where the comparison is claimed;
- scanner and provenance are not one-to-one;
- sample or slide and provenance are not one-to-one for cross-sample attribution;
- folds do not isolate a provenance level entirely unless the planned test is explicitly level-held-out;
- missingness is not determined by scanner, category, or outcome; and
- all scanner views of one region remain in the same split and permutation block.

Two independent units per level is only a bare algebraic minimum and is not adequate confirmatory support. A confirmatory design needs a pre-specified power/precision analysis and enough independent units per `provenance x category` cell to leave units out while retaining positive and negative matches. No universal numeric threshold is asserted here.

The preferred design crosses biology and preparation directly: repeated or serial material from the same specimen processed under different preparation conditions, plus multiple independent specimens within each preparation condition. Paired scanners alone do not create this crossing.

## Current canine held-out support

The geometry-qualified canine anchor contains:

- 4,025 observations;
- 805 regions, each with five scanner views;
- 44 samples;
- seven recorded categories; and
- five sample-disjoint folds.

Eligibility is evaluated inside each held-out fold at two distinct levels. All counts below include only anchor-capable sample-category cells and their regions. Rows used only as matched negatives are excluded from every cell, region, and observation support count.

### Candidate-discovery eligibility

A sample-category cell is candidate-discovery eligible when it contains at least two distinct regions and the same fold/category contains a different sample with at least one region for matched negatives.

| Held-out fold | Anchor cells | Anchor-positive regions | Anchor-positive observations | Non-estimable in this fold |
|---:|---:|---:|---:|---|
| 0 | 25 | 141 | 705 | none |
| 1 | 26 | 183 | 915 | none |
| 2 | 21 | 130 | 650 | Bone |
| 3 | 20 | 148 | 740 | none |
| 4 | 20 | 157 | 785 | Cartilage |
| **Across-fold support total** | **112** | **759** | **3,795** | **Globally absent: Cartilage** |

### Replicated-anchor confirmatory eligibility

A fold/category is replicated-anchor confirmatory eligible only when at least two independent samples are anchor-capable, meaning that each sample-category cell contains at least two distinct regions.

| Held-out fold | Anchor cells | Anchor-positive regions | Anchor-positive observations | Non-estimable in this fold |
|---:|---:|---:|---:|---|
| 0 | 25 | 141 | 705 | none |
| 1 | 26 | 183 | 915 | none |
| 2 | 21 | 130 | 650 | Bone |
| 3 | 20 | 148 | 740 | none |
| 4 | 19 | 155 | 775 | Bone, Cartilage |
| **Across-fold support total** | **111** | **757** | **3,785** | **Globally absent: Cartilage** |

Fold-4 Bone is candidate-discovery eligible but confirmatory-ineligible because only one Bone sample is anchor-capable. Bone remains confirmatory-eligible in other folds and is not globally absent. `Globally absent` means that a category has no eligible anchor cell in any held-out fold; it is not the union of fold-specific exclusions.

These are algebraic support counts only; they do not clear lineage, leakage, power, or precision gates. Per-scanner-direction support must still be checked before metric execution.

## Current technical proxies

Canine metadata include:

- annotation and region rank;
- bounding-box area and dimensions;
- source-image dimensions;
- adaptive crop side;
- inside-image and padding fractions;
- region-level maximum padding;
- orientation normalization;
- affine/registration summaries;
- file inventory and size; and
- annotation-history fields.

These are useful controls, not ground-truth non-biological labels. Crop size and geometry may encode morphology. Orientation is largely scanner-specific. File size mixes content, dimensions, and encoding. Annotation order may reflect spatial or category ordering.

## Metadata absent from current anchors

Neither current dataset provides explicit, validated fields for:

- site or laboratory;
- fixation or processing batch;
- tissue block and section lineage;
- staining protocol, batch, or reagent lot;
- preparation operator or preparation date;
- coverslip or mounting batch; or
- a crossed preparation-by-specimen design.

SCORPION also lacks category/tissue labels, so it cannot test whether residual structure is unexplained by recorded category.

## Quality and missingness gates

Before any metric run:

1. Assert unique composite row keys.
2. Assert exact archive-to-manifest key coverage.
3. Assert category is constant across scanner views of a region.
4. Assert sample and all scanner views of a region do not cross splits.
5. Record category-by-fold-by-sample support and exclude non-estimable strata before seeing metric values.
6. Record provenance missingness by scanner, category, sample, and fold.
7. Reject one-to-one aliases masquerading as independent provenance labels.
8. Freeze a representation manifest and checksum list.

## Interpretation ceiling by metadata level

| Available metadata | Maximum interpretation |
|---|---|
| Scanner + region only | Scanner-shared region structure |
| Scanner + sample + region | Scanner-shared sample/region structure |
| Category added | Structure unexplained by the recorded coarse category |
| Geometry/QC proxies added | Association with measured technical proxies |
| Valid crossed site/preparation labels added | Evidence consistent with measured non-scanner provenance correlates; operational scanner invariance additionally requires G2 |

No metadata tier alone proves a causal artifact source.
