# Paired-Acquisition Neural Factorization Identity Audit CLI v0

## Purpose

`pathoalign_identity_audit.py` is the first runnable version of the Paired-Acquisition Neural Factorization Oncology Identity Benchmark. It audits frozen pathology representations for the core question:

> Has the representation learned disease biology, or has it learned institutional and acquisition identity?

The script does not train Paired-Acquisition Neural Factorization. It evaluates existing feature tables against the benchmark contract and the measurement validation protocol.

## Command

```bash
python scripts/pathoalign_identity_audit.py \
  --features features.csv \
  --metadata metadata.csv \
  --out results/identity_audit
```

Recommended blocked run:

```bash
python scripts/pathoalign_identity_audit.py \
  --features features.csv \
  --metadata metadata.csv \
  --out results/identity_audit \
  --block-column sample_id
```

Federated or client-identity run:

```bash
python scripts/pathoalign_identity_audit.py \
  --features features.csv \
  --metadata metadata.csv \
  --out results/identity_audit_client \
  --shortcut-columns client_id,site_id,scanner_id \
  --biology-columns sample_id,task_label \
  --block-column client_id
```

## Inputs

The audit requires two CSV files.

### Feature CSV

The feature table should contain one row per representation unit and numeric feature columns.

Example:

| unit_id | f0 | f1 | f2 | ... |
|---|---:|---:|---:|---:|
| u0001 | 0.12 | -0.20 | 1.31 | ... |
| u0002 | 0.17 | -0.14 | 1.22 | ... |

The script auto-detects an identifier column from common names such as `unit_id`, `feature_id`, `patch_id`, `region_id`, `sample_id`, `slide_id`, or `id`.

When feature columns are not auto-detected, pass them explicitly:

```bash
--feature-columns f0,f1,f2,f3
```

### Metadata CSV

The metadata table should contain the same identifier column or have the same row order as the feature table.

Recommended metadata columns:

| Column | Role |
|---|---|
| `unit_id` | Join key linking features to metadata. |
| `sample_id` | Biological blocking unit. |
| `patient_id` | Stronger biological blocking unit when available. |
| `region_id` | Same-region retrieval identity when paired regions exist. |
| `scanner_id` | Acquisition shortcut identity. |
| `site_id` | Institution shortcut identity. |
| `stain_id` | Stain/acquisition shortcut identity. |
| `client_id` | Federated-client shortcut identity. |
| `biology_label` | Optional biological class or morphology label. |
| `task_label` | Optional WSI or task label. |

## Outputs

The command writes:

```text
identity_audit_summary.json
identity_audit_summary.csv
shortcut_probe_predictions.csv
```

### `identity_audit_summary.json`

Full structured audit output containing input metadata, detected columns, collapse checks, shortcut probe results, biology retrieval results, cross-acquisition consistency results, and the benchmark validity rule.

### `identity_audit_summary.csv`

Flattened table for quick inspection and paper/report inclusion.

### `shortcut_probe_predictions.csv`

Cross-validated predictions from shortcut probes, one row per evaluated unit and shortcut column.

## Measurements

### Shortcut probes

For each shortcut column, the script trains a cross-validated probe on frozen representations.

Default shortcut columns include:

- `scanner_id`
- `site_id`
- `hospital_id`
- `stain_id`
- `client_id`
- `lab_id`
- `cohort_id`
- `dataset_id`
- `annotation_source`
- `annotator_id`

Lower probe accuracy in a biological representation is better only when biology preservation and collapse checks remain valid.

### Random-label probe control

For every shortcut probe, the script also runs a permuted-label control. The random-label probe should approach chance. If it does not, the split or probe may be leaking information.

### Biology retrieval

For biological identity columns, the script computes nearest-neighbor same-label retrieval.

Default biology columns include:

- `region_id`
- `sample_id`
- `patient_id`
- `case_id`
- `tissue_id`
- `biology_label`
- `disease_label`
- `tumor_label`
- `task_label`
- `slide_label`

Higher retrieval is better. A shortcut probe decrease is not valid evidence if biological retrieval collapses.

### Cross-acquisition consistency

When both biology columns and shortcut columns are present, the script measures cosine similarity among same-biological-identity units across different acquisition identities.

Example: same `region_id`, different `scanner_id`.

Higher cross-acquisition cosine indicates that matched biological units are closer across acquisition changes.

### Collapse checks

The script reports:

- total variance
- mean per-dimension variance
- zero-variance dimension fraction
- effective rank
- numerical rank
- top singular value fraction
- mean and standard deviation of L2 norms

A representation with low shortcut probe accuracy but degenerate variance or rank is not valid evidence for biological learning.

## Split and blocking behavior

When a block column is available, the script uses blocked cross-validation and blocked bootstrap confidence intervals when possible.

Recommended block columns:

| Setting | Recommended block column |
|---|---|
| paired-region study | `sample_id` or `region_id` depending on claim |
| patient-level task | `patient_id` |
| sample-level task | `sample_id` |
| federated study | `client_id` |
| external site study | `site_id` |

If no block column is provided, the script tries `sample_id`, `patient_id`, `case_id`, `client_id`, then `site_id`.

## Validity rule

A result is credible only when:

1. shortcut identity decreases,
2. biology remains recoverable,
3. task utility does not collapse when task utility is evaluated,
4. evaluation is blocked by the relevant biological or client unit,
5. random-label and collapse controls behave correctly.

The audit is a measurement tool, not a clinical validation tool.

## Current limitation

Version 0 audits frozen features and metadata. It does not yet train Paired-Acquisition Neural Factorization, run TransnnMIL, run federated training, or perform prospective clinical validation.
