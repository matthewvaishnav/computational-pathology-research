# WSI-NCA PANDA data-access engineering gate

## Question

Can the tracked Windows-local, coordinate-bearing PANDA feature cohort be made portable without
silently changing slide membership, labels, features, or coordinates?

This is an engineering gate. It is not a PANDA model result and does not promote any WSI-NCA
pathology claim.

## Repository audit at `398cec01`

| Item | Observed value |
|---|---:|
| Label rows in `panda/train.csv` | 10,616 |
| Rows in `panda_phikon_manifest.csv` | 10,616 |
| HDF5 files reported by extraction summary | 10,615 |
| Manifest-valid rows with a feature path | 10,614 |
| Missing feature rows | 1 |
| Manifest-invalid compressed HDF5 rows | 1 |
| Additional known unreadables in baseline exclusion CSV | 3 |
| Final eligible transfer cohort | **10,611** |
| Recorded feature dimension | **768** |

The missing slide is `3790f55cad63053e956fb73027179707`. The manifest-invalid HDF5 is
`001d865e65ef5d2579c190a0e0350d8f`. The three additional known unreadables are
`0032bfa835ce0f43a92ae0bbab6871cb`, `003a91841da04a5a31f808fb5c21538a`, and
`004391d48d58b18156f811087cd38abf`.

The HDF5 contract used by the current trainer is:

- `features`: rank-2 array with shape `(num_patches, 768)`;
- `coordinates`: rank-2 array with shape `(num_patches, 2)`;
- matching first dimensions and at least two patches;
- optional `slide_id` attribute, which must match the manifest ID when present.

Labels come from `isup_grade` in `panda/train.csv`, joined into the tracked manifest by
`image_id`. The tracked feature paths use the Windows-local convention
`D:\panda\features_phikon\<image_id>.h5`.

`panda/splits.json` contains an older 1,365-slide split (955 train, 204 validation, 206 test).
The current WSI-NCA Phase A trainer does not consume it: it performs a deterministic stratified
train/validation split from the supplied manifest for each seed. The transfer tool only selects
and packages slides; it does not define the eventual scientific split.

## Data-access observation

No `.h5`/`.hdf5` file or PANDA feature archive was present in the execution workspace. Exact
filename/type searches of connected Library storage and searches of connected Drive storage for
PANDA, `panda_phikon`, HDF5, coordinate features, and feature archives also returned no usable
cohort. Therefore no genuine PANDA training run was executed in this gate.

## Implemented portability layer

`experiments/wsi_nca/prepare_panda_transfer_bundle.py` provides `create` and `validate`
subcommands. Creation:

- remaps the tracked Windows path root with `--source-root`;
- removes manifest-invalid/missing rows and the explicit known-unreadable CSV;
- supports deterministic proportional ISUP-grade selection with `--limit`, `--stratified`, and
  `--seed`;
- fully reads every selected `features` and `coordinates` array;
- checks HDF5 shapes against the recorded manifest metadata and `slide_id` when present;
- aborts the entire atomic build on any selected missing, unreadable, or misaligned file;
- copies each HDF5 byte-for-byte and verifies source/destination SHA256 equality;
- writes only relative `features/<image_id>.h5` paths in `manifest.csv`;
- records every included ID and every excluded ID with a reason in `bundle_summary.json`;
- writes `checksums.sha256` for the manifest, summary, and every HDF5 file.

The validator checks the exact file inventory, every SHA256, the manifest/summary agreement, all
HDF5 arrays, uniform feature dimensionality, patch totals, and optional source-manifest hash.
The PANDA trainer and frozen-coordinate-manifest preparer now resolve relative feature paths
against the input manifest directory.

For the requested 300-slide proportional cohort, the frozen selection counts are grades
0/1/2/3/4/5 = **82/75/38/35/35/35**.

## Exact local creation command

Run this one PowerShell command from the repository root on the Windows machine that contains the
HDF5 files:

```powershell
python .\experiments\wsi_nca\prepare_panda_transfer_bundle.py create --manifest .\results\panda_manifest\panda_phikon_manifest.csv --exclude-csv .\results\panda_attention_mil_baseline\unreadable_features.csv --source-root "D:\panda\features_phikon" --out-dir "D:\panda\transfer\panda_wsi_nca_300" --limit 300 --stratified --seed 42
```

After transfer, validate the directory before training:

```powershell
python .\experiments\wsi_nca\prepare_panda_transfer_bundle.py validate --bundle-dir "D:\panda\transfer\panda_wsi_nca_300" --source-manifest .\results\panda_manifest\panda_phikon_manifest.csv --report "D:\panda\transfer\panda_wsi_nca_300.validation.json"
```

## Observed validation

The transfer utility was exercised end to end on generated coordinate-bearing HDF5 fixtures.
Deterministic stratification, byte-identical copying, relative-path loading, manifest hash checks,
full-array validation, and trainer path resolution passed. Negative tests confirmed that a missing
`coordinates` dataset aborts creation and post-copy byte corruption fails validation.

The full 10,611-slide or 300-slide real bundle was not created because the source bytes were not
available in this environment. No real PANDA metric is reported.

## Claim boundary and next gate

Supported: the tracked metadata resolve to a 10,611-slide portable candidate cohort, and the new
tooling fails closed while preserving selected HDF5 bytes exactly on tested fixtures.

Not supported: successful transfer of the real cohort, a real-data training run, optimization
adequacy, WSI-NCA benefit on PANDA, clinical value, or any biological dynamics claim.

The next promotion gate is to create and validate the 300-slide bundle, then run an explicitly
engineering-only matched T0/T1/T4-tied/T4-untied end-to-end smoke before freezing the scientific
optimization protocol.
