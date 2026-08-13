# PA-NF trained-model release audit — 2026-08-12

## Decision

The complete release object is the 25-checkpoint registered `pathoalign_dep20`
family (5 folds × 5 seeds), not a post-hoc selected singleton. The original Work
audit found zero checkpoint files and zero of the five required fold
standardizers under `/workspace` or `/tmp`; that was an execution-environment
limitation rather than an absence of the trained artifacts.

On 2026-08-13, the Windows source machine successfully built the complete bundle
from the authoritative artifact index. The bundle builder verified all 25
checkpoint files and all five fold standardizers and returned `verified: true`.
No checkpoint was selected, retrained, substituted, or fabricated.

The PA-NF model release is licensed under Apache-2.0. This license applies to the
released PA-NF checkpoints and the researcher-authored release-specific
source/inference code distributed in the model package. Raw SCORPION images and
the frozen DINOv2 feature archive are not redistributed by the model release and
retain their own upstream terms.

## Checkpoint audit

The authoritative
`evidence/paired_acquisition/scorpion-capacity-matched-20260726/campaign/cell_artifact_index.csv`
contains exactly 25 valid `pathoalign_dep20` checkpoint rows with the expected
fold/seed grid. Their total expected size is 155,202,450 bytes. The complete
per-candidate Work-environment audit is in
`panf-model-checkpoint-audit-20260812.csv`.

The Windows transfer-bundle build subsequently validated the real bytes against
the same promoted index and authenticated cell manifests. Each checkpoint was
required to pass indexed size/SHA256 verification, `torch.load`, stored method,
seed, epoch, strict-determinism and config checks, exact state-dict key/shape
validation, finite-tensor validation, and the registered parameter count.

Every supplied checkpoint must contain this exact config:

```json
{
  "input_dim": 768,
  "biological_dim": 256,
  "acquisition_dim": 64,
  "hidden_dim": 512,
  "temperature": 0.1,
  "reconstruction_weight": 1.0,
  "variance_weight": 1.0,
  "covariance_weight": 0.01,
  "scanner_adversary_weight": 0.5,
  "scanner_acquisition_weight": 0.5,
  "scanner_dependence_weight": 20.0,
  "cross_covariance_weight": 0.05,
  "gradient_reversal_strength": 1.0
}
```

It must also store `method=pathoalign`, its indexed seed, `epochs=75`, and
`strict_determinism=true`. The state dict must have the exact 24-key architecture
defined by `src/models/scorpion_pathoalign.py`, finite tensors, and 1,550,026
parameters.

## Required preprocessing

Each checkpoint consumes raw 768-D frozen `facebook/dinov2-base` features, not
images and not already-standardized features. The required transform is
`(features - mean) / std` using
`fold_context/fold_X_standardization.npz`, fit on all non-test slides in that
registered fold.

The promoted artifact index authenticates each external `cell_manifest.json`.
Each cell manifest, in turn, records the fold standardization SHA256. The transfer
utility therefore verifies the cell-manifest hash first, requires one consistent
standardization hash across the five seeds in a fold, and then verifies the
corresponding `.npz` bytes, arrays, shapes, dtypes, finiteness, and copied bytes.
The Windows build verified all five required standardization files.

## Source-byte reconciliation

Four hashes embedded in `campaign/campaign_design.json` are hashes of the Windows
CRLF worktree bytes. The corresponding hashes in `release_manifest.json` are the
LF-normalized Git blob bytes. Converting the exact files at training commit
`0adea50f1ef22865969109f1834a3c175e3f8b43` from LF to CRLF reproduces all four
campaign hashes exactly. This line-ending difference does not identify different
source content. The transfer utility accepts either authenticated line-ending
form for the exact model definition and rejects any other hash.

## Package boundary

The intended Hub model repository is
`MatthewVaishnav/paired-acquisition-neural-factorization`. A verified bundle
contains:

- all 25 co-equal fixed-final-epoch `pathoalign_dep20` checkpoints;
- five fold-specific standardization files;
- 25 authenticated source cell manifests and the filtered artifact index;
- the exact model definition and checksum-aware inference helper;
- `model-manifest.json`, model card, Apache-2.0 license file, and
  `checksums.sha256`.

It excludes raw SCORPION images, the frozen DINOv2 archive, projected feature
arrays, metrics/training-history debris, and all six non-release variants.

The existing evidence dataset remains unchanged and separate: model bytes belong
in the model repository; registered analyses, metrics, manifests, and the canine
negative comparison remain in
`MatthewVaishnav/paired-acquisition-factorization-evidence`.

## Current release state

The artifact and preprocessing gates are cleared. Public publication has not yet
occurred. The remaining operational steps are to rebuild the bundle from the
latest branch state so it contains the Apache-2.0 card/license metadata, rerun
local and release-folder verification, publish from an authenticated Hugging
Face environment with the explicit public-release flag, redownload the remote
repository, verify all checksums, and record the immutable Hub revision in the
release registry.
