---
license: other
pretty_name: PANDA Phikon WSI Spatial Features
language:
  - en
tags:
  - computational-pathology
  - prostate-cancer
  - whole-slide-imaging
  - feature-extraction
  - non-commercial
  - private-transfer
---

# PANDA Phikon WSI Spatial Features

Status: **private release scaffold; not populated**.

This repository is intended to transport a frozen 300-slide cohort of genuine
coordinate-preserving PANDA Phikon HDF5 bags for non-commercial remote research.
No fixture or synthetic file may be represented as PANDA data.

## Data origin

The source cohort is the 10,616-slide PANDA development set from Radboud
University Medical Center and Karolinska Institutet. Labels come from PANDA
`train.csv`. Each intended derived bag was produced from a source WSI with the
repository extractor and `owkin/phikon`.

Original study: W. Bulten et al., *Artificial intelligence for diagnosis and
Gleason grading of prostate cancer: the PANDA challenge*, Nature Medicine 28,
154–163 (2022), <https://doi.org/10.1038/s41591-021-01620-2>.

## Unit of observation

One HDF5 file per prostate-biopsy WSI. Each file contains a variable-length bag
of patch representations and the matching level-0 WSI coordinate for every
representation row.

## Schema

| Field | Type / shape | Meaning |
|---|---|---|
| `features` | float, `(N, 768)` | Frozen Phikon CLS representations. |
| `coordinates` | integer/float, `(N, 2)` | `(x, y)` patch origins in the level-0 WSI coordinate frame. |
| HDF5 `slide_id` | string attribute | Must equal the manifest `image_id` when present. |
| `manifest.csv:image_id` | string | PANDA slide identifier. |
| `manifest.csv:isup_grade` | integer 0–5 | Source slide-level ISUP grade. |
| `manifest.csv:feature_path` | relative path | `features/<image_id>.h5`; absolute machine paths are prohibited. |

`features` and `coordinates` must have identical first dimensions and at least
two rows. Non-finite or misaligned arrays fail validation.

## Sample counts

- Source label/manifest rows: **10,616**.
- Eligible coordinate-feature bags after tracked exclusions: **10,611**.
- Frozen private transfer cohort: **300** slides.
- Intended ISUP 0/1/2/3/4/5 counts: **82 / 75 / 38 / 35 / 35 / 35**.
- Feature dimension: **768**.
- Maximum intended extracted patches per source slide: **600**.

These are intended and audited counts. The repository is not populated until
remote file inventory and hashes verify all 300 real bags.

## Preprocessing

The tracked extractor at the WSI-NCA source commit uses:

- OpenSlide pyramid level 1;
- 224 × 224 pixel patches with stride 224 at that level;
- a 2048 × 2048 thumbnail tissue mask with mean intensity between 20 and 220;
- minimum tissue-mask fraction 0.1;
- up to 600 accepted patches per slide;
- resize to 224 × 224, tensor conversion, and ImageNet mean/std normalization;
- frozen `owkin/phikon` loaded as a ViT without a pooling layer;
- first-token/CLS output as the 768-dimensional representation.

The repository configuration records the extractor revision as `main`; an exact
immutable Phikon Hub revision was not persisted. This is an explicit provenance
limitation and a blocker to public release.

## Coordinates

Coordinates are patch origins converted back to the OpenSlide level-0 frame.
They preserve feature/topology correspondence for spatial graph construction.
They are not annotations, cell locations, tumor outlines, or a common physical
coordinate system across slides.

## Exclusions

The transfer tool removes one missing feature row, one manifest-invalid compressed
HDF5 row, and three additional known unreadable bags. It then performs
deterministic proportional selection with seed 42. Every selected HDF5 is fully
read before atomic bundle completion. Raw WSIs, masks, trained aggregators,
predictions, and fixture data are excluded.

## Checksums

The completed local bundle must include SHA256 for `manifest.csv`,
`bundle_summary.json`, and all 300 HDF5 files. Source and copied HDF5 bytes must
match. After upload, every remote file must be downloaded at the immutable HF
revision and rehashed before the release registry may move from `prepared` to
`private`.

Exact manifest and HDF5 hashes are intentionally absent from this scaffold
because the real bundle is not available in Work.

## Licensing

This dataset uses `license: other` because two non-commercial sources apply and
the final derived-feature redistribution basis still requires explicit review:

- The PANDA paper states that the development set is available for
  non-commercial research under a Creative Commons BY-SA-NC 4.0 formulation and
  requires citation of the source paper.
- `owkin/phikon` is distributed under the Owkin non-commercial license, which
  places non-commercial conditions on use and sharing of results.

Private visibility does not erase source-license duties. Access must remain
limited to permitted non-commercial research, with attribution and any
share-alike obligations preserved. This card does not grant rights beyond the
source licenses.

## Intended use

- checksum-validated remote execution of WSI-NCA engineering and scientific
  experiments;
- spatial MIL/GNN research using fixed patch features and coordinates;
- reproducible non-commercial analysis of the frozen 300-slide cohort.

## Limitations

- This is a non-commercial derived representation cohort, not raw pathology.
- Slide-level ISUP labels are weak supervision and may contain grading noise.
- At most 600 patches are retained and the thumbnail tissue heuristic may omit
  relevant tissue.
- Exact Phikon Hub revision is unrecorded.
- The cohort is not an official PANDA split and is not external validation.
- Private remote storage is not clinical governance, consent, security, or
  regulatory validation.

## Provenance

- GitHub repository:
  <https://github.com/matthewvaishnav/computational-pathology-research>
- WSI-NCA source branch: `research/wsi-nca-phase-a-20260807`
- Source commit: `cb48cfda8c47307c54b97273d69c87004a1d3108`
- Pull request: <https://github.com/matthewvaishnav/computational-pathology-research/pull/84>
- Source manifest: `results/panda_manifest/panda_phikon_manifest.csv`
- Exclusion list: `results/panda_attention_mil_baseline/unreadable_features.csv`
- Transfer tool: `experiments/wsi_nca/prepare_panda_transfer_bundle.py`
- Extractor model identifier: `owkin/phikon@main` (exact revision unrecorded)

The completed card must add the exact source-manifest SHA256, bundle-manifest
SHA256, HDF5 inventory hash, extraction environment, creation timestamp, and
immutable HF revision.

## Citation

Users must cite the PANDA source paper above and the Phikon paper/model as
required by their licenses. Cite this release only after a populated immutable
revision exists.

## Claim boundary

This release, if populated, would establish only a validated portable byte set
of coordinate-bearing frozen representations. It would not establish WSI-NCA
performance, spatial benefit, recurrent benefit, tissue-topology causality,
pathology utility, scanner invariance, diagnostic validity, or clinical value.
