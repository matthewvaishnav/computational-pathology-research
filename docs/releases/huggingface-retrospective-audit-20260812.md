# Hugging Face retrospective release audit — 2026-08-12

## Audit basis

This audit binds release decisions to the current program hub at
`edf8b2b96fbdb8b21fbecc03b8a19ac0351e1dce`, the WSI-NCA branch at
`cb48cfda8c47307c54b97273d69c87004a1d3108`, open PR #84, open migration
issues #87–#89, the root claim boundary, the foundations manuscript claim
ledger, and the tracked machine-readable evidence packages.

Hugging Face is a curated distribution layer. GitHub remains authoritative for
source, tests, experiments, CI, negative results, repairs, and claim history.

## Authentication and existing remote state

The official `huggingface_hub` 1.27.0 client reported no active credential.
Repository creation, upload, visibility mutation, card update, Collection
creation, and remote checksum verification were therefore not attempted. No Hub
release is reported as created or populated by this audit.

## Decisions

| Research line / artifact | Decision | Evidence and artifact finding | Release consequence |
|---|---|---|---|
| PA-NF corrected + SCORPION capacity-matched evidence | **RELEASE NOW** | Both tracked evidence families are promoted and provenance-complete. The SCORPION campaign has 175/175 valid registered fits; the corrected canine evidence preserves the distinct bounded negative result. No raw images, external feature arrays, or checkpoints are required in the compact bundle. | Prepare one public HF **dataset** repository, not a model repository: `MatthewVaishnav/paired-acquisition-factorization-evidence`. It remains `prepared` until authenticated upload and remote checksum verification. |
| PA-NF trained factorizer / frozen representations | **DEFER** | No checkpoint is tracked in the hub or three study repositories. The evidence manifests explicitly identify external feature/checkpoint artifacts, but their bytes and a redistributable release license are unavailable here. | Do not imply that the evidence dataset is a trained model. |
| PANDA Phikon coordinate-bearing 300-slide cohort | **PREPARE PRIVATE** | PR #84 verifies 10,616 manifest rows, 10,611 eligible bags, 768 dimensions, an HDF5 `features` + `coordinates` contract, deterministic grade allocation 82/75/38/35/35/35, and fail-closed transfer tooling. The real HDF5 files are absent. | Prepare `MatthewVaishnav/panda-phikon-wsi-spatial-features` as private and blocked. Do not populate it with fixtures or mark it complete. |
| WSI-NCA Phase A | **DEFER** | PR #84 contains bounded synthetic depth-required mechanism evidence. It has no real PANDA result and no trained pathology checkpoint. | No public HF model. Retain card/release infrastructure only. |
| TransnnMIL repaired canonical path | **DEFER** | Repair code exists, but matched repaired reruns and a checkpoint with exact data/config/hash provenance are not present. | No model repository. |
| PathologyFL / FAIR-WEIGHTS-H | **DO NOT RELEASE** | The reusable object is implementation infrastructure already belonging on GitHub. No trained model or standalone dataset exists; smoke/integration execution is not multicenter clinical evidence. | Do not manufacture an HF model object. |
| Historical TransnnMIL fusion/topology outputs, debug checkpoints, fixtures, CI artifacts, smoke outputs | **DO NOT RELEASE** | Withdrawn, intermediate, duplicated, or non-scientific artifacts. | Preserve only in Git history/evidence ledgers as applicable. |

## PA-NF evidence release boundary

The release card must keep both promoted statements intact:

1. On the registered SCORPION structured-separation objective, the full model
   has a controlled comparative advantage over the equal-capacity two-branch
   neural control: tissue-branch scanner balanced accuracy is reduced by
   `0.3108` with fold-aware 95% interval `[-0.3346, -0.2858]`, registered
   same-region retrieval noninferiority is preserved, and acquisition-branch
   scanner information remains strong.
2. Under the corrected canine fixed five-category estimand, no additional neural
   feature-space improvement over the strongest simple scanner-removal baselines
   was established.

These are different comparators and endpoints. Neither permits a global claim of
universal superiority or global lack of advantage.

## PANDA and Phikon licensing/provenance finding

The PANDA source paper states that the 10,616-slide development set is available
for non-commercial research under a Creative Commons BY-SA-NC 4.0 formulation,
with attribution by citing the paper:
<https://www.nature.com/articles/s41591-021-01620-2#data-availability>.

The exact configured extractor is `owkin/phikon`, whose model card and linked
license identify the Owkin non-commercial license. That license expressly
defines outputs as “Results” and permits sharing them only for non-commercial
purposes under its conditions:
<https://huggingface.co/owkin/phikon> and
<https://github.com/owkin/HistoSSLscaling/blob/main/LICENSE.txt>.

The extraction source records `revision: main`, not an immutable Phikon Hub
commit. Consequently, the private dataset card identifies the model as
`owkin/phikon@main (exact revision unrecorded)` rather than inventing a version.
Public redistribution remains blocked until the derived-feature license terms,
attribution/share-alike obligations, and exact extractor revision are explicitly
resolved. The private transfer repository may be used for the immediate
non-commercial remote-execution workflow only after the genuine byte-validated
bundle is available.

## Artifact inventory findings

- Hub and three study repositories contain no `.pt`, `.pth`, `.ckpt`, or
  `.safetensors` checkpoint suitable for release.
- `paired-acquisition-neural-factorization` does not yet exist; issue #89 tracks
  the history-preserving reusable-core extraction.
- SCORPION, canine SCC, and allocation study repositories already exist on
  GitHub and should not be blindly mirrored.
- PR #84 remains draft and synthetic/engineering-only; it does not change the
  promoted PA-NF evidence record.
- The PANDA source bytes remain outside Work under the Windows-local feature
  root recorded by the repository.

## Collection manifest

Create the public Collection only after the first remotely verified public
release exists.

```yaml
title: Computational Pathology Research — Matthew Vaishnav
namespace: MatthewVaishnav
visibility: public
ordered_items:
  - type: dataset
    id: MatthewVaishnav/paired-acquisition-factorization-evidence
    add_when: remotely populated and checksum-verified
  - type: dataset
    id: MatthewVaishnav/panda-phikon-wsi-spatial-features
    add_when: public redistribution is resolved and visibility is intentionally changed
  - type: model
    id: null
    note: Add WSI-NCA or TransnnMIL only after a provenance-complete pathology checkpoint crosses its evidence gate.
```

Deferred and private objects must not be added merely to enlarge the Collection.

## Resume gate

The single external authentication action is:

```bash
hf auth login
```

Use a write-capable token for the `MatthewVaishnav` account. Do not place the
token in a command argument, source file, log, release folder, or Git history.
