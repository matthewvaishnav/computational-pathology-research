# Paired-acquisition evidence archive

This directory contains the historical evidence archive and separately
versioned forward-valid corrected evidence for Paired-Acquisition Neural
Factorization.

## Corrected evidence

`corrected-20260726/` is the current corrected-evidence release for the
fixed-estimand canine SCC audit and fold-aware SCORPION inference. It contains
only validated summaries, design and environment records, hashes, and release
metadata. Raw and large artifacts remain external and are bound by SHA-256.

The corrected release does not overwrite or reactivate historical evidence.
Its claim boundary is subordinate to `CLAIM_BOUNDARY.md`.

## Historical archive

The files at this directory level index the text evidence consolidated from the
15 historical commits used by the claim ledger.

## Scope

- 14 final experiment runners under `experiments/paired_acquisition/`.
- 104 designs, CSV/JSON metrics, reports, and run logs under their canonical
  `results/paired_acquisition_factorization_*/` paths.
- 15 full source commit SHAs in `claim_source_manifest.csv`.
- 118 artifact-level source Git blob IDs and SHA-256 checksums in
  `artifact_manifest.csv`.

Artifacts are referenced byte-for-byte from their recorded source Git blobs.
The source and consolidated Git blob IDs are required to match for every row.

## Collision rule

The only duplicate path across the source commits was
`experiments/paired_acquisition/run_pair_structure_boundary_test.py`.
The later cross-backbone source commit
`d018c924757c90a56ab5d515c4ecc02110286df6` is the consolidated version.
The earlier commit remains recorded as a claim source, and its result directory
is preserved.

## Exclusions

This archive intentionally excludes raw images, WSI files, NPZ feature
archives, checkpoints, and other large local artifacts. It does not resolve the
350 metadata-lineage conflicts or the absence of historical output-hash binding
reported by `benchmarks/paired_acquisition_provenance_manifest/`.

## Claim boundary

This is an evidence-preservation and repository-consolidation change. It does
not rerun experiments, alter metrics, create a new scientific result, establish
clinical validity, or prove complete biological/acquisition disentanglement.

## Retirement gate

Evidence branches may be retired only after:

1. all 15 source commits and 118 artifact rows validate;
2. every consolidated Git blob matches its source Git blob;
3. every SHA-256 checksum is present and unique paths are enforced; and
4. the claim-ledger and figure/table source paths resolve on the consolidated
   branch.
