# Real paired-acquisition producer provenance adoption

## Scope

`experiments/paired_acquisition/run_provenance_bound_bottleneck_cell.py`
adapts one existing real producer—the canine SCC acquisition-bottleneck
separation frontier—to the forward-valid provenance contract introduced by
Issue #50.

This is an execution primitive, not a new scientific result. It runs exactly
one bottleneck-dimension × cross-covariance cell, disables reuse of historical
projections, and refuses to publish a release unless:

- the Git checkout is clean and remains on one commit throughout execution;
- the canonical 4,025 × 768 canine DINOv2 archive and requested fold manifest
  are present;
- training emits both biological and acquisition branches;
- every required metric is finite;
- projected features and the checkpoint both exist;
- the completed release passes the fail-closed validator.

## Bound inputs and outputs

Each successful release copies and hashes:

- the source DINOv2 feature archive;
- the fold-specific patch/split manifest;
- the exact configuration payload;
- Python, platform, NumPy, pandas, scikit-learn, PyTorch, CUDA, and device
  metadata;
- the producer command, seed, and exact 40-character Git commit;
- projected biological/acquisition features;
- the model checkpoint;
- branch-level metrics and the producer run log.

The release writer derives one immutable `parun-v1-*` identifier, writes all
required component documents with that identifier, constructs the release
manifest, validates all SHA-256 bindings, and only then moves the release into
its requested final path.

## First real smoke execution

From a clean checkout containing the canonical canine inputs:

```powershell
python experiments/paired_acquisition/run_provenance_bound_bottleneck_cell.py `
  --release-dir results/paired_acquisition_provenance_release/real-smoke-dim8-xcov005-fold0-seed911 `
  --acquisition-dim 8 `
  --cross-covariance-weight 0.05 `
  --fold 0 `
  --seed 911 `
  --epochs 1 `
  --device cuda
```

The producer validates the release before returning. It can also be checked
again explicitly:

```powershell
python scripts/provenance/validate_paired_acquisition_release.py `
  results/paired_acquisition_provenance_release/real-smoke-dim8-xcov005-fold0-seed911/release_manifest.json
```

## Issue boundaries

Issue #50 remains open until a real execution is completed, its release
manifest is retained with the artifacts, validation passes from the clean
producing checkout, and the release tag/checksum boundary is published.

Issue #51 must not call the legacy sweep directly for its locked factorial.
Its smoke gate should invoke this one-cell primitive for every preregistered
cell, combine the validated run records into one complete release, verify the
expected cell count and unique run identifiers, and stop before the full run
if any cell is missing, non-finite, or provenance-incomplete.

The 350 unresolved historical artifacts remain outside this release boundary.
Nothing in this adoption path reconstructs or upgrades their lineage.
