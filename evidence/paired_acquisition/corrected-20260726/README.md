# Corrected paired-acquisition evidence — 2026-07-26

This release promotes corrected, bounded evidence from the fixed-estimand
external canine SCC audit and the fold-aware SCORPION analysis. It binds the
analysis source tree, commands, input and output hashes, fold designs,
configuration, validation environment, and authoritative claim boundary.

## Included

- Canine five-fold seed-averaged metrics, descriptive summary, fixed-category
  support, and experiment design.
- SCORPION fold-aware contrasts and analysis design.
- A claim-boundary snapshot, environment snapshot, and machine-readable release
  manifest.

Raw metrics, feature archives, projected features, and slide-level contrast
records remain outside Git. Their canonical repository-relative paths, sizes,
row counts where applicable, and SHA-256 hashes are recorded in
`release_manifest.json`.

Tracked text artifacts use canonical LF line endings. The release manifest
separately binds the untouched source-output bytes, so this publication
normalization does not replace or rewrite the external result directories.

## Evidence boundary

The corrected evidence supports partial structured separation under the tested
conditions. It does not support pure biological factors, causal identification
beyond the paired design, information-theoretic independence, complete
disentanglement, complete scanner invariance, diagnostic or clinical
improvement, patient benefit, or deployment readiness.

Historical canine category metrics and slide-independent SCORPION sign-flip
p-values remain withdrawn and preserved. This release does not overwrite,
delete, rename, or promote those historical artifacts.

## Validation

From the repository root:

```bash
python scripts/provenance/validate_corrected_paired_acquisition_evidence.py
```

On the publication workstation, where the external source artifacts are
available:

```bash
python scripts/provenance/validate_corrected_paired_acquisition_evidence.py \
  --require-external-inputs
```
