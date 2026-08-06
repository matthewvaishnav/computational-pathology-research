# Real paired-scanner biological-bottleneck allocation validation

## Question and frozen motivation

This forward-valid experiment asks whether allocating more of a matched neural
parameter budget directly to the biological bottleneck improves meaningful
feature accessibility on observed paired-scanner pathology data without
increasing scanner recoverability or damaging paired-region preservation.

The motivating synthetic factorial remains frozen as
`complete_capacity_gain_with_scanner_tradeoff`. Its conclusion is not changed:
the low-budget, parameter-matched 64-dimensional candidate improved accessible
synthetic biological information more consistently than extra hidden width,
whereas the high-budget 64-dimensional family incurred a scanner trade-off.
That result motivated exactly one real-data candidate, not another synthetic
architecture search.

## Readiness boundary

The runner audits concrete, hash-verifiable DINOv2-Base arrays and all five
frozen fold manifests for SCORPION and canine squamous-cell carcinoma before a
neural module can be initialized. Every row must align exactly across the
feature archive and manifests. Regions and slides/specimens must remain wholly
inside one split. Every region must occur under at least two scanners, and all
numeric features must be finite.

Layer 1 uses frozen feature space only. Layer 2 is independently gated on
existing biological and acquisition branch archives, actual decoder weights,
source-region mapping, target-scanner mapping, swap assignment metadata, and
fold-consistent provenance. The presence of a checkpoint alone is insufficient.

The experiment never opens pixels or whole-slide images. Feature archives do
not imply registration, local patch correspondence, pixel supervision, or
pixel reconstruction. Pixel-space work remains prohibited without independently
verified image paths, coordinates, registration confidence, and QC exclusions.

## Fixed architectures and objective

Two and only two unsupervised scanner-prototype factorizer families are used:

| Family | Biological dimension | Hidden width | Acquisition dimension |
|---|---:|---:|---:|
| `real_b32_reference` | 32 | 128 | 8 |
| `real_b64_parameter_matched` | 64 | dataset-specific integer match | 8 |

For feature dimension `D`, biological dimension `B`, hidden width `H`,
acquisition dimension `A`, and scanner count `S`, the audited parameter formula
is

`P(D,B,H,A,S) = H² + (2D + 2B + 2A + 9)H + B + SA + D`.

The B64 hidden width is selected by exhaustive integer search before training,
and the relative parameter-count difference must be below 0.5%. The formula is
then checked against the actual PyTorch parameter count.

The fixed loss contains self-feature reconstruction, crossed-target paired
feature reconstruction, same-region biological consistency, a biological
variance floor, prototype centering, and prototype separation. There is no
consensus target, consensus loss, auxiliary head, routed-consensus path, or
downstream supervision. Category, tissue, slide, diagnosis, and biological
labels never enter factorizer training. Only scanner and paired-region metadata
construct training pairs.

## Splits, controls, and evaluation

All five frozen dataset folds and seeds 2201--2205 are used. The primary grid is
two datasets × five folds × five seeds × two families when both datasets pass
readiness. Feature scaling is fit on training rows only. Validation loss alone
selects the checkpoint; test rows remain untouched until selection is complete.

A predeclared broken-region-pair control uses fold 0 and seed 2201 for both
families. It deranges region assignments inside the training fold while
preserving scanner counts and never modifies the primary grid. Frozen original
features, centroid/QR scanner-subspace removal, fixed PCA-component removal, a
training-pair-only linear scanner transform to a canonical scanner, and a
deterministic scanner-balanced random control are evaluated on the same arrays
and folds. Historical results with a different architecture or estimand remain
contextual.

Layer-1 evaluation includes paired-null linear and fixed nonlinear scanner
probes, acquisition scanner recovery, overall and ordered-pair region retrieval,
same/different-region cosine separation, and training-only PCA/spectral
diagnostics. Category accessibility and acquisition-category leakage are
reported only where validated category labels exist. The unprojected biological
code remains the primary endpoint; no best PCA result replaces it.

For inference, model seeds are averaged within fold first. Fold-level effects
remain explicit, and deterministic bootstrap intervals resample fold-level
specimen-blocked aggregates. Scanner views and feature rows are not treated as
independent biological samples.

## Predeclared interpretation

A within-dataset B64 biological-accessibility benefit requires a category
balanced-accuracy increase of at least 0.02, positive effects in at least four
folds, no biological-scanner increase above 0.02, no overall or worst-pair
retrieval decrease below −0.02, no acquisition-category leakage increase above
0.02, and complete integrity. Without category labels the experiment cannot
call biological accessibility improved.

A scanner trade-off is reported separately when the B64 candidate increases
biological scanner recovery above 0.02 or damages overall/worst-pair retrieval
below −0.02. A paired method is unsupported unless the true-pair condition
outperforms its broken-pair control on region retrieval, similarity margin, or
an independently eligible decoder-space paired swap.

Poor scientific performance is a result, not an execution failure.

## Claim boundaries

- The synthetic factorial remains `complete_capacity_gain_with_scanner_tradeoff`.
- This is a new forward-valid test of its low-budget candidate.
- No biological or category label enters representation training.
- Frozen pathology features are not raw histology images.
- Feature-space evidence does not establish pixel-space reconstruction.
- Category preservation is not diagnostic or clinical validation.
- SCORPION and canine SCC cannot establish universal scanner generalization.
- Scanner, site, stain, cohort, and endpoint generalization are distinct.
- No FDA, clinical-deployment, or patient-care claim is supported.
- No public manuscript claim is modified automatically.

## Command

```powershell
py -m experiments.paired_acquisition.run_real_paired_scanner_bottleneck_allocation_validation `
  --synthetic-factorial-result "results\biological_bottleneck_capacity_allocation_factorial_20260803T150254\biological_bottleneck_capacity_allocation_factorial_result.json" `
  --repository-root . `
  --device cuda `
  --output-root "results\real_paired_scanner_bottleneck_allocation_validation_<timestamp>"
```
