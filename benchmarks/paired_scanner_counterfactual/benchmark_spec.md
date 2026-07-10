# Benchmark Spec

## Setup

A paired scanner counterfactual dataset consists of observations:

```text
x(region_id, scanner_id)
```

where `region_id` identifies the same biological tissue region and `scanner_id`
identifies the scanner or acquisition condition. Each counterfactual pair or set
contains the same region observed under at least two scanner labels.

## Allowed datasets

Allowed datasets must provide observed paired scanner acquisitions. Synthetic
pairs, unpaired scanner pools, or scanner labels without same-region pairing are
not sufficient for paired-counterfactual claims.

Allowed for Layer 1:

- paired scanner metadata;
- frozen feature arrays or frozen feature summaries;
- scanner labels;
- paired region/sample identifiers;
- category or tissue labels when category preservation is evaluated.

Allowed for Layer 2:

- Layer 1 metadata;
- biological branch embeddings;
- acquisition branch embeddings;
- decoder/composition weights;
- pair-assignment or swap metadata.

Allowed for Layer 3:

- actual paired patch images or WSI paths;
- scanner-specific paired acquisitions;
- local region correspondence;
- patch coordinates or crop definitions;
- registration confidence or QC metadata;
- category or tissue labels when morphology preservation is evaluated by label.

## Required metadata

Required fields:

- `dataset_id`;
- `scanner_id`;
- `region_id` or equivalent pair identifier;
- split/fold identifier;
- feature or patch path when the layer consumes files;
- source scanner;
- target scanner.

Optional but recommended fields:

- category or tissue label;
- slide/sample identifier;
- acquisition date or scanner protocol;
- registration confidence;
- QC exclusion reason;
- image resolution and magnification;
- patch coordinates in a declared coordinate frame.

## Pair identifiers

Pair identifiers must be stable across scanner labels. Valid identifiers include
same-region IDs, sample-region IDs, or explicit pair-group IDs. A slide ID alone
is insufficient unless it is combined with a local region identifier.

## Scanner labels

Scanner labels must identify acquisition condition at the scanner or scanner
protocol level. Site, center, or client labels are not equivalent unless the
dataset documentation states that they map directly to scanner acquisition.

## Biological/category labels

Category or tissue labels are optional for pair retrieval, neighborhood purity,
and scanner recoverability. They are required for category preservation,
category leakage, and category/scanner tradeoff metrics.

## Feature-space metrics

- Scanner recoverability in biological representation.
- Category/tissue recoverability in biological representation.
- Category/tissue leakage in acquisition representation.
- Pair/top-1 retrieval across scanners.
- Neighborhood purity by region, scanner, and category.
- Scanner/category tradeoff curves.
- Oldstyle centroid/QR scanner-erasure baseline scorecard.

## Decoder-space metrics

- Scanner probe on swapped decoded features.
- Category probe on swapped decoded features.
- Source-category nearest-neighbor rate.
- Target-scanner nearest-neighbor rate.
- Branch-space vs decoder-space discrepancy.
- Type-A same-region swap, where the biological branch comes from the source
  region and the acquisition branch comes from the target scanner observation of
  that same region.

## Pixel-space metrics

These are future-work metrics unless the audit confirms actual pixel-level
paired data and registration/QC readiness.

- Registered paired-patch reconstruction error.
- SSIM/PSNR when registration permits.
- Perceptual/pathology-feature distance.
- Morphology preservation.
- Scanner-target classifier/probe.
- Stain/color distribution match.
- Registration confidence stratification.

## Required baselines

Feature-space baselines:

- original frozen features;
- PCA removal;
- oldstyle centroid/QR projection;
- linear scanner-pair transform when feasible;
- shuffled/broken-pair controls;
- scanner-balanced random controls.

Decoder-space baselines:

- true-pair baseline;
- bottlenecked variants;
- random acquisition swap;
- shuffled control when available;
- oldstyle marked as not directly swappable unless a valid residual analogue
  exists.

Pixel-space future baselines:

- raw source patch;
- standard stain/color normalization;
- scanner-pair color transform;
- paired image translation without factorization;
- unpaired image translation;
- explicit biological/acquisition factor model.

## Failure modes

- Scanner labels are recoverable from the biological branch.
- Category labels are recoverable from the acquisition branch.
- Pair retrieval fails after scanner exchange.
- Neighborhoods collapse by scanner rather than region.
- Decoded features match the target scanner while losing source category.
- Decoder-space behavior contradicts branch-space audit results.
- Pixel reconstructions match color while altering morphology.
- Pixel metrics are reported despite missing registration/QC evidence.
- Shuffled-pair controls perform similarly to true-pair models.
