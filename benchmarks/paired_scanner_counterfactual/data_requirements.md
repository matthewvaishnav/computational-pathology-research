# Data Requirements

## Layer 1: Feature-space

Required:

- feature arrays;
- scanner IDs;
- paired region/sample identifiers;
- train/evaluation split metadata;
- category labels where category or tissue preservation is evaluated.

Feature arrays may be frozen encoder features, projected feature archives, or
documented feature summaries that point to concrete arrays. Scanner IDs and pair
identifiers must align row-by-row with the feature arrays.

## Layer 2: Decoder-space

Required:

- biological branch embeddings;
- acquisition branch embeddings;
- decoder/composition weights;
- paired swap metadata;
- scanner IDs;
- paired region/sample identifiers;
- split metadata.

The biological branch must be traceable to a source region. The acquisition
branch must be traceable to a target scanner observation. Decoder weights must be
the actual saved weights used to compose branch embeddings into decoded feature
space.

## Layer 3: Pixel-space

Required:

- original WSI or patch image paths;
- scanner-specific paired acquisitions;
- local region correspondence;
- patch coordinates;
- registration confidence or equivalent QC metadata;
- tissue/category labels if evaluating biological preservation by label;
- QC rules for rejecting misregistered patches.

Pixel-space reconstruction is not feasible from feature arrays alone. If image
paths exist without coordinates or registration/QC evidence, the dataset may be
useful for a future registration audit but not for a reconstruction benchmark.

## Registration and QC

A pixel-level benchmark must define:

- coordinate frame for each patch;
- scanner-to-scanner local correspondence;
- confidence or residual thresholds;
- exclusion rules for tissue loss, padding, background, blur, and misalignment;
- held-out region split policy;
- scanner-pair stratification.

## Minimum evidence for feasibility

Feature-space feasibility requires row-aligned features, scanner IDs, and region
pairs.

Decoder-space feasibility requires feature-space feasibility plus branch
embeddings and decoder/composition weights.

Pixel-space feasibility requires actual paired image data plus coordinates and
registration/QC evidence. Without those, Layer 3 remains future work.
