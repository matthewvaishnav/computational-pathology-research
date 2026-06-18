# PathoAlign external validation on multi-scanner canine SCC

## Status

**Result:** passed locked external validation.

This note freezes the external paired-acquisition validation of PathoAlign on the public Multi-Scanner Canine Cutaneous Squamous Cell Carcinoma histopathology dataset. The study uses a geometry-qualified paired-scanner subset rather than all released polygons, because the P1000 raster orientation and crop boundary required explicit qualification before feature extraction.

## Dataset and preprocessing boundary

- Dataset: Multi-Scanner Canine Cutaneous Squamous Cell Carcinoma histopathology.
- Biological samples: 44.
- Scanners: `cs2`, `gt450`, `nz20`, `nz210`, `p1000`.
- Released views inspected: 220 TIFF images, 6,215 annotation views, 1,243 candidate matched regions.
- Geometry-qualified subset: 805 complete five-view regions, 4,025 image views.
- Qualification rule: retain a region only when every scanner view requires at most 10% adaptive-crop padding.
- P1000 handling: sample-specific inverse orientation normalization derived from affine correspondence with CS2.
- Raw TIFFs, extracted JPEG patches, frozen feature archives, and checkpoints are intentionally excluded from git.

## Geometry and registration audit

The public annotations and SlideRunner database were audited before modeling:

- TIFF orientation tags were not sufficient to explain the P1000 discrepancy.
- Raw TIFF dimensions matched COCO dimensions.
- SQLite polygon bounds matched COCO geometry within less than one pixel.
- No hidden registration, transform, crop, offset, or matrix fields were found in the SQLite schema.
- P1000 annotation centers followed a coherent sample-specific affine map relative to CS2.
- P1000 median affine rotation was approximately -90 degrees with subpixel-to-low-pixel residuals.

This supports treating the dataset as a valid paired-acquisition benchmark after deterministic orientation normalization and geometry qualification.

## Frozen encoder triage on fold-0 validation

All encoders used the same 805-region geometry-qualified patch set and the same fold-0 sample-blocked split.

| Encoder | Pair cosine avg | Pair cosine worst | Retrieval top-1 avg | Retrieval top-1 worst | Scanner probe accuracy | Effective rank |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 ImageNet | 0.821860 | 0.749598 | 0.853704 | 0.653439 | 0.857143 | 74.743 |
| DINOv2-Base | 0.908476 | 0.862341 | 0.852381 | 0.727513 | 0.841270 | 51.599 |
| Phikon | 0.810269 | 0.778612 | 0.776190 | 0.640212 | 0.979894 | 59.094 |

**Frozen substrate selected:** DINOv2-Base.

Rationale: DINOv2 had the strongest cross-scanner cosine consistency and best worst-case retrieval while still retaining scanner signal far above chance.

## Fold-0 external PathoAlign validation

Setup:

- Frozen DINOv2-Base features.
- Fold 0 only.
- Train split used for fitting.
- Validation split used for evaluation.
- Test rows were not projected.
- Seeds: 901--905.
- Variants: `paired_reference` and `pathoalign_dep20`.
- Schedule: 75 epochs, region batch size 32, learning rate 3e-4, weight decay 1e-4.
- Hyperparameters: locked from SCORPION development.

Seed-matched validation means:

| Metric | Paired reference | PathoAlign dep20 | PathoAlign minus paired |
|---|---:|---:|---:|
| Scanner probe accuracy | 0.747937 | 0.365714 | -0.382222 |
| Pair cosine average | 0.693994 | 0.722137 | +0.028142 |
| Pair cosine worst | 0.603945 | 0.627117 | +0.023172 |
| Retrieval top-1 average | 0.939947 | 0.941587 | +0.001640 |
| Retrieval top-1 worst | 0.881481 | 0.878836 | -0.002646 |

Interpretation: PathoAlign reduced scanner identifiability by roughly 0.38 absolute while preserving same-region retrieval and improving cross-scanner cosine.

## Locked five-fold external test

Setup:

- Frozen DINOv2-Base features.
- Five sample-blocked folds.
- Each fold uses all non-test samples for fitting and evaluates the held-out fold exactly once.
- Seeds: 911--915.
- Variants: `paired_reference` and `pathoalign_dep20`.
- Total fits: 50.
- Hyperparameters remained locked.

Descriptive run means:

| Metric | Paired reference | PathoAlign dep20 | PathoAlign minus paired |
|---|---:|---:|---:|
| Scanner probe accuracy | 0.752868 | 0.361408 | -0.391460 |
| Pair cosine average | 0.696022 | 0.729961 | +0.033939 |
| Pair cosine worst | 0.627300 | 0.656736 | +0.029437 |
| Retrieval top-1 average | 0.930637 | 0.933392 | +0.002756 |
| Retrieval top-1 worst | 0.881242 | 0.884431 | +0.003189 |
| Effective rank | 79.779 | 74.044 | -5.734 |

Sample-blocked contrasts over 44 biological samples:

| Metric | Mean difference | 95% bootstrap CI | Favorable samples | Monte Carlo sign-flip p |
|---|---:|---:|---:|---:|
| Scanner probe accuracy | -0.380347 | [-0.399616, -0.361107] | 44/44 | 0.000004 |
| Pair cosine average | +0.033256 | [+0.030935, +0.035530] | 44/44 | 0.000004 |
| Pair cosine worst | +0.033104 | [+0.028789, +0.037550] | 44/44 | 0.000004 |
| Retrieval top-1 average | +0.002326 | [+0.000005, +0.004789] | 24/44 | 0.064600 |
| Retrieval top-1 worst | +0.001731 | [-0.005823, +0.009250] | 17/44 | 0.660573 |

Factorization audit means for PathoAlign:

| Metric | Mean |
|---|---:|
| Acquisition scanner probe | 0.865098 |
| Acquisition tissue retrieval | 0.180627 |
| Acquisition effective rank | 13.756 |
| Cross-covariance RMS | 0.089831 |

## Predefined success criteria

All criteria passed:

- Scanner probe reduction at least 0.15.
- Scanner-probe confidence interval below zero.
- Mean retrieval noninferiority within 0.02.
- Worst-pair retrieval noninferiority within 0.02.
- Mean pair-cosine confidence interval above zero.
- Worst pair-cosine confidence interval above zero.
- All biological dimensions nonzero.

## Claim boundary

Supported claim:

> On an independent external paired-scanner canine SCC benchmark, a PathoAlign projection with hyperparameters locked from SCORPION reduced scanner identifiability by approximately 0.38 absolute while preserving same-region retrieval and improving cross-scanner cosine consistency.

Do not overstate this as clinical validation or diagnostic performance. This is a representation-identifiability and paired-acquisition validation study.

## Recommended standalone study package

This result is mature enough to become its own standalone repository and short paper:

- Repository: `pathoalign-external-caninescc`
- Paper focus: external paired-scanner validation of PathoAlign.
- Main repository role: research-program hub that links to the standalone PDF, code, and results.
