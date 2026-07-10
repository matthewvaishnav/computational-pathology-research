# Metrics

## Feature-space metrics

### Scanner recoverability in biological representation

Train a scanner probe on the biological representation and report accuracy,
balanced accuracy, chance accuracy, and confidence intervals where possible.
Lower recoverability is better only when biological utility is preserved.

### Category/tissue recoverability in biological representation

Train a category or tissue probe on the biological representation when labels
exist. This guards against erasing the biological signal while suppressing
scanner signal.

### Category/tissue leakage in acquisition representation

Train a category or tissue probe on the acquisition representation. Lower
category recoverability is preferred when scanner recoverability remains high in
the acquisition branch.

### Pair/top-1 retrieval

For each source scanner observation, retrieve the same region in another scanner
by nearest neighbor. Report directional and averaged top-1 accuracy and mean
reciprocal rank.

### Neighborhood purity

Measure whether local neighborhoods are enriched for the same region, scanner,
or category. A useful biological space should enrich for region/category and
avoid scanner-only clustering.

### Scanner/category tradeoff

Report scanner recoverability against category/tissue recoverability across
representation variants and baselines. Useful methods should reduce scanner
recoverability without collapsing biological/category signal.

### Oldstyle centroid/QR scanner erasure baseline

Evaluate the oldstyle centroid/QR projection as a non-neural scanner-erasure
baseline. It is a feature-space baseline and should not be described as a
decoder-space swap method unless a valid residual analogue is explicitly built.

## Decoder-space metrics

### Scanner probe on swapped decoded features

After composing a source biological branch with a target acquisition branch,
train or apply a scanner probe to determine whether decoded features behave like
the target scanner.

### Category probe on swapped decoded features

Evaluate whether the decoded features retain the source region's category or
tissue label when labels exist.

### Source-category NN rate

Find nearest neighbors of swapped decoded features and report how often the
nearest category matches the source observation category.

### Target-scanner NN rate

Find nearest neighbors of swapped decoded features and report how often the
nearest scanner matches the target acquisition scanner.

### Branch-space vs decoder-space discrepancy

Compare branch-level probe/retrieval behavior with decoded-feature behavior.
Large disagreement indicates that branch separation does not necessarily survive
composition through the decoder.

### Type-A same-region swap

The cleanest swap test uses the same region observed by source and target
scanners. The source biological branch is combined with the target scanner's
acquisition branch, and the decoded output is compared with the real target
scanner observation at the feature level.

## Pixel-space future metrics

### Registered paired-patch reconstruction error

Compute pixel error only when source and target patches are registered closely
enough for pixel comparison.

### SSIM/PSNR where registration permits

Report SSIM and PSNR only for patch pairs that pass registration confidence and
QC thresholds.

### Perceptual/pathology-feature distance

Compare generated and real target patches using frozen pathology features or a
declared perceptual feature model.

### Morphology preservation

Measure whether structures from the source region remain present after scanner
translation. This can use category labels, feature distances, or registered
region masks when available.

### Scanner-target classifier/probe

Use a scanner classifier to test whether generated patches resemble the target
scanner. This metric is not sufficient by itself because it does not verify
biological preservation.

### Stain/color distribution match

Compare color and stain statistics between generated and real target scanner
patches, stratified by scanner pair.

### Registration confidence stratification

Report metrics by registration confidence/QC bins. Pixel metrics should degrade
or be withheld for poorly registered pairs.
