# Baselines

## Feature-space baselines

### Original frozen features

Evaluate frozen encoder features before any scanner-erasure or factor-like
projection. This is the main reference for scanner recoverability and pair
retrieval.

### PCA removal

Remove scanner-associated principal components using training data only. Report
the number of components removed and the effect on scanner and category metrics.

### Oldstyle centroid/QR projection

Use scanner centroids and QR projection to remove scanner-associated directions.
This baseline is valid for feature-space erasure and should be compared against
frozen features and learned branch models.

### Linear scanner-pair transform

When enough paired regions exist for each scanner pair, fit a linear transform
from source-scanner features to target-scanner features. Use held-out regions for
evaluation.

### Shuffled/broken-pair controls

Break region pairings while preserving scanner counts. A valid paired method
should lose same-region retrieval or swap quality under these controls.

### Scanner-balanced random controls

Use scanner-balanced random projections or random pair assignments to estimate
chance behavior under balanced scanner marginals.

## Decoder-space baselines

### True-pair baseline

Use real same-region source/target pairs as the clean reference for swap audits.

### Bottlenecked variants

Evaluate acquisition bottleneck sizes and leakage controls. Report both scanner
recoverability and biological/category preservation.

### Random acquisition swap

Combine a source biological branch with a randomly selected acquisition branch
from the target scanner. This tests whether the target scanner signal dominates
without same-region support.

### Shuffled control

Use shuffled pair assignments when available. The shuffled control should reduce
same-region consistency if the model depends on true pairing.

### Oldstyle scope boundary

Oldstyle centroid/QR is not directly swappable because it does not produce a
separate acquisition branch. It can enter decoder-space comparisons only if an
explicit residual analogue is constructed and documented.

## Pixel-space future baselines

### Raw source patch

Use the unmodified source scanner patch as a lower baseline for target-scanner
appearance.

### Standard stain/color normalization

Apply a conventional color or stain normalization method from source to target
scanner style.

### Scanner-pair color transform

Fit a scanner-pair color transform using training pairs and evaluate on held-out
regions.

### Paired image translation without factorization

Train a paired image-to-image model that maps source scanner patches to target
scanner patches without explicit biological/acquisition branches.

### Unpaired image translation

Train an unpaired scanner-translation model when unpaired scanner pools are
available. It should be clearly separated from observed paired-counterfactual
claims.

### Explicit biological/acquisition factor model

Train a model with separate biological and acquisition branches only when paired
pixel data and registration/QC evidence are available.
