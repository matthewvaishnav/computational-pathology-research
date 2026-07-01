# CAMELYON17 Center Leakage: Adversarial Removal Fails, Linear Center-Subspace Projection Partially Works

Date: 2026-06-22

This note records the current CAMELYON17 mechanism result for the Paired-Acquisition Neural Factorization branch. It should be treated as a representation-mechanism diagnostic, not as a deployable clinical model result.

## Question

CAMELYON17 source-center identity remains strongly decodable from the frozen feature substrate. The question was whether this leakage can be removed while preserving tumor information.

The experiments tested two mechanism families:

1. adversarial/subtractive learned removal, and
2. supervised linear center-subspace projection.

## Short conclusion

The adversarial/subtractive variants did not materially reduce post-hoc center leakage. A supervised linear projection of the center-discriminant subspace partially reduced center leakage while leaving tumor AUC essentially unchanged.

This suggests that the failure of the adversarial variants is not evidence that center information is impossible to remove. Instead, it indicates that the learned nuisance-subtraction objectives did not recover the removable center directions.

## Negative adversarial/subtractive results

### v3: tumor-preserving nuisance branch

After fixing the nuisance-center loss wiring, the nuisance branch became center-decodable. However, the cleaned features remained nearly as center-decodable as the original features, and the nuisance component remained tumor-rich.

### v4: component-level center objective

The decoded component became center-decodable, but subtracting it from the original feature vector did not substantially reduce center decodability. The nuisance component also retained strong tumor signal.

### v5: cleaned-feature center adversary

A direct adversary on cleaned features also failed to meaningfully reduce post-hoc center leakage. The best cleaned-feature center accuracy remained approximately original-level.

Representative v5 smoke result:

| Representation | Center accuracy | Tumor AUC |
| --- | ---: | ---: |
| Original features | 0.9171 | 0.9940 |
| Best v5 cleaned features | 0.9160 | 0.9940 |
| Teacher diagnostic representation | 0.8571 | 0.9835 |

Interpretation: v5 preserved tumor signal mostly because the cleaned representation barely moved with respect to center leakage.

## Positive linear projection diagnostic

The v6b diagnostic fit a supervised linear center classifier on one internal stratified subset, projected out the classifier's center-discriminant subspace, and evaluated center and tumor probes on an independent internal stratified probe split.

Across five seeds, the best setting used `C=0.01` for the supervised center projection.

| Representation | Center accuracy mean | Center accuracy std | Tumor AUC mean | Tumor AUC std | Tumor accuracy mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original scaled features | 0.8946 | 0.0122 | 0.9903 | 0.0018 | 0.9550 |
| Center-residualized features | 0.7636 | 0.0122 | 0.9903 | 0.0015 | 0.9560 |

The center accuracy drop was approximately 0.131 absolute, while tumor AUC was essentially unchanged.

## Rank-sweep diagnostic

The v6c rank sweep removed the top `k` supervised center-discriminant directions for `k = 0..4`. Results across five seeds:

| Removed rank | Center accuracy mean | Center accuracy std | Tumor AUC mean | Tumor AUC std | Tumor accuracy mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8946 | 0.0122 | 0.9903 | 0.0018 | 0.9550 |
| 1 | 0.8530 | 0.0117 | 0.9903 | 0.0017 | 0.9564 |
| 2 | 0.8256 | 0.0124 | 0.9903 | 0.0017 | 0.9564 |
| 3 | 0.8092 | 0.0138 | 0.9901 | 0.0015 | 0.9562 |
| 4 | 0.7636 | 0.0122 | 0.9903 | 0.0015 | 0.9560 |

Interpretation: the center signal is distributed across the full four-dimensional center-discriminant subspace rather than concentrated in a single dominant direction.

## v7 explicit projection baseline

The v7 baseline formalizes the v6b/v6c diagnostic as a reproducible script: draw a stratified CAMELYON17 feature subset, split it into an internal projection/calibration half and an independent probe half, fit a linear center classifier on the projection split, remove the top-k right-singular directions of the center classifier weights, then fit independent center and tumor probes on the probe split.

With `C=0.01`, ranks `0..4`, and seeds `911..915`, v7 reproduces the v6c pattern: center accuracy decreases monotonically as more center-discriminant directions are removed, while tumor AUC remains essentially flat.

## What this does and does not show

This shows:

- center information in CAMELYON17 frozen features is partly linearly removable,
- tumor signal can remain stable after removing supervised center directions,
- learned adversarial/subtractive variants v3/v4/v5 did not recover this removable structure,
- the effective center-discriminant projection rank is four for five centers after centering the one-vs-rest classifier weights.

This does not show:

- complete source-center invariance,
- clinical deployment readiness,
- improved held-out-center tumor classification,
- prospective validation,
- scanner/site invariance outside this feature-level diagnostic.

## Next experimental implication

The next useful model family should not simply increase adversarial weight. The v7 explicit center-subspace projection baseline is now the clean reference point.

A stricter follow-up should evaluate whether the projection directions can be fit only on source/train or calibration data, then reused without leakage into held-out evaluation. A learnable constrained projection layer should only be considered after this explicit projection baseline is fully characterized.

The target claim should remain conservative: CAMELYON17 contains a partially removable center subspace that can be attenuated without collapsing tumor signal in frozen features.
