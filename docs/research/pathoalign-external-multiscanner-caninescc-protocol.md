# PathoAlign external paired-scanner validation protocol

**Status:** preregistered before dataset-specific outcomes are inspected  
**External benchmark:** Multi-Scanner Canine Cutaneous Squamous Cell Carcinoma Histopathology Dataset  
**Reported release structure:** 44 tissue samples, each digitized on five whole-slide scanners, with local cross-scanner correspondences  
**Primary purpose:** test whether the frozen SCORPION-developed PathoAlign objective transfers to a different laboratory, tissue collection, scanner set, image scale, and species

## Why this benchmark

The SCORPION result now transfers across DINOv2-Base, Phikon, and ImageNet ResNet50, but all three feature families were evaluated on the same 48 human slides and five SCORPION scanners. The next threat to validity is dataset dependence rather than backbone dependence.

The selected external benchmark was produced independently and provides biology-preserving local correspondences across five scanners. It is publicly described as 44 canine cutaneous squamous-cell-carcinoma samples scanned on five devices. This is a strong acquisition-shift stress test because it changes the laboratory, tissue collection, scanner systems, raw image format, and species.

It is not an external-human validation set. A successful result will support cross-dataset paired-acquisition transfer, not direct human clinical generalization.

## Frozen method

No PathoAlign objective or optimization parameter may be tuned on this dataset.

```text
biological dimension                 256
acquisition dimension                 64
hidden dimension                     512
temperature                           0.1
scanner adversary weight              0.5
scanner acquisition weight            0.5
direct scanner-dependence weight     20.0
biological/acquisition covariance      0.05
gradient-reversal strength             1.0
reconstruction weight                  1.0
variance weight                        1.0
biological covariance weight           0.01
epochs                                75
region batch size                     32
AdamW learning rate                 3e-4
AdamW weight decay                  1e-4
```

The comparator remains paired consistency with the same biological projection, pairing loss, reconstruction, variance, covariance, optimizer, and schedule, but without the acquisition factor or scanner-separation terms.

## Staged data contract

The external release must pass all of the following gates before model training:

1. Exactly 44 biological sample identifiers are recoverable, unless the downloaded release documents a revised count.
2. Exactly five scanner identifiers are recoverable.
3. Every retained paired unit has one view from every scanner.
4. Cross-scanner correspondence metadata are available or can be reproduced from released transforms.
5. Train/test partitions are grouped by original biological sample.
6. No patch, region, registration target, or scanner view from one sample may cross partitions.
7. Images are readable without silent scanner-dependent filtering.
8. The manifest records file hashes, image dimensions, resolution metadata when available, and the transformation used to define each local correspondence.

If the release does not contain sufficiently reliable correspondence metadata, the experiment stops rather than substituting unverified same-coordinate crops.

## Representation families

The primary external test uses the same three frozen feature families:

- DINOv2-Base;
- Phikon;
- ImageNet ResNet50.

Feature extraction preprocessing must be identical across scanner views within a backbone. Native magnification and scanner resolution differences must be normalized through the released local-correspondence geometry, not by selecting visually convenient crops after inspecting outcomes.

## Split and inference design

The original biological sample is the independent unit.

- Five deterministic rotating sample folds.
- Each sample serves as test exactly once per seed and method.
- All scanner views and all local correspondences from a sample remain in the same fold.
- New seeds `801, 802, 803, 804, 805` are used for every backbone.
- The objective is fitted on all non-test samples because no model selection remains.
- Optimization seeds are averaged within biological sample before inference.
- Bootstrap intervals and sign-flip tests use the 44 biological samples as matched blocks.

If fewer than 44 complete samples remain after correspondence audit, the exact retained count and exclusion reasons must be frozen before training.

## Primary metrics

For the biological representation:

- held-out scanner-probe balanced accuracy;
- mean same-tissue cross-scanner cosine;
- worst scanner-pair cosine;
- mean bidirectional correspondence retrieval;
- worst scanner-pair retrieval;
- effective rank;
- nonzero-variance fraction.

For the acquisition representation:

- held-out scanner-probe balanced accuracy;
- correspondence retrieval;
- effective rank;
- normalized biological/acquisition cross-covariance RMS.

## Predefined per-backbone success criteria

Each backbone passes only if all conditions hold relative to paired consistency:

1. Biological scanner-probe accuracy decreases by at least `0.15` absolute.
2. The 95% sample-bootstrap interval for scanner-probe change lies entirely below zero.
3. The 95% lower bound for mean correspondence-retrieval change is at least `-0.02`.
4. The 95% lower bound for worst-pair retrieval change is at least `-0.02`.
5. The 95% interval for mean paired-cosine change lies entirely above zero.
6. The 95% interval for worst-pair cosine change lies entirely above zero.
7. Every biological dimension retains nonzero test-fold variance in every PathoAlign run.

The external cross-dataset claim passes only if at least two of the three frozen backbones pass all seven criteria and the third preserves retrieval within the non-inferiority margin without collapse. Results for every backbone will still be reported.

## Stronger and weaker conclusions

### Supported by a pass

The frozen paired-acquisition factorization transfers beyond SCORPION to an independently collected multi-scanner histopathology dataset and remains effective under changes in laboratory, scanners, tissue collection, and species.

### Not supported by a pass

- external-human clinical validation;
- diagnostic improvement;
- prospective robustness;
- transfer to unseen stains or laboratories without paired data;
- universal biological/acquisition identifiability;
- scanner equivalence for clinical use.

## Reproducibility and deterministic execution

All future CUDA runs must set

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

before importing Torch. Dataset hashes, manifests, folds, feature archives, training designs, and analysis outputs must be retained. Raw WSIs remain outside Git.

## Planned artifacts

```text
data/external_multiscanner_caninescc/
results/external_multiscanner_caninescc/audit/
results/external_multiscanner_caninescc/features/
results/external_multiscanner_caninescc/pathoalign_crossfold/
results/external_multiscanner_caninescc/pathoalign_crossfold_analysis/
```
