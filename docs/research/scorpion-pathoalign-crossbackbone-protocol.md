# SCORPION PathoAlign cross-backbone transfer protocol

**Status:** preregistered before Phikon or ResNet50 transfer results are inspected  
**Development backbone:** DINOv2-Base  
**Transfer backbones:** Phikon and ImageNet ResNet50  
**Dataset:** SCORPION, 48 original H&E slides, 480 aligned tissue regions, five scanners, 2,400 real-human-tissue patches

## Scientific question

Does the frozen `pathoalign_dep20` objective transfer beyond the DINOv2 representation family, or was the five-fold result specific to one pretrained feature space?

## Frozen components

The following values were selected during DINOv2 development and will not be changed for either transfer backbone:

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

The comparison remains the same paired-consistency projection with the scanner-separation terms disabled. No backbone-specific loss weight, architecture width, training duration, checkpoint rule, fold definition, or evaluation threshold may be tuned.

## Evaluation design

- Five rotating original-slide test folds.
- All non-test slides are used for fitting because model selection is already complete.
- New optimization seeds `701, 702, 703, 704, 705` are shared across Phikon and ResNet50.
- Each original slide serves as test exactly once per seed, method, and backbone.
- Each backbone requires `5 folds x 5 seeds x 2 methods = 50` fits.
- Phikon and ResNet50 results must be analyzed before any paper claim is changed.
- DINOv2 results are retained as the development-derived reference and are not re-estimated.

## Independent unit and inference

Optimization seeds are averaged within each original slide. The 48 original slides are then treated as the matched blocks for bootstrap intervals and sign-flip tests. Patches, scanner views, and tissue regions are not treated as independent biological replicates.

A cross-backbone summary may average backbone-specific method differences within slide, but this is a repeated-measures summary over the same 48 slides rather than evidence from 144 independent slides.

## Predefined per-backbone success criteria

A transfer backbone passes only if every condition below is met:

1. Mean biological scanner-probe reduction is at least `0.15` absolute.
2. The 95% slide-bootstrap interval for scanner-probe change lies entirely below zero.
3. The 95% lower bound for mean tissue-retrieval change is at least `-0.02`.
4. The 95% lower bound for worst-pair tissue-retrieval change is at least `-0.02`.
5. The 95% interval for mean paired-cosine change lies entirely above zero.
6. The 95% interval for worst-pair paired-cosine change lies entirely above zero.
7. Every biological dimension retains nonzero test-fold variance in every PathoAlign run.

The cross-backbone transfer claim passes only if both Phikon and ResNet50 satisfy all seven criteria without retuning.

## Secondary factor audit

For each transfer backbone, report:

- acquisition scanner-probe accuracy;
- acquisition tissue retrieval;
- acquisition effective rank;
- normalized biological/acquisition cross-covariance RMS;
- biological effective rank and variance audit.

These quantities diagnose whether reduced scanner leakage reflects factor separation rather than destruction of the frozen representation.

## Claim boundary

Passing this protocol would support transfer across three pretrained representation families on the same SCORPION slides and scanner systems. It would not establish generalization to a new dataset, laboratory, tissue type, staining process, scanner model outside SCORPION, diagnostic task, or prospective clinical workflow.

## Planned artifacts

```text
results/scorpion/pathoalign_crossbackbone_transfer/phikon/
results/scorpion/pathoalign_crossbackbone_transfer/resnet50/
results/scorpion/pathoalign_crossbackbone_transfer_analysis/phikon/
results/scorpion/pathoalign_crossbackbone_transfer_analysis/resnet50/
results/scorpion/pathoalign_crossbackbone_transfer_analysis/combined/
```
