# SCORPION frozen PathoAlign cross-backbone transfer

**Status:** completed; all preregistered criteria passed  
**Development backbone:** DINOv2-Base  
**Transfer backbones:** Phikon and ImageNet ResNet50  
**Dataset:** SCORPION, 48 original H&E slides, 480 aligned tissue regions, five scanners, 2,400 real-human-tissue patches  
**Transfer seeds:** 701--705  
**Evaluation:** five rotating original-slide test folds, paired consistency versus frozen `pathoalign_dep20`

## Protocol status

The transfer protocol was written before inspecting Phikon or ResNet50 outcomes. No backbone-specific loss weight, architecture width, optimizer setting, fold definition, training duration, checkpoint rule, or success threshold was changed.

The frozen objective used:

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

## Backbone-specific descriptive means

### DINOv2-Base

| Method | Mean cosine | Worst cosine | Mean retrieval | Worst retrieval | Biological scanner probe | Effective rank |
|---|---:|---:|---:|---:|---:|---:|
| Paired consistency | 0.847591 | 0.820211 | 0.999867 | 0.999111 | 0.782489 | 56.9326 |
| PathoAlign | **0.878856** | **0.850166** | 0.999787 | 0.998733 | **0.398907** | 54.5017 |

### Phikon

| Method | Mean cosine | Worst cosine | Mean retrieval | Worst retrieval | Biological scanner probe | Effective rank |
|---|---:|---:|---:|---:|---:|---:|
| Paired consistency | 0.773992 | 0.736356 | 0.999064 | 0.994044 | 0.954284 | 55.3623 |
| PathoAlign | **0.864493** | **0.830021** | **0.999680** | **0.997444** | **0.520044** | 46.7507 |

### ResNet50

| Method | Mean cosine | Worst cosine | Mean retrieval | Worst retrieval | Biological scanner probe | Effective rank |
|---|---:|---:|---:|---:|---:|---:|
| Paired consistency | 0.628590 | 0.571348 | 0.964467 | 0.935556 | 0.682791 | 87.6049 |
| PathoAlign | **0.654441** | **0.597813** | **0.972620** | **0.945489** | **0.314462** | 85.0875 |

## Slide-blocked contrasts

Difference definition:

```text
PathoAlign minus paired consistency
```

| Backbone | Scanner probe | Mean cosine | Worst cosine | Mean retrieval | Worst retrieval |
|---|---:|---:|---:|---:|---:|
| DINOv2-Base | **-0.385417** | **+0.031035** | **+0.031096** | -0.000083 | -0.000417 |
| Phikon | **-0.435250** | **+0.090023** | **+0.096274** | **+0.000604** | **+0.004792** |
| ResNet50 | **-0.368500** | **+0.025502** | **+0.029389** | **+0.008083** | **+0.016875** |

All three scanner-probe confidence intervals were entirely below zero. All mean- and worst-cosine confidence intervals were entirely above zero. Every backbone satisfied both predefined retrieval non-inferiority margins and every biological dimension retained nonzero test-fold variance.

## Transfer-backbone repeated-measures summary

Phikon and ResNet50 method differences were averaged within each of the same 48 original slides. This is a repeated-measures summary, not 96 independent biological units.

| Metric | Mean difference | 95% slide-bootstrap interval | Favorable slides |
|---|---:|---:|---:|
| Scanner-probe accuracy | **-0.401875** | **[-0.414458, -0.389458]** | 48/48 |
| Mean paired cosine | **+0.057762** | **[+0.053949, +0.061526]** | 48/48 |
| Worst paired cosine | **+0.062832** | **[+0.058452, +0.067158]** | 48/48 |
| Mean top-1 retrieval | **+0.004344** | **[+0.003313, +0.005427]** | 41/48 |
| Worst top-1 retrieval | **+0.010833** | **[+0.007083, +0.014583]** | 36/48 |

All Monte Carlo sign-flip tests were significant at or below `0.000008` with 250,000 draws.

## Factorization audit

| Backbone | Acquisition scanner probe | Acquisition tissue retrieval | Acquisition effective rank | Cross-covariance RMS |
|---|---:|---:|---:|---:|
| DINOv2-Base | 0.860151 | 0.104180 | 5.4446 | 0.09280 |
| Phikon | 0.971093 | 0.073918 | 8.6024 | 0.08887 |
| ResNet50 | 0.784533 | 0.170524 | 9.2414 | 0.08957 |

Across all three representation families, scanner identity remains strongly available in the compact acquisition branch, while tissue identity is concentrated in the biological branch. The effect is therefore consistent with factor separation rather than simple representational destruction.

## Preregistered decision

```text
crossbackbone_transfer_claim_passed = true
```

Both transfer backbones independently passed all seven criteria, and the pooled transfer-backbone summary also passed.

## Supported conclusion

The same frozen PathoAlign objective transfers across a general self-supervised vision transformer, a pathology-native transformer, and an ImageNet residual network on the SCORPION paired-scanner benchmark. Relative to paired consistency alone, it consistently reduces linearly recoverable scanner identity, improves same-tissue cross-scanner agreement, and preserves or improves cross-scanner tissue retrieval.

The result is strongest as evidence against a DINOv2-specific explanation. It shows representation-family transfer on the same 48 original slides and five scanners.

## Reproducibility warning

The Windows CUDA runs emitted PyTorch warnings because deterministic algorithms were enabled without setting `CUBLAS_WORKSPACE_CONFIG`. The multi-seed, slide-blocked effects are large and directionally consistent, so this does not negate the inferential result. It does mean the completed runs should not be described as guaranteed bitwise deterministic. Future runners must set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch or initializing CUDA.

## Claim boundary

This result does not establish:

- external-dataset generalization;
- transfer to unseen laboratories, tissues, staining procedures, or scanner models outside SCORPION;
- clinical safety, diagnostic equivalence, or improved patient outcomes;
- end-to-end whole-slide task performance;
- perfect biological/acquisition disentanglement.

## Artifacts

```text
docs/research/scorpion-pathoalign-crossbackbone-protocol.md
experiments/scorpion/run_pathoalign_crossbackbone_transfer.py
scripts/scorpion/analyze_pathoalign_crossbackbone_transfer.py
results/scorpion/pathoalign_crossbackbone_transfer/
results/scorpion/pathoalign_crossbackbone_transfer_analysis/
```
