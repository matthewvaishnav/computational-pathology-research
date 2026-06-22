# Camelyon17 External-Center Validation: Early Feature-Level Evidence

**Status:** early external-validation note / feature-level baseline  
**Scope:** Camelyon17/WILDS, frozen ResNet18 features, logistic-regression weighting baselines, center-leakage mechanism diagnostics  
**Clinical status:** research-only; not clinically validated; not diagnostic software; not intended for patient-care use

---

## One-sentence claim

A 5-seed Camelyon17/WILDS feature-level baseline shows that FedAvg-style equal-patch weighting can look better on source-like validation while performing substantially worse on a held-out external test center, and a separate center-subspace projection mechanism branch shows that supervised linear center-subspace projection can partially attenuate source-center leakage while preserving tumor signal.

---

## Why this note exists

The dominant-site / site-signal alignment work originally used simulated federations over pathology-derived features. Camelyon17 adds a more natural multi-center pathology validation target: source centers are used for training, separate source-domain validation is available, and two centers are held out as out-of-distribution validation/test domains.

This note documents the first external-center evidence layer. It is intentionally conservative: these are fast feature-level baselines and representation-mechanism diagnostics, not full clinical or production federated-learning results.

---

## Dataset audit

Camelyon17/WILDS metadata was audited and converted into an explicit federated-client manifest.

```text
Total examples: 455,954
Classes: 227,977 negative / 227,977 positive
Selected FL client column: center
Centers: 0, 1, 2, 3, 4
```

Split structure:

```text
train: 302,436 examples from centers 0, 3, 4
id_val: 33,560 examples from centers 0, 3, 4
val: 34,904 examples from center 1
test: 85,054 examples from center 2
```

Interpretation:

```text
source-domain training clients: centers 0, 3, 4
source-domain validation: centers 0, 3, 4
OOD validation: center 1
OOD test: center 2
```

This structure is useful for testing whether source-domain weighting choices generalize to held-out pathology centers and whether center-identifying representation directions can be audited separately from tumor information.

---

## Image-feature smoke baseline

A first image-based smoke test used frozen ImageNet ResNet18 features with logistic regression. The purpose was not to claim final model performance. The purpose was to verify that image loading, center splits, source-domain training, and OOD evaluation were wired correctly.

Single-smoke-test result using 8,000 sampled patches:

```text
train source centers 0/3/4: accuracy 0.9823, AUC 0.9984
id_val source centers 0/3/4: accuracy 0.9383, AUC 0.9805
OOD val center 1: accuracy 0.8820, AUC 0.9397
OOD test center 2: accuracy 0.8210, AUC 0.9497
```

The smoke baseline established that the real Camelyon17 image pipeline was working and that external-center evaluation produced visible distribution-shift effects.

---

## Center-weighting baselines

The next experiment compared three source-center weighting policies using the same frozen ResNet18 feature / logistic-regression setup:

1. `fedavg_equal_patch`: every sampled patch has equal weight, so larger source centers have more influence.
2. `equal_client`: each source center receives equal total weight.
3. `downweight_dominant_center`: each source center is balanced, then the dominant source center is further downweighted.

A 5-seed run showed the following mean accuracies:

| Policy | id_val accuracy | val accuracy | test accuracy |
|---|---:|---:|---:|
| `fedavg_equal_patch` | 0.9303 | 0.8668 | 0.8312 |
| `equal_client` | 0.9149 | 0.8258 | 0.9132 |
| `downweight_dominant_center` | 0.9125 | 0.8246 | 0.9094 |

Key held-out test-center differences against FedAvg-style equal-patch weighting:

```text
equal_client - fedavg_equal_patch = +0.0820 accuracy = +8.20 percentage points

downweight_dominant_center - fedavg_equal_patch = +0.0782 accuracy = +7.82 percentage points
```

Interpretation:

```text
FedAvg-style equal-patch weighting performs better on source-like validation and center-1 validation.
Equal-client weighting performs substantially better on the held-out test center.
```

This is exactly the kind of tradeoff the site-signal alignment hypothesis predicts: sample-volume weighting can optimize apparent source-domain performance while harming external-center generalization.

---

## Validation-aware detector switch

A simple detector-switch rule was then tested using only validation diagnostics.

Decision inputs:

```text
id_val: source-domain validation centers
val: OOD validation center
```

Held-out evaluation:

```text
test: held-out OOD test center
```

The detector did not use test performance when choosing a policy.

Rule:

```text
Switch from FedAvg-style equal-patch weighting to the alternative policy when:

alternative val accuracy - FedAvg val accuracy >= min_val_gain

and

FedAvg id_val accuracy - alternative id_val accuracy <= max_id_val_cost

Otherwise keep FedAvg-style equal-patch weighting.
```

With `min_val_gain = -0.05` and `max_id_val_cost = 0.03`, the detector switched in 4 of 5 seeds for both alternatives.

Mean held-out test-center gains:

```text
equal-client detector switch:
  test accuracy delta = +0.0658 = +6.58 percentage points
  test macro-F1 delta = +0.0681

 downweight-dominant detector switch:
  test accuracy delta = +0.0628 = +6.28 percentage points
  test macro-F1 delta = +0.0651
```

Interpretation:

```text
The detector result is weaker than always using equal-client weighting because it keeps FedAvg in one seed.
That is expected: the detector is making a validation-aware choice without seeing the held-out test center.
The important point is that the detector still recovers a large held-out-center improvement.
```

---

## Camelyon17-trained supervised ResNet18 feature validation

The frozen ImageNet feature result is useful as a fast external-center baseline, but a natural objection is that ImageNet features are not pathology-specific. I therefore trained an ImageNet-initialized ResNet18 on Camelyon17 source-domain centers and used the resulting penultimate-layer features for the same center-weighting analysis.

Training and checkpoint selection:

- Training split: source centers 0, 3, and 4.
- Checkpoint selection: source-domain validation behavior.
- Held-out test center: not used for checkpoint selection.

The 5-seed supervised-feature result showed the same direction with smaller but cleaner gains.

FedAvg-style equal-patch weighting:

- Train accuracy: 0.9991
- Source-domain validation accuracy: 0.9641
- OOD validation accuracy: 0.8986
- Held-out OOD test accuracy: 0.9052

Equal-client weighting:

- Train accuracy: 0.9683
- Source-domain validation accuracy: 0.9698
- OOD validation accuracy: 0.9232
- Held-out OOD test accuracy: 0.9318
- Held-out test gain versus FedAvg-style equal-patch weighting: +0.0266 accuracy, or +2.66 percentage points.

Downweight-dominant-center weighting:

- Train accuracy: 0.9681
- Source-domain validation accuracy: 0.9692
- OOD validation accuracy: 0.9228
- Held-out OOD test accuracy: 0.9322
- Held-out test gain versus FedAvg-style equal-patch weighting: +0.0270 accuracy, or +2.70 percentage points.

Interpretation:

The effect survives when moving from frozen ImageNet features to Camelyon17-trained supervised features. The gain is smaller because the learned features are stronger overall, but the generalization pattern is cleaner: FedAvg-style equal-patch weighting nearly saturates source training accuracy, while equal-client and downweight-dominant weighting improve source-domain validation, OOD validation, and held-out OOD test performance.

---

## Center-subspace projection mechanism branch

A separate center-subspace projection branch used the Camelyon17-trained supervised ResNet18 feature substrate to ask a different question: can source-center identity be attenuated without collapsing tumor signal?

This branch produced a negative-to-positive mechanism result:

1. v3/v4/v5 learned adversarial or subtractive nuisance-removal objectives did not materially reduce post-hoc center leakage.
2. v6b/v6c showed that a supervised linear center-discriminant subspace is partially removable.
3. v7 formalized the explicit projection baseline.

The v7 projection protocol is intentionally simple:

1. draw a stratified Camelyon17 feature subset,
2. split it into an internal projection/calibration half and an independent probe half,
3. fit a linear center classifier on the projection half,
4. remove the top `k` right-singular directions of the center-classifier weight matrix,
5. evaluate independent center and tumor probes on held-out probe features.

With `C=0.01`, ranks `0..4`, and seeds `911..915`, the formal v7 baseline reproduced the v6c result:

| Removed rank | Center accuracy mean | Center accuracy std | Tumor AUC mean | Tumor AUC std | Tumor accuracy mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8946 | 0.0122 | 0.9903 | 0.0018 | 0.9550 |
| 1 | 0.8530 | 0.0117 | 0.9903 | 0.0017 | 0.9564 |
| 2 | 0.8256 | 0.0124 | 0.9903 | 0.0017 | 0.9564 |
| 3 | 0.8092 | 0.0138 | 0.9901 | 0.0015 | 0.9562 |
| 4 | 0.7636 | 0.0122 | 0.9903 | 0.0015 | 0.9560 |

Interpretation:

```text
The adversarial/subtractive objectives failed, but this was not evidence that center information was inseparable from tumor information. Explicit supervised center-subspace projection partially reduced center leakage while preserving tumor AUC.
```

This mechanism branch does not claim improved clinical performance or full center invariance. It supports a narrower representation claim: a partially removable center subspace exists in the frozen Camelyon17 feature substrate, and removing the effective four-dimensional center-discriminant subspace attenuates center decodability without collapsing tumor signal.

See the dedicated mechanism note:

```text
docs/benchmarks/camelyon17_center_projection_negative_to_positive_mechanism.md
```

---

## Threshold sweep

To test whether the detector result depended on one hand-picked threshold, a local threshold sweep was run.

Sweep grid:

```text
min_val_gain: [-0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.00]
max_id_val_cost: [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
alternatives: equal_client, downweight_dominant_center
```

Total detector settings evaluated:

```text
112
```

Robust-positive definition:

```text
mean held-out test accuracy improvement > +0.03
mean held-out test macro-F1 improvement > +0.03
switch rate >= 40%
```

Result:

```text
43 / 112 settings were robust-positive.
```

Best equal-client settings achieved:

```text
test accuracy delta = +0.0820
test macro-F1 delta = +0.0848
test AUC delta = +0.0083
```

Best downweight-dominant settings achieved:

```text
test accuracy delta = +0.0782
test macro-F1 delta = +0.0810
test AUC delta = +0.0075
```

Interpretation:

```text
The detector-switch result is not only a single-threshold artifact.
A meaningful neighborhood of validation-aware settings preserves positive held-out-center gains.
```

---

## Claim boundaries

Supported by this note:

- Camelyon17/WILDS provides a natural multi-center validation structure for the site-signal alignment question.
- Frozen ResNet18 feature baselines show real external-center degradation.
- FedAvg-style equal-patch weighting can perform better on source-like validation while performing worse on a held-out external test center.
- Equal-client and downweighted-dominant policies improve held-out test-center accuracy in 5-seed feature-level baselines.
- A validation-aware detector-switch rule can recover substantial held-out test-center improvement without using the test center to choose the policy.
- A threshold sweep suggests the detector-switch finding is not purely a one-threshold artifact.
- The center-subspace projection mechanism branch shows that explicit supervised center-subspace projection partially attenuates center decodability while preserving tumor AUC in internal feature-level probes.

Not supported by this note:

- clinical readiness
- diagnostic safety
- real hospital federated deployment performance
- superiority of any final architecture
- universal calibration of the detector
- institutional ranking or claims about hospital/pathologist quality
- proof that equal-client weighting is always better
- complete source-center invariance
- proof of improved held-out-center tumor classification from center-subspace projection

---

## Reproducibility artifacts

Scripts:

```text
scripts/camelyon17/audit_camelyon17_wilds.py
scripts/camelyon17/build_fl_client_manifest.py
scripts/camelyon17/run_metadata_baselines.py
scripts/camelyon17/run_resnet18_feature_smoke.py
scripts/camelyon17/run_center_weighting_baselines.py
scripts/camelyon17/run_detector_switch_from_weighting_results.py
scripts/camelyon17/run_detector_switch_threshold_sweep.py
scripts/camelyon17/run_pathoalign_v6c_center_projection_rank_sweep.py
scripts/camelyon17/run_pathoalign_v7_center_projection_baseline.py
```

Key result artifacts:

```text
results/camelyon17/camelyon17_dataset_audit.md
results/camelyon17/fl_client_manifest.md
results/camelyon17/metadata_baselines.md
results/camelyon17/resnet18_feature_smoke_results.md
results/camelyon17/center_weighting_5seed_summary.md
results/camelyon17/center_weighting_5seed_delta_summary.md
results/camelyon17/detector_switch_validation_aware_summary.md
results/camelyon17/detector_switch_threshold_sweep_summary.md
docs/benchmarks/camelyon17_center_projection_negative_to_positive_mechanism.md
```

Large raw dataset files, checkpoints, and generated run directories are not tracked in the repository.

---

## Next steps

1. Replace fast feature-level baselines with larger confirmatory runs where practical.
2. Compare the center-projection baseline against stricter source/train-only or calibration-only projection fitting.
3. Add stronger conditional probes for residual center leakage after projection.
4. Compare against real FL baselines such as FedProx, SCAFFOLD, FedBN, and robust aggregation.
