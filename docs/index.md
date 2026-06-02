---
layout: doc
aside: false
outline: deep
---

# Computational Pathology AI Research Framework

## Complete technical report: PCam, PANDA/TransnnMIL, PathologyFL, FAIR-WEIGHTS-H, and dominant-site federated pathology

**Matthew Vaishnav**  
Independent research and engineering technical report  
Research-only. Not clinically validated. Not diagnostic software. Not intended for patient-care use.

---

## Abstract

This report consolidates the current evidence stack in this repository. The project is a computational pathology AI research framework spanning PCam patch-level tumor classification, PANDA slide-level prostate grading with Phikon features, TransnnMIL stabilization, PathologyFL federated learning infrastructure, FAIR-WEIGHTS-H institutional weighting research, and a dominant-site federated pathology stress result.

The strongest current paper-style artifact is the dominant-site result: in simulated federated pathology experiments over PANDA-derived Phikon features, raw sample count is not equivalent to task-specific site-signal alignment. A fixed label-noise-calibrated detector transfers to conservative ordinal threshold shift, keeps clean-regime switching low at 13.3%, and produces statistically positive gains at 35% and 45% conservative shift across global QWK, macro-F1, and worst-site QWK.

The broader framework evidence includes PCam full-dataset validation with 85.26% test accuracy and 0.9394 AUC on 32,768 test samples, PANDA slide-level Phikon MIL benchmarks with AttentionMIL and TransnnMIL, a stabilized TransnnMIL learning-rate grid, PCam simulated-site federated smoke tests, FAIR-WEIGHTS-H execution and aggregation-behavior validation, and explicit claim-boundary guardrails.

---

## 1. Repository-level purpose

This repository is an independent research and engineering framework for computational pathology AI. It combines model research, real pathology data workflows, benchmark automation, federated learning, mathematical validation tooling, and documentation.

The project connects three layers:

1. **Model research:** patch classifiers, attention MIL, transformer MIL, TransMIL-style reasoning, TransnnMIL, and foundation encoders.
2. **Real-data workflow:** PCam, PANDA, Camelyon planning, WSI preprocessing, patch extraction, metrics, thresholds, and failure analysis.
3. **Research infrastructure:** tests, reports, reproducibility commands, federated learning workflows, privacy hooks, robustness checks, and claim-status guardrails.

The project is not a clinical product. It is a research system for making computational pathology experiments reproducible, inspectable, and extensible.

---

## 2. Validation ladder

| Stage | Status | Meaning |
|---|---|---|
| Synthetic smoke validation | Complete | Basic plumbing and numerical stability checks |
| PCam patch-level validation | Complete | Real pathology patch benchmark validation |
| PCam federated smoke tests | Complete | Federated pipeline runs on real PCam patches split into simulated sites |
| PCam balanced federated benchmark | Complete | Weighting strategies compared under balanced simulated sites |
| PCam heterogeneous benchmark | Complete | Different weights produced, but patch-level performance was insensitive to those differences |
| PANDA slide-level prostate benchmark | Complete | Slide-level MIL over Phikon feature bags |
| PANDA TransnnMIL ablations | Complete | Patch cap, learning rate, dropout, and stabilized LR grid documented |
| Dominant-site federated pathology stress | Complete current paper artifact | Sample-volume / site-signal alignment failure mode and detector switch |
| Camelyon16/17 validation | Planned | Real multi-center / WSI validation target |
| Clinical validation | Not completed | Requires clinical workflow validation, governance, and regulatory review |

---

## 3. PCam public-dataset benchmark

The PCam work validates the patch-level pathology pipeline on real pathology images. The model was trained and evaluated on the full PatchCamelyon dataset, achieving **85.26% test accuracy** and **0.9394 AUC** on the complete 32,768-sample test set.

| Metric | Value | 95% CI Lower | 95% CI Upper |
|---|---:|---:|---:|
| Accuracy | 85.26% | 84.83% | 85.63% |
| AUC | 0.9394 | 0.9369 | 0.9418 |
| F1 Score | 0.8507 | 0.8464 | 0.8543 |
| Precision (macro) | 0.8718 | 0.8680 | 0.8751 |
| Recall (macro) | 0.8526 | 0.8486 | 0.8561 |

### 3.1 PCam benchmark comparison

| Rank by AUC | Method | Accuracy | AUC | F1 | Parameters | AUC Difference |
|---:|---|---:|---:|---:|---:|---:|
| 1 | This model | 0.8526 | 0.9394 | 0.8507 | ~12M | — |
| 2 | Swin-Transformer (2021) | — | 0.9312 | — | — | +0.0082 |
| 3 | ConvNeXt (2022) | — | 0.9298 | — | — | +0.0096 |
| 4 | ViT-Base (2021) | — | 0.9287 | — | — | +0.0107 |
| 5 | PathViT (2023) | — | 0.9267 | — | — | +0.0127 |
| 6 | MedViT (2023) | — | 0.9234 | — | — | +0.0160 |
| 7 | HistoNet (2022) | — | 0.9198 | — | — | +0.0196 |
| 8 | EfficientNet-B0 (2019) | — | 0.9134 | — | — | +0.0260 |
| 9 | ResNet-50 (2016) | — | 0.9021 | — | — | +0.0373 |
| 10 | DenseNet-121 (2017) | — | 0.8967 | — | — | +0.0427 |
| 11 | ResNet-18 (2018) | — | 0.8890 | — | — | +0.0504 |

### 3.2 PCam confusion matrix and threshold tuning

```text
              Predicted
              Normal  Tumor
Actual Normal  15,837    554
       Tumor    4,276 12,101
```

At the default threshold, tumor precision is high but tumor recall is lower. Screening-style threshold optimization at threshold **0.051** yields **90.0% sensitivity**, **80.3% specificity**, and reduces missed tumor predictions from **4,276** to **1,639**, a **61.7% relative reduction**.

### 3.3 PCam limitations

PCam is patch-level classification. It does not validate whole-slide aggregation, real hospital deployment, pathologist comparison, or clinical workflow performance.

---

## 4. PANDA slide-level prostate grading and TransnnMIL

The PANDA work validates slide-level prostate grading over Phikon feature bags. The readable feature subset contains **10,611 slide-level feature files** after HDF5 read verification, with feature dimension **768**.

| Model | Best validation QWK |
|---|---:|
| Mean-pooled Phikon + MLP | 0.7274 |
| Gated AttentionMIL | 0.8100 |
| Tuned TransnnMIL, seed 42 | 0.8155 |
| Tuned TransnnMIL, seed 123 | 0.8225 |
| Tuned TransnnMIL, seed 2025 | 0.8086 |

### 4.1 TransnnMIL direction

TransnnMIL is the custom whole-slide MIL direction in the project. It is intended to combine global transformer-style attention over patch embeddings with local diagnostic-region reasoning, hierarchical spatial pooling, topology-aware tissue-structure modeling, graph-inspired reasoning over spatial neighborhoods, and optional adaptive pruning.

### 4.2 Stabilized TransnnMIL learning-rate grid

Earlier PANDA TransnnMIL ablations showed optimizer sensitivity. A stabilized recipe using AdamW, warmup-cosine learning-rate scheduling, two warmup epochs, gradient clipping at norm 1.0, early stopping, and repeated seeds widened the stable learning-rate regime.

| Learning rate | Runs | Seeds | Mean best val QWK | Std | Min | Max | Mean best epoch |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1e-4 | 3 | 42, 123, 2025 | 0.8257 | 0.0169 | 0.8087 | 0.8425 | 17.67 |
| 2e-4 | 3 | 42, 123, 2025 | 0.8245 | 0.0192 | 0.8077 | 0.8455 | 14.67 |
| 3e-4 | 3 | 42, 123, 2025 | 0.8238 | 0.0160 | 0.8127 | 0.8422 | 18.67 |
| 1e-3 | 3 | 42, 123, 2025 | 0.8160 | 0.0170 | 0.7998 | 0.8337 | 15.00 |
| 5e-4 | 3 | 42, 123, 2025 | 0.8158 | 0.0144 | 0.8042 | 0.8319 | 16.67 |
| 7e-4 | 3 | 42, 123, 2025 | 0.8117 | 0.0163 | 0.7994 | 0.8301 | 16.00 |

Safe claim: stabilized TransnnMIL remained competitive across 18 full-PANDA runs spanning six learning rates and three seeds, with mean best validation QWK ranging from approximately 0.812 to 0.826. This does not prove architecture superiority over AttentionMIL.

---

## 5. PathologyFL federated learning framework

PathologyFL is the federated learning layer for computational pathology experiments. It supports privacy-preserving multi-site training, pathology-specific federated workflows, secure aggregation and differential privacy hooks, robustness to client heterogeneity and site imbalance, and FAIR-WEIGHTS-H institutional weighting.

Coordinator components include:

- orchestrator
- aggregator
- client registry
- privacy engine
- monitoring system
- byzantine detection

Client-side workflow:

```text
Local pathology data
    -> local training
    -> update serialization
    -> secure communication
    -> aggregation
```

Core federated integration tests pass and smoke tests execute on PCam-derived pathology data. This validates end-to-end execution but not real multi-center clinical deployment.

---

## 6. FAIR-WEIGHTS-H

FAIR-WEIGHTS-H is the hybrid institutional weighting protocol. It replaces prestige-style institutional weighting with an auditable hybrid protocol combining contribution estimation, diagnostic and process quality, distributional uniqueness, representation and subgroup-safety constraints, uncertainty penalties, anomaly monitoring, and fallback modes.

### 6.1 Key design principle

A single scalar institution weight cannot by itself guarantee fairness or safety. FAIR-WEIGHTS-H separates three concepts:

```math
w_i^{train} != w_i^{val} != w_i^{monitor}
```

A site may have a low training weight because of uncertain updates while still receiving high validation and monitoring priority if it represents an underserved or clinically important population.

### 6.2 Institutional signal vector

For institution `i`, define:

```math
z_i = [A_i^{adj}, Q_i, phi_i^{Owen}, JS_i, F_i, V_i, -S_i]
```

| Symbol | Meaning | Status |
|---|---|---|
| A_i^{adj} | Difficulty-adjusted reference-case diagnostic quality | Proposed |
| Q_i | Process and pathology quality composite | Proposed |
| phi_i^{Owen} | Group-aware counterfactual contribution estimate | Proposed |
| JS_i | Jensen-Shannon distributional uniqueness | Proposed |
| F_i | Underserved-population representation score | Proposed |
| V_i | Bounded/sublinear volume factor | Proposed |
| S_i | Uncertainty, instability, or anomaly penalty | Proposed |

Gradient alignment is not used as a primary contribution factor because it can encode status quo bias. It may be used only for anomaly detection and drift monitoring.

### 6.3 Integrity gate

```math
I_i = G_i^{data} G_i^{label} G_i^{safety}
```

The gate is used only for integrity and safety failures, not for prestige or raw quality. Low resource level, rural status, or case difficulty must not directly trigger exclusion.

### 6.4 Difficulty-adjusted quality

```math
Y_ij ~ Bernoulli(p_ij)
logit(p_ij) = alpha + beta^T X_ij + b_i
A_i^{adj} = E_{X ~ P_ref}[Pr(Y = 1 | X, i)]
```

This adjustment must be calibrated and audited.

### 6.5 Useful uniqueness

```math
D_i^{useful} = JS_i * A_i^{adj} * U_i^{subgroup}
```

Distributional uniqueness is useful only when paired with quality and subgroup utility.

### 6.6 Counterfactual contribution

```math
phi_i = E_{S subset N \ {i}}[U(S union {i}) - U(S)]
hat_phi_i^{Owen} = sum_g m_ig hat_phi_{i|g}
```

Grouped or multi-membership Owen-style sampling avoids forcing ambiguous hospitals into a single administrative category.

### 6.7 Training weight objective

```math
w_t = argmax_{w in W} sum_i w_i(
    hat_phi_{i,t}^{Owen}
    + lambda_D D_{i,t}^{useful}
    + lambda_F F_{i,t}
    + lambda_Q Q_{i,t}
    - lambda_S S_{i,t}
)
```

subject to:

```math
sum_i w_i = 1
w_i^{min} <= w_i <= w_i^{max}
C_g(w) >= C_g^{min} for all underserved groups g
Perf_g(w) >= Perf_g^{min} for all clinical groups g
|w_{i,t} - w_{i,t-1}| <= Delta_i
```

The fairness and subgroup safety requirements are constraints, not optional score boosts.

### 6.8 Empirical status

FAIR-WEIGHTS-H has been empirically tested for execution stability and aggregation behavior through synthetic Camelyon-like smoke validation, PCam federated smoke validation, PCam all-strategy smoke validation, PCam balanced federated benchmark, and PCam heterogeneous federated benchmark. These tests show that FAIR-WEIGHTS-H runs without numerical failure, produces distinct weight trajectories under heterogeneous simulated sites, and does not degrade performance in the current patch-level PCam setup.

They do **not** yet show a consistent performance or fairness advantage over simpler aggregation baselines.

### 6.9 Fallback modes

| Mode | Trigger | Action |
|---|---|---|
| Normal | No major anomaly | Full hybrid weighting |
| Conservative | Moderate anomaly or high uncertainty | Freeze coefficients, reduce caps, rely more on verified quality |
| Safety | Systemic manipulation or validation failure | Suspend uniqueness/contribution bonuses; use externally verified metrics only |
| Emergency | Severe corruption or safety violation | Temporarily exclude affected institution updates pending remediation |

### 6.10 Regulatory-safe claim language

Use:

> FAIR-WEIGHTS-H enforces pre-specified representation and subgroup-performance constraints and requires prospective validation before clinical deployment.

Do not use:

> FAIR-WEIGHTS-H proves fairness.

---

## 7. Dominant-site federated pathology paper

The dominant-site result is the current strongest integrated paper artifact. It studies sample-volume / site-signal alignment failure modes in simulated federated pathology experiments over PANDA-derived Phikon features. Validation labels are kept clean while the largest simulated client's training signal is perturbed through dominant-site label corruption and systematic ordinal threshold shift.

The working thesis is that raw sample count is not the same as task-specific site-signal alignment. FedAvg can become less safe when the largest simulated pathology client has a training-label process that is misaligned with the validation objective, and dominance-aware aggregation or switching can reduce that risk under controlled stress.

### 7.1 Fixed detector transfer result

| Conservative shift | Trigger rate | Global QWK delta | 95% CI | Macro-F1 delta | 95% CI | Worst-site QWK delta | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 13.3% | -0.00025 | [-0.00113, 0.00062] | +0.00009 | [-0.00039, 0.00057] | +0.00151 | [-0.00162, 0.00463] |
| 25% | 33.3% | +0.00129 | [-0.00175, 0.00432] | +0.00290 | [-0.00184, 0.00765] | +0.00353 | [-0.00428, 0.01133] |
| 35% | 60.0% | +0.00542 | [0.00062, 0.01022] | +0.00838 | [0.00272, 0.01405] | +0.00991 | [0.00169, 0.01813] |
| 45% | 73.3% | +0.01053 | [0.00239, 0.01866] | +0.01512 | [0.00819, 0.02204] | +0.01290 | [0.00547, 0.02034] |

![Fixed detector transfer](../figures/dominant-site-figure-3-detector-transfer.png)

### 7.2 Diagnostic frequency

| Diagnostic | Count |
|---|---:|
| mean_abs_error_high | 44 |
| worst_site_qwk_low | 31 |
| global_qwk_low | 27 |
| severe_error_rate_high | 22 |
| site_qwk_spread_high | 12 |

The detector is driven primarily by ordinal-error and QWK degradation signals, not by site-spread alone.

### 7.3 Leave-one-diagnostic-family-out ablation

| Variant | Mean trigger rate | Mean global QWK delta | Mean macro-F1 delta | Mean worst-site QWK delta | Significant global-QWK regimes | Positive global-QWK regimes |
|---|---:|---:|---:|---:|---:|---:|
| only_mean_abs_error_high | 96.7% | +0.00865 | +0.01462 | +0.01054 | 2 | 2 |
| full | 66.7% | +0.00797 | +0.01175 | +0.01141 | 2 | 2 |
| minus_mean_abs_error_high | 50.0% | +0.00701 | +0.01003 | +0.00856 | 1 | 2 |
| only_site_qwk_spread_high | 23.3% | +0.00128 | +0.00234 | +0.00228 | 0 | 2 |

Removing `mean_abs_error_high` reduces trigger rate but does not collapse the transfer result.

### 7.4 Calibration sensitivity

```text
low_quantile = 0.05, 0.10, 0.15
high_quantile = 0.75, 0.80, 0.85, 0.90
min_trigger_count = 2, 3, 4
```

Evaluated configurations: **36**  
Robust positive configurations: **29**

The original fixed rule, `low_0.1__high_0.8__min_3`, remains robust-positive, but it is not uniquely special.

---

## 8. Mathematical notation

### 8.1 FedAvg

Let there be `K` simulated clients. Client `k` has `n_k` local samples and produces update `Delta theta_k`. FedAvg uses:

```math
w_k = n_k / sum_j n_j
Delta theta = sum_k w_k Delta theta_k
```

The failure mode is that `n_k` is a sample-volume term, not an alignment term.

### 8.2 Site-signal alignment

Let `A_k` denote the task-specific alignment of client `k`'s training signal with the declared validation objective. The problematic condition is high `n_k` but reduced `A_k`.

### 8.3 Detector trigger

Let `d_1, ..., d_m` be validation diagnostics calibrated from clean FedAvg runs. Let `I_i(r)=1` if diagnostic `d_i` leaves its clean-calibrated safe range in run `r`, else `0`. The fixed detector triggers when:

```math
sum_i I_i(r) >= 3
```

For the headline detector:

```text
low_quantile = 0.10
high_quantile = 0.80
min_trigger_count = 3
use_entropy = false
```

### 8.4 Metric deltas

For metric `M`:

```math
Delta M(r) = M_switch(r) - M_clean(r)
```

Positive `Delta M` means the detector-switch policy improved over staying with the clean strategy.

---

## 9. Claim boundaries

Supported claims:

1. PCam patch-level pipeline works on real PCam data and reaches strong documented patch-level AUC.
2. PANDA Phikon feature bags support slide-level MIL benchmarking.
3. Stabilized TransnnMIL is competitive with AttentionMIL in the current PANDA feature-bag setup.
4. PathologyFL executes simulated-site federated workflows on real PCam-derived pathology data.
5. FAIR-WEIGHTS-H executes and produces distinct weight trajectories, but superiority over simpler baselines is not yet demonstrated.
6. FedAvg has a sample-volume / site-signal alignment failure mode in these simulated federated pathology experiments.
7. A fixed label-noise-calibrated detector transfers to conservative threshold-shift stress with low clean-regime switching and positive 35% / 45% gains.

Unsupported claims:

1. clinical readiness
2. diagnostic safety
3. real hospital federated deployment performance
4. universal detector calibration
5. FAIR-WEIGHTS-H proves fairness
6. institutional ranking or institutional reliability judgment
7. architecture superiority of TransnnMIL over all MIL baselines
8. that any real hospital, pathologist, or institution is inherently more or less trustworthy than another

---

## 10. Reproducibility artifacts

Primary dominant-site result files:

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_summary.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_run_diagnostics.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_diagnostic_summary/diagnostic_frequency_by_stress.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_leave_one_out/diagnostic_ablation_headline_35_45.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity/calibration_sensitivity_headline.csv
```

PANDA/TransnnMIL artifacts:

```text
docs/results/panda-transnnmil-stability.md
results/transnnmil_stability_summary/by_lr_summary.csv
scripts/experiments/aggregate_transnnmil_stability.py
```

PCam artifacts:

```text
docs/results/pcam-results.md
results/pcam_real/metrics.json
results/pcam_real/confusion_matrix.png
results/pcam_real/roc_curve.png
```

Framework documentation:

```text
docs/overview/index.md
docs/federated/pathologyfl.md
docs/theory/fair-weights-h.md
docs/research/dominant-site-federated-pathology-paper.md
docs/research/detector-diagnostic-ablation.md
```

Source repository:

```text
https://github.com/matthewvaishnav/computational-pathology-research
```

---

## 11. Short technical summary

The complete project is broader than the dominant-site paper. PCam validates the patch-level image pipeline, PANDA validates slide-level feature-bag MIL, TransnnMIL stabilization shows optimizer-sensitive but competitive behavior, PathologyFL provides federated execution infrastructure, FAIR-WEIGHTS-H defines an auditable institutional weighting protocol with conservative empirical status, and the dominant-site result provides the strongest current paper-style claim: raw sample count is not equivalent to task-specific site-signal alignment in simulated federated computational pathology.
