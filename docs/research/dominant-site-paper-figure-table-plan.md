# Dominant-Site Federated Pathology Paper Figure / Table Plan

**Status:** planning document for paper-style consolidation  
**Scope:** simulated federated PANDA experiments over pathology-derived Phikon features  
**Clinical status:** research-only; not clinically validated; not diagnostic software; not intended for patient-care use

---

## Core paper claim

Raw sample count is not the same as task-specific site-signal alignment. In simulated federated computational pathology over PANDA-derived Phikon features, FedAvg can become less safe when the largest simulated client's training signal is misaligned with the validation objective. Cross-site blending and a dominance-aware detector switch can reduce this risk under dominant-site label corruption and conservative ordinal threshold shift.

The strongest detector result is:

> A fixed label-noise-calibrated detector rule transfers to conservative ordinal threshold-shift stress. It keeps clean-regime switching low, produces statistically positive gains at 35% and 45% shift, survives leave-one-diagnostic-family-out ablation, and remains qualitatively stable across nearby calibration settings.

---

## Recommended paper structure

```text
1. Introduction
   - Problem: FedAvg silently equates sample count with aggregation authority.
   - Pathology risk: labels can vary by site due to thresholding, workflow, scanning, case mix, or grading practice.
   - Contribution: controlled simulated-site stress tests over real pathology-derived features.

2. Methods
   - Dataset/features: PANDA prostate grading, Phikon feature vectors, 10,611 readable feature files.
   - Federation simulation: multi-site split, dominant-site perturbation, clean validation labels.
   - Stress modes: random dominant-site label corruption and systematic ordinal threshold shift.
   - Strategies: FedAvg, cross-site blend, detector switch.
   - Detector: clean-calibrated diagnostics and trigger rule.

3. Results
   - FedAvg dominant-site failure mode under label-noise stress.
   - Transfer to conservative ordinal threshold shift.
   - Fixed detector transfer result.
   - Detector diagnostic ablation and calibration sensitivity.

4. Discussion
   - Sample-volume / site-signal alignment as an audit problem.
   - Why this is not hospital ranking.
   - Limitations and external validation needs.
```

---

## Figure 1: Problem schematic

**Title:** Sample volume is not the same as site-signal alignment

**Goal:** Explain the mechanism visually before showing metrics.

**Panel A:** Standard FedAvg assumption

```text
Client A: many samples  -> high aggregation weight
Client B: fewer samples -> lower aggregation weight
Client C: fewer samples -> lower aggregation weight
```

**Panel B:** Pathology failure mode

```text
Dominant client has many samples but shifted training labels.
Validation objective remains clean.
FedAvg amplifies the shifted signal because it weights by sample count.
```

**Panel C:** Detector-switch idea

```text
Monitor clean-calibrated validation diagnostics.
If enough diagnostics leave the safe range, switch away from sample-size dominance.
Otherwise keep FedAvg.
```

**Suggested file output:**

```text
figures/dominant-site-figure-1-problem-schematic.png
```

**Caption draft:**

> Standard FedAvg uses client sample count as aggregation authority. In computational pathology, a high-volume client can have a training-label process that is less aligned with the declared validation objective. The proposed detector switch treats sample-size dominance as an auditable assumption rather than an automatic guarantee of reliability.

---

## Figure 2: Dominant-site stress result overview

**Title:** Dominant-site training-signal shift exposes FedAvg vulnerability

**Goal:** Show the broad stress result across perturbation levels.

**Panels:**

- **A:** Label-noise stress: FedAvg vs cross-site blend global QWK by corruption level.
- **B:** Label-noise stress: worst-site QWK by corruption level.
- **C:** Conservative threshold shift: FedAvg vs cross-site blend global QWK by shift level.
- **D:** Conservative threshold shift: macro-F1 or worst-site QWK by shift level.

**Data sources likely needed:**

```text
results/fair_weights_h_stress_noise_*_aggregate/
results/threshold_shift_panda_all_conservative_*_15seed_aggregate/
```

**Suggested file output:**

```text
figures/dominant-site-figure-2-stress-overview.png
```

**Caption draft:**

> Across simulated dominant-site stress settings, sample-size-weighted aggregation becomes less reliable when the largest client's training signal is perturbed while validation labels remain clean. Cross-site blending improves robustness in corrupted regimes, especially under conservative ordinal threshold shift.

---

## Figure 3: Fixed detector transfer result

**Title:** A fixed label-noise-calibrated detector transfers to conservative threshold shift

**Goal:** Make the headline detector result visually clear.

**Panel A:** Trigger rate by conservative threshold-shift level

```text
0%: 13.3%
25%: 33.3%
35%: 60.0%
45%: 73.3%
```

**Panel B:** Global QWK delta vs clean baseline

```text
0%: -0.00025, CI crosses zero
25%: +0.00129, CI crosses zero
35%: +0.00542, CI [0.00062, 0.01022]
45%: +0.01053, CI [0.00239, 0.01866]
```

**Panel C:** Macro-F1 delta

```text
35%: +0.00838
45%: +0.01512
```

**Panel D:** Worst-site QWK delta

```text
35%: +0.00991
45%: +0.01290
```

**Data source:**

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_summary.csv
```

**Suggested file output:**

```text
figures/dominant-site-figure-3-detector-transfer.png
```

**Caption draft:**

> A fixed detector rule calibrated under dominant-site label-noise stress transferred to conservative ordinal threshold-shift stress. Clean-regime switching remained low at 13.3%, while 35% and 45% conservative shift produced statistically positive improvements across global QWK, macro-F1, and worst-site QWK.

---

## Figure 4: Detector interpretability and ablation

**Title:** Detector signal is distributed across ordinal-error and QWK diagnostics

**Goal:** Show the detector is not a single-feature trick.

**Panel A:** Diagnostic frequency bar chart

```text
mean_abs_error_high: 44
worst_site_qwk_low: 31
global_qwk_low: 27
severe_error_rate_high: 22
site_qwk_spread_high: 12
```

**Panel B:** Leave-one-diagnostic-family-out global QWK delta at 35% / 45%

Key rows:

```text
full: +0.00797
minus_mean_abs_error_high: +0.00701
only_mean_abs_error_high: +0.00865
only_site_qwk_spread_high: +0.00128
```

**Panel C:** Calibration-sensitivity summary

```text
Evaluated nearby configurations: 36
Robust-positive configurations: 29
```

Could be shown as a compact heatmap over:

```text
low_quantile x high_quantile x min_trigger_count
```

or as a stacked count:

```text
29 robust-positive / 36 tested
```

**Data sources:**

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_diagnostic_summary/diagnostic_frequency_by_stress.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_leave_one_out/diagnostic_ablation_headline_35_45.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity/calibration_sensitivity_headline.csv
```

**Suggested file output:**

```text
figures/dominant-site-figure-4-detector-ablation.png
```

**Caption draft:**

> Detector triggers were driven primarily by ordinal-error increase and QWK degradation. Removing the most frequent diagnostic family, mean absolute ordinal error, reduced trigger rate but did not collapse the 35% / 45% transfer result. A calibration-sensitivity sweep found that 29 of 36 nearby detector settings preserved low clean-regime switching and positive 35% / 45% gains.

---

## Table 1: Dataset and experiment setup

**Purpose:** Put all setup facts in one reproducibility table.

| Category | Value |
|---|---|
| Dataset | PANDA prostate cancer grading |
| Feature extractor | Phikon |
| Readable feature files | 10,611 |
| Feature dimension | 768 |
| Task | ISUP grade prediction, classes 0-5 |
| Federation | simulated multi-site setting |
| Validation labels | kept clean during stress experiments |
| Perturbed client | largest simulated client |
| Main metrics | global QWK, worst-site QWK, mean-site QWK, macro-F1, accuracy |
| Seeds | 15-seed stress studies |

---

## Table 2: Fixed detector transfer summary

**Purpose:** This is the headline quantitative table.

| Shift | Trigger rate | Global QWK delta | Global QWK CI | Macro-F1 delta | Macro-F1 CI | Worst-site QWK delta | Worst-site QWK CI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 13.3% | -0.00025 | [-0.00113, 0.00062] | +0.00009 | [-0.00039, 0.00057] | +0.00151 | [-0.00162, 0.00463] |
| 25% | 33.3% | +0.00129 | [-0.00175, 0.00432] | +0.00290 | [-0.00184, 0.00765] | +0.00353 | [-0.00428, 0.01133] |
| 35% | 60.0% | +0.00542 | [0.00062, 0.01022] | +0.00838 | [0.00272, 0.01405] | +0.00991 | [0.00169, 0.01813] |
| 45% | 73.3% | +0.01053 | [0.00239, 0.01866] | +0.01512 | [0.00819, 0.02204] | +0.01290 | [0.00547, 0.02034] |

**Source:**

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_summary.csv
```

---

## Table 3: Diagnostic ablation summary

**Purpose:** Support the claim that the detector is interpretable and not one hand-picked diagnostic.

| Variant | Mean trigger rate 35/45 | Mean global QWK delta 35/45 | Mean macro-F1 delta 35/45 | Mean worst-site QWK delta 35/45 | Significant global-QWK regimes |
|---|---:|---:|---:|---:|---:|
| full | 66.7% | +0.00797 | +0.01175 | +0.01141 | 2 |
| minus_mean_abs_error_high | 50.0% | +0.00701 | +0.01003 | +0.00856 | 1 |
| only_mean_abs_error_high | 96.7% | +0.00865 | +0.01462 | +0.01054 | 2 |
| only_site_qwk_spread_high | 23.3% | +0.00128 | +0.00234 | +0.00228 | 0 |

**Source:**

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_leave_one_out/diagnostic_ablation_headline_35_45.csv
```

---

## Table 4: Calibration sensitivity

**Purpose:** Support the claim that the result is not a single-threshold artifact.

| Sweep component | Values |
|---|---|
| low_quantile | 0.05, 0.10, 0.15 |
| high_quantile | 0.75, 0.80, 0.85, 0.90 |
| min_trigger_count | 2, 3, 4 |
| total configurations | 36 |
| robust-positive configurations | 29 |
| clean trigger-rate cutoff | <= 20% |
| target stress levels | 35%, 45% conservative shift |

**Source:**

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity/calibration_sensitivity_headline.csv
```

---

## Supplementary tables

### Supplementary Table S1: Full per-seed detector diagnostics

**Source:**

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_run_diagnostics.csv
```

### Supplementary Table S2: Full leave-one-out per-run diagnostics

**Source:**

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_leave_one_out/diagnostic_ablation_per_run.csv
```

### Supplementary Table S3: Full calibration sweep by stress level

**Source:**

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity/calibration_sensitivity_summary_by_stress.csv
```

---

## Claims supported by this figure/table stack

Supported:

1. FedAvg sample-size dominance can be unsafe under simulated dominant-site training-signal misalignment.
2. The conservative ordinal threshold-shift setting is the strongest systematic-bias transfer result.
3. A fixed label-noise-calibrated detector transfers to conservative threshold-shift stress.
4. Clean-regime switching is low for the fixed rule.
5. 35% and 45% conservative shift produce statistically positive detector gains across global QWK, macro-F1, and worst-site QWK.
6. The detector is interpretable: it is mainly driven by ordinal-error and QWK degradation diagnostics.
7. The detector is not a single-diagnostic artifact: removing `mean_abs_error_high` does not collapse the positive 35% / 45% transfer result.
8. The detector is not a single-threshold artifact: 29 of 36 nearby calibration settings remain robust-positive.

Not supported:

1. Clinical readiness.
2. Real hospital deployment performance.
3. Universal detector calibration.
4. Institutional reliability ranking.
5. Architecture superiority claims unrelated to the federated detector result.

---

## Immediate next implementation step

Create one figure-generation script:

```text
scripts/figures/make_dominant_site_paper_figures.py
```

It should produce:

```text
figures/dominant-site-figure-3-detector-transfer.png
figures/dominant-site-figure-4-detector-ablation.png
```

Start with Figures 3 and 4 because their data sources are already compact and committed. Figure 2 can follow after deciding exactly which aggregate folders to curate.

---

## Recommended next command

After pulling the repo, generate and commit the calibration-sensitivity outputs first:

```powershell
git pull

git add results\threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity\calibration_sensitivity_summary_by_stress.csv
git add results\threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity\calibration_sensitivity_headline.csv
git add results\threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity\calibration_sensitivity_thresholds.csv
git add results\threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity\calibration_sensitivity_config.json

git commit -m "Add detector calibration sensitivity outputs"
git push
```

Then implement the figure-generation script.
