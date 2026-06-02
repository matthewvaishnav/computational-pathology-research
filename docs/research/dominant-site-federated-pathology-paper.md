# When More Data Is Less Trustworthy

## Site-signal alignment failure modes in federated computational pathology

**Author:** Matthew Vaishnav  
**Status:** independent technical report / working paper draft  
**Scope:** simulated federated learning experiments over pathology-derived feature vectors  
**Clinical status:** research-only; not clinically validated; not diagnostic software; not intended for patient-care use

---

## Abstract

Federated learning is often proposed for medical AI because institutions can collaborate without centralizing raw patient data. A common aggregation baseline, FedAvg, weights client updates by sample count. This is convenient, but in computational pathology it can silently encode a risky assumption: more samples should imply more aggregation authority. Pathology labels may reflect local grading thresholds, staining and scanning workflows, case mix, annotation practices, pathologist disagreement, or historical reporting policy. A high-volume site can therefore contribute many samples while also producing a training signal that is less aligned with the declared validation objective.

This report studies that failure mode using simulated federations over PANDA-derived Phikon feature representations for prostate cancer grading. Validation labels are kept clean while the largest simulated client's training signal is perturbed through dominant-site label corruption and systematic ordinal threshold shift. The central question is whether sample-size-dominant aggregation becomes less safe when the dominant client's training signal is misaligned, and whether dominance-aware switching can reduce that risk.

The strongest result is a fixed detector-switch rule calibrated under dominant-site label-noise stress and evaluated on conservative ordinal threshold-shift stress. The rule keeps clean-regime switching low at 13.3% and produces statistically positive improvements at 35% and 45% conservative shift across global QWK, macro-F1, and worst-site QWK. Diagnostic analysis shows that detector triggers are mainly driven by ordinal-error increase and QWK degradation, not by a single site-spread heuristic. Leave-one-diagnostic-family-out ablation shows that removing the most frequent diagnostic, mean absolute ordinal error, reduces trigger rate but does not collapse the positive 35% / 45% transfer result. Calibration-sensitivity analysis finds that 29 of 36 nearby detector settings preserve low clean-regime switching and positive 35% / 45% gains across global QWK, macro-F1, and worst-site QWK.

These results do not establish clinical readiness or real hospital deployment performance. They support a narrower claim: in simulated federated pathology experiments over real pathology-derived features, raw sample count is not equivalent to task-specific site-signal alignment, and sample-size dominance should be treated as an auditable modeling assumption rather than an automatic guarantee of aggregation safety.

---

## 1. Introduction

Federated learning is attractive in medicine because raw patient data is difficult, risky, and often impossible to centralize. In pathology, the motivation is especially clear: whole-slide images and derived feature representations may be governed by institutional policy, patient privacy requirements, storage constraints, and clinical governance. Federated learning offers a way for multiple sites to train collaboratively while keeping raw local data inside each environment.

However, federated learning does not remove the question of influence. It changes the question from "who owns the data?" to "how much should each site shape the shared model?" Standard FedAvg answers that question with a simple rule: clients with more samples receive more aggregation influence.

That rule is not obviously safe in computational pathology. More samples can coexist with a training signal that is less aligned with the declared validation objective. A large site may differ because of grading thresholds, staining or scanning workflow, case mix, annotation source, label workflow, local reporting practice, or patient population. These differences are not evidence that an institution is worse or less trustworthy. They are evidence that sample volume and task-specific training-signal alignment are different quantities.

This report studies a controlled version of that problem. The experiments simulate multi-site federated learning over PANDA-derived Phikon feature vectors. The largest simulated site is perturbed while validation labels remain clean. The goal is to test whether FedAvg becomes vulnerable when its sample-size assumption is violated, and whether a detector-switch mechanism can identify when sample-size dominance has become unsafe.

The working thesis is:

> In federated computational pathology, raw sample count is not the same as task-specific site-signal alignment. FedAvg can become less safe when the largest simulated pathology client has a training-label process that is misaligned with the validation objective, and dominance-aware aggregation or switching can reduce that risk under controlled stress.

---

## 2. Contributions

This report makes five main contributions.

1. **A sample-volume / site-signal alignment failure-mode framing.** The report reframes dominant-client failure not as institutional reliability ranking, but as an audit of a modeling assumption that FedAvg already makes silently: larger client equals more influence.

2. **A simulated federated pathology stress test over real pathology-derived features.** The experiments use PANDA-derived Phikon feature vectors and perturb the dominant simulated site's training signal while keeping validation labels clean.

3. **Two stress modes.** The study examines dominant-site label corruption and systematic ordinal threshold shift. The latter is intended to approximate a more pathology-plausible failure mode than purely random label corruption.

4. **A fixed detector-switch transfer result.** A detector rule calibrated under label-noise stress transfers to conservative ordinal threshold shift, with low clean-regime switching and positive 35% / 45% shift gains.

5. **Diagnostic and calibration robustness checks.** The detector result is supported by diagnostic-frequency analysis, leave-one-diagnostic-family-out ablation, and calibration-sensitivity analysis over nearby detector settings.

---

## 3. Ethical framing: site-signal alignment, not institutional worth

This work does **not** claim that some hospitals, pathologists, or institutions are inherently more reliable, competent, or trustworthy than others.

The term **site-signal alignment** is used in a narrow modeling sense: whether a simulated client's training signal appears aligned with the declared validation objective under a specific experimental setup.

A client may appear misaligned for many non-blameworthy reasons, including:

- different grading thresholds
- staining or scanning protocol differences
- local case mix
- patient-population differences
- annotation workflow differences
- label-source differences
- historical reporting practice
- local clinical policy
- pathologist disagreement

Dominance-aware aggregation should therefore not be interpreted as an institutional ranking system. It is an audit mechanism for a modeling assumption that FedAvg already makes silently:

> larger client = more influence.

The ethical purpose of this work is to make that assumption visible, stress-testable, and contestable. Any real deployment would require governance, local clinical review, pathologist input, bias auditing, institutional agreement on validation objectives, prospective validation, security review, and regulatory review.

---

## 4. Experimental setup

The experiments use PANDA-derived Phikon feature representations and simulate multi-site federated learning over pathology-derived feature vectors.

| Category | Value |
|---|---|
| Dataset | PANDA prostate cancer grading |
| Feature extractor | Phikon |
| Readable feature files | 10,611 |
| Feature dimension | 768 |
| Task | ISUP grade prediction, classes 0-5 |
| Federation | simulated multi-site setting |
| Perturbed client | largest simulated client |
| Validation labels | kept clean during stress experiments |
| Metrics | global QWK, worst-site QWK, mean-site QWK, macro-F1, accuracy |
| Seeds | 15-seed stress studies |

Quadratic weighted kappa (QWK) is used because ISUP prostate grading is ordinal. Confusing adjacent grades is less severe than confusing grade 0 with grade 5, and QWK better reflects that ordinal structure than plain accuracy alone.

The experimental design keeps validation labels clean. Perturbations are applied to the largest simulated site's training labels. This isolates the question of whether aggregation remains safe when a high-volume client's training signal becomes less aligned with the target validation objective.

---

## 5. Stress modes

### 5.1 Dominant-site label corruption

The first stress mode corrupts labels at the largest simulated site. This creates a controlled failure mode where the site with the most samples remains influential under FedAvg even though its training labels become less aligned with the validation objective.

The observed pattern is conditional:

- clean setting: FedAvg remains strong and should not be automatically replaced
- corrupted dominant-site setting: cross-site blending improves robustness
- detector-switch setting: a clean-calibrated rule can switch away from FedAvg in unsafe regimes

This result should not be interpreted as "cross-site blending is always better." The claim is narrower: when the dominant client's training signal is made less aligned, pure sample-size weighting becomes less safe.

### 5.2 Systematic ordinal threshold shift

Random corruption is useful for exposing a mechanism, but pathology disagreement is often systematic rather than random. The second stress mode applies ordinal threshold shift to the dominant site's training labels.

Two directions are considered:

```text
Aggressive shift:
  selected labels move upward by one ISUP grade when possible

Conservative shift:
  selected labels move downward by one ISUP grade when possible
```

The conservative threshold-shift result is the strongest transfer setting. It is treated as the headline systematic-bias result in this report. Aggressive threshold shift is weaker and should not be presented as the main claim.

---

## 6. Aggregation and detector-switch logic

FedAvg encodes a simple statistical assumption:

> more samples = more authority.

The detector-switch approach treats that as an assumption to audit rather than an axiom. The detector follows this logic:

```text
1. Calibrate normal FedAvg validation behavior on clean runs.
2. Monitor validation diagnostics.
3. If enough diagnostics leave the clean-calibrated safe range, switch away from sample-size dominance.
4. Otherwise, keep FedAvg.
```

The fixed detector rule evaluated in the conservative threshold-shift transfer experiment was:

```text
low_quantile = 0.10
high_quantile = 0.80
min_trigger_count = 3
use_entropy = false
```

The detector diagnostics include global QWK, worst-site QWK, site-QWK spread, mean absolute ordinal error, and severe ordinal error rate. Prediction entropy is available but was not used in the fixed headline rule.

The detector is intended to answer a narrow question:

> Can observable validation diagnostics indicate when sample-size dominance has become unsafe under site-specific training-signal shift?

---

## 7. Fixed detector transfer result

A fixed label-noise-calibrated detector rule was evaluated on conservative threshold-shift stress. The result is summarized below.

| Conservative shift | Trigger rate | Global QWK delta | 95% CI | Macro-F1 delta | 95% CI | Worst-site QWK delta | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 13.3% | -0.00025 | [-0.00113, 0.00062] | +0.00009 | [-0.00039, 0.00057] | +0.00151 | [-0.00162, 0.00463] |
| 25% | 33.3% | +0.00129 | [-0.00175, 0.00432] | +0.00290 | [-0.00184, 0.00765] | +0.00353 | [-0.00428, 0.01133] |
| 35% | 60.0% | +0.00542 | [0.00062, 0.01022] | +0.00838 | [0.00272, 0.01405] | +0.00991 | [0.00169, 0.01813] |
| 45% | 73.3% | +0.01053 | [0.00239, 0.01866] | +0.01512 | [0.00819, 0.02204] | +0.01290 | [0.00547, 0.02034] |

Deltas are detector-switch performance minus the clean-strategy baseline. Positive values indicate that the detector-switch policy improved over staying with the clean strategy.

The clean 0% conservative-shift regime has low switching at 13.3%, with near-zero global QWK cost and a confidence interval crossing zero. At 25% shift, the result is directionally positive but not statistically clean because the confidence intervals cross zero. The 35% and 45% regimes are the headline positive transfer results: both show statistically positive improvements across global QWK, macro-F1, and worst-site QWK.

The supported interpretation is:

> A single fixed detector rule calibrated under dominant-site label-noise stress transfers to conservative ordinal threshold-shift stress, with low clean-regime switching and statistically positive gains at 35% and 45% shift across global QWK, macro-F1, and worst-site QWK.

---

## 8. Detector diagnostic analysis

The next question is whether the detector is interpretable. If it only triggers because of one arbitrary metric, the result is less convincing. The diagnostic-frequency analysis summarizes which clean-calibrated diagnostics were crossed by the fixed detector.

| Diagnostic | Count |
|---|---:|
| mean_abs_error_high | 44 |
| worst_site_qwk_low | 31 |
| global_qwk_low | 27 |
| severe_error_rate_high | 22 |
| site_qwk_spread_high | 12 |

The detector is driven primarily by ordinal-error and QWK degradation signals, not by site-spread alone. This is mechanistically sensible in a conservative ordinal threshold-shift setting. If labels are systematically shifted downward, mean absolute ordinal error should be an informative warning signal.

Triggered runs also show increasing shift severity. Trigger rates rise from 13.3% at 0% shift to 73.3% at 45% shift. When the detector triggers, it usually crosses approximately four diagnostic thresholds.

| Conservative threshold shift | Runs | Triggered runs | Trigger rate | Mean failure count | Mean failure count when triggered |
|---:|---:|---:|---:|---:|---:|
| 0% | 15 | 2 | 13.3% | 0.87 | 3.50 |
| 25% | 15 | 5 | 33.3% | 2.07 | 3.80 |
| 35% | 15 | 9 | 60.0% | 2.87 | 3.89 |
| 45% | 15 | 11 | 73.3% | 3.27 | 3.91 |

At 35% and 45% shift, triggered runs are strongly positive across global QWK, macro-F1, worst-site QWK, mean-site QWK, and global accuracy. The clean 0% regime shows a small clean-regime global-QWK cost among triggered cases, which is expected in a detector-switch setting.

---

## 9. Leave-one-diagnostic-family-out ablation

The most frequent diagnostic is `mean_abs_error_high`. The key ablation question is whether the fixed-rule transfer result collapses if this diagnostic is removed.

Headline comparison over the 35% and 45% conservative threshold-shift regimes:

| Variant | Mean trigger rate | Mean global QWK delta | Mean macro-F1 delta | Mean worst-site QWK delta | Significant global-QWK regimes | Positive global-QWK regimes |
|---|---:|---:|---:|---:|---:|---:|
| only_mean_abs_error_high | 96.7% | +0.00865 | +0.01462 | +0.01054 | 2 | 2 |
| full | 66.7% | +0.00797 | +0.01175 | +0.01141 | 2 | 2 |
| minus_site_qwk_spread_high | 60.0% | +0.00797 | +0.01124 | +0.01092 | 2 | 2 |
| only_global_qwk_low | 60.0% | +0.00797 | +0.01124 | +0.01092 | 2 | 2 |
| minus_worst_site_qwk_low | 53.3% | +0.00770 | +0.01011 | +0.00940 | 2 | 2 |
| only_severe_error_rate_high | 50.0% | +0.00757 | +0.00958 | +0.00847 | 2 | 2 |
| minus_severe_error_rate_high | 63.3% | +0.00728 | +0.01167 | +0.01057 | 1 | 2 |
| only_worst_site_qwk_low | 76.7% | +0.00709 | +0.01357 | +0.00974 | 1 | 2 |
| minus_global_qwk_low | 56.7% | +0.00701 | +0.01054 | +0.00904 | 2 | 2 |
| minus_mean_abs_error_high | 50.0% | +0.00701 | +0.01003 | +0.00856 | 1 | 2 |
| only_site_qwk_spread_high | 23.3% | +0.00128 | +0.00234 | +0.00228 | 0 | 2 |

The answer is clear: removing `mean_abs_error_high` reduces trigger rate but does not collapse the transfer result. The `minus_mean_abs_error_high` variant remains positive across the 35% and 45% conservative-shift regimes:

```text
minus_mean_abs_error_high:
  mean trigger rate, 35/45: 50.0%
  mean global QWK delta, 35/45: +0.00701
  mean macro-F1 delta, 35/45: +0.01003
  mean worst-site QWK delta, 35/45: +0.00856
  positive global-QWK regimes: 2 / 2
```

This suggests that the detector is not a one-feature shortcut. The signal is distributed across multiple diagnostics, especially QWK degradation and severe ordinal-error signals.

The weakest single-family variant is `only_site_qwk_spread_high`, which produces much smaller mean gains and no significant global-QWK regimes. This weakens the simplistic interpretation that the detector is primarily a site-spread detector. In this transfer setting, site spread is secondary.

---

## 10. Calibration-sensitivity analysis

The fixed detector should not be treated as convincing if it only works at one hand-picked threshold setting. To test this, nearby detector settings were swept:

```text
low_quantile = 0.05, 0.10, 0.15
high_quantile = 0.75, 0.80, 0.85, 0.90
min_trigger_count = 2, 3, 4
```

This produced 36 detector configurations. A configuration was counted as robust-positive if it preserved:

```text
clean trigger rate <= 20%
positive global-QWK deltas at both 35% and 45% conservative shift
positive macro-F1 deltas at both 35% and 45% conservative shift
positive worst-site-QWK deltas at both 35% and 45% conservative shift
```

Result:

```text
Evaluated configurations: 36
Robust positive configurations: 29
```

Top configurations by 35% / 45% global-QWK transfer strength included:

| Config | Clean trigger rate | Mean target trigger rate | Mean global QWK delta | Mean macro-F1 delta | Mean worst-site QWK delta | Significant global-QWK regimes | Robust positive? |
|---|---:|---:|---:|---:|---:|---:|---|
| low_0.15__high_0.8__min_3 | 20.0% | 70.0% | +0.00807 | +0.01246 | +0.01129 | 2 | yes |
| low_0.15__high_0.85__min_3 | 20.0% | 66.7% | +0.00802 | +0.01209 | +0.01149 | 2 | yes |
| low_0.15__high_0.9__min_3 | 13.3% | 66.7% | +0.00802 | +0.01209 | +0.01149 | 2 | yes |
| low_0.05__high_0.75__min_2 | 20.0% | 80.0% | +0.00799 | +0.01263 | +0.01103 | 2 | yes |
| low_0.05__high_0.8__min_2 | 20.0% | 80.0% | +0.00799 | +0.01263 | +0.01103 | 2 | yes |
| low_0.1__high_0.8__min_3 | 13.3% | 66.7% | +0.00797 | +0.01175 | +0.01141 | 2 | yes |

The original fixed rule, `low_0.1__high_0.8__min_3`, remains robust-positive, but it is not uniquely special. Many nearby configurations preserve the same qualitative behavior.

This strengthens the result because the detector-transfer finding is not tightly dependent on one exact threshold configuration.

---

## 11. Figures and tables

The report is designed around four primary figures and four main tables.

### Figure 1: Problem schematic

**Purpose:** Explain the mechanism visually before showing metrics.

```text
Standard FedAvg:
  more samples -> more aggregation weight

Pathology failure mode:
  dominant client has many samples but shifted training labels
  validation objective remains clean
  FedAvg amplifies shifted signal because it weights by sample count

Detector switch:
  monitor clean-calibrated diagnostics
  switch away from sample-size dominance when enough diagnostics leave safe range
```

### Figure 2: Dominant-site stress result overview

**Purpose:** Show the broad stress result across perturbation levels.

Candidate panels:

- label-noise stress: FedAvg vs cross-site blend global QWK
- label-noise stress: worst-site QWK
- conservative threshold shift: FedAvg vs cross-site blend global QWK
- conservative threshold shift: macro-F1 or worst-site QWK

### Figure 3: Fixed detector transfer result

**Purpose:** Show the headline detector-transfer result.

Data source:

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_summary.csv
```

Suggested output:

```text
figures/dominant-site-figure-3-detector-transfer.png
```

### Figure 4: Detector interpretability and ablation

**Purpose:** Show that the detector is not a one-feature or one-threshold trick.

Data sources:

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_diagnostic_summary/diagnostic_frequency_by_stress.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_leave_one_out/diagnostic_ablation_headline_35_45.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity/calibration_sensitivity_headline.csv
```

Suggested output:

```text
figures/dominant-site-figure-4-detector-ablation.png
```

---

## 12. Discussion

The main scientific point is not that cross-site blending is always superior to FedAvg. The main point is conditional: FedAvg can become less safe when the dominant client's training signal becomes misaligned with the validation objective, because FedAvg continues to assign that client high influence based on sample count.

In computational pathology, this matters because site-level training signals can differ for reasons that are not captured by sample count. A large site can contribute valuable data, but volume alone does not prove alignment with a particular validation objective. This is why aggregation influence should be auditable.

The detector-switch result suggests one possible audit mechanism. Rather than replacing FedAvg everywhere, the detector preserves FedAvg in clean regimes and switches away when enough clean-calibrated diagnostics indicate degraded alignment. The clean-regime switch rate remains low in the conservative threshold-shift transfer result, while 35% and 45% shift regimes show positive gains.

The diagnostic analyses make the detector more credible. It is mainly driven by ordinal-error and QWK degradation signals, which are mechanistically connected to ordinal threshold shift. It does not collapse when the most frequent diagnostic is removed. It also remains stable across a neighborhood of calibration settings.

This combination is more convincing than a single headline table because it addresses three failure modes of analysis:

1. The detector might be a one-metric shortcut.
2. The detector might be one hand-picked threshold.
3. The detector might only help by over-switching in clean regimes.

The current evidence reduces, but does not eliminate, those concerns.

---

## 13. Limitations

The limitations are substantial.

First, this is a simulated federation over pathology-derived feature vectors, not a real multi-hospital deployment. Site identity is simulated, and the perturbations are controlled. The result may not hold unchanged under naturally occurring site distributions.

Second, the data are derived from PANDA prostate cancer grading. A stronger claim would require external validation on real multi-center pathology data, such as Camelyon17 or another dataset with natural center identity.

Third, the detector uses validation diagnostics. Any real deployment would need to define how validation data are governed, where diagnostics are computed, and whether those diagnostics are allowed to leave institutional environments.

Fourth, the conservative threshold-shift result is stronger than the aggressive threshold-shift result. The paper should headline conservative shift and treat aggressive shift as weaker or supplementary.

Fifth, the detector is not universally calibrated. The calibration-sensitivity analysis shows robustness in a local neighborhood of settings for this experiment. It does not prove universal detector calibration across datasets, diseases, institutions, scanners, or label policies.

Sixth, the experiments do not establish clinical readiness, diagnostic safety, regulatory compliance, or deployment suitability.

---

## 14. Claim boundaries

Supported claims:

1. FedAvg has a sample-volume / site-signal alignment failure mode in these simulated federated pathology experiments.
2. The failure mode appears when the largest simulated client's training signal is made less aligned with the validation objective while validation labels remain clean.
3. Cross-site blending improves robustness under dominant-site label corruption and conservative ordinal threshold shift.
4. A fixed label-noise-calibrated detector transfers to conservative threshold-shift stress with low clean-regime switching and positive 35% / 45% shift gains.
5. The detector is interpretable: it is mainly driven by ordinal-error increase and QWK degradation diagnostics.
6. Leave-one-diagnostic-family-out ablation shows that removing `mean_abs_error_high` does not collapse the 35% / 45% transfer result.
7. Calibration-sensitivity analysis shows that 29 of 36 nearby detector settings preserve the qualitative result.

Unsupported claims:

1. clinical readiness
2. diagnostic safety
3. real hospital federated deployment performance
4. universal detector calibration
5. institutional ranking or institutional reliability judgment
6. that any real hospital, pathologist, or institution is inherently more or less trustworthy than another
7. that the same effect will hold unchanged across every pathology dataset

---

## 15. Reproducibility artifacts

Primary result files:

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_config.json
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_summary.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_thresholds.json
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_run_diagnostics.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/detector_grid_summary.csv
```

Diagnostic summary files:

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_diagnostic_summary/trigger_summary_by_stress.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_diagnostic_summary/diagnostic_frequency_by_stress.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_diagnostic_summary/diagnostic_outcomes_by_stress.csv
```

Leave-one-out ablation files:

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_leave_one_out/diagnostic_ablation_per_run.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_leave_one_out/diagnostic_ablation_summary_by_stress.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_leave_one_out/diagnostic_ablation_headline_35_45.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_leave_one_out/diagnostic_ablation_config.json
```

Calibration-sensitivity files:

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity/calibration_sensitivity_summary_by_stress.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity/calibration_sensitivity_headline.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity/calibration_sensitivity_thresholds.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity/calibration_sensitivity_config.json
```

Figure-generation script:

```text
scripts/figures/make_dominant_site_paper_figures.py
```

Planned figures:

```text
figures/dominant-site-figure-3-detector-transfer.png
figures/dominant-site-figure-4-detector-ablation.png
```

---

## 16. Short technical summary

FedAvg is useful, but it silently equates sample count with aggregation authority. In pathology, sample volume and task-specific site-signal alignment can diverge. These experiments simulate that divergence by perturbing the dominant client's training signal while keeping validation labels clean. Under conservative ordinal threshold shift, a fixed label-noise-calibrated detector switch keeps clean-regime switching low and improves global QWK, macro-F1, and worst-site QWK at 35% and 45% shift. Diagnostic ablation and calibration-sensitivity analysis suggest that the detector is not merely a one-feature or one-threshold artifact. The result supports sample-volume / site-signal alignment auditing as a research direction for federated computational pathology.
