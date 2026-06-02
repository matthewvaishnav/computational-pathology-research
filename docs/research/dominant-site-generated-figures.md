# Generated Dominant-Site Paper Figures

**Status:** generated figure index for the dominant-site federated pathology paper draft  
**Clinical status:** research-only; not clinically validated; not diagnostic software; not intended for patient-care use

---

## Figure 1: Problem schematic

![Figure 1: Sample volume is not the same as site-signal alignment](../../figures/dominant-site-figure-1-problem-schematic.png)

**Source:**

```text
scripts/figures/make_dominant_site_schematic.py
```

**Caption draft:**

Standard FedAvg uses sample count as aggregation authority. In computational pathology, a high-volume client can have a training-label process that is less aligned with the declared validation objective. The detector-switch framing treats sample-size dominance as an auditable modeling assumption rather than an automatic guarantee of reliability.

---

## Figure 2: Dominant-site stress overview

![Figure 2: Dominant-site stress overview](../../figures/dominant-site-figure-2-stress-overview.png)

**Source data:**

```text
results/threshold_shift_panda_all_conservative_0_15seed_aggregate/aggregate_summary.csv
results/threshold_shift_panda_all_conservative_25_15seed_aggregate/aggregate_summary.csv
results/threshold_shift_panda_all_conservative_35_15seed_aggregate/aggregate_summary.csv
results/threshold_shift_panda_all_conservative_45_15seed_aggregate/aggregate_summary.csv
results/ordinal_harm_fair_weights_h_panda_all_noise_0_aggregate/aggregate_summary.csv
results/ordinal_harm_fair_weights_h_panda_all_noise_25_aggregate/aggregate_summary.csv
results/ordinal_harm_fair_weights_h_panda_all_noise_35_aggregate/aggregate_summary.csv
results/ordinal_harm_fair_weights_h_panda_all_noise_45_aggregate/aggregate_summary.csv
```

**Caption draft:**

Dominant-site stress overview across label-noise and conservative threshold-shift settings. The figure summarizes strategy performance relative to FedAvg across perturbation levels, highlighting the conditional nature of the result: cross-site or adaptive alternatives are not automatically superior in clean regimes, but become useful when dominant-site training-signal misalignment is present.

---

## Figure 3: Fixed detector transfer result

![Figure 3: Fixed detector transfer to conservative ordinal threshold shift](../../figures/dominant-site-figure-3-detector-transfer.png)

**Source data:**

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_summary.csv
```

**Caption draft:**

A fixed detector rule calibrated under dominant-site label-noise stress transferred to conservative ordinal threshold-shift stress. Clean-regime switching remained low at 13.3%, while 35% and 45% conservative shift produced statistically positive improvements across global QWK, macro-F1, and worst-site QWK.

---

## Figure 4: Detector interpretability, ablation, and calibration robustness

![Figure 4: Detector interpretability, ablation, and calibration robustness](../../figures/dominant-site-figure-4-detector-ablation.png)

**Source data:**

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_diagnostic_summary/diagnostic_frequency_by_stress.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_leave_one_out/diagnostic_ablation_headline_35_45.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity/calibration_sensitivity_headline.csv
```

**Caption draft:**

Detector triggers were driven primarily by ordinal-error increase and QWK degradation. Removing the most frequent diagnostic family, mean absolute ordinal error, reduced trigger rate but did not collapse the 35% / 45% transfer result. A calibration-sensitivity sweep found that 29 of 36 nearby detector settings preserved low clean-regime switching and positive 35% / 45% gains.

---

## Interpretation

These figures support the paper's dominant-site result stack:

1. sample-size dominance and site-signal alignment are different quantities;
2. dominant-site stress can expose FedAvg vulnerability when those quantities diverge;
3. the fixed detector transfers to conservative threshold shift;
4. clean-regime switching remains low;
5. 35% and 45% shift gains are statistically positive;
6. detector diagnostics are interpretable;
7. the result does not collapse when the strongest diagnostic family is removed;
8. the result is stable across nearby calibration settings.

---

## Next figure work

The four primary figures are now generated.

Remaining optional figure work:

```text
Supplementary Figure S1: aggressive threshold-shift detector results
Supplementary Figure S2: full calibration-sensitivity heatmap
Supplementary Figure S3: per-seed detector-regret distribution
```
