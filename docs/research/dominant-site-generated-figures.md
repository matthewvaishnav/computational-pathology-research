# Generated Dominant-Site Paper Figures

**Status:** generated figure index for the dominant-site federated pathology paper draft  
**Clinical status:** research-only; not clinically validated; not diagnostic software; not intended for patient-care use

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

These figures support the paper's detector-switch result stack:

1. the fixed detector transfers to conservative threshold shift;
2. clean-regime switching remains low;
3. 35% and 45% shift gains are statistically positive;
4. detector diagnostics are interpretable;
5. the result does not collapse when the strongest diagnostic family is removed;
6. the result is stable across nearby calibration settings.

---

## Next figure work

The remaining primary figures are:

```text
Figure 1: problem schematic
Figure 2: dominant-site stress overview
```

Figure 2 should be built after selecting the exact lightweight aggregate files to curate for label-noise and threshold-shift stress overview results.
