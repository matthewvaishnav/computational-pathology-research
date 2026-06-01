# Detector Diagnostic Ablation Summary

**Status:** diagnostic summary and leave-one-diagnostic-family-out ablation over fixed conservative threshold-shift detector transfer  
**Scope:** simulated federated PANDA experiments over pathology-derived Phikon features  
**Clinical status:** research-only; not clinically validated; not diagnostic software; not intended for patient-care use

---

## Purpose

This note summarizes which clean-calibrated FedAvg diagnostics actually drive the fixed dominance-aware detector in the conservative threshold-shift transfer experiment.

The detector rule was:

```text
low_quantile = 0.10
high_quantile = 0.80
min_trigger_count = 3
use_entropy = false
```

The diagnostic summary was produced with:

```powershell
python scripts\experiments\summarize_detector_diagnostics.py `
  --diagnostics results\threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed\best_detector_run_diagnostics.csv `
  --out-dir results\threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_diagnostic_summary
```

The leave-one-diagnostic-family-out ablation was produced with:

```powershell
python scripts\experiments\ablate_detector_diagnostics.py `
  --diagnostics results\threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed\best_detector_run_diagnostics.csv `
  --thresholds results\threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed\best_detector_thresholds.json `
  --out-dir results\threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_leave_one_out
```

---

## Trigger-rate pattern

| Conservative threshold shift | Runs | Triggered runs | Trigger rate | Mean failure count | Mean failure count when triggered |
|---:|---:|---:|---:|---:|---:|
| 0% | 15 | 2 | 13.3% | 0.87 | 3.50 |
| 25% | 15 | 5 | 33.3% | 2.07 | 3.80 |
| 35% | 15 | 9 | 60.0% | 2.87 | 3.89 |
| 45% | 15 | 11 | 73.3% | 3.27 | 3.91 |

The detector becomes more likely to trigger as conservative threshold shift increases. When it does trigger, it usually crosses approximately four diagnostic thresholds.

---

## Metric impact when triggered

| Conservative threshold shift | Mean global QWK delta when triggered | Mean macro-F1 delta when triggered | Mean worst-site QWK delta when triggered | Mean global accuracy delta when triggered |
|---:|---:|---:|---:|---:|
| 0% | -0.00191 | +0.00065 | +0.01131 | -0.00353 |
| 25% | +0.00386 | +0.00871 | +0.01059 | +0.00668 |
| 35% | +0.00903 | +0.01397 | +0.01652 | +0.01213 |
| 45% | +0.01436 | +0.02062 | +0.01759 | +0.01746 |

The clean 0% regime shows the expected tradeoff: a small clean-regime global-QWK cost among triggered cases, but no broad evidence of harmful switching. At 35% and 45% shift, triggered runs are strongly positive across global QWK, macro-F1, worst-site QWK, mean-site QWK, and global accuracy.

---

## Diagnostic frequency

Across the fixed-detector conservative threshold-shift transfer runs, trigger diagnostics appeared with the following frequencies:

| Diagnostic | Count |
|---|---:|
| mean_abs_error_high | 44 |
| worst_site_qwk_low | 31 |
| global_qwk_low | 27 |
| severe_error_rate_high | 22 |
| site_qwk_spread_high | 12 |

The detector is therefore driven primarily by ordinal-error and QWK degradation signals, not by site-spread alone.

Most frequent signal:

```text
mean_abs_error_high
```

Secondary signals:

```text
worst_site_qwk_low
global_qwk_low
severe_error_rate_high
```

Least frequent signal in this summary:

```text
site_qwk_spread_high
```

---

## Leave-one-diagnostic-family-out result

The key ablation question was:

> Does the fixed-rule transfer result still hold if the strongest diagnostic, `mean_abs_error_high`, is removed?

The answer is yes: removing `mean_abs_error_high` reduced trigger rate, but did not collapse the 35% / 45% transfer signal.

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

---

## Interpretation

The detector is **not** dependent on a single diagnostic family.

`mean_abs_error_high` is the most frequent and strongest single diagnostic in this conservative ordinal-shift setting. The `only_mean_abs_error_high` variant has the highest trigger rate and strong 35% / 45% mean deltas.

However, the `minus_mean_abs_error_high` variant remains positive:

```text
minus_mean_abs_error_high:
  mean trigger rate, 35/45: 50.0%
  mean global QWK delta, 35/45: +0.00701
  mean macro-F1 delta, 35/45: +0.01003
  mean worst-site QWK delta, 35/45: +0.00856
  positive global-QWK regimes: 2 / 2
```

That means the transfer signal is distributed across multiple diagnostics, especially QWK degradation and severe ordinal-error signals.

The weakest single-family variant is `only_site_qwk_spread_high`, which produces much smaller mean gains and no significant global-QWK regimes. This weakens the simplistic interpretation that the detector is primarily a site-spread detector. In this transfer setting, site spread is a secondary signal.

---

## Scientific meaning

The fixed detector appears to be a multi-signal degradation detector rather than a one-feature shortcut.

The conservative threshold-shift mechanism is ordinal, so it is expected that `mean_abs_error_high` is highly informative. But leave-one-out ablation shows that removing that diagnostic still leaves a positive detector-transfer result at 35% and 45% shift.

This strengthens the detector story:

> The detector is driven mainly by ordinal-error and QWK degradation signals, and its 35% / 45% transfer result does not collapse when the most frequent diagnostic family is removed.

---

## Claim boundary

Supported wording:

> In the conservative threshold-shift transfer setting, the fixed detector was driven mainly by ordinal-error increase and QWK degradation diagnostics. Leave-one-diagnostic-family-out ablation showed that removing the strongest diagnostic, `mean_abs_error_high`, reduced trigger rate but did not collapse the positive 35% / 45% transfer result.

Avoid wording:

> The diagnostic set is fully optimized.

> Site spread is irrelevant in all settings.

> The detector is universally calibrated.

> The detector is clinically validated.

This is still a simulated-federation stress test over pathology-derived features. The result supports detector plausibility, not deployment readiness.

---

## Next ablation to run

The next useful detector test is a calibration-sensitivity analysis:

```text
min_trigger_count = 2, 3, 4
low_quantile = 0.05, 0.10, 0.15
high_quantile = 0.75, 0.80, 0.85, 0.90
```

The key question is:

> Is the detector-transfer result robust to nearby threshold choices, or does it depend tightly on one hand-picked configuration?

A robust result should preserve the main qualitative pattern: low clean-regime switching and positive 35% / 45% conservative-shift gains across a neighborhood of detector settings.
