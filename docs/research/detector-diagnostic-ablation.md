# Detector Diagnostic Ablation Summary

**Status:** initial diagnostic summary over fixed conservative threshold-shift detector transfer  
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

Most important signal:

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

## Interpretation

The detector does not appear to be triggering because of one noisy diagnostic alone. In triggered runs, the mean number of failed diagnostics is approximately 3.5 to 3.9 across stress levels.

The most frequent diagnostic is `mean_abs_error_high`, which is mechanistically sensible for ordinal threshold-shift stress. Conservative threshold shift changes ordinal grading behavior, so increased mean absolute ordinal error is expected to be a strong warning signal.

The QWK-based diagnostics also contribute strongly:

- `worst_site_qwk_low`
- `global_qwk_low`

This supports the interpretation that the detector is capturing a combination of global degradation, worst-site degradation, and ordinal-error increase.

The less frequent `site_qwk_spread_high` result is useful because it weakens an overly simplistic story that the detector only monitors site disparity. In this transfer setting, the stronger signals are ordinal error and QWK degradation.

---

## Claim boundary

Supported wording:

> In the conservative threshold-shift transfer setting, the fixed detector was driven mainly by ordinal-error increase and QWK degradation diagnostics. Trigger probability increased with shift severity, and triggered runs at 35% and 45% shift produced positive metric deltas across global QWK, macro-F1, worst-site QWK, and accuracy.

Avoid wording:

> The diagnostic set is fully optimized.

> Site spread is irrelevant in all settings.

> The detector is universally calibrated.

> The detector is clinically validated.

This is a first diagnostic summary over one fixed transfer experiment. A stronger ablation would rerun detector selection after removing one diagnostic family at a time.

---

## Next ablation to run

The next useful experiment is a leave-one-diagnostic-family-out detector ablation:

```text
full detector
minus global_qwk_low
minus worst_site_qwk_low
minus mean_abs_error_high
minus severe_error_rate_high
minus site_qwk_spread_high
```

The key question is:

> Does the fixed-rule transfer result still hold if the strongest diagnostic, mean_abs_error_high, is removed?

If removing `mean_abs_error_high` collapses the transfer result, the detector is mostly an ordinal-error detector in this setting. If the result persists, the signal is distributed across multiple independent diagnostics.
