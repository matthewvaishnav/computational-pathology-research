# Dominance Detector Transfer Results

**Status:** completed 15-seed transfer summary  
**Scope:** simulated federated PANDA experiments over pathology-derived Phikon features  
**Clinical status:** research-only; not clinically validated; not diagnostic software; not intended for patient-care use

---

## Purpose

This note documents the strongest detector-switch transfer result in the dominant-site / site-signal-alignment study.

The experiment asks whether a detector rule calibrated in one stress setting can transfer to another stress setting:

```text
Calibration setting:
  dominant-site label-noise stress

Transfer setting:
  dominant-site conservative ordinal threshold shift
```

The goal is not to prove a universal detector. The goal is narrower:

> Can a fixed dominance-aware detector rule, calibrated under dominant-site label-noise stress, still identify unsafe sample-size-dominance behavior under a more systematic ordinal grading shift?

---

## Fixed detector rule

The transferred detector rule was:

```text
low_quantile = 0.10
high_quantile = 0.80
min_trigger_count = 3
use_entropy = false
```

Committed result files:

```text
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_config.json
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_summary.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_thresholds.json
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_run_diagnostics.csv
results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/detector_grid_summary.csv
```

---

## Summary result

The detector was evaluated on 15-seed conservative threshold-shift runs.

| Conservative shift | Trigger rate | Global QWK delta | 95% CI | Macro-F1 delta | 95% CI | Worst-site QWK delta | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 13.3% | -0.00025 | [-0.00113, 0.00062] | +0.00009 | [-0.00039, 0.00057] | +0.00151 | [-0.00162, 0.00463] |
| 25% | 33.3% | +0.00129 | [-0.00175, 0.00432] | +0.00290 | [-0.00184, 0.00765] | +0.00353 | [-0.00428, 0.01133] |
| 35% | 60.0% | +0.00542 | [0.00062, 0.01022] | +0.00838 | [0.00272, 0.01405] | +0.00991 | [0.00169, 0.01813] |
| 45% | 73.3% | +0.01053 | [0.00239, 0.01866] | +0.01512 | [0.00819, 0.02204] | +0.01290 | [0.00547, 0.02034] |

Deltas are detector-switch performance minus the clean-strategy baseline. Positive values indicate that the detector-switch policy improved over staying with the clean strategy.

---

## Interpretation

The strongest claim is:

> A single fixed detector rule calibrated under dominant-site label-noise stress transfers to conservative ordinal threshold-shift stress, with low clean-regime switching and statistically positive gains at 35% and 45% shift across global QWK, macro-F1, and worst-site QWK.

The clean 0% conservative-shift regime had a trigger rate of 13.3%, with near-zero global QWK cost and a confidence interval crossing zero.

The 35% and 45% conservative-shift regimes are the important positive transfer results:

```text
35% conservative threshold shift:
  trigger rate: 60.0%
  global QWK delta: +0.00542, CI [0.00062, 0.01022]
  macro-F1 delta: +0.00838, CI [0.00272, 0.01405]
  worst-site QWK delta: +0.00991, CI [0.00169, 0.01813]

45% conservative threshold shift:
  trigger rate: 73.3%
  global QWK delta: +0.01053, CI [0.00239, 0.01866]
  macro-F1 delta: +0.01512, CI [0.00819, 0.02204]
  worst-site QWK delta: +0.01290, CI [0.00547, 0.02034]
```

The 25% regime is directionally positive but should not be headlined because confidence intervals cross zero.

---

## Scientific meaning

FedAvg silently assumes:

> more samples = more influence.

This detector-transfer result suggests a practical audit mechanism for that assumption. When the dominant simulated client's training signal becomes sufficiently misaligned with the declared validation objective, a fixed diagnostic rule can detect enough warning signs to switch away from sample-size dominance.

The key transfer point is that the detector was not tuned specifically to conservative threshold shift. It was evaluated as a fixed label-noise-calibrated rule, then applied to systematic ordinal threshold bias.

That makes the result more compelling than a separately tuned detector, because separately tuning on each shift type risks overfitting the detector to the evaluation regime.

---

## Claim boundaries

Supported wording:

> In simulated PANDA federations over pathology-derived feature vectors, a fixed dominance-aware detector rule calibrated under dominant-site label-noise stress transferred to systematic conservative ordinal threshold-shift stress. It kept clean-regime switching low at 13.3% and produced statistically positive improvements at 35% and 45% shift across global QWK, macro-F1, and worst-site QWK.

Avoid wording:

> The detector is universally calibrated.

> The detector proves real hospitals can be ranked by reliability.

> Conservative grading institutions are worse.

> This is clinically validated.

> The method is ready for deployment.

This is not an institutional ranking system. It is an audit of the sample-size dominance assumption under controlled simulated site-signal misalignment.

---

## Relationship to the broader research note

This note supports the detector-switch section of:

```text
docs/research/dominant-site-federated-pathology-research-note.md
```

The broader claim remains:

> Raw sample count is not the same as task-specific site-signal alignment. In federated computational pathology, sample-size-weighted aggregation can become unsafe when the largest client has a label process that is misaligned with the declared validation objective.

The detector-transfer result strengthens the story because it shows that a fixed diagnostic rule can transfer from random dominant-site label noise to a more pathology-plausible systematic ordinal threshold shift.
