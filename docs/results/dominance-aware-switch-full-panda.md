# Dominance-Aware Switch on Full PANDA Phikon Features

## Executive summary

This page documents the strongest current result in the project:

> FedAvg is not universally weak. It is vulnerable when client sample count and client reliability diverge.

Using the full readable PANDA Phikon pooled feature cache, a dominance-aware switch preserves FedAvg under clean conditions and switches to cross-site blending when observable FedAvg validation diagnostics indicate a dominant-site failure mode.

The tuned detector result is the strongest version:

- Full readable PANDA Phikon cache: 10,611 slide feature vectors, 768 dimensions
- Seeds: 15
- Federation: five simulated sites
- Failure mode: label noise applied only to the largest simulated site
- Clean strategy: `fedavg`
- Corrupted-regime strategy: `cross_site_blend_50`
- Tuned detector rule: switch only when at least three clean-calibrated FedAvg diagnostics are abnormal

The tuned detector reduced clean false switching from 33.3% to 6.7%, while still triggering on 86.7% to 93.3% of corrupted runs.

---

## Why this matters

FedAvg weights clients by sample count. This is reasonable when larger clients are also reliable. However, in clinical federated learning, sample size and reliability can diverge because of:

- site-specific label policies
- pathologist threshold differences
- scanner or staining differences
- historical labels with variable quality
- grade-distribution shift
- one institution contributing far more slides than others

The stress test here isolates that risk. The largest simulated site is made unreliable, while FedAvg continues to give it large influence because it has the most samples.

The discovery is not simply that `cross_site_blend_50` can beat FedAvg. The stronger discovery is:

> A system can keep FedAvg when validation behavior looks clean, then switch away from sample-size dominance when FedAvg diagnostics become abnormal.

---

## Dataset and experiment setup

### Feature cache

The full readable feature cache was built from the PANDA Phikon manifest:

```text
attempted rows: 10,614
cached rows: 10,611
bad file count: 3
feature shape: [10611, 768]
```

The three bad files failed HDF5 reads and were excluded from the pooled cache.

### Federated simulation

- Task: ISUP grade prediction
- Classes: 6 ordinal classes, ISUP 0 to 5
- Sites: five simulated sites
- Dominant site: site 0, largest training set
- Noise levels: 0%, 25%, 35%, 45% label flipping on the dominant site
- Seeds: 15 total
- Metrics: global QWK, worst-site QWK, mean-site QWK, global accuracy, macro-F1

### Strategies

- `fedavg`: standard sample-size-weighted aggregation
- `cross_site_blend_50`: partial reduction of dominant-site influence through cross-site blending
- `dominance-aware oracle switch`: FedAvg at 0% noise, `cross_site_blend_50` at nonzero noise
- `observable dominance detector`: detector based only on FedAvg validation diagnostics
- `tuned observable detector`: stricter detector requiring at least three abnormal diagnostics

---

## 15-seed FedAvg vs cross-site blending result

The first full-dataset result compares `fedavg` against `cross_site_blend_50` across 15 seeds.

| Noise | Metric | FedAvg mean | Cross-site mean | Delta vs FedAvg | 95% CI | Positive seeds |
|---:|---|---:|---:|---:|---:|---:|
| 0% | Global QWK | 0.720137 | 0.719777 | -0.000360 | [-0.004765, 0.004045] | 7/15 |
| 0% | Worst-site QWK | 0.684941 | 0.678207 | -0.006734 | [-0.012821, -0.000648] | 4/15 |
| 25% | Global QWK | 0.694833 | 0.703110 | +0.008276 | [0.001604, 0.014949] | 12/15 |
| 35% | Global QWK | 0.689265 | 0.697572 | +0.008307 | [0.005116, 0.011498] | 14/15 |
| 35% | Macro-F1 | 0.468275 | 0.474811 | +0.006536 | [0.000703, 0.012370] | 11/15 |
| 45% | Worst-site QWK | 0.633666 | 0.644754 | +0.011088 | [0.001122, 0.021054] | 11/15 |

Interpretation:

- Cross-site blending is not universally better.
- Under clean conditions, FedAvg retains a significant worst-site-QWK advantage.
- Under dominant-site corruption, cross-site blending significantly improves global QWK at 25% and 35% noise.
- At the highest corruption level, cross-site blending significantly improves worst-site QWK.

This supports the mechanism: FedAvg becomes vulnerable when the largest client becomes less reliable.

---

## Oracle switch result

The oracle switch tests the upper bound of a perfect detector:

```text
if noise == 0:
    use fedavg
else:
    use cross_site_blend_50
```

This switch preserves FedAvg under clean conditions and takes the cross-site gains in corrupted regimes.

| Noise | Metric | Chosen strategy | Delta vs FedAvg | 95% CI | Positive seeds |
|---:|---|---|---:|---:|---:|
| 0% | Global QWK | `fedavg` | 0.000000 | [0.000000, 0.000000] | 0/15 |
| 0% | Worst-site QWK | `fedavg` | 0.000000 | [0.000000, 0.000000] | 0/15 |
| 25% | Global QWK | `cross_site_blend_50` | +0.008276 | [0.001604, 0.014949] | 12/15 |
| 35% | Global QWK | `cross_site_blend_50` | +0.008307 | [0.005116, 0.011498] | 14/15 |
| 35% | Macro-F1 | `cross_site_blend_50` | +0.006536 | [0.000703, 0.012370] | 11/15 |
| 45% | Worst-site QWK | `cross_site_blend_50` | +0.011088 | [0.001122, 0.021054] | 11/15 |

The oracle result motivates the actual method: replace knowledge of `noise` with observable validation diagnostics.

---

## Observable detector

The first observable detector used only FedAvg validation behavior. It was calibrated from clean runs and triggered if any diagnostic left the clean-calibrated range.

Diagnostics included:

- FedAvg global QWK below clean-calibrated lower bound
- FedAvg worst-site QWK below clean-calibrated lower bound
- FedAvg site-QWK spread above clean-calibrated upper bound
- FedAvg mean absolute ordinal error above clean-calibrated upper bound
- FedAvg severe ordinal error rate above clean-calibrated upper bound

The initial detector triggered too often in clean runs but recovered the corrupted regimes well:

| Noise | Trigger rate |
|---:|---:|
| 0% | 33.3% |
| 25% | 93.3% |
| 35% | 86.7% |
| 45% | 100.0% |

It significantly improved global QWK at 25% and 35% noise and worst-site QWK at 45% noise, but the clean false-trigger rate was too high for a polished method.

---

## Tuned detector

The tuned detector searched simple rule configurations. The best rule was:

```text
low_quantile = 0.10
high_quantile = 0.80
min_trigger_count = 3
use_entropy = False
```

In words:

> Switch only when at least three FedAvg diagnostics violate clean-calibrated thresholds.

This made the detector much more conservative under clean conditions.

### Tuned trigger rates

| Noise | Trigger rate |
|---:|---:|
| 0% | 6.7% |
| 25% | 86.7% |
| 35% | 86.7% |
| 45% | 93.3% |

### Tuned detector results

| Noise | Metric | FedAvg mean | Detector mean | Oracle mean | Detector delta vs FedAvg | 95% CI | Regret vs oracle |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0% | Global QWK | 0.720137 | 0.720262 | 0.720137 | +0.000126 | [-0.000144, 0.000395] | +0.000126 |
| 0% | Worst-site QWK | 0.684941 | 0.684990 | 0.684941 | +0.000049 | [-0.000057, 0.000155] | +0.000049 |
| 25% | Global QWK | 0.694833 | 0.702842 | 0.703110 | +0.008009 | [0.001345, 0.014672] | -0.000268 |
| 35% | Global QWK | 0.689265 | 0.696747 | 0.697572 | +0.007482 | [0.003919, 0.011046] | -0.000825 |
| 45% | Worst-site QWK | 0.633666 | 0.643379 | 0.644754 | +0.009713 | [-0.000257, 0.019683] | -0.001375 |

Interpretation:

- Clean false switching dropped from 33.3% to 6.7%.
- The detector preserved statistically significant global-QWK gains at 25% and 35% dominant-site corruption.
- At 45% noise, the detector preserved most of the oracle worst-site-QWK gain, though the confidence interval narrowly crosses zero.
- The detector has low regret versus the oracle switch in corrupted regimes.

---

## Current strongest claim

Supported:

> On the full 10,611-slide readable PANDA Phikon feature cache across 15 seeds, FedAvg is vulnerable when the largest simulated site becomes label-corrupted. Cross-site blending significantly improves global QWK at 25% and 35% dominant-site corruption and significantly improves worst-site QWK at 45% corruption. A tuned observable detector, using only FedAvg validation diagnostics, reduces clean false switching to 6.7% while preserving high corrupted-regime trigger rates and significant global-QWK gains at 25% and 35% noise.

Not supported yet:

- This is not proof on real hospital federations.
- This is not clinical validation.
- This is not a claim that hospitals normally have 25% to 45% random label flips.
- This does not yet prove the detector generalizes outside the simulated PANDA stress test.
- This does not yet outperform FedAvg under every possible heterogeneity pattern.

Best framing:

> Dominant-site label corruption is a controlled stress test for a realistic risk: sample volume and client reliability can diverge. FedAvg fails because sample count remains high even when the dominant client becomes less reliable. A dominance-aware detector can preserve FedAvg under clean conditions and switch away from sample-size dominance when validation behavior becomes abnormal.

---

## Reproduction commands

Run the oracle switch analysis:

```powershell
python scripts\experiments\analyze_dominance_aware_switch.py `
  --patterns `
    "results/ordinal_harm_fair_weights_h_panda_all_noise_*_seed_*\summary.csv" `
    "results/cross_site_fedavg_panda_all_15seed_noise_*_seed_*\summary.csv" `
  --output-dir results\dominance_aware_switch_panda_all_15seed `
  --clean-strategy fedavg `
  --corrupted-strategy cross_site_blend_50
```

Run the observable detector:

```powershell
python scripts\experiments\analyze_dominance_detector_switch.py `
  --patterns `
    "results/ordinal_harm_fair_weights_h_panda_all_noise_*_seed_*\predictions.csv" `
    "results/cross_site_fedavg_panda_all_15seed_noise_*_seed_*\predictions.csv" `
  --output-dir results\dominance_detector_switch_panda_all_15seed `
  --clean-strategy fedavg `
  --corrupted-strategy cross_site_blend_50
```

Run detector tuning:

```powershell
python scripts\experiments\tune_dominance_detector_switch.py `
  --patterns `
    "results/ordinal_harm_fair_weights_h_panda_all_noise_*_seed_*\predictions.csv" `
    "results/cross_site_fedavg_panda_all_15seed_noise_*_seed_*\predictions.csv" `
  --output-dir results\dominance_detector_switch_tuned_panda_all_15seed `
  --clean-strategy fedavg `
  --corrupted-strategy cross_site_blend_50
```

---

## Next validation steps

1. Add a concise research note with abstract, method, results, limitations, and claim boundary.
2. Validate on a non-label-flip heterogeneity stress test: scanner/site shift, class-prior shift, or pathologist-threshold shift.
3. Test whether the detector works when the corrupted client is not always the largest site.
4. Test whether the detector still works when corruption affects two medium-sized clients instead of one large client.
5. Repeat on Camelyon17 or another real multi-center pathology benchmark.
