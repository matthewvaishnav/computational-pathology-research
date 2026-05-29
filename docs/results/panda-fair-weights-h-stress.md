# PANDA FAIR-WEIGHTS-H Noisy-Site Stress Result

**Status:** completed PANDA-3000 noisy large-site stress benchmark  
**Dataset source:** PANDA-derived Phikon slide feature cache  
**Clinical status:** simulated-federation stress benchmark only; not real multi-center clinical validation; not diagnostic software

---

## Research question

When does FedAvg stop being the best aggregation rule for PANDA-derived pathology features?

This experiment stress-tests the standard sample-size assumption in FedAvg. FedAvg gives more influence to sites with more training samples. That is usually reasonable when sample count is a good proxy for reliability, but it can become harmful if the largest institution has noisy labels or unreliable updates.

The stress question is:

> If the largest simulated institution becomes increasingly label-noisy, does contribution-aware weighting become preferable to FedAvg?

---

## Experimental setup

The benchmark used the 3,000-slide PANDA-derived Phikon feature cache:

| Property | Value |
|---|---:|
| Attempted cached slides | 3,000 |
| Readable cached slides | 2,999 |
| Feature dimension | 768 |
| Labels | ISUP grade 0–5 |
| Feature pooling | mean pooling over per-slide Phikon patch features |
| Seeds | `42`, `123`, `2025`, `7`, `99` |
| Simulated sites | 5 |
| Site proportions | `0.45, 0.15, 0.15, 0.125, 0.125` |

The largest simulated site received controlled training-label noise. Validation labels remained clean.

Noise levels tested:

```text
0%, 15%, 25%, 35%, 45%
```

Strategies compared:

| Strategy | Description |
|---|---|
| `fedavg` | Standard sample-size-weighted federated averaging |
| `cross_site_full` | Full contribution-aware cross-site weighting |
| `cross_site_blend_25` | 25% contribution-aware weights, 75% FedAvg weights |
| `cross_site_blend_50` | 50% contribution-aware weights, 50% FedAvg weights |
| `cross_site_blend_75` | 75% contribution-aware weights, 25% FedAvg weights |

---

## Reproduction commands

Run noisy-site sweeps:

```powershell
foreach ($noise in 15,25,35,45) {
  foreach ($s in 42,123,2025,7,99) {
    python scripts\experiments\run_fair_weights_h_panda_feature_stress.py `
      --feature-cache C:\panda_cache\panda_phikon_mean_features_3000.npz `
      --output-dir "results\fair_weights_h_panda_feature_stress_3000_noise_$noise`_seed_$s" `
      --rounds 5 `
      --large-site-label-flip "0.$noise" `
      --seed $s `
      --device cuda `
      --strategies fedavg cross_site_full cross_site_blend_25 cross_site_blend_50 cross_site_blend_75
  }
}
```

Aggregate results:

```powershell
foreach ($noise in 15,25,35,45) {
  python scripts\experiments\aggregate_fair_weights_h_results.py `
    --pattern "results\fair_weights_h_panda_feature_stress_3000_noise_$noise`_seed_*\summary.csv" `
    --output-dir "results\fair_weights_h_panda_feature_stress_3000_noise_$noise`_aggregate" `
    --baseline fedavg
}
```

The clean 0% condition was also run separately with `--large-site-label-flip 0.0`.

---

## Aggregate results

### Best strategy by noise level

| Largest-site label noise | Best global QWK | Best worst-site QWK | Best mean-site QWK | Best accuracy | Best macro F1 |
|---:|---|---|---|---|---|
| 0% clean | `fedavg` | `fedavg` | `fedavg` | `fedavg` | `fedavg` |
| 15% | `cross_site_blend_75` | `cross_site_blend_25` | `cross_site_blend_75` | `cross_site_blend_25` | `cross_site_blend_25` |
| 25% | `cross_site_blend_50` | `cross_site_blend_50` | `cross_site_blend_50` | `cross_site_blend_50` | `cross_site_blend_50` |
| 35% | `fedavg` | `cross_site_blend_75` | `fedavg` | `fedavg` | `fedavg` |
| 45% | `cross_site_full` | `cross_site_blend_50` | `cross_site_full` | `fedavg` | `fedavg` |

### Aggregator outputs

Clean 0% noise:

```text
Best global_qwk: fedavg mean=0.6659
Best worst_site_qwk: fedavg mean=0.5769
Best mean_site_qwk: fedavg mean=0.6727
Best global_accuracy: fedavg mean=0.4829
Best macro_f1: fedavg mean=0.4709
```

15% noisy largest site:

```text
Best global_qwk: cross_site_blend_75 mean=0.6527
Best worst_site_qwk: cross_site_blend_25 mean=0.5586
Best mean_site_qwk: cross_site_blend_75 mean=0.6572
Best global_accuracy: cross_site_blend_25 mean=0.4719
Best macro_f1: cross_site_blend_25 mean=0.4578
```

25% noisy largest site:

```text
Best global_qwk: cross_site_blend_50 mean=0.6474
Best worst_site_qwk: cross_site_blend_50 mean=0.5340
Best mean_site_qwk: cross_site_blend_50 mean=0.6524
Best global_accuracy: cross_site_blend_50 mean=0.4649
Best macro_f1: cross_site_blend_50 mean=0.4505
```

35% noisy largest site:

```text
Best global_qwk: fedavg mean=0.6403
Best worst_site_qwk: cross_site_blend_75 mean=0.5497
Best mean_site_qwk: fedavg mean=0.6432
Best global_accuracy: fedavg mean=0.4579
Best macro_f1: fedavg mean=0.4448
```

45% noisy largest site:

```text
Best global_qwk: cross_site_full mean=0.6259
Best worst_site_qwk: cross_site_blend_50 mean=0.5514
Best mean_site_qwk: cross_site_full mean=0.6353
Best global_accuracy: fedavg mean=0.4512
Best macro_f1: fedavg mean=0.4371
```

---

## Main finding

FAIR-WEIGHTS-H-style contribution-aware weighting is **not** a universal FedAvg replacement.

The clean PANDA-3000 result showed that FedAvg remains strongest when the simulated federation is clean and institution size remains a reasonable proxy for useful contribution.

The noisy-site sweep shows a different pattern:

```text
FedAvg dominates the clean setting.
Contribution-aware blends become competitive or superior when the largest site becomes label-noisy.
```

The clearest conditional win occurs at 25% label noise in the largest simulated site, where `cross_site_blend_50` is best across all aggregate metrics.

This supports the narrower claim:

> Contribution-aware weighting appears most useful as a stress-regime mechanism, not as a universal replacement for FedAvg.

---

## Interpretation

FedAvg is sample-size driven. It assumes that larger sites should exert more influence because they contain more data. In clean conditions, this works well.

However, when the largest simulated site becomes unreliable, sample size and reliability diverge. In those conditions, contribution-aware weighting can reduce dependence on raw volume and shift aggregation toward updates that generalize better across sites.

The observed pattern suggests an adaptive design:

| Federation condition | Preferred behavior |
|---|---|
| Clean, aligned sites | Use FedAvg |
| Mild-to-moderate large-site label noise | Blend FedAvg with contribution-aware weighting |
| Severe reliability mismatch | Consider stronger contribution-aware or harm-aware weighting |
| Worst-site robustness degrades | Prefer the blend that protects worst-site QWK, even if global accuracy remains FedAvg-favored |

---

## Implication for FAIR-WEIGHTS-H Adaptive

The next version should not try to replace FedAvg everywhere.

Instead, the system should detect when FedAvg is becoming risky and switch aggregation mode only under measurable stress.

Candidate adaptive rule:

```text
if federation appears clean:
    use FedAvg
else if largest-site reliability is questionable:
    use contribution-aware blend
else if worst-site performance is degrading:
    increase harm-aware weighting
```

The next research step is to log per-site and per-grade predictions so that the switch can be driven by observed failure signals, not only global QWK.

---

## Claim boundary

This is simulated-federation evidence on cached PANDA-derived Phikon features. It is not clinical validation.

Supported claim:

> On PANDA-derived 3,000-slide simulated federations, FedAvg is strongest in the clean setting, while contribution-aware blends become competitive or superior under controlled large-site label noise, especially around 15–25% noise and for worst-site robustness.

Unsupported claims:

- This does not prove FAIR-WEIGHTS-H universally beats FedAvg.
- This does not prove real hospital deployment readiness.
- This does not establish clinical utility for prostate cancer diagnosis.
- This does not replace validation on real multi-center datasets.

---

## Next steps

1. Add prediction logging by seed, site, grade, and strategy.
2. Analyze per-grade recall and high-grade ISUP 4/5 failure modes.
3. Build a FedAvg risk detector using cross-site harm, worst-site degradation, and high-grade recall drop.
4. Implement `adaptive_fair_weights_h`, which uses FedAvg in clean conditions and switches to contribution-aware/harm-aware weighting under stress.
5. Compare FedAvg, static FAIR-WEIGHTS-H, adaptive FAIR-WEIGHTS-H, and an oracle switch upper bound.
