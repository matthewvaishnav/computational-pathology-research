# FAIR-WEIGHTS-H Stress Scenario Result

**Status:** synthetic research stress test  
**Clinical status:** not clinical validation; not diagnostic software  
**Purpose:** identify a condition where cross-site contribution weighting is expected to outperform sample-size weighting/FedAvg

---

## Research question

Can contribution-aware institutional weighting outperform FedAvg when the largest simulated institution has severe label noise?

This stress scenario was designed to test a specific weakness of sample-size weighting: if the largest site is unreliable, FedAvg can overweight the noisy site and degrade global/worst-site performance.

---

## Scenario design

Five synthetic institutions were created:

| Site | Role | Construction |
|---|---|---|
| Site 0 | Large noisy site | 3.0x sample volume with severe label noise |
| Site 1 | Small clean site | smaller clean balanced site |
| Site 2 | Medium clean site | medium clean balanced site |
| Site 3 | Rare-signal site | smaller positive-enriched useful-signal site |
| Site 4 | Shifted clean site | smaller shifted but clean site |

The experiment compares FedAvg against cross-site contribution weighting, where each candidate local update is evaluated by whether it improves validation behavior across all simulated sites.

---

## Strategies compared

| Strategy | Description |
|---|---|
| `fedavg` | sample-size weighting baseline |
| `cross_site_full` | full cross-site contribution weighting |
| `cross_site_blend_25` | 75% FedAvg + 25% cross-site contribution correction |
| `cross_site_blend_50` | 50% FedAvg + 50% cross-site contribution correction |
| `cross_site_blend_75` | 25% FedAvg + 75% cross-site contribution correction |

---

## Aggregate result across 5 seeds

Seeds: `42`, `123`, `2025`, `7`, `99`

| Strategy | Mean global AUC | Mean worst-site AUC | Interpretation |
|---|---:|---:|---|
| `fedavg` | ~0.5928 | ~0.5274 | sample-size weighting is vulnerable in this stress scenario |
| `cross_site_full` | ~0.6486 | ~0.5856 | best global AUC and best worst-site AUC |
| `cross_site_blend_25` | lower than full cross-site | above FedAvg in most runs | partial correction helps but underuses the contribution signal |
| `cross_site_blend_50` | lower than full cross-site | above FedAvg in most runs | stronger correction improves over FedAvg but remains below full cross-site |
| `cross_site_blend_75` | close to full cross-site | below full cross-site | strong correction helps but full cross-site performed best here |

The aggregator reported:

```text
Best global AUC: cross_site_full mean=0.6486
Best worst-site AUC: cross_site_full mean=0.5856
```

---

## Supported interpretation

This result supports the narrow claim:

> In a controlled synthetic stress scenario where the largest simulated institution has severe label noise, cross-site contribution weighting outperforms FedAvg across five seeds, improving both global AUC and worst-site AUC.

This does **not** support the broad claim that FAIR-WEIGHTS-H is generally superior to FedAvg.

---

## What changed from earlier experiments

Earlier synthetic experiments showed:

- local FAIR-WEIGHTS-H scoring did not beat FedAvg,
- FedAvg/FAIR blends mostly matched but did not improve FedAvg,
- naive hard-site upweighting hurt performance,
- balanced synthetic setups were FedAvg-favorable.

The stress scenario shows that contribution-aware weighting becomes useful under a specific condition:

> the largest site is not necessarily the most reliable site.

This is the strongest current evidence for the FAIR-WEIGHTS-H direction.

---

## Claim boundary

This is synthetic research evidence only. It does not establish real-world hospital performance, clinical utility, or regulatory readiness.

Next validation steps:

1. Preserve the per-seed summaries and aggregate summary.
2. Add real PCam or PANDA-derived feature simulations once available.
3. Test label-noise and site-size sweeps.
4. Confirm whether the result holds under multiple stress magnitudes.
5. Compare against additional robust FL baselines.
