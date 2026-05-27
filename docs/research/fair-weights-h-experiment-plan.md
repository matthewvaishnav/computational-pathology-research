# FAIR-WEIGHTS-H Experiment Plan

**Status:** planned controlled validation  
**Clinical status:** research-only; not clinically validated  
**Purpose:** simplify FAIR-WEIGHTS-H into a testable hypothesis and compare it against simple federated site-weighting baselines

---

## Core hypothesis

In federated computational pathology, weighting institutions only by sample count is not enough. Site influence should be based on validated cross-site contribution, uncertainty, and subgroup coverage, with caps to prevent domination by any single institution.

This document intentionally narrows FAIR-WEIGHTS-H from a broad multi-term framework into a controlled experiment.

---

## What problem this tests

Federated oncology learning usually has institution-level heterogeneity:

- different data volumes,
- different scanners and staining distributions,
- different class balances,
- label noise,
- rare subgroups,
- site-specific artifacts,
- variable reliability of updates.

A naive aggregation rule can overweight a large but redundant/noisy site or underweight a small site that carries useful rare-signal information.

The experiment asks whether a simplified FAIR-WEIGHTS-H rule can improve generalization or robustness compared with simpler baselines.

---

## Experimental setup

Simulate 5 institutions from PCam or PANDA-derived features.

Initial implementation should start with the fastest reproducible setup before moving to full slide-level PANDA:

1. **PCam patch-level simulation** for quick iteration.
2. **PANDA feature-level simulation** once the weighting logic is stable.
3. **Camelyon17 / multi-center validation** later if available.

---

## Simulated institution types

Create five controlled synthetic sites:

| Site | Purpose | Construction idea |
|---|---|---|
| Site A: large clean site | high-volume baseline | large balanced sample, low/no label noise |
| Site B: small high-quality site | tests whether small useful sites get ignored | small balanced sample, low label noise |
| Site C: noisy-label site | tests robustness to bad updates | inject label noise into a percentage of examples |
| Site D: rare-subgroup site | tests subgroup coverage value | oversample rare class/subgroup or hard positives |
| Site E: shifted site | tests domain shift handling | create scanner/stain/feature shift or class-balance shift |

The exact construction must be logged so the experiment is reproducible.

---

## Baselines to compare

Compare FAIR-WEIGHTS-H against simple rules first:

| Method | Description |
|---|---|
| Equal weighting | each institution gets equal aggregation weight |
| Sample-size weighting / FedAvg | weight by local sample count |
| Inverse-loss weighting | higher weight for lower validation/local loss |
| Uncertainty-weighted aggregation | lower weight for high-uncertainty updates |
| Leave-one-site-out contribution weighting | estimate contribution by performance change when including/excluding a site |
| Simplified FAIR-WEIGHTS-H | contribution-aware, uncertainty-constrained, subgroup-aware weighting with caps |

Do not claim FAIR-WEIGHTS-H is superior until it beats these baselines under controlled conditions.

---

## Simplified FAIR-WEIGHTS-H rule

Use a constrained weighting rule, not an overloaded weighted sum.

Candidate formulation:

```text
maximize:    cross-site validation utility
subject to:  minimum subgroup coverage contribution
             maximum site weight cap
             uncertainty penalty
             anomaly/noise penalty
             entropy floor to prevent single-site domination
```

Practical first implementation:

```text
raw_score_i = contribution_i + subgroup_bonus_i - uncertainty_penalty_i - anomaly_penalty_i
weight_i = softmax(raw_score_i / temperature)
weight_i = cap_and_renormalize(weight_i, max_weight=w_max, min_entropy=H_min)
```

Where:

- `contribution_i` estimates whether a site improves validation utility.
- `subgroup_bonus_i` rewards useful rare/hard-case coverage.
- `uncertainty_penalty_i` reduces influence of unstable updates.
- `anomaly_penalty_i` reduces influence of suspicious/noisy updates.
- `max_weight` prevents one institution from dominating.
- `min_entropy` keeps the effective number of institutions above a threshold.

---

## Metrics

Primary metrics:

- global validation AUC or QWK,
- worst-site validation performance,
- mean cross-site validation performance,
- subgroup / rare-class performance,
- calibration error,
- weight stability across rounds,
- effective number of institutions,
- sensitivity to label noise.

For PANDA, use QWK as the primary task metric. For PCam, use AUC.

---

## Success criteria

FAIR-WEIGHTS-H is promising if it shows at least one of the following without unacceptable global degradation:

1. better global validation performance than equal weighting and sample-size weighting,
2. better worst-site performance,
3. better rare-subgroup performance,
4. better robustness under noisy-label site conditions,
5. more stable weights than naive inverse-loss weighting,
6. improved calibration or uncertainty behavior.

If it does not beat simple baselines, the correct conclusion is that the theory is not yet empirically justified.

---

## Honest public claim boundary

Safe current claim:

> FAIR-WEIGHTS-H is a research hypothesis for contribution-aware institutional weighting in federated oncology learning. It has been tested for execution stability and aggregation behavior, but a performance or fairness advantage over simpler baselines still requires controlled validation.

Unsafe claim:

> FAIR-WEIGHTS-H is a proven superior federated weighting method for clinical pathology AI.

---

## Implementation checklist

1. Create synthetic institution splitter.
2. Implement baseline weighting rules.
3. Implement simplified FAIR-WEIGHTS-H scoring and cap/entropy projection.
4. Add logging for per-round site weights.
5. Add metrics for global, worst-site, subgroup, calibration, and weight stability.
6. Run PCam quick simulation.
7. Document results.
8. Only then port to PANDA feature-level simulation.

---

## Research question for the first run

> Under controlled label-noise, rare-subgroup, and distribution-shift simulations, does simplified FAIR-WEIGHTS-H improve robustness or worst-site/subgroup performance compared with equal weighting and sample-size weighting?
