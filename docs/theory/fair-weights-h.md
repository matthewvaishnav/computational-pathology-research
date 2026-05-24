# FAIR-WEIGHTS-H: Hybrid Institutional Weighting Protocol

**Status:** Proposed protocol and implementation specification  
**Scope:** Federated computational pathology research system  
**Validation status:** Empirically tested for execution stability and aggregation behavior; performance/fairness advantage over simpler baselines not yet demonstrated

---

## 1. Motivation

The current prestige-style institutional weighting pattern assigns higher influence to cancer centers and lower influence to rural or community hospitals based on institutional category. That is not scientifically defensible for federated pathology because institutional reputation is not the same as measured contribution, clinical reliability, subgroup coverage, or domain-shift value.

FAIR-WEIGHTS-H replaces prestige weighting with an auditable hybrid protocol that combines:

1. counterfactual contribution estimation,
2. diagnostic and process quality,
3. distributional uniqueness,
4. representation and subgroup-safety constraints,
5. uncertainty penalties,
6. anomaly monitoring and fallback modes.

The framework is intended as a research protocol. It does not claim clinical validation or regulatory clearance.

---

## 2. Key Design Principle

A single scalar institution weight cannot by itself guarantee fairness or safety. Therefore FAIR-WEIGHTS-H separates three concepts:

$$
w_i^{\mathrm{train}} \neq w_i^{\mathrm{val}} \neq w_i^{\mathrm{monitor}}
$$

where:

- $w_i^{\mathrm{train}}$: aggregation weight used during federated model updates,
- $w_i^{\mathrm{val}}$: validation representation priority,
- $w_i^{\mathrm{monitor}}$: post-market or research monitoring priority.

A site may have a low training weight because of uncertain or noisy updates while still receiving high validation and monitoring priority if it represents an underserved or clinically important population.

---

## 3. Institutional Signals

For institution $i$, define the feature vector:

$$
z_i = \left[A_i^{\mathrm{adj}}, Q_i, \phi_i^{\mathrm{Owen}}, JS_i, F_i, V_i, -S_i\right]
$$

where:

| Symbol | Meaning | Status |
|---|---|---|
| $A_i^{\mathrm{adj}}$ | Difficulty-adjusted reference-case diagnostic quality | Proposed |
| $Q_i$ | Process and pathology quality composite | Proposed |
| $\phi_i^{\mathrm{Owen}}$ | Group-aware counterfactual contribution estimate | Proposed |
| $JS_i$ | Jensen-Shannon distributional uniqueness | Proposed |
| $F_i$ | Underserved-population representation score | Proposed |
| $V_i$ | Bounded/sublinear volume factor | Proposed |
| $S_i$ | Uncertainty, instability, or anomaly penalty | Proposed |

Gradient alignment with the current global model is not used as a primary contribution factor because it can encode status quo bias. It may be used only for anomaly detection and drift monitoring.

---

## 4. Integrity Gate

FAIR-WEIGHTS-H uses a hard gate only for integrity and safety failures, not for prestige or raw quality.

$$
I_i = G_i^{\mathrm{data}} G_i^{\mathrm{label}} G_i^{\mathrm{safety}}
$$

where:

- $G_i^{\mathrm{data}} = 1$ only when data integrity checks pass,
- $G_i^{\mathrm{label}} = 1$ only when no severe label-corruption signal is detected,
- $G_i^{\mathrm{safety}} = 1$ only when there is no active safety violation.

Low resource level, rural status, or case difficulty must not directly trigger exclusion. Quality should be modeled with difficulty adjustment and uncertainty, not with a crude hard threshold.

---

## 5. Difficulty-Adjusted Quality

Raw accuracy on reference cases can be misleading when institutions serve different case mixes. FAIR-WEIGHTS-H therefore requires a pre-specified difficulty-adjustment model before using adjusted quality in weight computation.

One defensible model is:

$$
Y_{ij} \sim \mathrm{Bernoulli}(p_{ij})
$$

$$
\mathrm{logit}(p_{ij}) = \alpha + \beta^\top X_{ij} + b_i
$$

where:

- $Y_{ij}$: correct or incorrect diagnosis for case $j$ at site $i$,
- $X_{ij}$: tumor type, stage, slide quality, stain quality, scanner metadata, referral status, and case complexity,
- $b_i$: institution-level effect after adjustment.

The adjusted quality is evaluated on a standardized reference distribution:

$$
A_i^{\mathrm{adj}} = \mathbb{E}_{X \sim P_{\mathrm{ref}}}\left[\Pr(Y = 1 \mid X, i)\right]
$$

This adjustment must be calibrated and audited. It cannot simply be asserted.

---

## 6. Useful Uniqueness

Distributional uniqueness alone is not always beneficial. A site can be unique because it serves rare populations, but it can also be unique because of scanner artifacts, poor fixation, or systematic labeling errors.

Therefore uniqueness is treated as a weak signal unless paired with quality and subgroup utility:

$$
D_i^{\mathrm{useful}} = JS_i \cdot A_i^{\mathrm{adj}} \cdot U_i^{\mathrm{subgroup}}
$$

where $U_i^{\mathrm{subgroup}}$ measures whether the institution improves performance on the subgroup, morphology, or cancer subtype it uniquely represents.

---

## 7. Counterfactual Contribution

The preferred contribution signal is not local gradient alignment. It is counterfactual marginal contribution:

$$
\phi_i = \mathbb{E}_{S \subseteq N \setminus \{i\}}\left[U(S \cup \{i\}) - U(S)\right]
$$

For production feasibility, FAIR-WEIGHTS-H estimates this with grouped or multi-membership Owen-style sampling:

$$
\hat{\phi}_i^{\mathrm{Owen}} = \sum_g m_{ig}\hat{\phi}_{i\mid g}
$$

where $m_{ig} \in [0,1]$ allows institutions to belong partly to multiple groups, such as academic, rural-serving, specialty center, or network-affiliated.

This avoids forcing ambiguous hospitals into a single administrative category.

---

## 8. Training Weight Objective

The quarterly training weights are produced by a constrained optimization problem:

$$
w_t = \arg\max_{w \in \mathcal{W}} \sum_{i=1}^{K} w_i\left(\hat{\phi}_{i,t}^{\mathrm{Owen}} + \lambda_D D_{i,t}^{\mathrm{useful}} + \lambda_F F_{i,t} + \lambda_Q Q_{i,t} - \lambda_S S_{i,t}\right)
$$

subject to:

$$
\sum_i w_i = 1
$$

$$
w_i^{\min} \leq w_i \leq w_i^{\max}
$$

$$
C_g(w) \geq C_g^{\min} \quad \forall g \in \mathcal{G}_{\mathrm{underserved}}
$$

$$
\mathrm{Perf}_g(w) \geq \mathrm{Perf}_g^{\min} \quad \forall g \in \mathcal{G}_{\mathrm{clinical}}
$$

$$
\left|w_{i,t} - w_{i,t-1}\right| \leq \Delta_i
$$

The fairness and subgroup safety requirements are constraints, not optional score boosts.

---

## 9. Empirical Status

FAIR-WEIGHTS-H has been empirically tested for execution stability and aggregation behavior in this repository.

Completed checks include:

- synthetic Camelyon-like smoke validation,
- PCam federated smoke validation,
- PCam all-strategy smoke validation,
- PCam balanced federated benchmark,
- PCam heterogeneous federated benchmark.

These tests show that FAIR-WEIGHTS-H runs without numerical failure, produces distinct weight trajectories under heterogeneous simulated sites, and does not degrade performance in the current patch-level PCam setup. They do **not** yet show a consistent performance or fairness advantage over simpler aggregation baselines. That stronger claim requires the planned ablation and slide-level multi-center validation.

---

## 10. Quarterly Algorithm

1. Collect privacy-preserving aggregate signals from institutions.
2. Run integrity checks and anomaly detection.
3. Estimate difficulty-adjusted quality and uncertainty.
4. Estimate distributional uniqueness using aggregate profiles.
5. Estimate approximate Owen contribution using sampled warm-started coalitions.
6. Compute useful uniqueness.
7. Solve the constrained weight optimization problem.
8. Run subgroup safety and representation audits.
9. Produce an institution-level report card.
10. Version the weights, coefficients, inputs, and audit results.

---

## 11. Anomaly Monitoring

Continuous or between-quarter monitoring should detect:

- scanner drift,
- stain distribution shift,
- abrupt gradient drift,
- local validation collapse,
- unusual update variance,
- suspicious simultaneous score changes among affiliated institutions,
- distributional specialization that reduces coverage of community or underserved cases.

If severe anomalies are detected, weights should be throttled or frozen pending review.

---

## 12. Fallback Modes

FAIR-WEIGHTS-H should degrade safely rather than continue normal operation during instability.

| Mode | Trigger | Action |
|---|---|---|
| Normal | No major anomaly | Full hybrid weighting |
| Conservative | Moderate anomaly or high uncertainty | Freeze coefficients, reduce caps, rely more on verified quality |
| Safety | Systemic manipulation or validation failure | Suspend uniqueness/contribution bonuses; use externally verified metrics only |
| Emergency | Severe corruption or safety violation | Temporarily exclude affected institution updates pending remediation |

---

## 13. Validation Plan

FAIR-WEIGHTS-H should be compared against:

1. equal weighting,
2. volume weighting,
3. prestige weighting,
4. original multiplicative FAIR-WEIGHTS,
5. Shapley/Owen-only attribution,
6. FAIR-WEIGHTS-H.

Primary metrics should include:

- global AUC,
- balanced accuracy,
- calibration error,
- worst-group sensitivity,
- false-negative-rate disparity,
- subgroup non-inferiority,
- convergence stability,
- weight stability,
- robustness under missingness and simulated gaming.

---

## 14. Regulatory-Safe Claim Language

Use:

> FAIR-WEIGHTS-H enforces pre-specified representation and subgroup-performance constraints and requires prospective validation before clinical deployment.

Do not use:

> FAIR-WEIGHTS-H proves fairness.

Use:

> Shapley/Owen values provide an axiomatic counterfactual attribution benchmark whose conclusions depend on the chosen validation utility and reference distribution.
