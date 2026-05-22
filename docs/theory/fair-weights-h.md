# FAIR-WEIGHTS-H: Hybrid Institutional Weighting Protocol

**Status:** Proposed protocol and implementation specification  
**Scope:** Federated computational pathology research system  
**Validation status:** Requires empirical validation before clinical or regulatory claims

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

\[
w_i^{train} \neq w_i^{val} \neq w_i^{monitor}
\]

where:

- \(w_i^{train}\): aggregation weight used during federated model updates,
- \(w_i^{val}\): validation representation priority,
- \(w_i^{monitor}\): post-market or research monitoring priority.

A site may have a low training weight because of uncertain or noisy updates while still receiving high validation and monitoring priority if it represents an underserved or clinically important population.

---

## 3. Institutional Signals

For institution \(i\), define the feature vector:

\[
z_i = [A_i^{adj}, Q_i, \phi_i^{Owen}, JS_i, F_i, V_i, -S_i]
\]

where:

| Symbol | Meaning | Status |
|---|---|---|
| \(A_i^{adj}\) | Difficulty-adjusted reference-case diagnostic quality | Proposed |
| \(Q_i\) | Process and pathology quality composite | Proposed |
| \(\phi_i^{Owen}\) | Group-aware counterfactual contribution estimate | Proposed |
| \(JS_i\) | Jensen-Shannon distributional uniqueness | Proposed |
| \(F_i\) | Underserved-population representation score | Proposed |
| \(V_i\) | Bounded/sublinear volume factor | Proposed |
| \(S_i\) | Uncertainty, instability, or anomaly penalty | Proposed |

Gradient alignment with the current global model is not used as a primary contribution factor because it can encode status quo bias. It may be used only for anomaly detection and drift monitoring.

---

## 4. Integrity Gate

FAIR-WEIGHTS-H uses a hard gate only for integrity and safety failures, not for prestige or raw quality.

\[
I_i = \mathbf{1}[\text{data integrity OK}]\cdot\mathbf{1}[\text{no severe label corruption}]\cdot\mathbf{1}[\text{no active safety violation}]
\]

Low resource level, rural status, or case difficulty must not directly trigger exclusion. Quality should be modeled with difficulty adjustment and uncertainty, not with a crude hard threshold.

---

## 5. Difficulty-Adjusted Quality

Raw accuracy on reference cases can be misleading when institutions serve different case mixes. FAIR-WEIGHTS-H therefore requires a pre-specified difficulty-adjustment model before using adjusted quality in weight computation.

One defensible model is:

\[
Y_{ij} \sim \mathrm{Bernoulli}(p_{ij})
\]

\[
\mathrm{logit}(p_{ij}) = \alpha + \beta^\top X_{ij} + b_i
\]

where:

- \(Y_{ij}\): correct or incorrect diagnosis for case \(j\) at site \(i\),
- \(X_{ij}\): tumor type, stage, slide quality, stain quality, scanner metadata, referral status, and case complexity,
- \(b_i\): institution-level effect after adjustment.

The adjusted quality is evaluated on a standardized reference distribution:

\[
A_i^{\mathrm{adj}} = \mathbb{E}_{X \sim P_{\mathrm{ref}}} \left[ \Pr(Y = 1 \mid X, i) \right]
\]

This adjustment must be calibrated and audited. It cannot simply be asserted.

---

## 6. Useful Uniqueness

Distributional uniqueness alone is not always beneficial. A site can be unique because it serves rare populations, but it can also be unique because of scanner artifacts, poor fixation, or systematic labeling errors.

Therefore uniqueness is treated as a weak signal unless paired with quality and subgroup utility:

\[
D_i^{useful} = JS_i \cdot A_i^{adj} \cdot U_i^{subgroup}
\]

where \(U_i^{subgroup}\) measures whether the institution improves performance on the subgroup, morphology, or cancer subtype it uniquely represents.

---

## 7. Counterfactual Contribution

The preferred contribution signal is not local gradient alignment. It is counterfactual marginal contribution:

\[
\phi_i = \mathbb{E}_{S\subseteq N\setminus\{i\}}[U(S\cup\{i\}) - U(S)]
\]

For production feasibility, FAIR-WEIGHTS-H estimates this with grouped or multi-membership Owen-style sampling:

\[
\hat\phi_i^{Owen} = \sum_g m_{ig}\hat\phi_{i\mid g}
\]

where \(m_{ig}\in[0,1]\) allows institutions to belong partly to multiple groups, such as academic, rural-serving, specialty center, or network-affiliated.

This avoids forcing ambiguous hospitals into a single administrative category.

---

## 8. Training Weight Objective

The quarterly training weights are produced by a constrained optimization problem:

\[
w_t = \arg\max_{w\in\mathcal W}\sum_{i=1}^K w_i(\hat\phi_{i,t}^{Owen} + \lambda_D D_{i,t}^{useful} + \lambda_F F_{i,t} + \lambda_Q Q_{i,t} - \lambda_S S_{i,t})
\]

subject to:

\[
\sum_i w_i = 1
\]

\[
w_i^{min} \leq w_i \leq w_i^{max}
\]

\[
C_g(w) \geq C_g^{min}\quad\forall g\in\mathcal G_{underserved}
\]

\[
\mathrm{Perf}_g(w) \geq \mathrm{Perf}_g^{min}\quad\forall g\in\mathcal G_{clinical}
\]

\[
|w_{i,t} - w_{i,t-1}| \leq \Delta_i
\]

The fairness and subgroup safety requirements are constraints, not optional score boosts.

---

## 9. Quarterly Algorithm

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

## 10. Anomaly Monitoring

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

## 11. Fallback Modes

FAIR-WEIGHTS-H should degrade safely rather than continue normal operation during instability.

| Mode | Trigger | Action |
|---|---|---|
| Normal | No major anomaly | Full hybrid weighting |
| Conservative | Moderate anomaly or high uncertainty | Freeze coefficients, reduce caps, rely more on verified quality |
| Safety | Systemic manipulation or validation failure | Suspend uniqueness/contribution bonuses; use externally verified metrics only |
| Emergency | Severe corruption or safety violation | Temporarily exclude affected institution updates pending remediation |

---

## 12. Validation Plan

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

## 13. Regulatory-Safe Claim Language

Use:

> FAIR-WEIGHTS-H enforces pre-specified representation and subgroup-performance constraints and requires prospective validation before clinical deployment.

Do not use:

> FAIR-WEIGHTS-H proves fairness.

Use:

> Shapley/Owen values provide an axiomatic counterfactual attribution benchmark whose conclusions depend on the chosen validation utility and reference distribution.

Do not use:

> Shapley is an absolute ground truth.

Use:

> Distributional uniqueness is considered useful only when paired with quality, subgroup utility, and safety monitoring.

Do not use:

> Diversity alone increases institutional influence.

---

## 14. Known Limitations

- Owen/Shapley estimates can be noisy with small numbers of institutions.
- Validation sets can encode their own demographic and institutional bias.
- Jensen-Shannon uniqueness can be gamed by case selection or artificial specialization.
- Difficulty adjustment can fail if case complexity is incompletely observed.
- Multi-membership grouping requires governance and versioned definitions.
- Mathematics cannot resolve all policy tradeoffs between global accuracy and protected-subgroup safety.

---

## 15. Summary

FAIR-WEIGHTS-H should be understood as a locked, auditable, risk-controlled weighting protocol rather than a simple formula. Its core contribution is the integration of counterfactual contribution, useful uniqueness, subgroup constraints, and regulatory traceability into one deployable institutional weighting framework.
