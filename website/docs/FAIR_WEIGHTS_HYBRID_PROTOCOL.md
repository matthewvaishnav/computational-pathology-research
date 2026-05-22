# FAIR-WEIGHTS-H Hybrid Institutional Weighting Protocol

**Status:** Experimental research protocol. Not clinical validation. Not regulatory clearance.

FAIR-WEIGHTS-H is a hybrid institutional weighting framework for federated computational pathology. It replaces fixed hospital prestige multipliers with auditable signals based on contribution, quality, useful uniqueness, uncertainty, and subgroup-safety constraints.

## Core objective

```text
w_t = argmax sum_i w_i(phi_i^Owen + lambda_D D_i^useful + lambda_F F_i + lambda_Q Q_i - lambda_S S_i)
```

subject to normalization, weight caps, representation constraints, subgroup-performance constraints, and stability constraints.

## Implemented scaffold

The current repository contains:

- FAIR-WEIGHTS-H weighting engine: `src/features/federated/pathology_fl/weighting/fair_weights_h.py`
- explicit weighted aggregator: `src/features/federated/pathology_fl/aggregator/weighted.py`
- synthetic federation generator: `src/features/federated/pathology_fl/weighting/synthetic_federation.py`
- perturbation suite: `src/features/federated/pathology_fl/weighting/perturbations.py`
- benchmark runner: `src/features/federated/pathology_fl/weighting/benchmark.py`
- canonical experiment suite: `src/features/federated/pathology_fl/weighting/experiment_suite.py`
- report generator: `src/features/federated/pathology_fl/weighting/report_generator.py`

## What is not implemented yet

- full Owen/Shapley estimation,
- executable subgroup-constrained optimizer,
- real multi-institutional validation,
- clinical validation,
- regulatory clearance.

## Interpretation guardrail

Synthetic perturbation experiments are engineering checks only. They should not be interpreted as proof of fairness, clinical performance, or deployment readiness.
