# FAIR-WEIGHTS-H Scope and Limitations

FAIR-WEIGHTS-H is an experimental research implementation for studying institution-weighting rules in federated pathology. It is not clinical validation or deployment evidence.

## Implemented

- Hybrid weighting over supplied institution-level signals.
- Stable normalization, caps, and renormalization.
- Integrity gating and conservative-mode controls.
- Entropy and effective-institution-count diagnostics.
- Synthetic federation utilities and comparison baselines.
- Perturbation, experiment, reporting, and weighted-aggregation helpers.

The code and tests for the exact commit are the source of truth. Fixed test counts and coverage percentages are intentionally omitted because they become stale.

## Not implemented

- Owen or Shapley contribution estimation.
- A fitted case-difficulty adjustment model.
- Executable subgroup performance constraints.
- Validated automated drift detection or fallback selection.
- Production governance, scheduling, or policy versioning.

## Evidence boundary

Passing software tests does not establish improvement on real multi-institution data, clinical validity, fairness across patient groups, privacy guarantees, or deployment readiness.

Evaluation should include simple baselines, held-out institutions, uncertainty estimates, subgroup analysis, clean-regime behaviour, and explicit failure cases.

```bash
pytest tests/federated -q
```

Record the exact commit, environment, optional dependencies, dataset version, and command whenever reporting results.
