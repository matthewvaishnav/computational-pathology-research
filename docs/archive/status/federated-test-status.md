# Federated Test Scope

This document describes the intended scope of the federated-learning tests without publishing fixed pass counts that become stale as the repository changes.

## Source of truth

The current test result is the result produced by the test suite and continuous-integration workflow for the commit being evaluated. Historical counts in documentation must not be treated as current evidence.

Run the relevant tests from the repository root:

```bash
pytest tests/federated -q
```

Additional integration or experiment-specific tests should be run when their optional dependencies and datasets are available.

## What passing tests can establish

Tests can provide evidence that selected implementation contracts hold, including:

- local-training and aggregation interfaces;
- serialization and checkpoint behaviour;
- client-dropout handling;
- numerical and input-validation invariants;
- deterministic synthetic scenarios;
- expected behaviour of supported weighting rules.

## What passing tests do not establish

Passing tests do not establish:

- clinical validity or safety;
- generalization to unseen hospitals or patient populations;
- privacy guarantees beyond the assumptions of the tested mechanism;
- robustness to every adversarial or operational failure mode;
- superiority of one aggregation method on real multi-institution data;
- deployment readiness.

Optional-dependency skips, failures, and unavailable integration environments must be reported with the exact commit, environment, and command used. They must not be dismissed as having no impact without a case-specific analysis.
