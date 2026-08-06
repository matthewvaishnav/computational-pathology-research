# Task-benchmark instrument-power audit

## Scope

This protocol audits the operating characteristics of the probes and sample sizes used by the frozen synthetic task-defined biological-sufficiency benchmark. It initializes and trains no factorizer, does not initialize CUDA, and cannot change the frozen `complete_task_defined_biological_sufficiency_unsupported` status or any frozen task threshold.

The audit separates three questions that the original benchmark necessarily combined: whether a synthetic task is learnable at a given sample size, whether a probe and its validation selection are sufficiently powered, and whether a learned representation contains accessible task information. An oracle failure makes the corresponding representation-level failure inadmissible under that design.

## Frozen-result adjudication

Every serialized task record is extracted by dataset seed, renderer, factorizer seed, task, representation, identity budget, subset seed, probe seed, probe family, and selected epoch. Linear and nonlinear probe results remain separate. Available validation losses or metrics, unseen-test performance, epoch-zero status, checkpoint, change from the linear baseline, and worst-scanner performance are retained.

The original-design oracle control uses the frozen 32-training-identity, eight-validation-identity records. Regression requires median residual R² at least 0.80, median worst-scanner R² at least 0.70, and rejection of the identity-permuted control. Classification additionally requires complete five-class coverage in every training and validation split.

The linear task receives a dedicated subset-matched comparison across biological code, raw observation, scanner-centered observation, and oracle latent. Ridge and residual behavior are adjudicated independently. The frozen artifact did not serialize biological feature matrices or checkpoints. Consequently, biological feature covariance rank and condition number are explicitly marked unavailable; reconstructing them would require a prohibited factorizer rerun. Feature dimension, training count, feature-to-identity ratio, target-side evidence recoverable from task definitions, and all serialized performance and checkpoint diagnostics remain reportable.

## Independent calibration

The exact frozen linear matrix, neural teachers, interaction equations, normalization values, task-noise distribution, and classification quintiles are regenerated from their frozen seeds and verified hashes. Five independent standard-normal biological-latent pools use seeds 8501–8505. Each pool contains 512 nested training identities, 128 separate validation identities, and 4,096 independent test identities.

Training budgets are 8, 16, 32, 64, 128, 256, and 512 identities. Model selection is repeated with eight validation identities and with 128 validation identities. Split arrays and task calibration arrays are hashed, and the three identity pools are disjoint.

The frozen instruments are evaluated unchanged: Ridge alpha `1e-3`, residual seeds 7203 and 7204, logistic `C=1`, and shallow-classifier seeds 7301–7303. Separate validation-only regularization diagnostics use the predeclared Ridge and logistic grids. They do not replace or tune the frozen instruments.

Each task includes identity-permuted targets, independent Gaussian features, and a constant-mean or class-prior control. Classification also includes a deterministic class-count-preserving label permutation. Test results never affect hyperparameter or epoch selection.

## Power definitions

An instrument is powered at a budget only when at least 80% of independent generation/probe repeats meet performance 0.80, at least 95% beat every paired negative control, every result is finite, classification coverage is complete, and median absolute validation-to-test optimism is at most 0.10.

For each task, probe and validation regime, the audit reports median test performance, the 2.5–97.5% range, threshold and negative-control success probabilities, optimism, selected-epoch distribution, monotonicity, and the minimum powered training budget.

Original-design decisions distinguish admissibility, training underpower, validation underpower, both forms of underpower, suspected fixed-regularization failure, and failure to become learnable through 512 training identities. No task becomes admissible from one favorable seed.

## Counterfactual eligibility and claim boundaries

Each frozen counterfactual record is eligible only if its direct-code task performance first reaches 0.70. The original preservation flag remains unchanged; ineligible direct probes cannot establish semantic counterfactual failure.

The final claim adjudication reports supported and unresolved conclusions separately. Exact replication, acquisition shortcut behavior, acquisition exclusion, and permuted-control rejection remain factual frozen observations. Nonlinear, interaction, classification, and counterfactual claims depend on the calibrated oracle and direct-probe evidence. The linear representation failure can remain a genuine negative result if its predeclared cross-run and positive-control criteria pass.

Future synthetic task benchmarks should freeze task complexity, training size, validation size, probe family, regularization, and positive-control power before evaluating learned representations. These synthetic tasks are not real pathology endpoints.

## Execution

```powershell
py -m experiments.paired_acquisition.run_task_benchmark_instrument_power_audit `
  --task-benchmark-result "results\task_defined_biological_sufficiency_20260803T094641\task_defined_biological_sufficiency_result.json" `
  --output-root "results\task_benchmark_instrument_power_audit_<timestamp>"
```

The result JSON, heterogeneous summary CSV, and manifest are written atomically into a new output directory and contain canonical internal hashes plus before-and-after frozen-artifact verification.
