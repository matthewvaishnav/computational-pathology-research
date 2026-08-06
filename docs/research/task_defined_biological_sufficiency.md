# Task-defined biological sufficiency benchmark

## Claim scope

This protocol evaluates a synthetic crossed-target biological representation through predeclared downstream tasks. It does not claim that those synthetic tasks are real pathology endpoints, and it does not reopen or reinterpret any prior result. In particular, the frozen finite-sample audit remains `complete_partial_finite_sample_whitening_support`, the original fixed covariance cutoff remains failed, and canonical generator-latent coordinates are not a success requirement here.

The benchmark asks whether a representation retains task-relevant biological information without requiring a privileged coordinate system. It separately measures label efficiency, scanner-confounding robustness, acquisition-branch exclusion, and semantic preservation under explicit scanner counterfactuals.

## Frozen factorizer campaign

Exactly eight factorizer fits are scheduled: dataset seeds 4301 and 5301, linear and nonlinear renderers, and model seeds 2201 and 2202. Every fit uses only the original `crossed_target_prototype` family, a 32-dimensional biological code, no whitening, 40 training identities, and 20 entirely unseen identities. The original epochs, objectives, prototype mechanism, optimizer, and deterministic settings are loaded from the frozen calibrated diagnostic. Each fit must reproduce its matching calibrated run within the existing deterministic tolerance; replication or identity-split failure closes execution.

Task labels, oracle biological latents, and task teachers never enter factorizer fitting, checkpoint selection, reconstruction objectives, scanner prototypes, or representation normalization. They are evaluation-only.

## Frozen task calibration and task battery

A deterministic independent sample of 100,000 eight-dimensional standard-normal biological latents freezes task normalization, teacher normalization, and five population-quintile thresholds. It is not drawn from any experimental identity partition.

The four tasks are:

1. Four standardized linear targets `b @ A`, where the deterministic rank-four 8-by-4 matrix has singular values 0.8, 0.9333, 1.0667, and 1.2.
2. Four standardized targets from a fixed 8-16-16-4 GELU teacher plus deterministic Gaussian task noise with standard deviation 0.01.
3. Four standardized interaction targets: `b0*b1 + 0.5*b4`, `b2*b3 + 0.5*b5`, `b4*b5 + 0.5*b6`, and `b6*b7 + 0.5*b0`.
4. Five biological classes determined by population quintiles of a separate fixed 8-16-16-1 GELU teacher.

All teacher arrays are read-only, all seeds are predeclared, and calibration hashes and teacher parameter hashes are serialized.

## Identity partitions, views, and representation sources

The benchmark reproduces the calibrated geometry split: 32 probe-training identities, eight probe-validation identities, and 20 entirely unseen test identities. Label budgets are nested sets of 8, 16, and 32 identities under two deterministic subset seeds. A budget counts identities, never scanner rows, and identities cannot cross partitions.

Balanced training and validation select one target-independent scanner view per identity, with scanner counts differing by at most one. Unseen balanced evaluation uses all five views. The classification confounding regime selects `scanner_id = biological_class` for training and validation. Its anti-confounded evaluation selects `scanner_id = (biological_class + 2) mod 5`, while all-scanner and per-scanner test results remain visible.

Seven sources are evaluated: biological code, acquisition code, their concatenation, standardized raw observation, scanner-centered observation, evaluation-only oracle biological latent, and an identity-level permuted biological-code control. Scalers and scanner means use only the applicable labeled training identities. The combined code is reported but is not treated as scanner invariant.

## Probes and label efficiency

Each continuous task uses standardized Ridge with alpha `1e-3` and the frozen calibrated residual nonlinear regressor with seeds 7203 and 7204. Epoch zero is eligible and selection uses only identity-disjoint validation data. The classification task uses multinomial logistic regression and the frozen shallow classifier with seeds 7301, 7302, and 7303.

For each budget, source, and probe, the result records overall, per-output, per-scanner, worst-scanner, identity-averaged, and scanner-view-stability metrics as applicable. Label efficiency is the normalized trapezoidal area under performance versus log identity budget, plus the 8-to-32 gain and gaps to oracle, raw, and scanner-centered controls.

## Predeclared interpretations

Regression sufficiency requires two finite residual seeds, median test R² at least 0.80, no more than 0.10 below the oracle median, and median worst-scanner R² at least 0.70. Classification sufficiency analogously requires three finite nonlinear seeds, median balanced accuracy at least 0.80, no more than 0.10 below oracle, and median worst-scanner balanced accuracy at least 0.70.

Broad biological task sufficiency requires all four tasks, acquisition-code failure on at least three tasks, rejection of the permuted control on every task, and finite metrics. Acquisition-branch exclusion is stricter: acquisition performance must remain below R² 0.10 for every regression task and below chance plus 0.10 for classification.

Scanner-confounding robustness requires anti-confounded balanced accuracy at least 0.70, a drop from all-scanner accuracy no greater than 0.10, worst-scanner accuracy at least 0.65, and median scanner-view disagreement no greater than 0.10. Shortcut susceptibility requires validation accuracy at least 0.80 and a validation-to-anti-confounded drop of at least 0.20.

At the full budget, each unseen source observation is encoded, decoded under every requested target scanner prototype, and re-encoded before applying the fitted biological task probes. Counterfactual preservation requires a performance drop no greater than 0.10 and a worst ordered scanner-pair result above the task-specific minimum (0.70 for regression R² and classification balanced accuracy).

Scientific underperformance maps to one of the complete scientific statuses. The failure status is reserved for integrity, execution, exact-replication, identity-leakage, or non-finite-output failures. No outcome can overwrite a prior status or make canonical generator-coordinate recovery a requirement.

## Reproduction

```powershell
py -m experiments.paired_acquisition.run_task_defined_biological_sufficiency `
  --identifiability-audit "results\finite_sample_whitening_identifiability_audit_20260802T222234\finite_sample_whitening_identifiability_audit_result.json" `
  --device cuda `
  --output-root "results\task_defined_biological_sufficiency_<timestamp>"
```

The output directory must not already exist. The JSON result, heterogeneous summary CSV, and manifest are written atomically, with a canonical internal result hash and before/after frozen-artifact verification.
