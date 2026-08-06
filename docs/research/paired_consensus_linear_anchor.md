# Paired-consensus linear anchor

## Scientific scope

The frozen task benchmark remains `complete_task_defined_biological_sufficiency_unsupported`, and the instrument-power audit remains `complete_original_task_benchmark_partially_instrument_valid`. Only the frozen linear biological task is admissible as a primary representation endpoint. Nonlinear-teacher, interaction, classification, and ineligible counterfactual conclusions remain unresolved.

This experiment asks whether a label-free paired-acquisition consensus objective can make simple identity-dependent observation content accessible from the unchanged 32-dimensional biological code. Success would demonstrate task-agnostic linear accessibility in this synthetic paired-acquisition setting. It would not establish canonical biological coordinates or pathology, clinical, stain, site, cohort, vendor, or endpoint generalization. Failure would show only that this simple consensus anchor is insufficient.

## Consensus target

For each scanner, its mean observation is fitted using only the 40 factorizer-training identities. Each observation is scanner-centered, and the five centered views of an identity are averaged once to form a 64-dimensional identity consensus. The per-dimension consensus mean and scale are fitted from the 40 distinct training-identity consensus vectors, with a fixed scale floor of `1e-6`. The resulting standardized target is repeated across the five views of its identity and detached before training.

The constructor accepts only the synthetic dataset. Biological latents, task labels, teachers, classes, and oracle representations are absent. Scanner means, scaler arrays, consensus arrays, identity lists, and per-view targets are hashed.

Before any model initialization, the consensus target must reproduce the frozen linear task under the exact 32-training/8-validation/20-unseen identity design. The preflight uses the frozen Ridge and residual probes, raw, scanner-centered and oracle controls, and an identity-permuted consensus. Failure produces the scientific target-inadmissible status and zero factorizer fits.

## Isolated model families

The campaign contains exactly eight runs for each of three families:

1. `crossed_target_baseline`: the exact original scanner-prototype factorizer, with no head or consensus loss.
2. `nonlinear_consensus_anchor`: the unchanged factorizer plus a 32-128-128-64 two-GELU head.
3. `linear_consensus_anchor`: the unchanged factorizer plus one biased 32-to-64 affine layer.

Only the head and its MSE contribution differ. The anchor weight is fixed at `0.25`. The head output never enters the FiLM observation decoder, scanner prototypes, acquisition code, biological consistency, or primary task probe. The original decoder always receives the original biological code.

Training records the original objective, raw and weighted consensus losses, total objective, encoder gradient norm attributable to the consensus loss, head parameter count, prediction variance, training consensus R², and final epoch history.

## Evaluation

All 24 runs retain the calibrated operational evaluation: two-axis intervention, confidence-interval transfer evidence, ordered scanner pairs, repeated observed/permuted-null scanner probes, acquisition exclusion, prototype invariance, overall and worst-pair identity retrieval, and independent diagnostic decoding. The eight baseline runs must reproduce the frozen calibrated references or execution fails closed.

The primary linear task evaluates biological code, isolated head prediction, acquisition code, combined code, raw observation, scanner-centered observation, oracle latent, and identity-permuted biological code at 8, 16 and 32 labeled identities under two nested subset seeds. Ridge and both residual seeds remain separate. Full-budget sufficiency, acquisition exclusion, label efficiency, worst-scanner performance, identity-averaged performance, and scanner-view prediction variance use their predeclared thresholds.

Anchored head predictions are evaluated against consensus targets on training, validation and unseen identities. The mechanism diagnostic includes overall and per-dimension R², MSE, worst scanner, identity averaging, view variance and covariance rank. For the linear head, numerical composition verifies that an affine task map after the head equals one affine map from the biological code, with no hidden nonlinearity.

Counterfactual linear-task preservation is evaluated only if direct biological-code R² first reaches 0.70. Eligible runs decode each unseen source under every requested non-source scanner prototype, re-encode the generated observation and apply the fitted full-budget linear probes. Ineligible results are not called failures.

## Interpretation boundaries

Family-level interpretation requires the complete eight-run pattern. A linear mechanism requires all linear-anchor conditions, including generalized consensus prediction, task sufficiency and efficiency, operational preservation, acquisition exclusion, and eligible preserved counterfactuals. Nonlinear-head success can establish a paired-consensus objective effect but cannot be labeled a linear-accessibility mechanism. Improvement accompanied by operational damage is a trade-off; heterogeneous conditions produce a mixed status; poor performance is a scientific result rather than execution failure.

No prior result, threshold, task, architecture, or artifact is modified.

## Execution

```powershell
py -m experiments.paired_acquisition.run_paired_consensus_linear_anchor `
  --power-audit "results\task_benchmark_instrument_power_audit_20260803T102605\task_benchmark_instrument_power_audit_result.json" `
  --device cuda `
  --output-root "results\paired_consensus_linear_anchor_<timestamp>"
```

The output directory must be new. Result JSON, heterogeneous summary CSV, and manifest are written atomically with canonical internal hashes and before/after frozen-artifact verification.
