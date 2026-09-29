# PA-NF shortcut-susceptibility and acquisition-conflict preregistration

**Frozen:** 2026-09-29, before inspection of outcomes from these two tests.

This campaign reuses the completed SCORPION capacity-matched PA-NF projections and the frozen DINOv2 source features. It does not retrain PA-NF and does not introduce a biological-category or diagnostic label that SCORPION does not provide.

## Experiment 1: acquisition-shortcut susceptibility

The downstream task is binary same-region versus different-region matching within an original slide. Training pairs come only from non-test slides. Scanner identity is made spuriously predictive of the pair label at frozen correlation levels `0.2, 0.4, 0.6, 0.8, 1.0`; `0.2` is exactly scanner-uniform and `1.0` is deterministic. Held-out test pairs are scanner-balanced and label-balanced.

Primary candidate: `pathoalign_dep20` biological branch. Primary capacity-matched comparator: `two_branch_no_scanner_objectives` biological branch. Raw DINOv2 is a second comparator. The PA-NF acquisition branch is descriptive only.

The primary endpoint is held-out balanced accuracy at correlation `1.0`, with the additional registered requirements that PA-NF exhibit a smaller `0.2 -> 1.0` degradation than the capacity control and remain noninferior to that control at `0.2` within `0.02`.

## Experiment 2: adversarial acquisition-vs-identity retrieval

Every held-out SCORPION image is used exactly once as a query. Correct candidates are the other four scanner views of the same tissue region. Shortcut candidates are all wrong regions from the same original slide acquired on the query scanner. Cosine similarity is used after row-wise normalization.

The primary endpoint is conflict top-1 accuracy: the best correct cross-scanner match must outrank the best wrong same-scanner match. The hard-negative cosine margin is also frozen as a required endpoint.

## Inference

Both experiments use a paired hierarchical bootstrap over the five frozen projection seeds and 48 original slides, with 100,000 draws and seed `20260929`. Patch rows and region pairs are not treated as independent biological replicates.

## Claim boundary

A passing result may support reduced acquisition-shortcut susceptibility and stronger acquisition-conflict tissue-identity retrieval on the registered SCORPION paired-scanner protocol. It does not establish diagnostic robustness, clinical utility, external-site generalization, pure biological disentanglement, or universal superiority.

The machine-readable specification is:

`experiments/paired_acquisition/pa_nf_shortcut_conflict_spec_20260929.json`
