# Paired-Acquisition Neural Factorization Measurement Validation Protocol v0

## Purpose

The Paired-Acquisition Neural Factorization Oncology Identity Benchmark asks whether pathology representations have learned disease biology or institutional and acquisition shortcuts. This protocol defines how to decide whether those measurements are trustworthy.

A lower scanner, site, stain, or client probe is not sufficient by itself. A representation can make shortcut identity harder to decode by collapsing, discarding useful morphology, or destroying all task-relevant signal. A valid Paired-Acquisition Neural Factorization result must show shortcut suppression together with biological preservation and downstream utility.

## Core counterfactual design

The strongest validation design is a blocked or paired counterfactual:

| Counterfactual | Purpose |
|---|---|
| Same biology, different acquisition | Tests whether the representation follows tissue biology rather than scanner, stain, or site. |
| Different biology, same acquisition | Tests whether the representation avoids grouping unrelated tissue merely because acquisition identity is shared. |
| Same task label, different site or client | Tests whether slide-level utility transfers beyond institutional identity. |
| Different task label, same site or client | Tests whether the model is not using site/client identity as a proxy for the target. |

When paired acquisition is available, matched tissue regions or biological units should be evaluated across scanner, stain, or site changes. When exact pairing is unavailable, patient-, sample-, slide-, or client-blocked splits should be used to avoid leakage.

## Valid measurement rule

A Paired-Acquisition Neural Factorization measurement should only support a representation-identifiability claim if all required conditions hold:

1. Shortcut identity decreases in the biological representation.
2. Biological preservation is maintained or improved.
3. Whole-slide or task utility is maintained or improved, when a downstream task is part of the experiment.
4. Splits, resampling, and confidence intervals are blocked by the biological unit or federated client.
5. Positive and negative controls behave as expected.
6. Collapse checks show that the representation remains informative.

If shortcut probe accuracy decreases but biological preservation also collapses, the result is not valid evidence for Paired-Acquisition Neural Factorization. It is evidence of information destruction.

## Required controls

| Control | Expected behavior | Failure mode detected |
|---|---|---|
| Random shortcut labels | Probe should fall to chance. | Probe leakage, overfitting, or invalid split. |
| Random biological labels | Biological classifier or retrieval should fall to chance. | Leakage or accidental identity duplication. |
| Raw-feature positive control | Scanner, site, stain, or client identity should often be decodable before correction. | No measurable shortcut signal or broken probe. |
| Paired-region positive control | Same-region retrieval should be above random in raw or aligned features. | Broken pairing, broken feature extraction, or invalid region matching. |
| Collapse check | Embedding variance, rank, and retrieval should remain non-degenerate. | Trivial shortcut suppression by dead embeddings. |
| Acquisition-branch check | Acquisition branch should retain acquisition identity if a split representation is used. | Deletion rather than separation. |
| Biological-branch check | Biological branch should suppress acquisition identity while preserving biological identity. | Failed disentanglement. |
| Sample-blocked evaluation | Effects should persist when grouped by sample or patient. | Patch-level leakage or pseudoreplication. |
| Client-blocked evaluation | Effects should persist across held-out clients or institutions where applicable. | Federated client overfitting. |

## Required split discipline

Measurements must avoid patch-level leakage. The minimum acceptable split unit is the biological unit relevant to the claim.

| Claim type | Required blocking unit |
|---|---|
| Patch or region alignment | Region, slide, sample, or patient as appropriate. |
| Paired-acquisition validation | Biological sample or matched region group. |
| Whole-slide task utility | Patient, slide, or sample depending on dataset structure. |
| Federated client identity | Client, institution, or hospital. |
| External validation | Held-out scanner, site, cohort, or client when possible. |

For Paired-Acquisition Neural Factorization claims, confidence intervals and sign tests should be computed over independent biological units, not over patches alone.

## Measurement bundle

Each benchmark run should report the following bundle together. No single metric is sufficient.

| Measurement | Required interpretation |
|---|---|
| Shortcut probe accuracy | Should decrease in the biological representation. |
| Biological preservation | Should remain stable or improve. |
| Same-region or same-sample retrieval | Should remain stable or improve under acquisition changes. |
| Cross-acquisition consistency | Should improve for matched biological units. |
| Effective rank or variance | Should remain non-degenerate. |
| Downstream WSI utility | Should remain stable or improve when slide tasks are evaluated. |
| Calibration shift | Should decrease or remain stable across sites or clients. |
| Acquisition-branch probe | Should retain acquisition signal when separation is claimed. |

## Probe validity requirements

Shortcut probes should be trained only on frozen representations. Probe training must be separated from representation training and from final evaluation where possible.

Required probe reporting:

- probe model family
- train/validation/test split
- biological or client blocking strategy
- class balance
- chance baseline
- confidence interval or blocked uncertainty estimate
- random-label control
- raw-feature comparison

A probe result is not interpretable unless its chance baseline and split discipline are explicit.

## Collapse checks

A lower shortcut probe is invalid if the representation has collapsed. Each run should include at least three of the following:

- embedding variance by dimension
- effective rank
- average pairwise distance
- nearest-neighbor retrieval
- biological label predictability
- reconstruction or branch-consistency diagnostic when available
- downstream WSI utility

A valid biological representation should not merely be shortcut-invariant. It must remain biologically informative.

## Paired-acquisition validity test

For paired acquisition studies, the expected pattern is:

| Space | Desired behavior |
|---|---|
| Raw feature space | Same biology may still separate by scanner, stain, or acquisition source. |
| Paired-Acquisition Neural Factorization biological space | Same biology should move closer across acquisition conditions. |
| Paired-Acquisition Neural Factorization biological space | Scanner, site, or stain probe should decrease. |
| Paired-Acquisition Neural Factorization acquisition space | Scanner, site, or stain probe should remain recoverable. |
| Biological retrieval | Same-region, same-sample, or same-tissue retrieval should be preserved or improved. |

This pattern supports separation. A decrease in scanner probe without biological retrieval preservation does not.

## Federated-client validity test

For federated oncology settings, each client may encode hospital, scanner, stain, cohort, annotation style, and label prevalence. A federated Paired-Acquisition Neural Factorization claim should test:

| Test | Desired behavior |
|---|---|
| Client probe on biological representation | Lower after alignment or separation. |
| Held-client task performance | Preserved or improved. |
| Calibration by client | Reduced shift or no degradation. |
| Client-specific shortcut control | Randomized or permuted client labels should fall to chance. |
| Personalization comparison | Paired-Acquisition Neural Factorization should be compared with standard FL and personalization baselines where possible. |

Federated success means more than training without centralizing data. It means the learned representation is not dominated by client identity.

## Evidence standard

A Paired-Acquisition Neural Factorization result should be labeled according to the evidence level it satisfies:

| Level | Evidence standard |
|---|---|
| Level 0 | Metric reported without controls. Exploratory only. |
| Level 1 | Shortcut probe and biological preservation reported with blocked splits. |
| Level 2 | Required controls pass, collapse is ruled out, and blocked confidence intervals are reported. |
| Level 3 | External scanner, site, cohort, or client validation passes. |
| Level 4 | Independent reproduction or third-party benchmark use. |

Current child evidence packages should be treated as benchmark-building evidence unless they meet the higher levels above.

## Claim boundary

This protocol validates representation measurements. It does not establish clinical diagnostic safety, prospective patient benefit, regulatory readiness, or deployment fitness. Clinical claims require separate clinical validation.

## Short validity rule

A Paired-Acquisition Neural Factorization result is credible only when shortcut identity decreases, biology remains recoverable, task utility does not collapse, and controls prove the measurement is not leakage or representation destruction.
