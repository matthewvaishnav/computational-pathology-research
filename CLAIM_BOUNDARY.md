# Claim Boundary

This document is the authoritative public claim boundary for the repository. It
overrides older manuscripts, reports, website pages, tables, captions, result
summaries, and child repositories whenever they conflict.

**Scientific audit status — updated 2026-08-02:** corrected paired-acquisition
evidence was promoted through forward-valid releases. Historical code fixes and
new exploratory branches do not retroactively validate older artifacts. Draft
PR outputs, smoke runs, and unpromoted analyses are not public claim evidence.

## Current safe central claim

> Under corrected biological-sample-blocked and fold-aware evaluation,
> Paired-Acquisition Neural Factorization substantially reduces linearly
> recoverable scanner identity in its tissue-oriented representation while
> preserving descriptive tissue-category structure and same-region retrieval.
> The acquisition branch retains strong scanner information. These results
> support partial structured separation under the tested conditions, not pure
> biological factors, complete independence, causal factor recovery, or clinical
> utility.

The repository does not claim novelty, priority, patentability, complete scanner
invariance, disease biology, diagnostic improvement, patient benefit, clinical
readiness, regulatory compliance, or deployment readiness.

## Evidence that remains active

The following result families remain within the current claim boundary:

- SCORPION experiments on 48 original H&E slides, 480 aligned regions, five
  scanners, and 2,400 image patches, using original-slide-blocked folds and the
  current fold-aware two-stage fold/slide bootstrap.
- The separately versioned 175-fit SCORPION capacity-matched campaign, including
  its equal-capacity two-branch control and registered objective ablations.
- The separately versioned 450-fit canine SCC dimensionality × cross-covariance
  factorial, including its negative result: no stable fold-intersection Pareto
  condition and no universal dimensionality or regularization law.
- Frozen transfer of the SCORPION objective across DINOv2-Base, Phikon, and
  ImageNet ResNet50 representations without backbone-specific tuning.
- External canine SCC paired-scanner evidence under biological-sample-blocked
  folds, including the corrected fixed five-category estimand, fit-only probes,
  fit-pool category neighbours with same-region and same-sample exclusions,
  scanner probes, same-region retrieval, pair-integrity controls, and explicitly
  documented geometry and resolution boundaries.
- Controlled synthetic latent-factor and pair-repeat experiments when described
  strictly under their synthetic-data boundaries.
- CAMELYON17 center-subspace and center-weighting studies when described as
  mechanism or centralized source-weighting experiments, not clinical evidence
  or full federated-learning validation.
- PCam as a single-split patch-classification and engineering benchmark only.

The original slide-bootstrap and slide-level sign-flip inference for SCORPION is
not the current inferential standard. The promoted fold-aware analysis averages
seeds within fold, slide, and method, then resamples folds before slides.
Historical intervals and extremely small p-values remain withdrawn as exact
independent-unit inference.

## Evidence not currently promoted

The following are not current public claim evidence:

1. **Draft crossed-target synthetic branches.** The scanner-prototype and
   unseen-identity PRs are exploratory software and synthetic diagnostics. Smoke
   completion or an open execution gate does not establish a final result, does
   not constitute pathology-domain evidence, and does not justify a novelty or
   causal-identification claim.
2. **Prospective paired affine comparison.** The protocol and implementation may
   be public, but no comparative numerical claim is allowed until the complete
   campaign, analysis, provenance, and forward-valid release are independently
   validated.
3. **Historical TransnnMIL fusion and topology results.** The historical
   one-query/one-key attention path was query-invariant, and historical topology
   mode created an unregistered projection during each forward pass. Historical
   QWK values do not validate genuine TransMIL–nnMIL fusion or a trained topology
   contribution. Repaired models require matched reruns.
4. **Historical canine biological-category audit.** Its category-probe,
   neighbourhood-purity, category/scanner-ratio, and derived bottleneck
   conclusions remain withdrawn. Only the separately published fixed-estimand
   replacement is active.
5. **Unified separation scoreboard rankings.** Cross-dataset and cross-protocol
   values are an evidence inventory, not a controlled leaderboard.
6. **PCam clinical or superiority claims.** No statement about diagnoses saved,
   lives saved, clinical benefit, clinical readiness, workflow burden,
   state-of-the-art performance, or statistical superiority over unrelated
   published models is allowed.
7. **Any claim that cosine differences prove biological preservation or tissue
   damage.** Cosine agreement is a representation-geometry metric and cannot by
   itself establish retained or destroyed biology.

## Paired-acquisition wording

Safe:

> The method substantially reduces linearly recoverable scanner identity in the
> tissue-oriented branch while retaining scanner information in an explicit
> acquisition branch.

Safe:

> Same-region retrieval and cross-scanner agreement are largely preserved under
> the tested paired-acquisition protocols.

Safe:

> Under the corrected five-category canine audit, descriptive category balanced
> accuracy and fit-pool category purity in the true-pair tissue-oriented branch
> remain close to the frozen representation.

Safe:

> The current evidence is consistent with partial structured separation under
> the tested conditions.

Not safe:

> The biological branch is scanner-free.

Not safe:

> The acquisition branch is biologically pure.

Not safe:

> The model proves biological and acquisition disentanglement.

Not safe:

> Paired acquisition is the best scanner-removal method.

Not safe:

> The architecture or factorization concept is proven novel, first, or
> patentable.

Under the corrected fixed-estimand protocol, the linear centroid/QR baseline
removes linearly recoverable scanner information more aggressively than the
neural tissue-oriented branch while retaining similar descriptive category
metrics. The neural factorization additionally retains an explicit acquisition
branch with strong scanner information. This is a bounded representation audit,
not a claim that either approach creates pure biology.

## Whole-slide modeling boundary

Safe:

> TransnnMIL is an active whole-slide architecture research line studying
> complementary transformer and gated-attention aggregation.

Safe:

> The repaired canonical implementation uses genuine branch-token fusion and
> must be evaluated against concat, gate, branch-attention, TransMIL, nnMIL, and
> AttentionMIL controls under matched splits, seeds, tuning budgets, and compute.

Not safe:

> The historical TransnnMIL scores prove that combining TransMIL and nnMIL
> improved PANDA grading.

Not safe:

> TransnnMIL is superior to AttentionMIL, TransMIL, or nnMIL.

## Institutional and federated-learning boundary

Safe:

> The PANDA studies evaluate institutional weighting and detector behavior under
> simulated site corruption and ordinal shift.

Safe:

> The CAMELYON17 weighted logistic-regression studies are centralized frozen-
> feature proxies for source-center influence on one held-out center.

Not safe:

> The studies validate a complete federated-learning algorithm, privacy
> guarantee, hospital workflow, or clinical deployment.

Not safe:

> One held-out CAMELYON17 center establishes a general law across hospitals.

## PCam boundary

Safe:

> The documented PCam model achieved 0.9394 ROC AUC and 0.8526 accuracy on one
> official patch-level test split.

Safe:

> Retrospective test-set threshold calculations illustrate a mathematical
> sensitivity-specificity trade-off.

Not safe:

> The selected threshold is validated or recommended for deployment.

Not safe:

> Patch-level false-negative differences correspond to cancers, patients, or
> diagnoses saved.

## Required next evidence

New promoted claims require:

- completion and forward-valid promotion of the paired affine and
  orthogonal-Procrustes comparison before any comparative harmonization claim;
- completion and independent validation of crossed-target synthetic studies,
  with the synthetic-only boundary retained;
- repaired TransnnMIL controlled reruns;
- forward-valid provenance releases for every additional numerical claim; and
- stronger external human-tissue validation before broader biological claims.

The corrected canine fixed-estimand audit, fold-aware SCORPION inference,
175-fit capacity-matched campaign, and 450-fit factorial are complete and
promoted only within the boundaries above. The public manuscript and PDF remain
on scientific-audit hold.

The detailed remediation ledger is
`docs/research/scientific-audit-remediation-20260725.md`.
