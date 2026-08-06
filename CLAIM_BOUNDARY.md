# Claim Boundary

This document is the authoritative public claim boundary for the repository. It
overrides older manuscripts, reports, tables, captions, website text, and result
summaries whenever they conflict.

**Corrected focused-preprint status:** corrected paired-acquisition evidence was
promoted on 2026-07-26 under the forward-valid record at
`evidence/paired_acquisition/corrected-20260726/release_manifest.json`, and the
corrected focused PA-NF preprint was released on 2026-08-06. Scientific-audit
remediation continues for unrelated research lines. Code fixes and corrected
replacements do not retroactively validate older artifacts.

## Current safe central claim

> Under corrected biological-sample-blocked and fold-aware evaluation,
> Paired-Acquisition Neural Factorization substantially reduces linearly
> recoverable scanner identity in its tissue-oriented representation while
> preserving descriptive tissue-category structure and same-region retrieval.
> The acquisition branch retains strong scanner information. These results
> support partial structured separation under the tested conditions, not pure
> biological factors, complete independence, or clinical utility.

This supports **partial structured separation under the tested conditions**. It
does not establish pure biological factors, information-theoretic independence,
complete scanner invariance, disease biology, diagnostic improvement, clinical
utility, or deployment readiness.

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
- Controlled synthetic latent-factor experiments where ground-truth factors are
  known.
- Pair-repeat allocation studies under their own matched-budget protocols.
- CAMELYON17 center-subspace and center-weighting studies when described as
  mechanism or centralized source-weighting experiments, not clinical evidence
  or full federated-learning validation.
- PCam as a single-split patch-classification and engineering benchmark only.

The original slide-bootstrap and slide-level sign-flip inference for SCORPION is
not the current inferential standard. The published fold-aware analysis uses
`scripts/scorpion/analyze_pathoalign_crossfold_v2.py`, averages seeds within
fold/slide/method, and resamples folds before slides. Historical intervals and
extremely small p-values remain withdrawn as exact independent-unit inference.

## Evidence withdrawn pending clean rerun

The following results are not current claim evidence:

1. **Historical TransnnMIL fusion and topology results.** The historical
   one-query/one-key attention path was query-invariant, and historical topology
   mode created an unregistered random projection inside each forward pass.
   Historical QWK values remain records of predictions from that old execution
   path; they do not validate genuine TransMIL–nnMIL fusion or a trained topology
   contribution. Repaired models require new matched reruns.
2. **Historical canine biological-category audit.** Its category-probe,
   category-purity, category/scanner-ratio, and derived bottleneck conclusions
   remain withdrawn. The old audit contained test-set preprocessing leakage in
   one baseline, scale-incomparable probes, same-region nearest-neighbour
   leakage, and a changing rare-class estimand across folds. Only the separately
   published fixed-estimand replacement is active.
3. **Unified separation scoreboard rankings.** Cross-dataset and cross-protocol
   values are an evidence inventory, not a controlled leaderboard.
4. **PCam clinical or superiority claims.** No statement about diagnoses saved,
   lives saved, clinical benefit, clinical readiness, workflow burden,
   state-of-the-art performance, or statistical superiority over unrelated
   published models is allowed.
5. **Any claim that cosine differences prove biological preservation or tissue
   damage.** Cosine agreement is a representation-geometry metric. It may be
   reported descriptively under a matched protocol but cannot by itself establish
   retained or destroyed biology.

## Paired-Acquisition wording

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

> The current evidence is consistent with partial structured separation.

Not safe:

> The biological branch is scanner-free.

Not safe:

> The acquisition branch is biologically pure.

Not safe:

> The model proves biological and acquisition disentanglement.

Not safe:

> Paired acquisition is the best scanner-removal method.

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

> The CAMELYON17 weighted logistic-regression studies validate a complete
> federated-learning algorithm.

Not safe:

> One held-out CAMELYON17 center establishes a general law across hospitals.

## PCam boundary

Safe:

> The documented PCam model achieved 0.9394 ROC AUC and 0.8526 accuracy on one
> official patch-level test split.

Safe:

> Retrospective test-set threshold calculations illustrate the mathematical
> sensitivity-specificity trade-off.

Not safe:

> The selected threshold is validated or recommended for deployment.

Not safe:

> Patch-level false-negative differences correspond to cancers, patients, or
> diagnoses saved.

## Required next evidence

New promoted claims require:

- execution and forward-valid promotion of the prospective paired affine and
  orthogonal-Procrustes comparison before any claim against FEATMAP-style
  harmonization;
- repaired TransnnMIL controlled reruns;
- forward-valid provenance releases for every additional numerical claim;
- stronger external human-tissue validation for broader biological claims.

The corrected canine fixed-estimand audit, fold-aware SCORPION inference,
175-fit capacity-matched campaign, and 450-fit factorial are complete and
promoted only within the boundaries above. The corrected focused preprint at
`https://matthewvaishnav.github.io/computational-pathology-research/paired-acquisition-neural-factorization.pdf`
is the current public PA-NF manuscript. Every earlier focused PA-NF PDF remains
superseded and on audit hold.

The detailed remediation ledger is
`docs/research/scientific-audit-remediation-20260725.md`.
