# Claim Boundary

This document is the authoritative public claim boundary for the repository. It
overrides older manuscripts, reports, tables, captions, website text, and result
summaries whenever they conflict.

**Scientific audit status:** remediation in progress as of 2026-07-25. Code fixes
do not retroactively validate artifacts produced by older code or analysis
procedures.

## Current safe central claim

> Paired-Acquisition Neural Factorization is a representation-auditing framework
> that uses matched acquisitions of the same tissue to learn a tissue-oriented
> branch and an explicit acquisition branch. Under the tested paired-scanner
> protocols, the tissue-oriented branch contains substantially less linearly
> recoverable scanner identity, the acquisition branch retains strong scanner
> information, and same-region tissue retrieval is largely preserved.

This supports **partial structured separation under the tested conditions**. It
does not establish pure biological factors, information-theoretic independence,
complete scanner invariance, disease biology, diagnostic improvement, clinical
utility, or deployment readiness.

## Evidence that remains active

The following result families remain within the current claim boundary:

- SCORPION experiments on 48 original H&E slides, 480 aligned regions, five
  scanners, and 2,400 image patches, using original-slide-blocked folds.
- Frozen transfer of the SCORPION objective across DINOv2-Base, Phikon, and
  ImageNet ResNet50 representations without backbone-specific tuning.
- External canine SCC paired-scanner evidence restricted to scanner probes,
  cross-scanner agreement, same-region retrieval, pair-integrity controls, and
  explicitly documented geometry and resolution boundaries.
- Controlled synthetic latent-factor experiments where ground-truth factors are
  known.
- Pair-repeat allocation studies under their own matched-budget protocols.
- CAMELYON17 center-subspace and center-weighting studies when described as
  mechanism or centralized source-weighting experiments, not clinical evidence
  or full federated-learning validation.
- PCam as a single-split patch-classification and engineering benchmark only.

The original slide-bootstrap and slide-level sign-flip inference for SCORPION is
not the current inferential standard. Fold-aware inference must be produced by
`scripts/scorpion/analyze_pathoalign_crossfold_v2.py`. Until that output is
published, the large descriptive effects may be reported, but the historical
intervals and extremely small p-values must not be treated as exact independent-
unit inference.

## Evidence withdrawn pending clean rerun

The following results are not current claim evidence:

1. **Historical TransnnMIL fusion and topology results.** The historical
   one-query/one-key attention path was query-invariant, and historical topology
   mode created an unregistered random projection inside each forward pass.
   Historical QWK values remain records of predictions from that old execution
   path; they do not validate genuine TransMIL–nnMIL fusion or a trained topology
   contribution. Repaired models require new matched reruns.
2. **Historical canine biological-category audit.** Category-probe,
   category-purity, category/scanner-ratio, and derived bottleneck category
   conclusions are withdrawn pending the v2 audit. The old audit contained
   test-set preprocessing leakage in one baseline, scale-incomparable probes,
   same-region nearest-neighbour leakage, and a changing rare-class estimand
   across folds.
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

> The current evidence is consistent with partial structured separation.

Not safe:

> The biological branch is scanner-free.

Not safe:

> The acquisition branch is biologically pure.

Not safe:

> The model proves biological and acquisition disentanglement.

Not safe:

> Paired acquisition is the best scanner-removal method.

The strongest historical raw scanner-removal result is the oldstyle centroid/QR
projection. Any future comparison with it must be rerun under the corrected,
matched v2 protocol before category-preservation numbers are promoted.

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

- the canine biological-label audit v2;
- fold-aware SCORPION inference;
- capacity-matched SCORPION objective ablations;
- repaired TransnnMIL controlled reruns;
- forward-valid provenance releases;
- stronger external human-tissue validation for broader biological claims.

The detailed remediation ledger is
`docs/research/scientific-audit-remediation-20260725.md`.
