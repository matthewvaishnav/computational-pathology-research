# Full-Program Publication Portfolio

**Date:** 2026-08-04
**Purpose:** which components warrant focused papers, what is ready now, what is
internally reviewable, what requires experiments, which papers share evidence,
and which manuscript should be released first.

## Flagship foundations manuscript

**Title:** *Accountable Neural Aggregation in Computational Pathology: From
Paired-Acquisition Representations to Whole-Slide and Institutional Learning*

**Package:** `manuscripts/computational-pathology-foundations-v1/`

A comprehensive manuscript or research monograph presenting the entire program
as a connected system of representation, slide, and institutional aggregation.
**Status:** internal-review ready (with explicit pending statuses). **Release
first** among the portfolio because it frames all focused papers and carries the
corrected evidence-maturity discipline.

## Focused Paper A — Paired-Acquisition Neural Factorization

- Exact paired acquisitions; explicit tissue-oriented and acquisition branches;
  corrected SCORPION and canine evidence; paired supervision; strong simple
  baselines; synthetic identifiability; the negative neural-increment result;
  the Layer-2 limitation.
- **Ready now:** corrected paired-supervision evidence, strong-baseline
  boundary, negative neural-increment result.
- **Internally reviewable:** yes.
- **Requires experiments before external submission:** validated Layer-2
  swapping metadata (for the swap claim), additional category datasets.
- **Shares evidence with:** flagship manuscript Level I section.
- **Release order:** after flagship (or in parallel, since its evidence is the
  flagship's Level I core).

## Focused Paper B — TransnnMIL

- Dual/multibranch whole-slide architecture; branch-token fusion; hierarchical
  pooling; topology branch; adaptive pruning; matched TransMIL, nnMIL,
  AttentionMIL, concat, gate, and learned-fusion controls; PANDA evaluation;
  computational efficiency and stability.
- **Ready now:** architecture implementation + tests + preliminary stability
  grids.
- **Requires experiments:** repaired matched controlled reruns. **Empirical
  release is pending** until those reruns are completed. Do not reuse historical
  QWK numbers as repaired-model evidence.
- **Shares evidence with:** flagship manuscript Level II section; PANDA studies.
- **Release order:** after the controlled reruns complete.

## Focused Paper C — PathologyFL and FAIR-WEIGHTS-H

- Pathology-specific FL infrastructure; separation of training, validation, and
  monitoring weights; counterfactual contribution; distributional uniqueness;
  uncertainty and anomaly penalties; representation and subgroup constraints;
  heterogeneity and corruption stress testing; PCam execution validation; PANDA
  and CAMELYON17 research boundaries.
- **Ready now:** infrastructure description + PCam smoke + PANDA simulated
  stress + CAMELYON17 centralized proxies (with explicit boundary that they are
  proxies).
- **Requires experiments:** prospective and multi-center validation; real
  multi-center PathologyFL deployment; implementing the specification-only
  protocol elements; dedicated tests for the FAIR-WEIGHTS-H aggregator.
- **Do not claim** demonstrated fairness or clinical readiness.
- **Shares evidence with:** flagship manuscript Level III sections.
- **Release order:** after infrastructure elements are implemented and tested.

## Focused Paper D — Scientific auditing and provenance

Evaluate whether the fail-closed evidence system, immutable releases, exact
result recovery, historical withdrawal, and claim validation warrant a dedicated
reproducibility or research-engineering paper. **Do not force this paper if the
inventory does not support a coherent scientific contribution.** The inventory
does support a coherent research-engineering contribution (the audit system is
itself validated and reproducible); a decision to write it can be made at
portfolio review.

## Shared-evidence disclosure rules

- No claim may be duplicated across papers without clear disclosure (each paper
  must bind its active numbers to the same immutable artifacts and state the
  shared source).
- The negative neural-increment result (flagship Level I, Paper A) is shared;
  each must report it identically and may not reinterpret it as architectural
  invalidation.
- TransnnMIL preliminary QWK numbers (flagship Level II, Paper B, PANDA studies)
  must carry the same "preliminary / pending controlled validation" label
  everywhere.
- FAIR-WEIGHTS-H conditional stress results must carry the same "no established
  superiority" boundary everywhere.

## Recommended release order

1. **Flagship foundations manuscript** (internal review) — frames the program.
2. **Focused Paper A** (PA-NF) — corrected evidence is ready.
3. **Focused Paper C** (PathologyFL / FAIR-WEIGHTS-H) — infrastructure +
   boundaries, after elements are implemented/tested.
4. **Focused Paper B** (TransnnMIL) — after repaired matched reruns.
5. **Focused Paper D** (provenance) — decision at portfolio review.

## Status summary

| Component | Ready now | Internally reviewable | Requires experiments |
| --- | --- | --- | --- |
| Flagship manuscript | yes (internal review) | yes | for public release |
| Paper A (PA-NF) | yes | yes | Layer-2 swap metadata; more datasets |
| Paper B (TransnnMIL) | architecture only | yes (as architecture) | matched controlled reruns |
| Paper C (FL/FAIR-WEIGHTS-H) | infrastructure | yes | implementation completion; multi-center |
| Paper D (provenance) | candidate | yes | portfolio decision |
