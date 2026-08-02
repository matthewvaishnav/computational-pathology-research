# Computational Pathology Research Repository Overview

**Last updated:** August 2, 2026

## Scope

This repository is an independent computational-pathology **research and
research-engineering codebase**. Its primary purpose is to develop and audit
neural-network experiments, not to claim a production clinical platform.

The repository includes model code, experiment runners, reproducibility tools,
metadata and provenance audits, software prototypes, and historical engineering
modules. The presence of an API, PACS adapter, security utility, deployment
configuration, or privacy-related component does not establish clinical,
regulatory, security, privacy, or production readiness.

The repository-root `CLAIM_BOUNDARY.md` is authoritative.

## Main research line

### Paired-Acquisition Neural Factorization

The central research question is whether matched acquisitions of the same tissue
region can support partial separation of tissue-associated and
scanner-associated information in frozen pathology embeddings.

The model includes:

- a tissue-oriented neural branch trained for matched-view agreement and reduced
  scanner recoverability;
- an acquisition branch trained to retain scanner information;
- a joint decoder that reconstructs the original embedding; and
- variance, covariance, cross-covariance, and dependence controls.

The current safe interpretation is **partial structured separation under the
tested conditions**. The model is not described as pure biology, complete
scanner invariance, a proven causal factorization, or a universally superior
harmonization method.

## Promoted evidence

### SCORPION

- 48 original H&E slides;
- 480 aligned tissue regions;
- five scanners and 2,400 image patches;
- original-slide-blocked evaluation;
- fold-aware fold-then-slide bootstrap inference; and
- a 175-fit capacity-matched objective campaign.

### Multi-Scanner Canine SCC

- biological-sample-blocked evaluation;
- corrected fixed five-category estimand;
- fit-only preprocessing and probes;
- same-region and same-sample neighbourhood exclusions; and
- a completed 450-cell dimensionality × cross-covariance factorial.

The factorial produced a bounded negative result: no stable universal bottleneck
or regularization optimum was supported.

## Current unpromoted studies

- a prospective paired affine and orthogonal-Procrustes comparison;
- crossed-target scanner-prototype synthetic diagnostics;
- identity-disjoint synthetic generalization; and
- repaired TransnnMIL controlled reruns.

These studies may exist as code or draft PRs without being current numerical
claim evidence.

## Secondary research and engineering areas

The repository also contains work on:

- whole-slide multiple-instance learning;
- foundation-model feature extraction;
- institutional weighting and source-influence analysis;
- metadata readiness and provenance;
- paired-design identifiability;
- PCam patch-level engineering benchmarks;
- experiment orchestration and validation; and
- software prototypes for APIs, WSI handling, clinical interfaces, security, and
  deployment.

These areas have different evidence levels. Software presence is not equivalent
to empirical or clinical validation.

## Historical modules

Large portions of the repository predate the July 2026 scientific audit. Older
pages may use obsolete names, platform framing, benchmark comparisons, or
clinical-deployment language. Such material is historical unless explicitly
reaffirmed by the current claim boundary and a forward-valid evidence package.

## Non-claims

The repository does not currently establish:

- novelty, priority, or patentability;
- production-grade clinical deployment;
- HIPAA, FDA, CE, or other regulatory compliance;
- verified operation with live hospital PACS or FHIR infrastructure;
- complete federated-learning or privacy validation;
- diagnostic or patient benefit; or
- state-of-the-art or universal superiority.

## Recommended reading order

1. `../CLAIM_BOUNDARY.md`
2. `CURRENT_STATUS.md`
3. `research/paired-acquisition-research-engineering-brief.md`
4. `research/scientific-audit-remediation-20260725.md`
5. the relevant versioned evidence package for any numerical claim
