---
layout: default
title: Current Status
---

# Current Research Status

**Last updated:** August 2, 2026

The authoritative public interpretation is the repository-root
[`CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md). This page summarizes current work; it
does not expand that boundary.

## Repository identity

This is an independent **computational-pathology research and research-engineering
repository**. It is not a clinically validated product, hospital platform,
regulated medical device, HIPAA compliance certification, or production
deployment claim.

Historical pages describing a production-grade clinical platform, benchmark
superiority, validated PACS operation, privacy guarantees, or hospital readiness
are obsolete and are not current evidence.

## Promoted paired-acquisition evidence

### SCORPION

The corrected SCORPION program uses 2,400 image patches from 480 aligned tissue
regions on 48 original H&E slides scanned by five devices. Evaluation is blocked
by original slide.

The capacity-matched campaign completed **175/175 registered fits** across seven
variants, five folds, and five seeds. Against the equal-capacity two-branch
control, the full model:

- reduced tissue-branch scanner balanced accuracy by `0.3108`, with a fold-aware
  95% interval of `[-0.3346, -0.2858]`;
- preserved average and worst same-region retrieval within the registered `0.02`
  noninferiority margin; and
- retained strong acquisition-branch scanner information (`0.8565` accuracy).

This supports partial structured separation under the tested protocol. It does
not establish pure biological factors, complete independence, or clinical value.

### External canine SCC

The corrected fixed-estimand audit uses biological-sample-blocked folds,
fit-only preprocessing and probes, and fit-pool neighbourhood evaluation with
same-region and same-sample exclusions.

The separate dimensionality × cross-covariance campaign completed **450/450
registered cells**. It found no stable fold-intersection Pareto condition and no
universal bottleneck dimension or regularization law. This negative result is
part of the promoted evidence record.

## Work that is active but not promoted evidence

### Paired affine comparison

A prospective comparison of centroid translation, orthogonal Procrustes,
unregularized affine, ridge affine, and the neural factorization is specified on
the frozen SCORPION folds. No comparative numerical claim is authorized until
the full run, analysis, provenance checks, and forward-valid release are
complete.

### Crossed-target synthetic diagnostics

Draft PR #74 tests whether scanner prototypes and explicit crossed-target
supervision recover known synthetic factors. Its smoke gate completed 16/16
small fits and opened execution of the full deterministic grid. That smoke result
is an engineering gate, not a final scientific result.

Draft PR #75 tests identity-disjoint generalization on synthetic data. It remains
post-confirmatory exploratory work and does not test pathology-domain features or
unseen scanners.

Neither PR establishes novelty, causal identification, or pathology-domain
validity.

### TransnnMIL

The canonical fusion and topology defects were repaired. Historical QWK results
remain withdrawn as evidence for genuine branch fusion or topology. New matched
reruns against standalone and controlled fusion baselines are still required.

### Institutional weighting and federated studies

PANDA studies remain simulated-institution stress tests. Current CAMELYON17
weighting studies are centralized frozen-feature source-weighting proxies on one
held-out center. They are not full federated-learning, privacy, fairness, or
clinical validations.

### PCam

PCam remains a single official patch-level test-split engineering benchmark. The
documented result is `0.9394` ROC AUC and `0.8526` accuracy. Cross-paper
superiority, clinical-threshold, diagnoses-saved, and deployment claims remain
withdrawn.

## Public manuscript status

The paired-acquisition manuscript and public PDF remain on scientific-audit hold.
They must be rebuilt from corrected forward-valid evidence. Older manuscripts,
PDFs, captions, tables, website pages, and child repositories are subordinate to
the current claim boundary.

## Current priorities

1. Complete and independently validate the paired affine comparison.
2. Complete the crossed-target synthetic grids while retaining the synthetic-only
   boundary.
3. Run repaired TransnnMIL controlled comparisons.
4. Produce forward-valid releases for every additional numerical claim.
5. Obtain stronger external human-tissue evidence before broader biological
   interpretation.
6. Keep public documentation synchronized with the authoritative claim boundary.

## Explicit non-claims

The repository does not currently establish:

- novelty, priority, or patentability;
- pure biology or complete disentanglement;
- clinical utility, patient benefit, hospital readiness, or regulatory status;
- complete federated-learning or privacy validation;
- universal superiority over other architectures or harmonization methods; or
- correctness of unpromoted smoke, draft-PR, or historical result artifacts.
