# Literature Positioning and Public Claim Discipline

**Updated:** August 2, 2026

This page no longer presents an exhaustive literature review or asserts that any
repository method is first, unique, novel, patentable, or without prior
comparators.

Literature coverage in a public code repository is necessarily incomplete and
can become stale. The repository therefore positions its work by **tested design,
evidence level, and limitations**, not by priority claims.

## Paired-Acquisition Neural Factorization

Publicly supported positioning:

- it studies frozen pathology representations using matched acquisitions of the
  same tissue regions;
- it learns a tissue-oriented branch with reduced scanner recoverability and an
  explicit scanner-retaining acquisition branch;
- it reconstructs the input representation jointly from both branches;
- it is evaluated with biological-unit-blocked splits, capacity controls,
  leakage checks, retrieval measures, scanner probes, and forward-valid evidence
  packages; and
- the current evidence supports partial structured separation under the tested
  protocols.

Unsupported positioning:

- first or unique factorization of anatomy, biology, modality, style, acquisition,
  nuisance, or content information;
- proof of pure biological factors or complete scanner invariance;
- universal superiority to harmonization, residualization, adversarial, affine,
  variational, generative, or other representation-learning approaches; or
- novelty, priority, patentability, or freedom-to-operate conclusions.

The prospective paired affine comparison and crossed-target synthetic studies
remain unpromoted until their complete evidence packages are validated.

## TransnnMIL

Publicly supported positioning:

- TransnnMIL is an active whole-slide architecture research line combining
  transformer-style and gated-attention branches;
- implementation defects in the historical fusion and topology paths were
  identified and repaired; and
- genuine architectural claims require new matched comparisons against
  standalone and controlled fusion baselines.

Unsupported positioning:

- historical QWK values as evidence that fusion improved performance;
- superiority over AttentionMIL, nnMIL, TransMIL, CLAM, or other methods; or
- uniqueness of hierarchical, topological, attention, or pruning components.

## Institutional weighting and federated studies

Publicly supported positioning:

- PANDA work studies simulated institutional corruption and ordinal shift;
- CAMELYON17 work studies centralized frozen-feature source influence on one
  held-out center; and
- these are mechanism and research-infrastructure studies.

Unsupported positioning:

- complete federated-learning validation;
- verified fairness, privacy, or security guarantees;
- hospital or multi-institutional clinical validation; or
- uniqueness or novelty of institutional weighting formulations.

## PCam

The documented PCam result is a single official patch-level test-split result:
`0.9394` ROC AUC and `0.8526` accuracy.

It must not be used to claim:

- state-of-the-art performance;
- statistical superiority over unrelated published systems;
- slide-level or patient-level performance;
- clinical thresholds, diagnoses saved, or workflow benefit; or
- clinical deployment readiness.

## Public literature-review policy

1. Treat repository literature summaries as non-exhaustive and date-bounded.
2. Prefer primary sources when making scientific comparisons.
3. Separate conceptual similarity from exact experimental equivalence.
4. Do not infer copying, ownership, novelty, or patentability from architectural
   overlap.
5. Do not add private research notes, legal analysis, unpublished dossiers, or
   competitor-specific strategy to the public repository.
6. Update the authoritative claim boundary before promoting any broader public
   interpretation.

The repository-root `CLAIM_BOUNDARY.md` remains authoritative.
