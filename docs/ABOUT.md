# About This Research Repository

**Updated:** August 2, 2026

This repository documents independent computational-pathology research and
research engineering by Matthew Vaishnav.

## Primary focus

The central research line is **Paired-Acquisition Neural Factorization**, a neural
representation-learning study using matched acquisitions of the same tissue
regions to examine how frozen pathology embeddings allocate tissue-associated
and scanner-associated information.

The current promoted evidence supports a bounded conclusion: partial structured
separation under the tested paired-acquisition protocols. It does not establish
pure biological factors, complete scanner invariance, causal factor recovery,
diagnostic improvement, or clinical value.

## What the repository contains

- PyTorch neural-network implementations and controlled ablations;
- same-region and biological-unit-blocked experimental designs;
- scanner, retrieval, geometry, leakage, and pair-integrity audits;
- deterministic experiment execution and provenance tooling;
- forward-valid evidence packages and independent validators;
- whole-slide model research, including repaired TransnnMIL implementations;
- metadata-readiness and identifiability audits; and
- historical software prototypes for WSI processing, APIs, PACS/FHIR adapters,
  security, deployment, and federated-learning infrastructure.

Historical software modules are code artifacts, not proof of production,
clinical, security, privacy, compliance, or hospital validation.

## Current promoted evidence

- **SCORPION:** 175/175 registered capacity-matched fits across seven variants,
  five original-slide-blocked folds, and five seeds.
- **Multi-Scanner Canine SCC:** corrected biological-sample-blocked audit and a
  completed 450/450 dimensionality × cross-covariance factorial.

The canine factorial did not support a universal bottleneck dimension or
regularization setting. Negative and null findings are retained as part of the
research record.

## Active but unpromoted work

- paired affine and orthogonal-Procrustes comparison;
- crossed-target scanner-prototype synthetic diagnostics;
- identity-disjoint synthetic generalization; and
- repaired TransnnMIL matched reruns.

Draft pull requests and smoke runs are not promoted pathology-domain evidence.

## Explicit non-claims

This repository does not currently claim:

- novelty, priority, or patentability;
- state-of-the-art or universal superiority;
- clinical diagnosis, patient benefit, or workflow improvement;
- live hospital, PACS, or FHIR deployment;
- HIPAA, FDA, CE, or other regulatory compliance;
- complete federated-learning, fairness, privacy, or security validation; or
- correctness of superseded historical metrics and platform descriptions.

The authoritative public interpretation is
[`CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md). The current project state is in
[`CURRENT_STATUS.md`](CURRENT_STATUS.md).
