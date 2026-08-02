# Repository Structure and Research Boundaries

**Updated:** August 2, 2026

## Purpose

This repository is an independent computational-pathology research and
research-engineering codebase. It contains current research packages alongside
historical software modules accumulated during earlier development phases.

Do not infer scientific validity, production readiness, clinical readiness, or
regulatory status from repository size, module count, test count, or the presence
of a software component.

## Current research areas

### Paired-acquisition representation learning

Primary locations include:

- `experiments/paired_acquisition/`
- `experiments/scorpion/`
- `scripts/scorpion/`
- `evidence/paired_acquisition/`
- `docs/research/paired-acquisition-*`

This line studies tissue-oriented and acquisition-oriented neural
representations using matched scanner views, biological-unit-blocked evaluation,
capacity controls, leakage audits, and forward-valid evidence packages.

### Whole-slide neural models

Relevant packages include AttentionMIL, nnMIL, TransMIL, CLAM, and TransnnMIL
implementations. Historical TransnnMIL performance interpretations are withdrawn
pending matched reruns of repaired fusion and topology code.

### Institutional and center-effect studies

PANDA and CAMELYON17/WILDS packages study simulated institutional corruption,
ordinal shift, source weighting, and center-associated representation structure.
They are not complete federated-learning, fairness, privacy, or clinical
validation.

### Provenance and experiment infrastructure

The repository includes:

- deterministic run and release identities;
- source, configuration, input, and artifact hashing;
- append-only ledgers;
- fail-closed validation and resume behavior;
- registered-grid execution;
- metadata-readiness and identifiability audits; and
- versioned evidence packages.

## High-level directory guide

```text
src/                 reusable models, data, training, inference, and utilities
experiments/         research runners and controlled studies
scripts/             analysis, validation, reporting, and maintenance tools
tests/               software, contract, and regression tests
evidence/            promoted forward-valid evidence packages
docs/research/       research protocols, results, audits, and boundaries
research/            standalone design, metadata, and provenance packages
website/             generated public documentation site
paper/               manuscript sources and historical paper artifacts
```

Directory contents evolve. Use Git and the relevant package README rather than
fixed historical module or line counts.

## Historical software areas

The repository may contain APIs, databases, WSI streaming, PACS/DICOM, FHIR,
security, privacy, deployment, federated-learning, multimodal, and monitoring
code. These modules may be useful implementation references or prototypes.

They do not establish:

- live hospital integration;
- production service reliability;
- HIPAA or other compliance;
- validated differential-privacy guarantees;
- FDA, CE, or regulatory readiness;
- clinical workflow benefit; or
- patient safety.

Earlier examples using named hospitals, institutional prestige multipliers, or
product-style “Distributed Medical Intelligence” descriptions are historical and
must not be treated as current research claims.

## Evidence hierarchy

1. Repository-root `CLAIM_BOUNDARY.md`
2. Current status and remediation ledger
3. Versioned forward-valid evidence packages
4. Current research protocols and result pages
5. Exploratory branches and draft PRs
6. Smoke outputs and software fixtures
7. Superseded historical documents and generated artifacts

A lower item cannot override a higher one.

## Current non-claims

The repository does not currently claim:

- novelty, priority, or patentability;
- state-of-the-art or universal superiority;
- complete disentanglement or pure biology;
- production, clinical, PACS, FHIR, privacy, security, or regulatory validation;
- a general law across hospitals or scanners; or
- that all historical files remain scientifically current.

## Recommended entry points

- [`../CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md)
- [`CURRENT_STATUS.md`](CURRENT_STATUS.md)
- [`DOCS_INDEX.md`](DOCS_INDEX.md)
- [`PORTFOLIO_SUMMARY.md`](PORTFOLIO_SUMMARY.md)
- [`research/paired-acquisition-research-engineering-brief.md`](research/paired-acquisition-research-engineering-brief.md)
