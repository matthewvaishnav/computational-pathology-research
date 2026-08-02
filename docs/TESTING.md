# Testing and Validation

**Updated:** August 2, 2026

## Scope

This repository contains a large and heterogeneous test suite covering research
code, experiment contracts, data handling, APIs, historical software modules,
and evidence validators.

Tests establish the behavior exercised by their fixtures and environments. They
do not by themselves establish clinical safety, production readiness, security
certification, privacy compliance, regulatory status, or scientific validity.

## Authoritative test status

Do not rely on fixed historical counts such as “3,006 tests,” “5,071 tests,” or a
single repository-wide coverage percentage. The suite changes over time, some
modules require optional dependencies or external data, and different workflows
run different subsets.

Use the current GitHub Actions results and run the relevant test command from the
commit being evaluated.

## Basic local checks

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
```

A full repository run may require optional system libraries, model packages,
external data, or platform-specific dependencies. Focused research packages
should document their own exact commands and required inputs.

## Evidence-contract testing

The paired-acquisition evidence system includes focused tests for:

- immutable configuration and run identities;
- input, source, and artifact hashes;
- complete registered-grid validation;
- missing, duplicate, unexpected, or corrupted cells;
- append-only execution ledgers;
- fail-closed resume behavior;
- checkpoint and finite-value validation;
- canonical text hashing across platforms;
- analysis-specification binding; and
- independent evidence-package validation.

Passing these tests supports reproducibility and provenance claims defined by the
contract. It does not prove the biological interpretation of a model.

## Scientific-validation rules

Scientific evaluation should additionally enforce:

- biological-unit- or slide-blocked splits;
- fit-only preprocessing, probe fitting, harmonization, and model selection;
- no same-region or same-sample leakage in neighbourhood metrics;
- seed averaging at the correct hierarchy;
- inference at the independent sampling unit;
- prospective thresholds and decision rules;
- equal-capacity or otherwise explicitly bounded controls; and
- separate promotion of smoke, exploratory, confirmatory, and public evidence.

## Smoke tests

A smoke test verifies that a small execution path runs and produces structurally
valid outputs. It is not scientific evidence, does not validate final metrics,
and must not be promoted as a pathology-domain result.

## Historical modules

Tests for PACS, FHIR, DICOM, security, privacy, deployment, or clinical-workflow
modules are software tests using mocks, fixtures, or controlled environments
unless explicitly documented otherwise. They do not establish live hospital
operation, HIPAA compliance, regulatory approval, or patient safety.

## Current public boundary

The repository-root [`CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md) is authoritative.
A passing test must not be used to broaden the scientific or deployment claim
beyond that document.
