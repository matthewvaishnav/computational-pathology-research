# GitHub Actions workflows

This directory contains the repository's automated engineering, security, publication, and release checks.

The workflow files are part of a **research codebase**. The presence of Docker, Kubernetes, cloud, API, security, or deployment automation does not establish clinical validation, hospital deployment, regulatory readiness, or production use. Public interpretation is bounded by [`CLAIM_BOUNDARY.md`](../../CLAIM_BOUNDARY.md).

## Core engineering checks

### `ci.yml` — CI

Runs on pushes and pull requests to `main` and `develop`, plus manual dispatch.

The current CI performs:

- changed-file Python formatting and lint checks;
- static type checking;
- security checks;
- a Linux/Windows Python test matrix;
- installed-package import validation, including the optional foundation-model research package;
- Docker image build validation;
- documentation/YAML checks on pull requests;
- a quick synthetic/demo smoke on `main`;
- coverage artifacts on `main`.

The primary test matrix currently uses Ubuntu with Python 3.10 and 3.11 plus Windows with Python 3.10. The comprehensive weekly matrix is separate.

### `quick-check.yml` — Quick Check

Fast Ubuntu/Python 3.10 feedback on pushes and pull requests to `main` and `develop`.

### `comprehensive-test.yml` — Comprehensive Test

Weekly and manually triggered broader platform test matrix. It is intentionally separate from the faster pull-request path.

### `repository-audit.yml` — Repository Audit

Runs on pull requests to `main` and on manual dispatch. It inventories tracked files, duplicate content, large/generated artifacts, and Python syntax. The audit is diagnostic; it does not delete or rewrite scientific artifacts.

### `root-layout.yml` — Root layout

Checks that root-level files follow the repository layout rules.

## Scientific provenance and publication

### `publish-panf-manuscript.yml` — PA-NF manuscript publication

Builds the canonical PA-NF manuscript and supplement from LaTeX, applies PDF/source quality gates, packages the arXiv source bundle, records checksums, and publishes the direct manuscript site on `gh-pages`.

This workflow owns the canonical public Pages publication path.

### `paired-acquisition-provenance.yml` — Paired-acquisition provenance

Validates the paired-acquisition provenance/release machinery and its protected evidence paths.

### `huggingface-release-validation.yml` — Hugging Face release validation

Validates release registry/tooling and model/evidence bundle tests when the Hugging Face release layer changes.

### `build-arxiv-paper.yml` — arXiv preprint build

Builds the separate arXiv source/PDF artifact for the path-filtered paper source under `paper/arxiv/`.

## Security

### `security.yml` — Security Audit

Runs on `main`, on a daily schedule, and manually.

### `security-scan.yml` — Security Scan

Runs on pushes/pull requests to `main`, weekly, and manually.

### `codeql.yml` — CodeQL Security Scan

Runs weekly and on manual dispatch.

### `dependency-review.yml` — Standalone dependency review

Retained for manual use. Pull-request dependency review is integrated into the main CI path.

## Release and deployment scaffolding

### `release.yml` — Release

Tag/manual release automation for packaged artifacts.

### `docker-publish.yml` — Docker Publish

Manual-only container publishing workflow. It is not evidence that a public or clinical deployment exists.

### `cd.yml` — CD

Research/deployment scaffolding retained for reproducibility and engineering experiments. Treat it as infrastructure code, not as evidence of a validated development/staging/production service.

### `pages.yml` and `mkdocs.yml`

Disabled compatibility stubs. They are retained because repository regression tests reference these workflow paths. They do not own deployment.

## Local verification

For the Python package:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[foundation]"
python -c "import src; import src.foundation.training_pipeline"
pytest tests/security -m "not property and not slow"
```

For root documentation tooling:

```bash
npm ci
npm run docs:build
```

For the historical Docusaurus tree under `website/`, use its own lockfile:

```bash
cd website
npm ci
npm run build
```

Do not deploy either documentation stack directly to `gh-pages`; the canonical publication workflow owns that branch.
