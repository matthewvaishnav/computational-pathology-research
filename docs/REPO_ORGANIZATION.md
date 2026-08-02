# Repository organization

This is a public research codebase. Dated notes, setup guides, recruiter material, experiment reports, and legacy product documents belong under `docs/`, not at the repository root.

## Canonical directories

```text
.github/      CI and publication workflows
configs/      active experiment configurations
data/         small manifests and split metadata
experiments/  reproducible runners and frozen protocols
docs/         canonical documentation and historical archive
figures/      paper and research figures
panda/        PANDA metadata and deterministic splits
paper/        LaTeX source and publication tooling
results/      compact evidence supporting research claims
scripts/      preparation, training, analysis, and audit tools
src/          maintained Python package
tests/        automated tests
```

## Root allowlist

The root is reserved for conventional project entry points and files directly required by tooling:

```text
README.md
LICENSE
CITATION.cff
CLAIM_BOUNDARY.md
CONTRIBUTING.md
SECURITY.md
CHANGELOG.md
VERSION
pyproject.toml
requirements*.txt
Dockerfile
docker-compose*.yml
.env*.example
mkdocs.yml
package.json
package-lock.json
alembic.ini
```

Do not add new loose Markdown reports to the root.

## Documentation layout

```text
docs/
  overview/     research-program orientation
  quickstart/   first-run guides
  models/       model and architecture documentation
  federated/    PathologyFL documentation
  validation/   evaluation design and claim boundaries
  results/      stable result summaries
  research/     protocols and technical research notes
  outreach/     recruiter and public communication material
  engineering/  implementation and maintenance documentation
  setup/        environment-specific setup guides
  roadmap/      active future work
  archive/      dated, superseded, commercial, or noncanonical material
```

Archiving preserves provenance; it does not validate historical claims.

## Rules

1. Put maintained code in `src/`, reproducible runners in `experiments/`, and command-line utilities in `scripts/`.
2. Put every new Markdown report under `docs/`.
3. Keep raw datasets, checkpoints, large feature archives, logs, and generated run directories outside Git.
4. Keep compact summaries and manifests only when they support a reproducible claim.
5. Keep Paired-Acquisition Neural Factorization/SCORPION, TransnnMIL/PANDA, and PathologyFL/CAMELYON17 visible as separate research lines.
6. Archive obsolete deployment, enterprise, and marketing material instead of presenting it as active research infrastructure.

Active Docker entry points remain at the root because CI and existing workflows reference them.
