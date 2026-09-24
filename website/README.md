# Historical Docusaurus documentation tree

This directory contains the repository's older Docusaurus documentation implementation.

It is **not** the deployment owner for the current public research site. The canonical PA-NF manuscript site is built and published by `.github/workflows/publish-panf-manuscript.yml`.

Two kinds of content remain here:

- the historical Docusaurus application, retained while documentation surfaces are consolidated;
- `foundations-site/index.html`, which is intentionally consumed by the canonical manuscript publication workflow as the lightweight landing page.

## Local Docusaurus build

Use the committed lockfile:

```bash
npm ci
npm run build
```

The generated site is for local validation only unless deployment ownership is explicitly changed in a reviewed commit.

Do **not** run `docusaurus deploy` against the repository's `gh-pages` branch. That branch is owned by the canonical PA-NF manuscript publication workflow.

See `docs/repository-cleanup/INVENTORY.md` for the ongoing documentation-surface consolidation audit.
