# Deployment

This page summarizes deployment-oriented documentation for the research platform.

## Current deployment scope

The repository supports engineering deployment workflows for:

- documentation through MkDocs Material and GitHub Pages
- experiment scripts and benchmark runners
- federated learning coordinator/client prototypes
- security and privacy-oriented infrastructure

## Documentation deployment

The public documentation is generated with MkDocs Material.

```bash
python -m mkdocs build
python -m mkdocs serve
```

GitHub Actions deploys the MkDocs site to GitHub Pages.

## Research deployment caution

This project is not clinically validated or regulatory cleared. Any hospital or clinical deployment would require additional review, validation, governance, and compliance work.
