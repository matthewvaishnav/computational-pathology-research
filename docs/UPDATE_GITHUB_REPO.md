# Current GitHub Repository Description Guidance

**Updated:** August 2, 2026

The former instructions on this page recommended a production-grade clinical
platform description. That wording is obsolete and must not be reused.

## Recommended repository description

```text
Independent computational pathology neural-network research on paired acquisition, representation auditing, whole-slide models, and reproducible experiment engineering.
```

This description intentionally does not claim:

- clinical deployment or hospital readiness;
- PACS or FHIR validation;
- HIPAA, FDA, CE, or regulatory compliance;
- fixed test counts or coverage figures;
- state-of-the-art or benchmark superiority;
- novelty, priority, or patentability; or
- complete federated-learning or privacy validation.

## Recommended topics

Use research- and implementation-oriented topics:

- `computational-pathology`
- `histopathology`
- `neural-networks`
- `representation-learning`
- `paired-data`
- `multiple-instance-learning`
- `whole-slide-imaging`
- `pytorch`
- `reproducible-research`
- `machine-learning`

Avoid product or validation topics such as `clinical-ai`, `hipaa-compliant`,
`fda-ready`, `production-ready`, or `pacs-integration` unless a future public
record independently validates those claims.

## Repository website

The GitHub Pages website should present the current audited research record and
link prominently to:

- `CLAIM_BOUNDARY.md`;
- `docs/CURRENT_STATUS.md`; and
- `docs/DOCS_INDEX.md`.

## Social-preview guidance

A social-preview image may use the repository title and neutral research terms.
Do not include stale test counts, clinical branding, superiority language,
regulatory icons, patent language, or unpromoted result numbers.

## Maintenance rule

Whenever the repository description, topics, website metadata, or social preview
is changed, verify it against the current repository-root `CLAIM_BOUNDARY.md`.
