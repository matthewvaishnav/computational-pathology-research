# Metadata requirements

## Core physical hierarchy

A candidate study must distinguish:

1. biological unit;
2. block or equivalent material anchor;
3. section;
4. preparation condition;
5. scanner/device or defensible acquisition condition;
6. post-preparation workflow exposure when that contrast is requested.

These labels are not interchangeable. In particular, a site label does not define a post-preparation workflow, and scanner model does not establish same-section paired acquisition.

## Preparation contrast

Confirmatory candidacy requires explicit and verified availability of:

- biological-unit identity;
- block identity;
- section identity;
- preparation condition;
- preparation batch;
- scanner identity;
- scan batch;
- matched serial-section relationship.

The design must also have a plausible path to multiple preparation levels within biological units or blocks and multiple scanners across preparation levels. This registry records metadata feasibility, not the final crossing proof.

## Scanner contrast

Confirmatory candidacy requires explicit and verified availability of:

- section identity;
- preparation condition;
- scanner device identity or another defensible scanner condition;
- same-section paired scans;
- scan batch;
- immutable source identity.

Scanner model alone cannot substitute for paired physical provenance.

## Post-preparation workflow contrast

Confirmatory candidacy requires explicit and verified availability of:

- section identity;
- preparation condition;
- scanner identity;
- prospectively defined post-preparation workflow;
- acquisition or workflow exposure order;
- scan batch;
- a same-section repeated workflow exposure or another explicitly justified physical bridge.

A generic site label alone fails this requirement.

## Evidence classes

Allowed evidence status values are:

- `verified_primary_source`
- `verified_repository_metadata`
- `reported_unverified`
- `inferred`
- `unknown`

Only the two `verified_*` states can support confirmatory readiness. `reported_unverified` may support descriptive inventory or candidate discovery. `inferred` and `unknown` never satisfy a confirmatory requirement.

## Interpretation rule

Absence of metadata is not evidence that the underlying factor was absent. The audit reports only what is explicitly supported by the cited evidence source.
