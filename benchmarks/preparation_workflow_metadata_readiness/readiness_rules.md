# Readiness rules

## Canonical availability values

Every availability field must be exactly one of:

- `yes`
- `no`
- `partial`
- `unknown`
- `not_applicable`

The audit fails closed on any other value.

## Readiness tiers

### descriptive_inventory

The candidate can be described, but the recorded metadata do not support the target contrast.

### candidate_discovery

The candidate has enough explicit metadata for exploratory structure discovery, but lacks one or more requirements for confirmatory readiness such as complete crossed provenance, immutable source identity, access/license resolution, or batch/order metadata.

### confirmatory_design_candidate

The candidate explicitly records every required identity for at least one target contrast, uses verified evidence, has resolved access and license status, and has no unresolved factor-nesting flag.

This tier is a metadata/design candidacy label, not an experimental result and not proof of statistical power.

## Blocking reasons

The audit may emit:

- `missing_biological_anchor`
- `missing_block_identity`
- `missing_section_identity`
- `missing_preparation_condition`
- `missing_preparation_batch`
- `missing_scanner_identity`
- `missing_scan_batch`
- `missing_workflow_definition`
- `site_not_equivalent_to_workflow`
- `missing_acquisition_order`
- `missing_serial_section_relationship`
- `missing_same_section_scanner_pairing`
- `missing_immutable_source_provenance`
- `factor_nesting_unresolved`
- `metadata_inferred_not_verified`
- `access_or_license_unresolved`

## Contrast-specific logic

### Preparation

Requires verified `yes` for biological unit, block, section, preparation condition, preparation batch, scanner identity, scan batch, and matched serial sections. Partial or unknown values block confirmatory readiness.

### Scanner

Requires verified `yes` for section identity, preparation condition, scanner device identity, same-section paired scans, scan batch, and immutable source identity.

### Post-preparation workflow

Requires verified `yes` for section identity, preparation condition, scanner identity, post-preparation workflow, acquisition order, scan batch, and a valid same-section physical bridge. In this registry, `paired_same_section_scans_available=yes` is the conservative bridge flag. `site_available=yes` cannot substitute for `post_preparation_workflow_available=yes`.

## Global gates

`access_status` and `license_status` must be `yes` for confirmatory candidacy. An evidence status other than `verified_primary_source` or `verified_repository_metadata` blocks confirmatory candidacy. A row whose notes contain the literal token `factor_nesting_unresolved` is treated as unresolved nesting and cannot be confirmatory-ready.

## Fail-closed contradictions

The audit rejects:

- confirmatory claims implied by invalid enum values;
- `paired_same_section_scans_available=yes` when section identity is not `yes`;
- `matched_serial_sections_available=yes` when block or section identity is not `yes`;
- workflow availability `yes` with site-only evidence explicitly marked in notes;
- missing evidence source;
- duplicate dataset IDs.
