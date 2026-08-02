# PACS Integration Prototype Notice

**Reclassified:** August 2, 2026

The repository contains historical DICOM and PACS integration code. The former
page described that code as production-ready, clinical-grade, multi-vendor
validated, HIPAA compliant, and suitable for automated hospital workflows. Those
claims are withdrawn.

## Current interpretation

The PACS-related modules are software prototypes and testable interfaces for
research engineering. They may include code paths for query/retrieve/store,
configuration, retries, vendor-specific adapters, notifications, audit records,
and workflow orchestration.

Code presence and mocked tests do not establish interoperability with any real
PACS installation or vendor product.

## Not validated

The repository does not currently establish:

- successful C-FIND, C-MOVE, or C-STORE operation against a live hospital PACS;
- conformance with a specific vendor implementation;
- TLS or certificate configuration in a production environment;
- reliable failover, dead-letter handling, concurrency, or transfer throughput;
- secure handling of protected health information;
- seven-year audit retention or legally sufficient audit records;
- HL7 notification delivery;
- automated storage of AI results for clinical review;
- clinical safety, regulatory readiness, or hospital workflow integration; or
- HIPAA compliance.

Named vendor support in historical code or documentation should be interpreted
as adapter intent, not vendor certification or validation.

## Requirements before real use

Any real integration would require a site-specific DICOM conformance review,
network and identity architecture, least-privilege service accounts, certificate
and key management, PHI governance, vendor testing, failure-mode analysis,
monitoring, incident response, legal review, and validation in the intended
clinical environment.

The repository’s promoted scientific evidence does not depend on PACS
integration.

Use the repository-root [`CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md) for the
current public boundary.
