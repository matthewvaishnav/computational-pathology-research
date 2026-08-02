---
layout: default
title: Security Engineering Notes
---

# Security Engineering Notes

**Claim boundary updated:** August 2, 2026

The repository contains security-oriented code, static-analysis workflows,
input-validation utilities, authentication prototypes, path-handling fixes, and
historical audit notes.

The former page stated that these measures ensured HIPAA compliance in
production clinical environments. That claim is withdrawn.

## Supported interpretation

Individual commits and tests may demonstrate that specific software defects were
identified and that bounded mitigations were implemented for the tested code
paths. Examples can include path normalization, parameterized queries, upload
validation, authentication checks, timeout handling, dependency scanning, and
safer serialization.

Such changes are useful software engineering. They do not establish the security
of the complete repository or any deployed system.

## Not established

The repository does not currently provide or certify:

- HIPAA compliance;
- a completed threat model;
- independent penetration testing;
- secure production configuration;
- vulnerability-free dependencies;
- correct identity, access, key, certificate, or secret management;
- legally sufficient audit logging or retention;
- safe processing of protected health information;
- validated encryption or privacy guarantees across a deployed workflow;
- incident-response readiness; or
- regulatory cybersecurity compliance.

A historical count of “vulnerabilities fixed” is not a current security rating.
The codebase, dependencies, deployment environment, and threat surface continue
to change.

## Safe engineering use

Before deploying any component:

1. define assets, actors, trust boundaries, and abuse cases;
2. minimize enabled services and privileges;
3. audit dependencies and build provenance;
4. configure secrets and cryptographic material outside the repository;
5. test authorization, isolation, logging, backup, and recovery;
6. perform independent code review and penetration testing; and
7. obtain legal, privacy, security, and organizational approval for the intended
   data and environment.

## Research boundary

Security-related tests and modules do not broaden the scientific evidence or
clinical-use boundary. The repository-root
[`CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md) remains authoritative.
