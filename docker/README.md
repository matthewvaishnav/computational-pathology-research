# Historical HistoCore container scaffolding

This directory preserves older container and monitoring work from the HistoCore
phase of the repository. It is **not** the canonical Docker entry point and it is
not evidence of a deployed production or clinical system.

For the maintained local research environment, use:

- [root Docker guide](../docs/DOCKER.md);
- [root Dockerfile](../Dockerfile);
- [local Compose stack](../docker-compose.yml);
- [secret-free environment template](../.env.docker.example).

## Historical files

Files under this directory may still be useful as engineering references, but
they should be treated as prototypes unless a current CI workflow explicitly
validates them.

`production.env.example` is a non-secret historical configuration template.
It deliberately disables clinical-integration flags and leaves secret values
empty. Do not commit a filled copy.

## Claim boundary

Container, monitoring, PACS, FHIR, cloud, or Kubernetes artifacts in this
repository do not establish deployment readiness, hospital operation, security
certification, regulatory compliance, or clinical utility. The repository-root
[`CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md) controls public interpretation.
