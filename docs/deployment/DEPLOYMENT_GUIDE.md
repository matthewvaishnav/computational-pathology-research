# Historical deployment scaffolding

**Status:** research/engineering reference only.

This repository contains Docker, Kubernetes, cloud, PACS, monitoring, and
deployment-oriented code accumulated during earlier engineering phases. Those
files do not establish an active production service, clinical deployment,
hospital integration, regulatory readiness, or validated security posture.

## Maintained local path

For local integration work, use the maintained research Docker path:

1. copy `.env.docker.example` to an untracked `.env`;
2. replace every `CHANGE_ME` value;
3. run `docker compose config`;
4. start the stack with `docker compose up -d --build`.

See [`../DOCKER.md`](../DOCKER.md) for the current container contract.

## Kubernetes and cloud directories

The `k8s/`, `kubernetes/`, and `cloud/` trees are historical or experimental
infrastructure code. Before using any of them, re-audit dependencies, secret
management, identity/permissions, networking, storage, rollback behavior, image
provenance, and health checks in the target environment.

Tracked Kubernetes Secret manifests intentionally contain **no secret values**.
Real credentials must be injected from an external secret manager or an
untracked local manifest.

## Historical environment template

`docker/production.env.example` is retained only as a historical configuration
reference. It is not a deployable production profile and contains no credential.

## Scientific boundary

None of the repository's promoted scientific results depend on deployment of
these infrastructure prototypes. Current scientific claims are controlled by
[`../../CLAIM_BOUNDARY.md`](../../CLAIM_BOUNDARY.md).
