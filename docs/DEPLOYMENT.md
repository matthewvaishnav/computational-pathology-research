# Historical Deployment Prototype Notice

**Retired as a production guide:** August 2, 2026

The former page described a development, staging, and production Kubernetes
pipeline as though it were an active validated deployment system. The repository
does not currently establish that those environments, clusters, credentials,
images, rollback paths, monitoring systems, or service-level properties exist or
have been validated.

## Current interpretation

Deployment-related files in this repository are historical engineering
prototypes, examples, or research infrastructure. They may illustrate Docker,
Kubernetes, CI/CD, packaging, configuration, or service orchestration patterns.

They do not establish:

- an active production service;
- successful deployment to a real cluster;
- high availability, rollback reliability, or disaster recovery;
- secure secret management;
- performance or load capacity;
- clinical or hospital deployment;
- regulatory readiness; or
- suitability for processing protected health information.

## Safe use

Before using any deployment artifact, review it from first principles, pin and
audit dependencies, apply least-privilege permissions, validate network and
secret boundaries, test rollback and recovery, and perform an independent
security review in the intended environment.

Do not copy historical commands that create cluster-admin service accounts or
embed broad credentials without a separate threat model and operational review.

## Research boundary

The repository’s promoted scientific evidence does not depend on a production
Kubernetes deployment. Experiment execution and evidence validation are separate
from clinical service deployment.

Use [`CURRENT_STATUS.md`](CURRENT_STATUS.md) and the repository-root
[`CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md) for the current record.
