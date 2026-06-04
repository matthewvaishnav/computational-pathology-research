# Pillar 2 Bounded Resolution: Communication Overhead

**Status:** bounded-resolution claim
**Pillar:** Massive communication overhead
**Scope:** Camelyon17/WILDS feature/head federation versus full ResNet18 federation
**Clinical status:** research-only; not clinically validated; not deployment-ready

---

## Claim

For the Camelyon17 feature/head federation regime studied in this repository, the communication-overhead barrier is resolved under explicit accounting assumptions.

Key result:

- 100-round fp32 full ResNet18 federation across 3 source clients: 24.98 GB communicated.
- 100-round fp32 feature/head federation across 3 source clients: 2.35 MB communicated.
- Full-model traffic is approximately 10,894x larger.

Within the same feature/head communication budget, better source-center weighting improves held-out external-center performance:

- FedAvg-style equal-patch held-out test accuracy: 0.9052.
- Equal-client held-out test accuracy: 0.9318, a +2.66 percentage-point gain.
- Downweight-dominant-center held-out test accuracy: 0.9322, a +2.70 percentage-point gain.

Therefore, in this bounded setting, communication overhead is not merely reduced; it is reduced by orders of magnitude while preserving and improving held-out external-center accuracy.

---

## Why this is the strongest pillar to resolve first

The heterogeneity pillar depends on empirical generalization and cannot be fully closed without broader clinical validation.

The privacy pillar cannot be closed without formal DP accounting or leakage attacks.

The infrastructure pillar cannot be closed without real deployment or containerized orchestration tests.

The communication pillar is different. Under a stated model size, precision, client count, and number of rounds, communication cost is exactly auditable.

---

## Assumptions

Communication cost is computed as:

    parameters x clients x rounds x bytes_per_parameter x 2

The factor of 2 counts:

- server-to-client model download
- client-to-server update upload

The bounded claim uses:

- clients: 3
- rounds: 100
- precision: fp32
- bytes per parameter: 4
- full model: binary ResNet18
- full model parameters: 11,177,538
- feature/head model: 512-to-2 logistic head
- feature/head parameters: 1,026

---

## Communication accounting

Full ResNet18 federation:

    11,177,538 parameters x 3 clients x 100 rounds x 4 bytes x 2
    = 26,826,091,200 bytes
    = 24.98 GB

Feature/head federation:

    1,026 parameters x 3 clients x 100 rounds x 4 bytes x 2
    = 2,462,400 bytes
    = 2.35 MB

Traffic ratio:

    24.98 GB / 2.35 MB ? 10,894x

---

## Accuracy-per-communication result

The Camelyon17-trained ResNet18 feature experiment provides the empirical performance side.

All three policies use the same feature/head communication budget:

    communication = 2.35 MB for 100 fp32 rounds across 3 clients

But held-out test accuracy differs:

| Policy | Held-out test accuracy | Gain vs FedAvg-style |
|---|---:|---:|
| FedAvg-style equal-patch | 0.9052 | 0.0000 |
| Equal-client | 0.9318 | +0.0266 |
| Downweight-dominant-center | 0.9322 | +0.0270 |

This means the accuracy gains are not purchased by additional communication. They come from better weighting under the same communication budget.

---

## What is solved

Solved in this bounded setting:

- Full-model communication cost is explicitly quantified.
- Feature/head communication cost is explicitly quantified.
- Feature/head communication is approximately 10,894x smaller than full ResNet18 communication under the same client/round/precision assumptions.
- Equal-client and downweight-dominant policies improve held-out test accuracy under the same feature/head communication budget.
- The communication-overhead objection is resolved for the feature/head Camelyon17 validation regime.

---

## What is not solved

This does not solve:

- full iterative FL communication for all pathology models
- foundation-model communication overhead
- hospital firewall constraints
- bandwidth variability
- secure aggregation overhead
- DP overhead
- real deployment wall-clock performance
- clinical deployment readiness

---

## Final bounded-resolution statement

Within the Camelyon17 feature/head federated validation regime, communication overhead is no longer an open blocker. The feature/head approach reduces communication by approximately 10,894x compared with full ResNet18 federation under matched round/client/precision assumptions, and improved weighting policies increase held-out external-center accuracy without increasing communication.

This resolves Pillar 2 only within the stated experimental scope.
