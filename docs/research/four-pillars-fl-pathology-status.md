# Four Pillars of Federated Learning Failure in Computational Pathology

**Status:** research-progress summary  
**Scope:** Camelyon17/WILDS, PANDA-derived feature experiments, communication/privacy/infrastructure probes  
**Clinical status:** research-only; not clinically validated; not diagnostic software

---

## Summary

This repository is organized around four major barriers to federated learning in computational pathology:

1. Severe data heterogeneity / Non-IID site shift
2. Massive communication overhead
3. Residual privacy risks
4. Implementation and infrastructure barriers

The current work does not fully solve all four. It establishes measurable research artifacts for each pillar and gives the strongest evidence so far for Pillar 1.

---

## Pillar status

| Pillar | Problem | Current status | Verdict |
|---|---|---|---|
| 1 | Severe data heterogeneity / Non-IID site shift | PANDA stress tests plus Camelyon17 external-center validation | Strong progress |
| 2 | Massive communication overhead | Bounded Camelyon17 feature/head communication resolution plus accuracy-per-communication proxy | Bounded resolution |
| 3 | Residual privacy risks | Coefficient-noise privacy-style robustness probe | Started |
| 4 | Implementation / infrastructure barriers | Infrastructure-friction simulation with client speed/dropout/straggler accounting | Started |

---

## Pillar 1: Severe data heterogeneity / Non-IID site shift

### Evidence

The strongest current result stack attacks the assumption that more local sample volume automatically means better federated signal.

Evidence layers:

- Simulated dominant-site pathology stress tests over PANDA-derived features.
- Detector transfer from label-noise calibration to conservative ordinal threshold shift.
- Detector ablation and calibration sensitivity.
- Camelyon17/WILDS natural external-center validation.
- Frozen ImageNet ResNet18 feature baselines.
- Camelyon17-trained supervised ResNet18 feature baselines.

### Camelyon17 frozen ImageNet ResNet18 result

Using frozen ImageNet ResNet18 features:

- FedAvg-style equal-patch held-out test accuracy: 0.8312
- Equal-client held-out test accuracy: 0.9132
- Equal-client gain: +8.20 percentage points
- Downweight-dominant gain: +7.82 percentage points
- Validation-aware detector-switch gain: +6.58 percentage points
- Threshold sweep: 43 / 112 settings robust-positive

### Camelyon17-trained supervised ResNet18 result

Using features from a Camelyon17 source-trained ResNet18 checkpoint:

- FedAvg-style equal-patch held-out test accuracy: 0.9052
- Equal-client held-out test accuracy: 0.9318
- Downweight-dominant held-out test accuracy: 0.9322
- Equal-client gain: +2.66 percentage points
- Downweight-dominant gain: +2.70 percentage points
- FedAvg-style source-train accuracy: 0.9991

### Interpretation

Pillar 1 is the most advanced pillar. The evidence supports the narrow claim that sample-volume weighting can overfit source-center structure and generalize worse to held-out centers. The effect appears both in simulated pathology-feature stress tests and in Camelyon17 external-center validation.

### Claim boundary

This does not prove clinical readiness, universal superiority of equal-client weighting, or final FL deployment safety.

---

## Pillar 2: Massive communication overhead

### Bounded-resolution status

Pillar 2 has a bounded resolution for the Camelyon17 feature/head federation regime.

This does not solve all FL communication overhead in computational pathology. It resolves the communication-overhead objection only under the explicit assumptions used in this repository: 3 source clients, 100 fp32 rounds, binary ResNet18 full-model comparison, and 512-to-2 feature/head federation.

The bounded-resolution artifact is documented here:

[../research/pillar2-communication-bounded-resolution.md](pillar2-communication-bounded-resolution.md)

### Evidence

A Camelyon17 communication-overhead analysis was added to quantify full-model versus feature/head communication.

For 100 fp32 rounds across 3 source clients:

- Full ResNet18 federation: 24.98 GB
- Feature/head federation: 2.35 MB
- Full-model traffic is approximately 10,894x larger.

A detector-style reduced-round accounting showed:

- Diagnose/switch after 5 rounds: 95% communication reduction versus 100-round full-model FL
- Diagnose/switch after 10 rounds: 90% reduction
- Diagnose/switch after 25 rounds: 75% reduction

### Accuracy per communication

Using Camelyon17-trained ResNet18 features under the same feature/head communication budget:

- FedAvg-style held-out test accuracy: 0.9052
- Equal-client held-out test accuracy: 0.9318
- Downweight-dominant held-out test accuracy: 0.9322

The equal-client and downweight-dominant gains are not purchased by additional communication because all three feature/head policies use the same communication budget.

### Interpretation

Pillar 2 is solved only in a bounded sense: communication overhead is no longer an open blocker for the Camelyon17 feature/head validation regime. The broader full-model FL communication problem remains open.

### Next step

Replace the full-model proxy with actual iterative FL runs and measured wall-clock/network costs.

---

## Pillar 3: Residual privacy risks

### Evidence

A Camelyon17 privacy-noise stress test was added.

Setup:

- Feature extractor: Camelyon17 source-trained ResNet18
- Classifier: logistic head
- Stress type: Gaussian noise added to classifier coefficients
- Noise levels: 0.0, 0.01, 0.03, 0.05, 0.10, 0.20
- Noise repeats: 5

At every tested noise level, equal-client and downweight-dominant policies remained positive versus FedAvg-style weighting on held-out test performance.

At the highest tested noise level, noise_std = 0.20:

- Equal-client test accuracy gain: +1.98 percentage points
- Downweight-dominant test accuracy gain: +1.90 percentage points

### Interpretation

Pillar 3 is not solved. This is not formal differential privacy and not a leakage audit. It is a privacy-noise robustness probe.

### Next step

Run a formal privacy-oriented experiment, such as:

- DP-SGD-style noise and clipping accounting
- membership-inference attack testing
- gradient/update inversion stress test
- secure aggregation simulation

---

## Pillar 4: Implementation and infrastructure barriers

### Evidence

A Camelyon17 infrastructure-barrier simulation was added.

It models:

- heterogeneous client compute speeds
- client dropout probability
- synchronous straggler delay
- asynchronous proxy behavior
- communication cost
- detector-style reduced-round operation

The simulation uses three illustrative source clients:

- center_0_fast: 4 minutes per local update
- center_3_medium: 7 minutes per local update
- center_4_slow: 13 minutes per local update

### Interpretation

Pillar 4 is not solved. This is not a real hospital deployment benchmark. It is a reproducible infrastructure-friction accounting simulation.

The current contribution is a framework for measuring:

- wall-clock burden
- failed rounds
- straggler sensitivity
- active-client counts
- communication cost

### Next step

Move from simulation to containerized deployment tests using Flower, FedML, or NVIDIA FLARE-style orchestration.

---

## Overall conclusion

The four pillars have not all been taken down, but Pillar 2 now has a bounded resolution.

The current honest status is:

- Pillar 1 has strong evidence and is the core research contribution.
- Pillar 2 has a bounded resolution for the Camelyon17 feature/head validation regime, plus measurable communication accounting and accuracy-per-communication proxy results.
- Pillar 3 has a first privacy-noise robustness probe but no formal privacy guarantee.
- Pillar 4 has a first infrastructure-friction simulation but no real deployment benchmark.

The project has moved from a single-method idea to a broader research program: site-signal alignment under heterogeneity, plus measurable communication, privacy, and infrastructure stress probes.

---

## Best current claim

In federated computational pathology, sample volume is not automatically equivalent to reliable site signal. Across simulated pathology-feature stress tests and Camelyon17 external-center validation, FedAvg-style sample-volume weighting can overfit source-center structure and generalize worse to held-out centers. Communication, privacy, and infrastructure analyses are now framed as measurable stress dimensions rather than solved deployment claims.
