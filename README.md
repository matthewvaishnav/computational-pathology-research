# Matthew Vaishnav Computational Pathology Research

Independent research and engineering work in computational pathology, multiple-instance learning, and federated medical AI validation.

The main public artifact is the paper PDF site:

**[When More Data Is Less Trustworthy: Site-Signal Alignment Failure Modes in Federated Computational Pathology](https://matthewvaishnav.github.io/computational-pathology-research/)**

Source paper draft:

**[dominant-site federated pathology paper draft](docs/research/dominant-site-federated-pathology-paper.md)**

This report studies a sample-volume / site-signal alignment failure mode in simulated federated pathology experiments over PANDA-derived Phikon features. The central question is whether FedAvg becomes less safe when the largest simulated client's training signal is misaligned with the validation objective, and whether dominance-aware switching can reduce that risk.

This repository is **research-only**. It is not clinically validated, not diagnostic software, and not intended for patient-care use.

---

## Research package

For readers who do not want to start with the full paper:

1. **Paper PDF site:** [public PDF](https://matthewvaishnav.github.io/computational-pathology-research/)
2. **Plain-English one-page summary:** [docs/outreach/plain-english-summary.md](docs/outreach/plain-english-summary.md)
3. **Technical one-page summary:** [docs/outreach/technical-summary.md](docs/outreach/technical-summary.md)
4. **Demo / figure thread:** [docs/outreach/demo-figure-thread.md](docs/outreach/demo-figure-thread.md)
5. **Outreach list:** [docs/outreach/outreach-list.md](docs/outreach/outreach-list.md)
6. **Research package index:** [docs/outreach/research-package-index.md](docs/outreach/research-package-index.md)
7. **Camelyon17 external-center validation note:** [docs/research/camelyon17-external-center-validation-note.md](docs/research/camelyon17-external-center-validation-note.md)
8. **Four pillars FL pathology status:** [docs/research/four-pillars-fl-pathology-status.md](docs/research/four-pillars-fl-pathology-status.md)
9. **Pillar 2 bounded communication resolution:** [docs/research/pillar2-communication-bounded-resolution.md](docs/research/pillar2-communication-bounded-resolution.md)

---

## Read first

1. **Main technical report:** [dominant-site federated pathology paper draft](docs/research/dominant-site-federated-pathology-paper.md)
2. **Figure/table plan:** [dominant-site paper figure and table plan](docs/research/dominant-site-paper-figure-table-plan.md)
3. **Detector transfer result:** [dominance detector transfer results](docs/research/dominance-detector-transfer-results.md)
4. **Detector ablation and calibration sensitivity:** [detector diagnostic ablation summary](docs/research/detector-diagnostic-ablation.md)
5. **PANDA TransnnMIL stabilization:** [PANDA TransnnMIL stabilization results](docs/results/panda-transnnmil-stability.md)
6. **Camelyon17 external-center validation:** [early feature-level validation note](docs/research/camelyon17-external-center-validation-note.md)
7. **Four pillars FL pathology status:** [current status report](docs/research/four-pillars-fl-pathology-status.md)
8. **Pillar 2 bounded communication resolution:** [communication-overhead bounded-resolution note](docs/research/pillar2-communication-bounded-resolution.md)
9. **Claim boundary:** [CLAIM_BOUNDARY.md](CLAIM_BOUNDARY.md)
10. **Recruiter / hiring manager quick read:** [RECRUITER_README.md](RECRUITER_README.md)

---

## Main result stack

The current strongest evidence is the dominant-site federated pathology result stack:

| Result | Evidence |
|---|---|
| Fixed detector transfer | Label-noise-calibrated detector transfers to conservative ordinal threshold shift |
| Clean-regime behavior | 13.3% trigger rate at 0% conservative shift, near-zero global-QWK cost |
| 35% conservative shift | +0.00542 global QWK, +0.00838 macro-F1, +0.00991 worst-site QWK |
| 45% conservative shift | +0.01053 global QWK, +0.01512 macro-F1, +0.01290 worst-site QWK |
| Diagnostic interpretability | Triggers mainly driven by ordinal-error and QWK degradation signals |
| Leave-one-out ablation | Removing `mean_abs_error_high` does not collapse the 35% / 45% transfer result |
| Calibration sensitivity | 29 of 36 nearby detector settings preserve low clean switching and positive 35% / 45% gains |

Supported claim:

> In simulated federated pathology experiments over real pathology-derived features, raw sample count is not equivalent to task-specific site-signal alignment. Sample-size dominance should be treated as an auditable modeling assumption rather than an automatic guarantee of aggregation safety.

### Early Camelyon17 external-center validation

A new Camelyon17/WILDS feature-level validation layer tests whether the site-signal alignment pattern appears under natural external-center shift.

| Result | Evidence |
|---|---:|
| Dataset audit | 455,954 Camelyon17/WILDS examples across 5 centers |
| Source-domain training centers | 0, 3, 4 |
| OOD validation center | 1 |
| Held-out OOD test center | 2 |
| Frozen ImageNet ResNet18: FedAvg-style test accuracy | 0.8312 |
| Frozen ImageNet ResNet18: equal-client test accuracy | 0.9132 |
| Frozen ImageNet ResNet18: equal-client gain vs FedAvg-style | +8.20 percentage points |
| Frozen ImageNet ResNet18: downweight-dominant gain vs FedAvg-style | +7.82 percentage points |
| Frozen ImageNet ResNet18: validation-aware detector switch gain | +6.58 percentage points held-out test accuracy |
| Frozen ImageNet ResNet18: detector threshold sweep | 43 / 112 settings robust-positive |
| Camelyon17-trained ResNet18: FedAvg-style test accuracy | 0.9052 |
| Camelyon17-trained ResNet18: equal-client test accuracy | 0.9318 |
| Camelyon17-trained ResNet18: downweight-dominant test accuracy | 0.9322 |
| Camelyon17-trained ResNet18: equal-client gain vs FedAvg-style | +2.66 percentage points |
| Camelyon17-trained ResNet18: downweight-dominant gain vs FedAvg-style | +2.70 percentage points |
| Camelyon17-trained ResNet18: FedAvg-style source-train accuracy | 0.9991 |

Interpretation:

> In early Camelyon17/WILDS feature-level baselines, FedAvg-style equal-patch weighting can fit the source training distribution more strongly while generalizing worse to held-out centers. The effect appears in both frozen ImageNet ResNet18 features and Camelyon17-trained ResNet18 features. With Camelyon17-trained features, FedAvg-style weighting nearly saturates source-train accuracy, while equal-client and downweight-dominant weighting improve held-out test accuracy.

Full note: [Camelyon17 external-center validation](docs/research/camelyon17-external-center-validation-note.md)

Unsupported claim:

> This is clinically validated or ready for real hospital deployment.

---

## PANDA / TransnnMIL evidence

The repository also contains slide-level prostate grading work on PANDA using Phikon feature bags and MIL models.

| Area | Result |
|---|---:|
| PANDA readable slide-level feature files | 10,611 after HDF5 read verification |
| PANDA mean-pooled Phikon + MLP | QWK 0.7274 |
| PANDA gated AttentionMIL | QWK 0.8100 |
| PANDA tuned TransnnMIL seed 42 | QWK 0.8155 |
| PANDA tuned TransnnMIL seed 123 | QWK 0.8225 |
| PANDA tuned TransnnMIL seed 2025 | QWK 0.8086 |
| PANDA stabilized TransnnMIL LR grid | Mean best val QWK 0.8117-0.8257 across 18 full-PANDA runs |
| PANDA stabilized TransnnMIL best LR mean | 1e-4 mean QWK 0.8257 ± 0.0169 across 3 seeds |

Interpretation:

> Stabilized TransnnMIL is competitive with gated AttentionMIL in the current PANDA Phikon-feature experiments, but the margin remains small. Stronger architecture-superiority claims require more controlled validation.

---

## Additional validation work

| Area | Status |
|---|---|
| PCam validation | 95.37% validation AUC |
| PCam public benchmark | 85.26% test accuracy, 0.9394 test AUC on the full 32,768-sample test set |
| PCam comparison | #1 by AUC among 11 compared PCam methods in the documented comparison table |
| Bootstrap validation | 1,000 bootstrap resamples reported for PCam metrics |
| Threshold optimization | Screening threshold analysis reduced missed tumor predictions by 61.7% in the documented PCam analysis |
| FAIR-WEIGHTS-H smoke/unit tests | Focused tests passing |
| PCam federated smoke tests | Equal, volume, prestige, and FAIR-WEIGHTS-H strategies completed on real PCam patches split into simulated sites |
| PCam balanced federated benchmark | Complete: FAIR-WEIGHTS-H stable, no performance degradation observed |
| PCam heterogeneous benchmark | Complete: strategies produced different weight trajectories, but patch-level performance was insensitive to those differences |
| Camelyon17 validation | Early external-center validation complete with frozen ImageNet ResNet18 and Camelyon17-trained ResNet18 feature baselines; pathology foundation-model features and full iterative FL remain next |
| Camelyon17 communication accounting | Full ResNet18 FL estimated at 24.98 GB for 100 fp32 rounds across 3 clients; feature/head federation estimated at 2.35 MB, about 10,894x smaller |
| Camelyon17 accuracy per communication | Same feature/head communication budget: equal-client improves held-out test accuracy by +2.66 points and downweight-dominant improves by +2.70 points over FedAvg-style weighting |
| Camelyon17 privacy-noise stress | Coefficient-noise probe, not formal DP: equal-client and downweight-dominant remain positive versus FedAvg-style across all tested noise levels up to noise_std 0.20 |
| Clinical validation | Not completed |

---

## Repository components

### PathologyFL

Federated learning research infrastructure for computational pathology:

- coordinator/client workflow
- local pathology training
- weighted aggregation
- differential privacy hooks
- secure aggregation direction
- byzantine/dropout robustness checks
- simulated dominant-site stress testing
- dominance-aware detector switching

See: [PathologyFL documentation](docs/federated/pathologyfl.md)

### FAIR-WEIGHTS-H

Mathematical institutional weighting framework for federated oncology learning.

The core idea is that in federated computational pathology, site influence should not be based only on sample count. It should be auditable and constrained by signals such as contribution, uncertainty, subgroup coverage, distributional usefulness, and caps that prevent a single institution from dominating.

See: [FAIR-WEIGHTS-H theory](docs/theory/fair-weights-h.md)

### TransnnMIL

Custom multiple-instance learning architecture direction for whole-slide pathology modeling using global attention and local diagnostic-region reasoning.

The current PANDA evidence supports a conservative claim: tuned and stabilized TransnnMIL-style modeling is competitive with AttentionMIL on Phikon feature bags, but not conclusively superior.

See: [TransnnMIL v2.0 documentation](docs/models/transnnmil-v2.md)

---

## Claim boundary

This is a research and engineering repository. It is:

- not clinically validated
- not diagnostic software
- not FDA-cleared or CE-marked
- not HIPAA-certified
- not deployed for patient care
- not an institutional ranking system

The long-term goal is responsible clinical translation after proper external validation, regulatory review, security review, usability testing, and deployment testing.

See [CLAIM_BOUNDARY.md](CLAIM_BOUNDARY.md).
