# Recruiter / Hiring Manager Quick Read

This file is a short guide to the evidence in this repository. The main artifact is not a product pitch; it is a preprint-style technical report.

## Start here

**Main technical report:** [`docs/research/dominant-site-federated-pathology-paper.md`](docs/research/dominant-site-federated-pathology-paper.md)

**Title:** *When More Data Is Less Trustworthy: Site-Signal Alignment Failure Modes in Federated Computational Pathology*

The report studies a federated learning failure mode in simulated computational pathology: FedAvg weights clients by sample count, but a high-volume site can have a training signal that is less aligned with the declared validation objective. The work tests this using PANDA-derived Phikon features, dominant-site stress experiments, detector switching, ablations, and calibration sensitivity.

This is research-only. It is not clinical software, not clinically validated, and not deployed for patient care.

---

## What to look at first

1. Main paper draft: [`docs/research/dominant-site-federated-pathology-paper.md`](docs/research/dominant-site-federated-pathology-paper.md)
2. Figure/table plan: [`docs/research/dominant-site-paper-figure-table-plan.md`](docs/research/dominant-site-paper-figure-table-plan.md)
3. Detector transfer result: [`docs/research/dominance-detector-transfer-results.md`](docs/research/dominance-detector-transfer-results.md)
4. Detector ablation and calibration sensitivity: [`docs/research/detector-diagnostic-ablation.md`](docs/research/detector-diagnostic-ablation.md)
5. PANDA TransnnMIL stabilization: [`docs/results/panda-transnnmil-stability.md`](docs/results/panda-transnnmil-stability.md)
6. Claim boundary: [`CLAIM_BOUNDARY.md`](CLAIM_BOUNDARY.md)

---

## Best current evidence

| Area | Current result |
|---|---:|
| Fixed detector transfer, conservative threshold shift | +0.00542 global QWK at 35%, +0.01053 at 45% |
| Fixed detector clean switching | 13.3% trigger rate at 0% conservative shift with near-zero global QWK cost |
| Detector diagnostic ablation | Removing `mean_abs_error_high` does not collapse the 35% / 45% transfer result |
| Detector calibration sensitivity | 29 of 36 nearby settings preserve low clean switching and positive 35% / 45% gains |
| PANDA readable slide-level feature files | 10,611 |
| PANDA mean-pooled Phikon + MLP | QWK 0.7274 |
| PANDA gated AttentionMIL | QWK 0.8100 |
| PANDA tuned TransnnMIL seed 42 | QWK 0.8155 |
| PANDA tuned TransnnMIL seed 123 | QWK 0.8225 |
| PANDA tuned TransnnMIL seed 2025 | QWK 0.8086 |
| PANDA stabilized TransnnMIL LR grid | Mean best val QWK 0.8117-0.8257 across 18 full-PANDA runs |
| PANDA stabilized TransnnMIL best LR mean | QWK 0.8257 +/- 0.0169 across 3 seeds |
| PCam validation AUC | 95.37% |

---

## What the federated pathology work shows

The federated pathology work studies a sample-volume / site-signal alignment failure mode. FedAvg gives more influence to clients with more samples, but more samples do not automatically imply better alignment with the declared validation objective.

The strongest current detector result is a fixed label-noise-calibrated rule transferred to conservative ordinal threshold shift. It kept clean-regime switching low and produced statistically positive improvements at 35% and 45% conservative shift across global QWK, macro-F1, and worst-site QWK.

Follow-up checks make the result more credible:

- detector triggers are mainly ordinal-error and QWK degradation signals
- removing the strongest diagnostic does not collapse the result
- 29 of 36 nearby calibration settings preserve the qualitative pattern

This should be read as simulated-federation robustness research, not an institutional ranking system and not clinical validation.

---

## What the PANDA / TransnnMIL work shows

The PANDA work moves the project beyond patch-level PCam validation into slide-level prostate pathology modeling.

The work includes:

- PANDA manifest construction and feature-file validation
- HDF5 read verification and exclusion of unreadable compressed feature files
- slide-level feature-bag training
- mean-pooling baseline
- gated AttentionMIL baseline
- tuned and stabilized TransnnMIL baselines
- repeated-seed validation
- controlled ablations for patch cap, learning rate, and dropout
- optimizer-stability testing across six learning rates and three seeds

The current interpretation is intentionally conservative: tuned and stabilized TransnnMIL is competitive with gated AttentionMIL in the current repeated-seed PANDA experiments, and stabilization widened the usable learning-rate regime, but stronger architecture-superiority claims require more controlled validation.

---

## What this project is not

This is not clinical software. It is not a deployed hospital system. It is not FDA-cleared, CE-marked, HIPAA-certified, or clinically validated.

See [`CLAIM_BOUNDARY.md`](CLAIM_BOUNDARY.md) for the exact public claim boundary.

---

## Roles this work is meant to support

Target role families:

- Computational Pathology AI Research Engineer
- Medical Imaging Machine Learning Engineer
- Healthcare AI Infrastructure Engineer
- Biomedical Research Software Engineer
- Federated / Privacy-Preserving Healthcare ML Engineer
- Research Engineer, Medical AI

---

## One-sentence summary

I am building evidence-focused computational pathology research infrastructure for whole-slide modeling and federated medical AI validation, with the current main result showing that sample-size-weighted aggregation can fail under dominant-site signal misalignment and that detector switching can reduce that risk in simulated PANDA federations.
