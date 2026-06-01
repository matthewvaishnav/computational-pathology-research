# Recruiter / Hiring Manager Quick Read

This file is the shortest path through the repository for recruiters, hiring managers, research engineers, and collaborators.

## What this project is

This is a research-focused computational pathology and oncology AI engineering framework. It combines whole-slide histopathology modeling, multiple-instance learning, benchmark validation, federated oncology learning experiments, and reproducible experiment documentation.

The current strongest direction is slide-level prostate cancer grading on the PANDA dataset using Phikon patch features and MIL models.

## What to look at first

- PANDA slide-level results: [`docs/results/panda-slide-level-baselines.md`](docs/results/panda-slide-level-baselines.md)
- PANDA TransnnMIL stabilization summary: [`docs/results/panda-transnnmil-stability.md`](docs/results/panda-transnnmil-stability.md)
- PANDA TransnnMIL ablation summary: [`results/panda_transnnmil_ablation/ablation_summary.csv`](results/panda_transnnmil_ablation/ablation_summary.csv)
- Claim boundary: [`CLAIM_BOUNDARY.md`](CLAIM_BOUNDARY.md)
- Main project documentation: [`docs/`](docs/)

## Best current evidence

| Area | Current result |
|---|---:|
| PCam validation AUC | 95.37% |
| PANDA readable slide-level feature files | 10,611 |
| PANDA mean-pooled Phikon + MLP | QWK 0.7274 |
| PANDA gated AttentionMIL | QWK 0.8100 |
| PANDA tuned TransnnMIL seed 42 | QWK 0.8155 |
| PANDA tuned TransnnMIL seed 123 | QWK 0.8225 |
| PANDA tuned TransnnMIL seed 2025 | QWK 0.8086 |
| PANDA stabilized TransnnMIL LR grid | Mean best val QWK 0.8117-0.8257 across 18 full-PANDA runs |
| PANDA stabilized TransnnMIL best LR mean | QWK 0.8257 +/- 0.0169 across 3 seeds |
| TransnnMIL high-LR ablation, lr=1e-3 | Previously QWK 0.7403 before stabilization |
| TransnnMIL dropout ablation, dropout=0.25 | QWK 0.8015 |

## What the PANDA work shows

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

## What this project is not

This is not clinical software. It is not a deployed hospital system. It is not FDA-cleared, CE-marked, HIPAA-certified, or clinically validated.

See [`CLAIM_BOUNDARY.md`](CLAIM_BOUNDARY.md) for the exact public claim boundary.

## Roles this work is meant to support

Target role families:

- Computational Pathology AI Research Engineer
- Medical Imaging Machine Learning Engineer
- Healthcare AI Infrastructure Engineer
- Biomedical Research Software Engineer
- Federated / Privacy-Preserving Healthcare ML Engineer
- Research Engineer, Medical AI

## One-sentence pitch

I build reproducible computational pathology AI research infrastructure for whole-slide histopathology modeling, MIL benchmarking, PANDA prostate cancer grading, TransnnMIL stability analysis, and federated oncology validation.
