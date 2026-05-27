---
layout: home

hero:
  name: Computational Pathology AI Research Framework
  text: Whole-slide pathology AI, MIL benchmarking, federated oncology validation, and mathematical validation infrastructure
  tagline: Research-focused framework for PCam validation, PANDA prostate cancer grading, TransnnMIL, PathologyFL, FAIR-WEIGHTS-H, and reproducible computational pathology experiments.
  actions:
    - theme: brand
      text: Start Reading
      link: /overview/
    - theme: alt
      text: View PANDA Results
      link: /results/panda-slide-level-baselines
    - theme: alt
      text: GitHub
      link: https://github.com/matthewvaishnav/computational-pathology-research

features:
  - title: PANDA Slide-Level Benchmarking
    details: Prostate cancer grading experiments using Phikon feature bags, 10,611 readable slide-level feature files, mean pooling, gated AttentionMIL, and tuned TransnnMIL.
  - title: TransnnMIL
    details: Custom MIL architecture direction combining transformer-style global attention with local diagnostic-region reasoning and topology-aware WSI modeling.
  - title: PathologyFL
    details: Federated learning infrastructure for computational pathology experiments with coordinator/client workflows, weighted aggregation, privacy hooks, and simulated-site benchmarks.
  - title: FAIR-WEIGHTS-H
    details: Experimental contribution-aware institutional weighting research using validated contribution, uncertainty, subgroup coverage, entropy constraints, and effective-institution diagnostics.
  - title: Mathematical Validation
    details: Evidence-bound reporting with QWK/AUC metrics, ablation summaries, entropy, N_eff, weight trajectories, worst-site performance, and explicit claim-status guardrails.
  - title: Validation Roadmap
    details: PCam patch validation and PANDA slide-level validation are complete; Camelyon16/17 and real multi-center validation remain future work.
---

## Current status

This project is a research and engineering framework for computational pathology AI, slide-level MIL benchmarking, federated oncology validation, and mathematical validation tooling. It is **not clinically validated**, **not diagnostic software**, and **not regulatory cleared**.

| Area | Status |
|---|---|
| PCam validation | Complete: 95.37% validation AUC; 85.26% test accuracy and 0.9394 test AUC on the full 32,768-sample test set |
| PANDA slide-level benchmark | Complete current research pass using Phikon feature bags and 10,611 readable slide-level feature files |
| PANDA mean pooling | Complete: mean-pooled Phikon + MLP QWK 0.7274 |
| PANDA gated AttentionMIL | Complete: QWK 0.8100 |
| PANDA tuned TransnnMIL | Complete repeated-seed results: QWK 0.8155 / 0.8225 / 0.8086 |
| PANDA TransnnMIL ablations | Complete: lr=1e-3 dropped to QWK 0.7403; dropout=0.25 reached QWK 0.8015 |
| TransnnMIL | Implemented research architecture and current PANDA benchmark target |
| PathologyFL | Federated workflow scaffold implemented and tested on simulated-site PCam experiments |
| FAIR-WEIGHTS-H | Experimental institutional weighting hypothesis; execution behavior tested, superiority over simpler baselines not yet demonstrated |
| Camelyon16/17 validation | Planned future multi-center / whole-slide validation target |
| Clinical validation | Not completed |

## Key findings so far

The PANDA slide-level experiments show that tuned TransnnMIL is competitive with gated AttentionMIL and slightly favorable across the current repeated-seed runs, beating AttentionMIL on 2 of 3 tested seeds. The advantage is small, so the supported claim is **competitive and slightly favorable**, not conclusive superiority.

The PANDA ablations show that TransnnMIL is highly optimization-sensitive in this setup. Lowering learning rate from `1e-3` to `3e-4` was a major contributor to competitive performance; higher dropout also reduced performance.

The PCam heterogeneous benchmark showed that weighting strategies can produce different aggregation weights without producing measurable performance differences in the current patch-level setup. This suggests benchmark insensitivity rather than proof that institutional weighting is ineffective.

## PANDA slide-level benchmarking

The project now includes slide-level prostate cancer grading experiments on PANDA using Phikon pathology features and multiple-instance learning.

| Model | Best validation QWK |
|---|---:|
| Mean-pooled Phikon + MLP | 0.7274 |
| Gated AttentionMIL | 0.8100 |
| Tuned TransnnMIL, seed 42 | 0.8155 |
| Tuned TransnnMIL, seed 123 | 0.8225 |
| Tuned TransnnMIL, seed 2025 | 0.8086 |

## Project map

- [Overview](/overview/) — what the framework is and what claims are currently supported.
- [PANDA Results](/results/panda-slide-level-baselines) — slide-level prostate cancer grading baselines and TransnnMIL comparison.
- [Models](/models/) — TransnnMIL, MIL models, and foundation encoders.
- [Federated Learning](/federated/pathologyfl) — PathologyFL, privacy, robustness, and FAIR-WEIGHTS-H.
- [Validation](/validation/) — smoke tests, PCam benchmarks, heterogeneous-site diagnostics, PANDA results, and Camelyon planning.
- [PCam Results](/results/pcam-results) — benchmark summaries and performance comparisons.
- [Roadmap](/roadmap/) — current limitations and next validation steps.

## Interpretation guardrail

Implemented systems, synthetic checks, PCam simulated-site experiments, and PANDA slide-level benchmarks are not the same as real hospital-level validation. PCam provides real pathology patches and PANDA provides real slide-level prostate pathology labels/features, but Camelyon16/17 or equivalent multi-center whole-slide validation is required before making real multi-institutional clinical claims.
