---
layout: home

hero:
  name: Computational Pathology AI Research Framework
  text: Whole-slide pathology AI, MIL benchmarking, federated oncology validation, and mathematical validation infrastructure
  tagline: Research-focused framework for PCam validation, PANDA prostate cancer grading, TransnnMIL stabilization, dominant-site federated pathology stress testing, and reproducible computational pathology experiments.
  actions:
    - theme: brand
      text: Read Dominant-Site Paper
      link: /research/dominant-site-federated-pathology-paper
    - theme: alt
      text: View PANDA Results
      link: /results/panda-slide-level-baselines
    - theme: alt
      text: GitHub
      link: https://github.com/matthewvaishnav/computational-pathology-research

features:
  - title: Dominant-Site Federated Pathology
    details: Paper-style result on sample-volume / site-signal alignment failure modes, with generated figures, fixed detector transfer, diagnostic ablation, and calibration-sensitivity analysis.
  - title: PANDA Slide-Level Benchmarking
    details: Prostate cancer grading experiments using Phikon feature bags, 10,611 readable slide-level feature files, mean pooling, gated AttentionMIL, and stabilized TransnnMIL.
  - title: TransnnMIL Stabilization
    details: Repeated-seed learning-rate stability grid showing stabilized TransnnMIL remains competitive across a broad optimizer range.
  - title: PathologyFL
    details: Federated learning infrastructure for computational pathology experiments with coordinator/client workflows, weighted aggregation, privacy hooks, and simulated-site benchmarks.
  - title: FAIR-WEIGHTS-H
    details: Experimental contribution-aware institutional weighting research using validated contribution, uncertainty, subgroup coverage, entropy constraints, and effective-institution diagnostics.
  - title: Claim Boundaries
    details: Explicit research-only guardrails: not clinically validated, not diagnostic software, not regulatory cleared, and not an institutional ranking system.
---

## Current status

This project is a research and engineering framework for computational pathology AI, slide-level MIL benchmarking, federated oncology validation, and mathematical validation tooling. It is **not clinically validated**, **not diagnostic software**, and **not regulatory cleared**.

| Area | Status |
|---|---|
| Dominant-site federated pathology paper | Complete working-paper artifact with generated Figures 1-4 |
| Fixed detector transfer | Label-noise-calibrated detector transfers to conservative threshold shift with low clean switching and positive 35% / 45% gains |
| Detector calibration sensitivity | 29 of 36 nearby detector settings preserve the qualitative result |
| PCam validation | Complete: 95.37% validation AUC; 85.26% test accuracy and 0.9394 test AUC on the full 32,768-sample test set |
| PANDA slide-level benchmark | Complete current research pass using Phikon feature bags and 10,611 readable slide-level feature files |
| PANDA mean pooling | Complete: mean-pooled Phikon + MLP QWK 0.7274 |
| PANDA gated AttentionMIL | Complete: QWK 0.8100 |
| PANDA tuned TransnnMIL | Complete repeated-seed results: QWK 0.8155 / 0.8225 / 0.8086 |
| PANDA stabilized TransnnMIL LR grid | Complete: mean best validation QWK 0.8117-0.8257 across 18 full-PANDA runs |
| Camelyon16/17 validation | Planned future multi-center / whole-slide validation target |
| Clinical validation | Not completed |

## Key findings so far

The strongest current research artifact is the dominant-site federated pathology paper: **When More Data Is Less Trustworthy**. It studies a sample-volume / site-signal alignment failure mode in simulated federated pathology, where FedAvg can become less safe when the largest simulated client's training signal is misaligned with the declared validation objective.

The fixed detector-transfer result is the main headline: a detector calibrated under dominant-site label-noise stress transfers to conservative ordinal threshold-shift stress, keeps clean-regime switching low at 13.3%, and produces statistically positive gains at 35% and 45% conservative shift across global QWK, macro-F1, and worst-site QWK.

The detector result is supported by diagnostic-frequency analysis, leave-one-diagnostic-family-out ablation, and calibration-sensitivity analysis. Removing the most frequent diagnostic family, `mean_abs_error_high`, does not collapse the positive 35% / 45% transfer result, and 29 of 36 nearby calibration settings preserve the qualitative result.

The PANDA slide-level experiments show that tuned and stabilized TransnnMIL is competitive with gated AttentionMIL, but the advantage is small, so the supported claim is **competitive**, not conclusive architecture superiority.

## Generated figures

- [Figure index](/research/dominant-site-generated-figures) — generated Figures 1-4 for the dominant-site paper.
- [Dominant-site paper](/research/dominant-site-federated-pathology-paper) — paper-style working draft.

## Project map

- [Dominant-site paper](/research/dominant-site-federated-pathology-paper) — main current research artifact.
- [Generated dominant-site figures](/research/dominant-site-generated-figures) — Figure 1-4 index.
- [Overview](/overview/) — what the framework is and what claims are currently supported.
- [PANDA Results](/results/panda-slide-level-baselines) — slide-level prostate cancer grading baselines and TransnnMIL comparison.
- [Models](/models/) — TransnnMIL, MIL models, and foundation encoders.
- [Federated Learning](/federated/pathologyfl) — PathologyFL, privacy, robustness, and FAIR-WEIGHTS-H.
- [Validation](/validation/) — smoke tests, PCam benchmarks, heterogeneous-site diagnostics, PANDA results, and Camelyon planning.
- [PCam Results](/results/pcam-results) — benchmark summaries and performance comparisons.
- [Roadmap](/roadmap/) — current limitations and next validation steps.

## Interpretation guardrail

Implemented systems, synthetic checks, PCam simulated-site experiments, and PANDA slide-level benchmarks are not the same as real hospital-level validation. PCam provides real pathology patches and PANDA provides real slide-level prostate pathology labels/features, but Camelyon16/17 or equivalent multi-center whole-slide validation is required before making real multi-institutional clinical claims.
