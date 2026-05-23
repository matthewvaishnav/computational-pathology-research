---
layout: home

hero:
  name: Matthew Vaishnav
  text: Computational Pathology, Federated Oncology Learning, and Mathematical Validation Infrastructure
  tagline: Whole-slide pathology AI, TransnnMIL v2.0, PathologyFL, FAIR-WEIGHTS-H institutional weighting, PCam/Camelyon validation, and multi-institutional oncology learning infrastructure.
  actions:
    - theme: brand
      text: Start Reading
      link: /overview/
    - theme: alt
      text: View Validation
      link: /validation/
    - theme: alt
      text: GitHub
      link: https://github.com/matthewvaishnav/computational-pathology-research

features:
  - title: TransnnMIL v2.0
    details: Custom whole-slide multiple-instance learning architecture combining TransMIL-style global attention, hierarchical spatial pooling, topology-aware graph processing, and adaptive pruning research.
  - title: PathologyFL
    details: Federated learning infrastructure for computational pathology with coordinator/client workflows, weighted aggregation, privacy hooks, secure aggregation work, and robustness checks.
  - title: FAIR-WEIGHTS-H
    details: Experimental institutional weighting scaffold using quality, useful uniqueness, fairness, contribution, volume, uncertainty, entropy, and effective-institution diagnostics.
  - title: PCam Validation
    details: Balanced and heterogeneous PCam federated benchmarks test weighting behavior on real pathology patches split into simulated institutions.
  - title: Mathematical Validation
    details: Evidence-bound reporting with entropy, N_eff, weight trajectories, worst-site performance, benchmark diagnostics, and explicit claim-status guardrails.
  - title: Camelyon Roadmap
    details: Future validation path toward real multi-center whole-slide experiments on Camelyon17 and broader oncology datasets.
---

## Current status

This project is a research and engineering platform for computational pathology and federated oncology learning. It is **not clinically validated** and is **not regulatory cleared**.

| Area | Status |
|---|---|
| TransnnMIL v2.0 | Implemented research architecture |
| PathologyFL | Federated workflow scaffold implemented |
| FAIR-WEIGHTS-H | Implemented and tested as an experimental institutional weighting scaffold |
| PCam smoke validation | Completed on real pathology patches split into simulated sites |
| PCam balanced benchmark | Completed across equal, volume, prestige, and FAIR-WEIGHTS-H strategies |
| PCam heterogeneous benchmark | Completed with informative null result: different weights, no measurable performance sensitivity |
| Camelyon17 validation | Planned future real multi-center validation |
| Clinical validation | Not completed |

## Key finding so far

The PCam heterogeneous benchmark showed that weighting strategies can produce different aggregation weights without producing measurable performance differences in the current patch-level setup. This suggests benchmark insensitivity rather than evidence that the institutional-weighting theory is ineffective.

## Project map

- [Overview](/overview/) — what the platform is and what claims are currently supported.
- [Models](/models/) — TransnnMIL v2.0, MIL models, and foundation encoders.
- [Federated Learning](/federated/pathologyfl) — PathologyFL, privacy, robustness, and FAIR-WEIGHTS-H.
- [Validation](/validation/) — smoke tests, PCam benchmarks, heterogeneous-site diagnostics, and Camelyon17 planning.
- [Results](/results/pcam-results) — benchmark summaries and performance comparisons.
- [Roadmap](/roadmap/) — current limitations and next validation steps.

## Interpretation guardrail

Implemented systems, synthetic checks, and PCam simulated-site experiments are not the same as real hospital-level validation. PCam provides real pathology patches, but Camelyon17 or equivalent multi-center whole-slide validation is required before making real multi-institutional claims.
