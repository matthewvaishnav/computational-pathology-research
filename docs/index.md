---
layout: default
title: Documentation
---

# Computational Pathology Research Documentation

A navigation-first guide to the repository: what the platform does, where the validated results are, how the federated learning stack works, and where to find implementation details.

<div class="callout callout-warning">
  <strong>Research Use Only:</strong> This framework is designed for research and engineering validation. It has not been validated for clinical diagnostic use.
</div>

---

## Start Here

| Goal | Read this |
|---|---|
| Understand the whole repository | [Repository Overview](REPOSITORY_OVERVIEW) |
| Understand the platform architecture | [Platform Overview](PLATFORM_OVERVIEW) |
| Install and run the code | [Getting Started](GETTING_STARTED) |
| Review the strongest benchmark result | [PCam Real Results](PCAM_REAL_RESULTS) |
| Understand institutional weighting | [FAIR-WEIGHTS-H Protocol](FAIR_WEIGHTS_HYBRID_PROTOCOL) |
| Review benchmark tooling | [Benchmark System](BENCHMARK_SYSTEM) |
| Review security work | [Security Hardening](SECURITY_HARDENING) |

---

## What Is Implemented

### Core modeling and pathology pipeline

- Whole-slide image processing and patch-level pipelines. See [Platform Overview](PLATFORM_OVERVIEW) and [Repository Overview](REPOSITORY_OVERVIEW).
- Multiple-instance learning models including attention-based MIL and TransMIL-style components. See [Repository Overview](REPOSITORY_OVERVIEW).
- Foundation-model integration pathways for pathology feature extraction. See [Foundation Models](FOUNDATION_MODELS).
- Benchmarking and statistical evaluation utilities. See [Benchmark System](BENCHMARK_SYSTEM) and [PCam Real Results](PCAM_REAL_RESULTS).

### Federated learning stack

- Pathology-aware federated aggregation modules. Source: `src/features/federated/pathology_fl/aggregator/`.
- Secure aggregation and privacy-oriented infrastructure. See [Security Hardening](SECURITY_HARDENING).
- Explicit weighted aggregation adapter for externally computed institutional weights. Source: `src/features/federated/pathology_fl/aggregator/weighted.py`.
- FAIR-WEIGHTS-H experimental institutional weighting engine. Source: `src/features/federated/pathology_fl/weighting/fair_weights_h.py`.

### FAIR-WEIGHTS-H research scaffold

Implemented components include:

- hybrid institutional weighting engine — `src/features/federated/pathology_fl/weighting/fair_weights_h.py`,
- explicit weighted aggregation adapter — `src/features/federated/pathology_fl/aggregator/weighted.py`,
- synthetic federation generator — `src/features/federated/pathology_fl/weighting/synthetic_federation.py`,
- equal / volume / prestige / FAIR-WEIGHTS-H baseline comparison — `src/features/federated/pathology_fl/weighting/benchmark.py`,
- perturbation suite for uncertainty, scanner shift, quality degradation, and rare-population enrichment — `src/features/federated/pathology_fl/weighting/perturbations.py`,
- canonical experiment suite — `src/features/federated/pathology_fl/weighting/experiment_suite.py`,
- markdown report generation for synthetic experiment summaries — `src/features/federated/pathology_fl/weighting/report_generator.py`.

FAIR-WEIGHTS-H is experimental and requires empirical validation before any clinical or regulatory claims.

---

## Key Results and Evidence Links

The repository documents the following benchmark claims:

| Claim | Evidence / status link |
|---|---|
| **0.9394 AUC** on PatchCamelyon | [PCam Real Results](PCAM_REAL_RESULTS) |
| **85.26% test accuracy** on the PCam test set | [PCam Real Results](PCAM_REAL_RESULTS) |
| **7x parameter efficiency** compared with Swin-Transformer in the reported benchmark context | [PCam Real Results](PCAM_REAL_RESULTS) |
| Camelyon17 federated attention audit across simulated hospital sites | [Current Status](CURRENT_STATUS_2026-05-14) and [Repository Overview](REPOSITORY_OVERVIEW) |
| 5,071+ automated tests reported in project documentation | [Current Status](CURRENT_STATUS_2026-05-14) and test suite paths under `tests/` |

For broader evaluation infrastructure, see [Benchmark System](BENCHMARK_SYSTEM).

---

## Federated Learning and DMI

The project originally included fixed institutional multipliers such as cancer center, teaching hospital, community hospital, and rural hospital weights. Those fixed prestige-style multipliers are now treated as **comparison baselines**, not the preferred research direction.

The current research direction is **FAIR-WEIGHTS-H**, a hybrid institutional weighting framework based on:

\[
w_t = \arg\max_{w\in\mathcal W}\sum_i w_i\left(\hat\phi_i^{Owen}+\lambda_DD_i^{useful}+\lambda_FF_i+\lambda_QQ_i-\lambda_SS_i\right)
\]

subject to normalization, caps, representation constraints, subgroup-performance constraints, and stability constraints.

See [FAIR-WEIGHTS-H Protocol](FAIR_WEIGHTS_HYBRID_PROTOCOL). Source implementation begins at `src/features/federated/pathology_fl/weighting/fair_weights_h.py`.

---

## Recommended Reading Order

1. [Repository Overview](REPOSITORY_OVERVIEW)
2. [Platform Overview](PLATFORM_OVERVIEW)
3. [Getting Started](GETTING_STARTED)
4. [PCam Real Results](PCAM_REAL_RESULTS)
5. [Benchmark System](BENCHMARK_SYSTEM)
6. [FAIR-WEIGHTS-H Protocol](FAIR_WEIGHTS_HYBRID_PROTOCOL)
7. [Security Hardening](SECURITY_HARDENING)

---

## Current Research Status

| Area | Status | Evidence / link |
|---|---|---|
| PCam benchmark | Documented result | [PCam Real Results](PCAM_REAL_RESULTS) |
| MIL / WSI pipeline | Implemented | [Repository Overview](REPOSITORY_OVERVIEW), [Platform Overview](PLATFORM_OVERVIEW) |
| Federated learning infrastructure | Implemented | `src/features/federated/pathology_fl/`, [Platform Overview](PLATFORM_OVERVIEW) |
| FAIR-WEIGHTS-H engine | Experimental implementation | `src/features/federated/pathology_fl/weighting/fair_weights_h.py`, [FAIR-WEIGHTS-H Protocol](FAIR_WEIGHTS_HYBRID_PROTOCOL) |
| Explicit weighted aggregation | Implemented adapter | `src/features/federated/pathology_fl/aggregator/weighted.py` |
| Synthetic perturbation experiments | Implemented scaffold | `src/features/federated/pathology_fl/weighting/experiment_suite.py` |
| FAIR-WEIGHTS-H tests | Implemented | `tests/federated/test_fair_weights_h.py`, `tests/federated/test_weighting_benchmark.py` |
| Real multi-institutional validation | Future work | [Roadmap to Real Datasets](ROADMAP_TO_REAL_DATASETS) |
| Clinical diagnostic validation | Not completed | Research-use notice in this page and [Platform Overview](PLATFORM_OVERVIEW) |
| Regulatory clearance | Not claimed | Research-use notice and [Security Hardening](SECURITY_HARDENING) for infrastructure only |

---

## Source Code Pointers

| Component | Path |
|---|---|
| Federated aggregators | `src/features/federated/pathology_fl/aggregator/` |
| FAIR-WEIGHTS-H engine | `src/features/federated/pathology_fl/weighting/fair_weights_h.py` |
| Synthetic federation | `src/features/federated/pathology_fl/weighting/synthetic_federation.py` |
| Perturbation experiments | `src/features/federated/pathology_fl/weighting/experiment_suite.py` |
| Weighting tests | `tests/federated/` |

---

## Contribution Notes

When updating documentation:

- Keep validated benchmark results separate from proposed research directions.
- Do not describe FAIR-WEIGHTS-H synthetic experiments as clinical validation.
- Treat legacy prestige multipliers as baselines only.
- Prefer clear implementation status labels: implemented, experimental, planned, or future work.
