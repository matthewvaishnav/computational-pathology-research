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

- Whole-slide image processing and patch-level pipelines.
- Multiple-instance learning models including attention-based MIL and TransMIL-style components.
- Foundation-model integration pathways for pathology feature extraction.
- Benchmarking and statistical evaluation utilities.

### Federated learning stack

- Pathology-aware federated aggregation modules.
- Secure aggregation and privacy-oriented infrastructure.
- Explicit weighted aggregation adapter for externally computed institutional weights.
- FAIR-WEIGHTS-H experimental institutional weighting engine.

### FAIR-WEIGHTS-H research scaffold

Implemented components include:

- hybrid institutional weighting engine,
- explicit weighted aggregation adapter,
- synthetic federation generator,
- equal / volume / prestige / FAIR-WEIGHTS-H baseline comparison,
- perturbation suite for uncertainty, scanner shift, quality degradation, and rare-population enrichment,
- markdown report generation for synthetic experiment summaries.

FAIR-WEIGHTS-H is experimental and requires empirical validation before any clinical or regulatory claims.

---

## Key Results

The repository documents the following benchmark claims:

- **0.9394 AUC** on PatchCamelyon.
- **85.26% test accuracy** on the PCam test set.
- **7x parameter efficiency** compared with Swin-Transformer in the reported benchmark context.
- Camelyon17 federated attention audit across simulated hospital sites.
- 5,071+ automated tests reported in the project documentation.

For details, see [PCam Real Results](PCAM_REAL_RESULTS) and the benchmark documentation.

---

## Federated Learning and DMI

The project originally included fixed institutional multipliers such as cancer center, teaching hospital, community hospital, and rural hospital weights. Those fixed prestige-style multipliers are now treated as **comparison baselines**, not the preferred research direction.

The current research direction is **FAIR-WEIGHTS-H**, a hybrid institutional weighting framework based on:

\[
w_t = \arg\max_{w\in\mathcal W}\sum_i w_i\left(\hat\phi_i^{Owen}+\lambda_DD_i^{useful}+\lambda_FF_i+\lambda_QQ_i-\lambda_SS_i\right)
\]

subject to normalization, caps, representation constraints, subgroup-performance constraints, and stability constraints.

See [FAIR-WEIGHTS-H Protocol](FAIR_WEIGHTS_HYBRID_PROTOCOL).

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

| Area | Status |
|---|---|
| PCam benchmark | Documented result |
| MIL / WSI pipeline | Implemented |
| Federated learning infrastructure | Implemented |
| FAIR-WEIGHTS-H engine | Experimental implementation |
| Synthetic perturbation experiments | Implemented scaffold |
| Real multi-institutional validation | Future work |
| Clinical diagnostic validation | Not completed |
| Regulatory clearance | Not claimed |

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
