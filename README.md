# Matthew Vaishnav Computational Pathology, Federated Oncology Learning, and Mathematical Validation Infrastructure

Research-focused computational pathology and oncology AI engineering framework for whole-slide histopathology modeling, multiple-instance learning, benchmark validation, and federated oncology validation experiments.

**Recruiter / hiring manager quick read:** [RECRUITER_README.md](RECRUITER_README.md)  
**Claim boundary:** [CLAIM_BOUNDARY.md](CLAIM_BOUNDARY.md)  
**Documentation:** https://matthewvaishnav.github.io/computational-pathology-research/  
**PANDA slide-level baselines:** [docs/results/panda-slide-level-baselines.md](docs/results/panda-slide-level-baselines.md)  
**PANDA TransnnMIL ablation summary:** [results/panda_transnnmil_ablation/ablation_summary.csv](results/panda_transnnmil_ablation/ablation_summary.csv)  
**Literature positioning:** https://matthewvaishnav.github.io/computational-pathology-research/research/literature-positioning  
**PCam results:** https://matthewvaishnav.github.io/computational-pathology-research/results/pcam-results

---

## What this repository is

This repository is my computational pathology and oncology AI research framework. It combines model development, federated learning, benchmark validation, statistical reporting, and documentation infrastructure for pathology AI experiments.

The current strongest direction is slide-level prostate cancer grading on the PANDA dataset using Phikon patch features and multiple-instance learning.

The work spans:

- patch-level and whole-slide pathology modeling
- multiple-instance learning for WSI classification
- PANDA prostate cancer grading with Phikon features
- mean-pooling, gated AttentionMIL, and tuned TransnnMIL baselines
- HDF5 feature-integrity validation
- repeated-seed validation and controlled ablations
- **PathologyFL** federated learning infrastructure
- **FAIR-WEIGHTS-H** institutional weighting research
- PCam, PANDA, and Camelyon validation workflows
- threshold analysis and statistical validation
- PubMed-grounded literature positioning
- testing, reporting, and deployment-oriented research infrastructure

This is a research and engineering framework. It is **not clinically validated**, **not diagnostic software**, and **not currently used for patient care**. The long-term goal is responsible clinical translation after proper validation, regulatory review, security review, usability testing, and deployment testing.

---

## Current evidence snapshot

| Area | Status |
|---|---|
| PCam validation | **95.37% validation AUC** |
| PCam public benchmark | **85.26% test accuracy**, **0.9394 test AUC** on the full 32,768-sample test set |
| PCam comparison | **#1 by AUC among 11 compared PCam methods** in the documented comparison table |
| PANDA data integrity | **10,611 readable slide-level feature files** after HDF5 read verification |
| PANDA mean-pooled Phikon + MLP | **QWK 0.7274** |
| PANDA gated AttentionMIL | **QWK 0.8100** |
| PANDA tuned TransnnMIL seed 42 | **QWK 0.8155** |
| PANDA tuned TransnnMIL seed 123 | **QWK 0.8225** |
| PANDA tuned TransnnMIL seed 2025 | **QWK 0.8086** |
| PANDA TransnnMIL ablation: lr=1e-3 | **QWK 0.7403**, showing high-LR instability |
| PANDA TransnnMIL ablation: dropout=0.25 | **QWK 0.8015**, showing higher dropout mildly hurts |
| Bootstrap validation | 1,000 bootstrap resamples reported for PCam metrics |
| Threshold optimization | Screening threshold analysis reduced missed tumor predictions by 61.7% in the documented PCam analysis |
| FAIR-WEIGHTS-H smoke/unit tests | Focused tests passing |
| PCam federated smoke tests | Equal, volume, prestige, and FAIR-WEIGHTS-H strategies completed on real PCam patches split into simulated sites |
| PCam balanced federated benchmark | Complete: FAIR-WEIGHTS-H stable, no performance degradation observed |
| PCam heterogeneous benchmark | Complete: strategies produced different weight trajectories, but patch-level performance was insensitive to those differences |
| FAIR-WEIGHTS-H empirical status | Tested for execution stability and aggregation behavior; performance/fairness advantage over simpler baselines not yet demonstrated |
| PubMed literature positioning | Added: related work, citation table, claim-strength table, and next experiment priorities |
| Camelyon16/17 validation | Planned next slide-level / multi-center validation target |
| Clinical validation | Not completed |

---

## PANDA interpretation

The PANDA slide-level experiments move the project beyond patch-level PCam validation into whole-slide prostate pathology modeling.

Current interpretation:

> Tuned TransnnMIL is competitive with gated AttentionMIL and slightly favorable across the current repeated-seed PANDA experiments, beating AttentionMIL on 2 of 3 tested seeds. The margin remains small, so stronger superiority claims require more controlled validation.

Ablation interpretation:

> TransnnMIL is highly optimization-sensitive in the current PANDA setup. Lowering learning rate from 1e-3 to 3e-4 was a major contributor to performance. Higher dropout mildly reduced performance. Patch cap increase was not the main driver in the seed-42 comparison.

---

## Core components

### TransnnMIL v2.0

Custom WSI multiple-instance learning architecture direction combining:

- TransMIL-style global attention
- hierarchical spatial pooling
- topology / graph-aware tissue structure modeling
- optional adaptive pruning

The current PANDA work provides slide-level benchmark evidence for tuned TransnnMIL-style modeling using Phikon feature bags. It is competitive with AttentionMIL in the current repeated-seed PANDA experiments, but not conclusively superior.

See: [TransnnMIL v2.0 documentation](docs/models/transnnmil-v2.md)

### PathologyFL

Federated learning infrastructure for computational pathology:

- coordinator/client workflow
- local pathology training
- weighted aggregation
- differential privacy hooks
- secure aggregation work
- byzantine/dropout robustness checks
- balanced and heterogeneous PCam federated benchmarks

See: [PathologyFL documentation](docs/federated/pathologyfl.md)

### FAIR-WEIGHTS-H

Mathematical institutional weighting framework for federated oncology learning.

FAIR-WEIGHTS-H does not simply assign weights from a checklist of institutional attributes. It formalizes institutional influence as a constrained optimization problem over mathematically defined signals, including difficulty-adjusted quality, Owen/Shapley-style counterfactual contribution, Jensen-Shannon distributional uniqueness, subgroup representation constraints, uncertainty penalties, entropy, and effective-institution diagnostics.

The method is designed to replace crude volume or prestige weighting with an auditable mathematical framework built from:

- difficulty-adjusted diagnostic quality models,
- group-aware Owen value / counterfactual contribution estimates,
- useful distributional uniqueness rather than raw domain difference,
- underserved-population and subgroup-performance constraints,
- bounded volume terms,
- uncertainty and anomaly penalties,
- entropy and effective-number diagnostics,
- constrained optimization with weight caps and temporal stability limits.

Current status: empirically tested for stability and aggregation behavior on synthetic and PCam federated benchmarks. It produces distinct weights under heterogeneity and does not degrade performance in the current patch-level setup. A performance/fairness advantage over simpler baselines still requires ablation and slide-level multi-center validation.

See: [FAIR-WEIGHTS-H theory](docs/theory/fair-weights-h.md)

---

## Validation ladder

```text
Synthetic smoke validation
  -> PCam patch-level validation
  -> PCam federated smoke validation
  -> PCam balanced federated benchmark
  -> PCam heterogeneous-site benchmark
  -> PANDA slide-level prostate benchmark
  -> Camelyon16 slide-level benchmark
  -> Camelyon17 real multi-center validation
  -> clinical validation
```

Current position: PCam patch-level/federated validation and PANDA slide-level prostate baselines are complete. Camelyon16/17 slide-level and real multi-center validation remain future work.

---

## Public claim boundary

See [CLAIM_BOUNDARY.md](CLAIM_BOUNDARY.md).

Short version:

> Research-only at this stage. Not clinically validated, not diagnostic software, and not currently used for patient care. Long-term goal is responsible clinical translation after proper validation, regulatory review, security review, usability testing, and deployment testing.
