# Computational Pathology Research Pipeline

An end-to-end computational pathology research system designed and engineered by **Matthew Vaishnav** for representation learning, whole-slide modeling, spatial tissue reasoning, scanner/site-effect analysis, and multi-institutional learning.

This repository is the central research pipeline: data preparation, pathology feature extraction, model development, controlled benchmarking, whole-slide aggregation, federated experimentation, result analysis, and release tooling live within one connected system rather than as isolated notebook experiments.

**Research only. Not clinical or diagnostic software.**

## Pipeline at a glance

```text
Public / paired-acquisition pathology data
                │
                ▼
      preprocessing + data contracts
                │
                ▼
   pathology foundation-model features
      (Phikon, DINOv2, other encoders)
                │
        ┌───────┴────────┐
        ▼                ▼
representation       whole-slide learning
learning              / MIL aggregation
(PA-NF)               (TransnnMIL + baselines)
        │                │
        └───────┬────────┘
                ▼
      spatial / tissue modeling
   (WSI-NCA / factorized dynamics)
                │
                ▼
      scanner + site-signal studies
                │
                ▼
   multi-institutional learning
(PathologyFL + FAIR-WEIGHTS-H)
                │
                ▼
 experiment, reproducibility, release,
      and scientific tooling
```

The pipeline is built around pretrained pathology/general-purpose encoders where appropriate; the surrounding research architecture, methods, training/evaluation code, aggregation systems, federated infrastructure, experiment machinery, and scientific tooling are developed in this program.

## Major research systems

### TransnnMIL — whole-slide neural aggregation

TransnnMIL is the custom whole-slide multiple-instance learning architecture developed in this project. It explores combinations of:

- global transformer-style attention over patch embeddings;
- local diagnostic-region reasoning;
- hierarchical spatial aggregation;
- topology-aware tissue modeling;
- graph-inspired neighborhood structure;
- branch-token fusion and matched fusion controls;
- adaptive pruning and efficiency experiments.

The PANDA pipeline supports slide-level Phikon feature bags and comparisons against mean pooling, gated AttentionMIL, TransMIL-style models, nnMIL-style baselines, and related MIL controls.

- [TransnnMIL v2 architecture](docs/models/transnnmil-v2.md)
- [TransnnMIL implementation](src/models/transnnmil/)
- [PANDA stabilization results](docs/results/panda-transnnmil-stability.md)

### Paired-Acquisition Neural Factorization (PA-NF) — representation learning across scanners

PA-NF uses matched acquisitions of the same tissue region to learn separate tissue-oriented and acquisition-oriented representations. The research line studies what can be separated when the same biological material is observed through multiple scanners rather than relying only on unpaired domain labels.

The program includes SCORPION paired-acquisition experiments, an external canine SCC study, allocation experiments, trained model releases, and frozen-feature experiments using DINOv2 representations.

- [SCORPION paired-acquisition study](https://github.com/matthewvaishnav/paired-acquisition-factorization-scorpion)
- [External canine SCC study](https://github.com/matthewvaishnav/paired-acquisition-factorization-caninescc)
- [Pair-repeat allocation study](https://github.com/matthewvaishnav/paired-acquisition-factorization-allocation)
- [PA-NF model release](https://huggingface.co/MatthewVaishnav/paired-acquisition-neural-factorization)
- [PA-NF evidence release](https://huggingface.co/datasets/MatthewVaishnav/paired-acquisition-factorization-evidence)

### WSI-NCA / Factorized Tissue Dynamics — learned tissue-state evolution

WSI-NCA is the spatial modeling line built around learned local update rules over whole-slide tissue representations. Instead of treating a slide as an unordered bag alone, this line investigates whether local state transitions and spatial neighborhoods can model tissue structure and multi-step interactions.

Current work includes synthetic mechanism tests, topology controls, tied/untied update studies, two-hop falsification experiments, and transfer infrastructure for PANDA feature bags.

- [WSI-NCA research branch](https://github.com/matthewvaishnav/computational-pathology-research/tree/research/wsi-nca-phase-a-20260807)
- [PANDA spatial-feature release specification](docs/releases/huggingface/panda-phikon-wsi-spatial-features/README.md)

### PathologyFL — federated computational pathology

PathologyFL is the multi-institutional learning layer of the pipeline. It provides pathology-oriented federated experiments with coordinator/client training, weighted aggregation, privacy hooks, robustness checks, and simulated-site evaluation on real pathology data.

Research components include:

- federated coordinator/client workflows;
- local pathology training loops;
- weighted model aggregation;
- differential-privacy integration;
- secure-aggregation work;
- client dropout and malformed-update robustness;
- site heterogeneity experiments;
- dominant-site / institutional signal analysis.

- [PathologyFL documentation](docs/federated/pathologyfl.md)
- [Federated source](src/features/federated/)
- [Federated experiment protocol](experiments/FEDERATED_ABLATION_PROTOCOL.md)

### FAIR-WEIGHTS-H — institutional aggregation

FAIR-WEIGHTS-H explores learned and constrained institutional weighting beyond ordinary sample-count FedAvg. The system studies how contribution, uncertainty, coverage, uniqueness, anomaly signals, entropy, and effective institution count can influence aggregation across simulated pathology sites.

- [FAIR-WEIGHTS-H documentation](docs/theory/fair-weights-h.md)

### Scanner and site-signal research

A second major thread of the program studies non-biological variation introduced by scanners, acquisition workflows, centers, and site structure.

This includes:

- paired-scanner representation experiments;
- scanner recoverability and feature-space analysis;
- dominant-site federated pathology detectors;
- site-signal alignment studies;
- CAMELYON17/WILDS external-center experiments;
- preparation/workflow metadata investigations.

The goal is to model these effects explicitly instead of silently allowing site identity to become a shortcut inside pathology representations.

### Pathology Pipeline Language / scientific compiler

The newest tooling line extends the pipeline beyond model code into machine-checkable experiment semantics. It introduces typed dataset identities, split constraints, paired-acquisition semantics, physical units, evidence objects, result ingestion, and bounded scientific statements.

The compiler work is designed so experimental assumptions that are normally buried in scripts or prose can be represented directly in the pipeline and checked before or after execution.

This work is being versioned as its own software line while remaining part of the broader computational-pathology research program.

## Data and benchmark layer

The pipeline currently spans several public and research pathology settings:

| Dataset / setting | Role in the pipeline |
|---|---|
| **PCam** | patch-level classification, training infrastructure, federated simulations |
| **PANDA** | whole-slide prostate grading, Phikon feature extraction, MIL and spatial modeling |
| **CAMELYON17 / WILDS** | multi-center and out-of-distribution site analysis |
| **SCORPION paired acquisitions** | scanner-aware paired representation learning |
| **External canine SCC paired acquisitions** | cross-study PA-NF evaluation |

Generated tensors and fixtures are used for software and mechanism tests; public pathology datasets are used for the corresponding scientific evaluations.

## Representative pipeline results

### PCam

The full PCam evaluation includes a 32,768-sample test split and has been used for patch-level modeling as well as simulated-site federated experiments.

| Metric | Result |
|---|---:|
| Validation AUC | 95.37% |
| Test AUC | 0.9394 |
| Test accuracy | 85.26% |
| F1 | 0.8507 |

### PANDA whole-slide learning

The slide-level pipeline validated **10,611 readable Phikon feature files** and supports repeated-seed MIL benchmarking.

| Model | Best validation QWK |
|---|---:|
| Mean-pooled Phikon + MLP | 0.7274 |
| Gated AttentionMIL | 0.8100 |
| Tuned TransnnMIL, seed 42 | 0.8155 |
| Tuned TransnnMIL, seed 123 | 0.8225 |
| Tuned TransnnMIL, seed 2025 | 0.8086 |

These values are benchmark results from the recorded experimental configurations, not claims of clinical performance.

## Repository map

```text
src/
  models/                 model and MIL architectures
  features/               pathology feature and federated components

scripts/
  training/               model training entry points
  experiments/            experiment and aggregation scripts

experiments/              registered / structured experiment protocols

docs/
  models/                  architecture documentation
  federated/              PathologyFL and privacy/robustness work
  results/                benchmark and experiment results
  research/               research notes and program structure
  releases/               model/data release specifications

paper/                    focused manuscript material
manuscripts/              longer-form research manuscripts
tests/                    software and experiment-infrastructure tests
```

## Research philosophy

The repository is organized as a **working scientific system**, not a single-model demo. New ideas are developed against matched baselines, repeated experiments, real pathology datasets where available, and explicit data/experiment contracts.

Reproducibility and provenance infrastructure are retained because they make the research easier to rerun and extend, but they are supporting machinery—not the scientific identity of the project.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
```

Large whole-slide images, generated feature archives, checkpoints, and run directories are kept outside Git where appropriate.

## Publications, models, and data releases

- [Focused PA-NF preprint](paper/paired_acquisition_preprint/README.md)
- [PA-NF trained model family](https://huggingface.co/MatthewVaishnav/paired-acquisition-neural-factorization)
- [PA-NF evidence dataset](https://huggingface.co/datasets/MatthewVaishnav/paired-acquisition-factorization-evidence)
- [Research documentation](docs/)

---

**Matthew Vaishnav — computational pathology, representation learning, whole-slide neural systems, spatial tissue modeling, and federated learning.**
