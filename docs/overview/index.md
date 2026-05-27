# Overview

This project is a **Computational Pathology AI Research Framework**: a research and engineering platform for building, testing, benchmarking, and documenting computational pathology systems.

It brings together whole-slide pathology AI, multiple-instance learning, federated learning, benchmark automation, clinical-data integration prototypes, and mathematical validation tooling. The goal is not just to train a model, but to build the surrounding infrastructure needed to make computational pathology experiments reproducible, inspectable, and extensible.

## What problem this solves

Computational pathology work often breaks down between three layers:

1. **Model research** — attention MIL, transformer MIL, topology-aware WSI modeling, patch classifiers, and foundation encoders.
2. **Real data workflow** — PCam/PANDA/Camelyon data loading, WSI preprocessing, patch extraction, metrics, thresholds, and failure analysis.
3. **Research infrastructure** — testing, DICOM/PACS/FHIR-style prototypes, federated learning, privacy hooks, robustness checks, and documentation.

This repository tries to connect those layers into one coherent research platform.

Instead of a notebook-only experiment, it provides a research system with:

- model implementations,
- benchmark scripts,
- federated learning experiments,
- validation reports,
- reproducibility commands,
- documentation pages,
- and explicit claim-status guardrails.

## Core research areas

### 1. Computational pathology modeling

The project supports both patch-level and whole-slide pathology workflows.

Key modeling areas include:

- PCam patch-level tumor classification,
- PANDA slide-level prostate cancer grading,
- whole-slide image classification workflows,
- attention-based multiple-instance learning,
- TransMIL-style global attention,
- CLAM-style attention learning,
- custom TransnnMIL development,
- feature extraction with pretrained CNN and pathology foundation-style encoders,
- and threshold tuning for screening-style sensitivity/specificity tradeoffs.

The strongest current evidence is the combination of PCam validation and PANDA slide-level MIL benchmarking. The PCam work reports **95.37% validation AUC** and **0.9394 test AUC** on the full 32,768-sample PCam test split. The PANDA work validates **10,611 readable slide-level Phikon feature files** and compares mean pooling, gated AttentionMIL, and tuned TransnnMIL.

### 2. TransnnMIL

TransnnMIL is the custom model direction in this project. It is intended to combine several complementary WSI modeling ideas:

- global transformer-style attention over patch embeddings,
- local diagnostic-region / nearest-neighbor style reasoning,
- hierarchical spatial pooling,
- topology-aware tissue-structure modeling,
- graph-inspired reasoning over spatial neighborhoods,
- and optional adaptive pruning to reduce computation.

The goal is to move beyond single-patch classification toward models that better represent whole-slide structure.

Current PANDA evidence shows tuned TransnnMIL is competitive with gated AttentionMIL and slightly favorable across the current repeated-seed experiments, but not conclusively superior.

Read more: [TransnnMIL v2.0](../models/transnnmil-v2.md)

### 3. PathologyFL

PathologyFL is the federated learning layer for computational pathology experiments.

It focuses on the situation where multiple hospitals or institutions should collaborate on model training without directly sharing raw patient data. The infrastructure includes:

- coordinator/client federated workflows,
- local training loops,
- weighted aggregation,
- differential privacy hooks,
- secure aggregation work,
- byzantine/dropout robustness checks,
- federated smoke tests,
- and PCam simulated-site benchmarks.

The current federated experiments use real PCam pathology patches split into simulated sites. This validates the federated pipeline on real image tensors, but it is not the same as real hospital-level multi-center validation.

Read more: [PathologyFL](../federated/pathologyfl.md)

### 4. FAIR-WEIGHTS-H

FAIR-WEIGHTS-H is the institutional weighting research component.

Standard federated averaging treats institutions uniformly or weights them by volume. FAIR-WEIGHTS-H explores a more auditable weighting scaffold based on signals such as:

- validated contribution,
- uncertainty,
- subgroup coverage,
- useful uniqueness,
- anomaly penalties,
- entropy,
- and effective number of institutions.

The current status is deliberately conservative: FAIR-WEIGHTS-H has been tested for execution stability and aggregation behavior, but a performance or fairness advantage over simpler baselines still requires controlled validation.

Read more: [FAIR-WEIGHTS-H](../theory/fair-weights-h.md)

## Main validation ladder

The project uses a staged validation ladder rather than treating every result as equal.

| Stage | Status | Meaning |
|---|---|---|
| Synthetic smoke validation | Complete | Basic plumbing and numerical stability checks |
| PCam patch-level validation | Complete | Real pathology patch benchmark validation |
| PCam federated smoke tests | Complete | Federated pipeline runs on real PCam patches split into simulated sites |
| PCam balanced federated benchmark | Complete | Weighting strategies compared under balanced simulated sites |
| PCam heterogeneous benchmark | Complete | Different weights produced, but no performance sensitivity observed |
| PANDA slide-level prostate benchmark | Complete | Slide-level MIL over Phikon feature bags |
| PANDA TransnnMIL ablations | Complete | Patch cap, learning rate, and dropout ablations documented |
| Camelyon16/17 validation | Planned | Real multi-center WSI validation target |
| Clinical validation | Not completed | Requires clinical workflow / patient-level validation and governance |

## Key results so far

### PCam public-dataset benchmark

| Metric | Value |
|---|---:|
| Validation AUC | 95.37% |
| Test accuracy | 85.26% |
| Test AUC | 0.9394 |
| F1 | 0.8507 |

### PANDA slide-level prostate benchmark

| Model | Best validation QWK |
|---|---:|
| Mean-pooled Phikon + MLP | 0.7274 |
| Gated AttentionMIL | 0.8100 |
| Tuned TransnnMIL, seed 42 | 0.8155 |
| Tuned TransnnMIL, seed 123 | 0.8225 |
| Tuned TransnnMIL, seed 2025 | 0.8086 |

### PANDA TransnnMIL ablation summary

| Run | Best validation QWK | Interpretation |
|---|---:|---|
| lr=3e-4, dropout=0.15, patch cap 600 | 0.8155 | tuned reference setting |
| lr=1e-3, dropout=0.15, patch cap 600 | 0.7403 | high learning rate unstable |
| lr=3e-4, dropout=0.25, patch cap 600 | 0.8015 | higher dropout mildly hurts |

## Claim boundary

Research-only at this stage. Not clinically validated, not diagnostic software, and not currently used for patient care. Long-term goal is responsible clinical translation after proper validation, regulatory review, security review, usability testing, and deployment testing.
