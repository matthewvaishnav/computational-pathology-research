# Overview

This project is **Matthew Vaishnav Computational Pathology, Federated Oncology Learning, and Mathematical Validation Infrastructure**: a research and engineering platform for building, testing, benchmarking, and documenting computational pathology systems.

It brings together whole-slide pathology AI, multiple-instance learning, federated learning, benchmark automation, clinical-data integration components, and mathematical validation tooling. The goal is not just to train a model, but to build the surrounding infrastructure needed to make computational pathology experiments reproducible, inspectable, and extensible.

## What problem this solves

Computational pathology work often breaks down between three layers:

1. **Model research** — attention MIL, transformer MIL, topology-aware WSI modeling, patch classifiers, and foundation encoders.
2. **Real data workflow** — PCam/Camelyon data loading, WSI preprocessing, patch extraction, metrics, thresholds, and failure analysis.
3. **Deployment-oriented infrastructure** — testing, PACS/DICOM/FHIR integration, federated learning, privacy hooks, robustness checks, and documentation.

This repository tries to connect those layers into one coherent platform.

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
- whole-slide image classification workflows,
- attention-based multiple-instance learning,
- TransMIL-style global attention,
- CLAM-style attention learning,
- custom TransnnMIL v2.0 development,
- feature extraction with pretrained CNN/foundation-style encoders,
- and threshold tuning for screening-style sensitivity/specificity tradeoffs.

The PCam work is the strongest completed public-dataset benchmark so far. The documented result is **85.26% test accuracy** and **0.9394 test AUC** on the full 32,768-sample PCam test split, with bootstrap confidence intervals.

### 2. TransnnMIL v2.0

TransnnMIL v2.0 is the custom model direction in this project. It is intended to combine several complementary WSI modeling ideas:

- global transformer-style attention over patch embeddings,
- hierarchical spatial pooling,
- topology-aware tissue-structure modeling,
- graph-inspired reasoning over spatial neighborhoods,
- and optional adaptive pruning to reduce computation.

The goal is to move beyond single-patch classification toward models that better represent whole-slide structure.

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

Standard federated averaging treats institutions uniformly or weights them by volume. Prestige-style weighting can over-favor large or famous institutions. FAIR-WEIGHTS-H explores a more auditable weighting scaffold based on signals such as:

- adjusted quality,
- useful uniqueness,
- fairness / representation,
- contribution,
- volume,
- uncertainty,
- entropy,
- and effective number of institutions.

The balanced PCam benchmark showed that FAIR-WEIGHTS-H is stable and does not degrade performance. The heterogeneous PCam benchmark produced an informative null result: strategies generated different weights, but performance did not meaningfully change in the current patch-level setup. That points to benchmark insensitivity, not proof that the weighting theory is ineffective.

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
| Camelyon17 validation | Planned | Real multi-center WSI validation target |
| Clinical validation | Not completed | Requires clinical workflow / patient-level validation and governance |

## Key results so far

### PCam public-dataset benchmark

| Metric | Value |
|---|---:|
| Test accuracy | 85.26% |
| Test AUC | 0.9394 |
| F1 | 0.8507 |
| Test samples | 32,768 |
| Training samples | 262,144 |
| Bootstrap samples | 1,000 |
| Hardware | RTX 4070 Laptop |

The PCam result ranked **#1 by AUC among 11 compared methods** in the restored comparison table.

### Threshold optimization

The default threshold was conservative and missed many tumor patches. A screening-oriented threshold shifted the operating point:

| Quantity | Value |
|---|---:|
| Screening threshold | 0.051 |
| Sensitivity | 90.0% |
| Specificity | 80.3% |
| Missed tumor predictions before tuning | 4,276 |
| Missed tumor predictions after tuning | 1,639 |
| Reduction in missed tumor predictions | 61.7% |

This is threshold analysis for research evaluation. It is not clinical deployment validation.

### Federated benchmark status

| Benchmark | Result |
|---|---|
| Synthetic Camelyon-like smoke | Passed |
| PCam equal-weight smoke | Passed |
| PCam all-strategy smoke | Passed |
| PCam balanced benchmark | All strategies similar; FAIR-WEIGHTS-H stable |
| PCam heterogeneous benchmark | Different weight trajectories, no measurable performance sensitivity |

## Architecture at a glance

The repository is organized around a layered research platform:

```text
Data / WSI processing
  -> patch extraction and feature encoding
  -> MIL and TransnnMIL models
  -> training and evaluation loops
  -> federated learning / PathologyFL
  -> validation reports and benchmark analysis
  -> documentation and deployment-oriented engineering
```

Major areas:

- `models/` — MIL architectures, TransnnMIL components, foundation encoders.
- `features/federated/` — PathologyFL, aggregation, privacy, robustness.
- `data/` — PCam/Camelyon and WSI processing utilities.
- `training/` — optimized training loops and experiment runners.
- `scripts/federated/` — smoke tests, benchmarks, and analysis scripts.
- `docs/` — VitePress documentation, validation reports, and project roadmap.

## What this project can currently claim

This project can claim:

- public pathology benchmark validation on PCam,
- strong PCam AUC with bootstrap confidence intervals,
- benchmarked training/inference optimization on consumer hardware,
- working federated smoke tests on real PCam patches,
- implemented FAIR-WEIGHTS-H weighting scaffold,
- documented balanced and heterogeneous PCam federated benchmark results,
- and production-oriented research infrastructure including tests, reports, and docs.

## What it should not claim yet

This project should not claim yet:

- clinical validation,
- regulatory clearance,
- proven hospital deployment performance,
- prospective patient-level validation,
- or real multi-center WSI validation until Camelyon17 or equivalent experiments are complete.

The work is legitimate and substantial, but the evidence level matters.

## Why this matters

The practical value of the project is that it moves computational pathology work away from fragile one-off scripts and toward a reusable research platform:

- experiments can be rerun,
- benchmarks can be compared,
- claims can be traced to reports,
- federated behavior can be inspected,
- thresholds can be tuned explicitly,
- and future Camelyon17-style validation has a clear path.

## Read next

- [Claim status](claim-status.md)
- [PCam results](../results/pcam-results.md)
- [Performance comparison](../results/performance-comparison.md)
- [TransnnMIL v2.0](../models/transnnmil-v2.md)
- [PathologyFL](../federated/pathologyfl.md)
- [FAIR-WEIGHTS-H](../theory/fair-weights-h.md)
- [Validation overview](../validation/index.md)
