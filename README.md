# Matthew Vaishnav Computational Pathology, Federated Oncology Learning, and Mathematical Validation Infrastructure

Whole-slide pathology AI, TransnnMIL v2.0, PathologyFL, FAIR-WEIGHTS-H institutional weighting, PCam/Camelyon validation, and multi-institutional oncology learning infrastructure.

**Documentation:** https://matthewvaishnav.github.io/computational-pathology-research/

---

## What this repository is

This repository is a computational pathology and oncology AI platform that combines:

- whole-slide and patch-level pathology modeling
- multiple-instance learning for WSI classification
- the custom **TransnnMIL v2.0** architecture
- **PathologyFL** federated learning infrastructure
- **FAIR-WEIGHTS-H** institutional weighting research
- PCam and Camelyon validation workflows
- testing, reporting, and deployment-oriented engineering infrastructure

The project is for research and engineering validation. It is **not clinically validated** and is **not regulatory cleared**.

---

## Current evidence snapshot

| Area | Status |
|---|---|
| PCam benchmark results | 85.26% accuracy and 93.94% AUC reported on PCam experiments |
| FAIR-WEIGHTS-H unit tests | Passing focused tests |
| PCam federated smoke tests | All four strategies completed on real PCam patches split into simulated sites |
| PCam balanced federated benchmark | Completed: equal, volume, prestige, and FAIR-WEIGHTS-H had similar global accuracy on balanced simulated sites |
| PCam heterogeneous benchmark | Running / in progress |
| Camelyon17 real multi-center validation | Planned / future work |
| Clinical validation | Not completed |

---

## Core components

### TransnnMIL v2.0

Custom WSI multiple-instance learning architecture combining:

- TransMIL-style global attention
- hierarchical spatial pooling
- topology / graph-aware tissue structure modeling
- optional adaptive pruning

See: [TransnnMIL v2.0 documentation](docs/models/transnnmil-v2.md)

### PathologyFL

Federated learning infrastructure for computational pathology:

- coordinator/client workflow
- local pathology training
- weighted aggregation
- differential privacy hooks
- secure aggregation work
- byzantine/dropout robustness checks

See: [PathologyFL documentation](docs/federated/pathologyfl.md)

### FAIR-WEIGHTS-H

Experimental institutional weighting engine for federated oncology learning.

It replaces simple volume or prestige weighting with an auditable weighting scaffold using signals such as quality, useful uniqueness, fairness, contribution, volume, and uncertainty.

See: [FAIR-WEIGHTS-H theory](docs/theory/fair-weights-h.md)

---

## Validation ladder

```text
Synthetic smoke validation
  -> PCam patch-level smoke validation
  -> PCam balanced federated benchmark
  -> PCam heterogeneous-site benchmark
  -> Camelyon17 real multi-center validation
  -> clinical validation
```

Current position: PCam patch-level validation is working; real multi-center validation remains future work.

---

## Quickstart

```bash
git clone https://github.com/matthewvaishnav/computational-pathology-research.git
cd computational-pathology-research
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Useful commands

Run focused FAIR-WEIGHTS-H tests:

```bash
pytest tests/federated/test_fair_weights_h.py
pytest tests/federated/test_weighted_aggregator.py
```

Run core federated integration tests:

```bash
pytest tests/federated/test_fl_integration.py
```

Run the balanced PCam federated benchmark:

```powershell
scripts\federated\run_pcam_benchmark.ps1
```

Analyze balanced benchmark results:

```bash
python scripts/federated/analyze_pcam_benchmark.py
```

Run VitePress documentation locally:

```bash
npm install
npm run docs:dev
```

Build documentation:

```bash
npm run docs:build
```

---

## Documentation map

- [Overview](docs/overview/index.md)
- [Claim status](docs/overview/claim-status.md)
- [Getting started](docs/getting-started.md)
- [Models](docs/models/index.md)
- [TransnnMIL v2.0](docs/models/transnnmil-v2.md)
- [PathologyFL](docs/federated/pathologyfl.md)
- [FAIR-WEIGHTS-H](docs/theory/fair-weights-h.md)
- [Validation overview](docs/validation/index.md)
- [PCam benchmark report](docs/validation/pcam-benchmark-report.md)
- [Engineering architecture](docs/engineering/architecture.md)
- [Roadmap](docs/roadmap/index.md)

---

## Interpretation guardrails

This repository contains substantial engineering and research infrastructure, but claims should be interpreted by evidence level:

- **Implemented** does not mean clinically validated.
- **Synthetic validation** does not mean real-world validation.
- **PCam simulated-site validation** uses real pathology patches but not real hospital-level site structure.
- **Camelyon17 validation** is required for real multi-center evidence.
- **Clinical validation and regulatory clearance** are separate future requirements.

---

## License and use

This repository is intended for research and engineering development. Clinical deployment requires additional validation, governance, security review, and regulatory assessment.
