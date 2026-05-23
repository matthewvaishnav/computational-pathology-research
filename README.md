# Matthew Vaishnav Computational Pathology, Federated Oncology Learning, and Mathematical Validation Infrastructure

Whole-slide pathology AI, TransnnMIL v2.0, PathologyFL, FAIR-WEIGHTS-H institutional weighting, PCam/PANDA/Camelyon validation, and multi-institutional oncology learning infrastructure.

**Documentation:** https://matthewvaishnav.github.io/computational-pathology-research/  
**Literature positioning:** https://matthewvaishnav.github.io/computational-pathology-research/research/literature-positioning  
**PCam results:** https://matthewvaishnav.github.io/computational-pathology-research/results/pcam-results

---

## What this repository is

This repository is my computational pathology and oncology AI research framework. It combines model development, federated learning, benchmark validation, statistical reporting, and documentation infrastructure for pathology AI experiments.

The work spans:

- patch-level and whole-slide pathology modeling
- multiple-instance learning for WSI classification
- the custom **TransnnMIL v2.0** architecture direction
- **PathologyFL** federated learning infrastructure
- **FAIR-WEIGHTS-H** institutional weighting research
- PCam, PANDA, and Camelyon validation workflows
- threshold analysis and statistical validation
- PubMed-grounded literature positioning
- testing, reporting, and deployment-oriented engineering infrastructure

This is a research and engineering framework. It is **not clinically validated** and is **not regulatory cleared**.

---

## Current evidence snapshot

| Area | Status |
|---|---|
| PCam public benchmark | **85.26% test accuracy**, **0.9394 test AUC** on the full 32,768-sample test set |
| PCam comparison | **#1 by AUC among 11 compared PCam methods** in the documented comparison table |
| Bootstrap validation | 1,000 bootstrap resamples reported for PCam metrics |
| Threshold optimization | Screening threshold analysis reduced missed tumor predictions by 61.7% in the documented PCam analysis |
| FAIR-WEIGHTS-H smoke/unit tests | Focused tests passing |
| PCam federated smoke tests | Equal, volume, prestige, and FAIR-WEIGHTS-H strategies completed on real PCam patches split into simulated sites |
| PCam balanced federated benchmark | Complete: FAIR-WEIGHTS-H stable, no performance degradation observed |
| PCam heterogeneous benchmark | Complete: strategies produced different weight trajectories, but patch-level performance was insensitive to those differences |
| FAIR-WEIGHTS-H empirical status | Tested for execution stability and aggregation behavior; performance/fairness advantage over simpler baselines not yet demonstrated |
| PubMed literature positioning | Added: related work, citation table, claim-strength table, and next experiment priorities |
| PANDA feature extraction | In progress on a separate RTX 3060 12GB machine; not yet marked complete |
| Camelyon16/17 validation | Planned next slide-level / multi-center validation target |
| Clinical validation | Not completed |

---

## Core components

### TransnnMIL v2.0

Custom WSI multiple-instance learning architecture direction combining:

- TransMIL-style global attention
- hierarchical spatial pooling
- topology / graph-aware tissue structure modeling
- optional adaptive pruning

Current status: design and implementation direction documented; slide-level WSI benchmark evidence is the next required step.

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

Current position: PCam patch-level and PCam federated validation are complete. PANDA feature extraction is nearly complete but still running on a separate RTX 3060 worker machine. Camelyon16/17 slide-level validation remains future work.

---

## Research positioning

The PubMed-grounded literature review positions this work at the intersection of:

1. **WSI MIL architecture research** — CLAM, NATMIL, SlideMamba, foundation-model MIL comparisons.
2. **Federated pathology infrastructure** — HistoFL and medical-imaging FL.
3. **Institutional weighting / fairness** — FAIR-WEIGHTS-H has no direct cited WSI-FL comparator across its full eight-dimensional weighting scheme.
4. **Benchmark and validation infrastructure** — PCam patch-level validation, PCam federated splits, PANDA/Camelyon roadmap.

The strongest current novelty signal is FAIR-WEIGHTS-H and PathologyFL as institutional weighting / federated validation infrastructure. The strongest completed benchmark result is PCam AUC. The biggest next evidence gap is slide-level validation.

See: [Literature positioning](docs/research/literature-positioning.md)

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

## Repository hygiene

Large local artifacts are intentionally ignored:

- raw datasets
- WSI tiles
- extracted features
- PANDA / PCam / Camelyon feature stores
- checkpoints
- model weights
- raw predictions
- temporary experiment outputs

Tracked artifacts should be small and reproducible:

- reports
- metrics summaries
- benchmark tables
- configuration files
- documentation
- source code

See: [results policy](results/README.md)

---

## Documentation map

- [Overview](docs/overview/index.md)
- [Literature positioning](docs/research/literature-positioning.md)
- [Claim status](docs/overview/claim-status.md)
- [Getting started](docs/getting-started.md)
- [Models](docs/models/index.md)
- [TransnnMIL v2.0](docs/models/transnnmil-v2.md)
- [PathologyFL](docs/federated/pathologyfl.md)
- [FAIR-WEIGHTS-H](docs/theory/fair-weights-h.md)
- [Validation overview](docs/validation/index.md)
- [PCam results](docs/results/pcam-results.md)
- [Performance comparison](docs/results/performance-comparison.md)
- [Engineering architecture](docs/engineering/architecture.md)
- [Roadmap](docs/roadmap/index.md)

---

## Interpretation guardrails

This repository contains substantial engineering and research infrastructure, but claims should be interpreted by evidence level:

- **Public pathology benchmark validation is real evidence.**
- **PCam is patch-level.** It does not replace Camelyon16/17 slide-level WSI validation.
- **PANDA and Camelyon are needed for stronger slide-level claims.**
- **FAIR-WEIGHTS-H has been empirically tested for stability and behavior.** It has not yet shown a consistent performance/fairness advantage over simpler baselines.
- **TransnnMIL v2.0 is not yet proven superior to CLAM, TransMIL, NATMIL, or SlideMamba at the slide level.**
- **Clinical validation and regulatory clearance are separate future requirements.**

---

## License and use

This repository is intended for research and engineering development. Clinical deployment requires additional validation, governance, security review, and regulatory assessment.
