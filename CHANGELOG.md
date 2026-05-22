# Changelog

All notable changes to this repository will be documented in this file.

This project uses a pragmatic changelog format inspired by [Keep a Changelog](https://keepachangelog.com/). Dates use ISO format.

---

## [Unreleased]

### Added

#### FAIR-WEIGHTS-H institutional weighting

- Added `docs/FAIR_WEIGHTS_HYBRID_PROTOCOL.md`, a regulatory-safe protocol for hybrid institutional weighting in federated computational pathology.
- Added experimental FAIR-WEIGHTS-H weighting engine:
  - `src/features/federated/pathology_fl/weighting/fair_weights_h.py`
- Added institutional weighting package exports:
  - `src/features/federated/pathology_fl/weighting/__init__.py`
- Added explicit weighted aggregation adapter:
  - `src/features/federated/pathology_fl/aggregator/weighted.py`
- Exported `ExplicitWeightedAggregator` from the federated aggregator package.
- Added FAIR-WEIGHTS-H diagnostics including normalized entropy, effective institution count, max/min weights, and integrity exclusion counts.
- Added conservative mode to reduce diversity/fairness influence under instability.

#### Synthetic weighting experiments

- Added deterministic synthetic federation profiles:
  - `src/features/federated/pathology_fl/weighting/synthetic_federation.py`
- Added baseline weighting strategies:
  - equal weighting,
  - volume weighting,
  - legacy prestige weighting,
  - FAIR-WEIGHTS-H weighting.
- Added synthetic perturbation utilities:
  - uncertainty spike,
  - scanner shift,
  - quality degradation,
  - rare-population enrichment.
- Added benchmark runner:
  - `src/features/federated/pathology_fl/weighting/benchmark.py`
- Added perturbation experiment runner:
  - `src/features/federated/pathology_fl/weighting/experiment_runner.py`
- Added canonical perturbation suite:
  - `src/features/federated/pathology_fl/weighting/experiment_suite.py`
- Added markdown reporting utilities:
  - `src/features/federated/pathology_fl/weighting/reporting.py`
  - `src/features/federated/pathology_fl/weighting/report_generator.py`

#### Tests

- Added unit tests for the FAIR-WEIGHTS-H engine.
- Added tests for explicit weighted aggregation.
- Added tests for synthetic federation baselines.
- Added tests for perturbation utilities.
- Added tests for the perturbation experiment runner.
- Added tests for markdown report generation.
- Added tests for the canonical experiment suite.

#### Documentation and website navigation

- Reworked `docs/index.md` into a navigation-first documentation hub.
- Added implementation/evidence links to key status claims in `docs/index.md`.
- Updated `docs/index.md` to distinguish documented benchmark results, experimental FAIR-WEIGHTS-H work, future validation, and non-claimed regulatory status.
- Updated `website/src/pages/index.tsx` to improve website navigation and reduce long paper-like scrolling.
- Added quick navigation cards for results, FAIR-WEIGHTS-H, getting started, and source code.
- Replaced legacy DMI prestige-weight framing with FAIR-WEIGHTS-H mathematical framing.

### Changed

- Legacy hospital-type multipliers such as cancer center 2.0x, teaching hospital 1.5x, community hospital 1.0x, and rural hospital 0.8x are now described as comparison baselines rather than preferred institutional weighting logic.
- DMI documentation now emphasizes evidence-based institutional weighting via contribution, quality, useful uniqueness, uncertainty, and subgroup-safety constraints.
- Website homepage now functions more like a research/project portal rather than a single long academic abstract.

### Notes

- FAIR-WEIGHTS-H is experimental and requires empirical validation before clinical, regulatory, or deployment claims.
- Synthetic perturbation experiments are engineering checks only; they are not evidence of clinical effectiveness.
- Existing reported PCam and infrastructure claims were preserved while adding clearer navigation and status links.

---

## Historical changes

Earlier changes were not tracked in this changelog. See Git history for prior development details.
