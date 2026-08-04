# Reconstruction Report

**Date:** 2026-08-04
**Purpose:** how this manuscript package corrects the earlier PA-NF-only
manuscript scope, and how the reconstruction was verified.

## The rejected prior scope

A previous proposed manuscript reconstruction treated Paired-Acquisition Neural
Factorization (PA-NF) as if it were the whole research program. That scope is
rejected. The repository contains a larger, connected computational-pathology
research program with multiple foundational contributions:

1. patch-level benchmark and evaluation infrastructure (PCam);
2. neural representation formation and acquisition-confounding research (PA-NF,
   paired-scanner design, SCORPION, canine SCC, scanner/center subspaces,
   simple baselines and controls);
3. whole-slide neural aggregation through TransnnMIL;
4. institutional and federated aggregation through PathologyFL;
5. institutional contribution, safety, and representation weighting through
   FAIR-WEIGHTS-H;
6. PANDA institutional-shift and ordinal-learning studies;
7. CAMELYON17 center-subspace and held-out-center studies;
8. synthetic identifiability and mechanism studies;
9. reproducibility, provenance, claim auditing, and fail-closed validation
   infrastructure.

Every one of these is inventoried and represented accurately
(`docs/research/full-program-scientific-inventory-20260804.{md,csv}`). No
foundational research line is silently omitted merely because its strongest
empirical claim is still pending.

## What the reconstruction does

- Builds a repository-wide scientific inventory with per-line classification
  (`active_corrected_empirical_evidence`, `implemented_architecture_pending_
  controlled_validation`, `implemented_research_infrastructure`, `proposed_
  protocol_with_execution_validation`, `synthetic_mechanism_evidence`,
  `negative_or_mixed_empirical_result`, `historical_withdrawn_evidence`,
  `future_protocol_only`, `prohibited_by_evidence_scope`).
- Separates architectural contribution, implemented infrastructure, theoretical
  protocol, execution validation, and demonstrated empirical performance.
- Creates a flagship foundations manuscript organized as three primary neural
  aggregation levels (representation, whole-slide, institutional) plus the
  benchmark and provenance infrastructure.
- Creates a publication portfolio identifying the flagship manuscript and
  focused papers A--D.
- Binds every active claim to an immutable result artifact, every architectural
  claim to source and tests, and every protocol claim to its specification.
- Preserves frozen statuses verbatim:
  `complete_mixed_real_paired_scanner_allocation_effects`,
  `fixed_estimand_adjudication_not_ready`,
  `complete_exact_real_bottleneck_representation_recovery`, and
  `complete_no_neural_feature_space_increment_supported`.

## What the reconstruction does not do

- It does not train models, launch experiments, or change numerical thresholds,
  category sets, results, or frozen evidence.
- It does not publish anything publicly.
- It does not overwrite or delete the withdrawn PA-NF manuscript
  (`paper/paired_acquisition_manuscript/manuscript_draft.md` remains intact) or
  the arxiv package (`paper/arxiv/`).
- It does not update the public website.

## Verification

- `validation/validate_manuscript.py` fails when the manuscript omits any
  foundational research line, treats PA-NF as the only line, presents projected
  or withdrawn numbers as observed, conflates architecture with empirical
  status, or modifies frozen artifacts.
- `validation/tests/` exercise the twenty required fail conditions.
- The PDF builds deterministically from the package sources
  (`latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`); the build
  report and PDF hash are recorded in `validation/build_report.json`.
