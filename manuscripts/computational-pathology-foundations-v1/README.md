# Accountable Neural Aggregation in Computational Pathology

**Public foundations preprint package (2026-08-06).**

This package contains the primary public foundations manuscript for the complete
computational-pathology research program. It is organized as a connected system
of three primary neural aggregation levels plus the cross-cutting benchmark and
provenance infrastructure that connects them:

1. **Level I — Representation formation** (patch and foundation-model
   representations; Paired-Acquisition Neural Factorization; scanner and center
   subspaces; synthetic identifiability).
2. **Level II — Whole-slide aggregation** (TransnnMIL and its architectural
   families).
3. **Level III — Institutional aggregation** (PathologyFL; FAIR-WEIGHTS-H;
   institutional weighting; PANDA and CAMELYON17 studies).
4. **Cross-cutting — Benchmark and scientific-audit infrastructure** (PCam
   patch evaluation; provenance, immutable releases, claim validation, exact
   artifact recovery).

The focused PA-NF manuscript remains available as a secondary supporting paper.
It does not replace this program-level foundations manuscript.

## Status vocabulary

Every research line is classified with one of these statuses, and architecture
vs. empirical status is never conflated:

- `active_corrected_empirical_evidence`
- `implemented_architecture_pending_controlled_validation`
- `implemented_research_infrastructure`
- `proposed_protocol_with_execution_validation`
- `synthetic_mechanism_evidence`
- `negative_or_mixed_empirical_result`
- `historical_withdrawn_evidence`
- `future_protocol_only`
- `prohibited_by_evidence_scope`

## Key current evidence statuses

- Real paired-scanner validation:
  `complete_mixed_real_paired_scanner_allocation_effects`.
- Exact 50-cell artifact recovery:
  `complete_exact_real_bottleneck_representation_recovery`.
- Fixed-estimand adjudication v2:
  `complete_no_neural_feature_space_increment_supported`.
- Corrected July 26 evidence release: immutable and bound to its snapshot;
  living claim-boundary status is reported explicitly.
- TransnnMIL: implemented authored architecture; controlled superiority evidence
  remains pending.
- PathologyFL: implemented research infrastructure; not a real multi-center
  deployment validation.
- FAIR-WEIGHTS-H: proposed and partially implemented protocol; fairness and
  performance superiority remain unestablished.

## Layout

- `main.tex` — main manuscript (two-column, sectioned source).
- `supplement.tex` — supplement with implementation inventories and per-fold
  tables.
- `references.bib` — bibliography.
- `sections/` — manuscript sections.
- `figures/` — reproducible figure sources, including the corrected full-width
  PA-NF architecture diagram.
- `tables/` — main-paper and supplement tables.
- `claims/` — claim ledger and prohibited-claim register.
- `evidence/` — evidence manifest and research-line bindings.
- `validation/` — manuscript validator and tests.
- `HOSTILE_REVIEW.md` — adversarial review register.
- `RECONSTRUCTION_REPORT.md` — scope reconstruction report.

## Build

The PA-NF SVG must first be converted to PDF for pdfLaTeX:

```sh
rsvg-convert -f pdf -o figures/pa_nf_architecture.pdf figures/pa_nf_architecture.svg
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error supplement.tex
```

The GitHub publication workflow performs these steps, validates the page counts
and LaTeX logs, and publishes the main PDF, supplement, source archive, and the
secondary focused PA-NF manuscript.
