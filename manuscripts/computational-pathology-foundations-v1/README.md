# Accountable Neural Aggregation in Computational Pathology

**Internal-review manuscript package v1 (2026-08-04). Not for public release.**

This package contains the flagship foundations manuscript for the complete
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

## Key current evidence statuses (2026-08-04)

- Real paired-scanner validation:
  `complete_mixed_real_paired_scanner_allocation_effects` (frozen, unchanged).
- Fixed-estimand adjudication v1:
  `fixed_estimand_adjudication_not_ready` (prior, preserved).
- Exact 50-cell artifact recovery:
  `complete_exact_real_bottleneck_representation_recovery` (all 50 cells
  replicated, projected features + checkpoints archived).
- Fixed-estimand adjudication v2:
  `complete_no_neural_feature_space_increment_supported` (new forward-valid
  result; not reinterpreted as architectural invalidation).
- Corrected July 26 evidence release: immutable and bound to its snapshot;
  living claim boundary reported informationally.

## Layout

- `main.tex` — main manuscript (twocolumn, input-per-section).
- `supplement.tex` — supplement with implementation inventories and per-fold
  tables.
- `references.bib` — bibliography (no fabricated citations).
- `sections/` — one `.tex` per manuscript section (abstract, introduction,
  three level sections, PANDA, CAMELYON17, synthetic, accountability framework,
  audit, discussion, limitations, conclusion).
- `figures/` — reproducible figure sources (TikZ / data tables); figures never
  invent results.
- `tables/` — main-paper and supplement tables as data files.
- `claims/` — `manuscript_claim_ledger.csv`, `prohibited_claims.txt`.
- `evidence/` — `manuscript_evidence_manifest.json`, `research_line_bindings.csv`.
- `validation/` — `validate_manuscript.py` and its tests.
- `HOSTILE_REVIEW.md` — adversarial review register.
- `RECONSTRUCTION_REPORT.md` — how this package corrects the earlier PA-NF-only
  manuscript scope.

## Build

```sh
# From this directory, with MiKTeX latexmk available:
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

The build must be reproducible from a clean checkout. The PDF is marked
internal review and must not be published.
