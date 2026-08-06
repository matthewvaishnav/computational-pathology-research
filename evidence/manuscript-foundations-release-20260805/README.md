# Manuscript-Foundations Release (2026-08-05)

Forward-valid evidence release for the **Accountable Neural Aggregation in
Computational Pathology** manuscript. This package hardens the evidence bindings
of the foundations manuscript: every active numerical claim binds to a real,
verified artifact hash; no placeholder hash remains.

## Contents

- `release_manifest.json` — hardened release manifest with every bound artifact,
  hash, size, source commit, and the canonical internal hash.
- `artifacts/` — copied immutable result artifacts:
  - `fixed_estimand_real_feature_space_adjudication_v2_result.json`
    (SHA-256 `2cccaad7…`)
  - `real_bottleneck_representation_recovery_result.json`
    (SHA-256 `2075a2aa…`)
- `claim_boundary_snapshot.md` — the claim boundary this release commits to.
- `citation_audit.md` — bibliography verification; no placeholders.
- `pathologyfl_test_report.md` — observed PathologyFL / FAIR-WEIGHTS-H test
  results (24 passed, 13 skipped for optional deps, 0 failed).
- `validation_environment.json` — environment and optional-dependency status.
- `build_log.md` — LaTeX build command, PDF hashes, log summary.

## Bound references (verified at their canonical paths)

- `manuscripts/computational-pathology-foundations-v1/` — manuscript source,
  supplement, figures, tables, claims ledger, evidence manifest, validation.
- `evidence/paired_acquisition/corrected-20260726/release_manifest.json` —
  immutable corrected paired-acquisition evidence (SHA-256 `a7a7e34d…`).
- `results/camelyon17/center_weighting_5seed_summary.md` — tracked CAMELYON17
  5-seed summary (SHA-256 `3637830e…`), labeled a bounded descriptive proxy.
- `docs/research/full-program-prior-art-review-20260804.md` — completed
  prior-art review (20 questions answered with primary sources).

## No modification

This release modifies no historical evidence package, no frozen scientific
result, and no public site. PCam is described as implementation/benchmark context
because its underlying numerical result artifacts are not tracked.
