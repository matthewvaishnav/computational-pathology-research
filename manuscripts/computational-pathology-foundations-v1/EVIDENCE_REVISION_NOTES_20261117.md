# PA-NF manuscript — working evidence revision (17 November 2026)

**Status:** working editorial branch, not a published manuscript release, journal submission, new experiment, or biological-purity confirmation.

**Canonical repository:** `matthewvaishnav/computational-pathology-research`  
**Original starting revision:** `f30f262bbee089b6eaa38ed12574d43e23eacd3f` (main at initial verification).  
**Revision branch:** `manuscript/2026-11-17-evidence-revision`.  
**Previously published September preprint/PDF:** preserved and untouched.

## Editorial corrections performed

1. Add a bounded, post-registration-selected **two-physical-slide ScanComp external retrieval** subsection for `kwo930` and `jye249`, including original frozen checkpoint/control contrasts, whole-panel exclusions, variance cautions, and separate prior exposed TRAIN-AUDIT mechanism data. No new inference was performed.
2. Correct outdated **TransnnMIL repaired-matched study pending** language. Report the already completed 7-arm × 5-seed PANDA confirmation: repaired TransnnMIL QWK 0.792195, TransMIL 0.805826, nnMIL 0.779618; primary repaired-minus-TransMIL mean -0.013631 (0/5 positive), with narrower nnMIL comparison +0.012577 (5/5 positive). Preserve 18-run pre-repair optimization records as historical only.
3. Clarify that documented PCam F1=0.8507 is **macro-F1**, not tumor-positive F1.
4. Distinguish conceptual FAIR-WEIGHTS-H hard eligibility/cap equations from verified historical implementation enforcement.
5. Amend discussion, abstract, conclusion, and limitations for correct scope; keep the full-pipeline manuscript architecture unchanged.

## Governing evidence

- [Original SCORPION/PA-NF claim and comparator ledger](https://app.notion.com/p/4026710bb9fa81daa9bae5304544e519).
- [September 2026 repaired TransnnMIL 35-fit source audit](https://app.notion.com/p/4016710bb9fa81148d66f0bcb8bea83a).
- [Frozen 35-cell PANDA checkpoint manifest at prior branch/commit](https://github.com/matthewvaishnav/computational-pathology-research/blob/48e791fa761f948f69498d0cb053026447aae4ec/experiments/transnnmil/external/frozen_panda_checkpoint_manifest_20260925.json).
- [ScanComp kwo930 completed frozen result](https://app.notion.com/p/4186710bb9fa8160ab16d26a791f19eb).
- [ScanComp jye249 completed frozen result](https://app.notion.com/p/4186710bb9fa8124abaed7a55d65ec49).
- [9 November original four-family checkpoint mechanism replay](https://app.notion.com/p/4126710bb9fa817191e7db63d2524626).
- [Original submission-production plan](https://app.notion.com/p/4136710bb9fa811db801f4aac5592085).

## Remaining verification before publication or merge

- Compile and review main LaTeX and supplement with all original figures, bibliography and local LaTeX dependencies; no complete PDF build was executed in this edit session.
- Reconcile figures, all displayed results, existing claims ledger and corresponding source-data tables against the latest amendment. Do not present two slides or 25 checkpoints per slide as population-level inference.
- Verify availability and distribution rights of frozen external source/model receipts prior to claiming full public release.
- Author review and approval required for any public manuscript update, repository merge, submission or fee.

The original public PDF and default branch were not modified as part of these revisions.
