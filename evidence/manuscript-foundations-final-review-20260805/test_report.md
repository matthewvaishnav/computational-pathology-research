# Final Manuscript Review — Test Report

**Date:** 2026-08-05
**Branch:** `research/foundations-manuscript-final-review-20260805`

## Full battery

**330 passed, 53 skipped, 0 failed** across:

- hardened manuscript validation tests (29 passed, 1 skipped — the skip is the
  symlink test on a platform without symlink privileges; the validator itself
  checks symlinks directly);
- corrected paired-acquisition evidence tests (10 passed);
- recovery tests (20 passed);
- fixed-estimand adjudication v1 and v2 (49 + 26 passed);
- PCam infrastructure tests (26 passed);
- CAMELYON17 proxy-validation tests (63 passed, 5 skipped);
- TransnnMIL architecture tests that require no new training (88 passed,
  34 skipped);
- PathologyFL integration tests (24 passed, 13 skipped);
- FAIR-WEIGHTS-H tests (8 passed).

## Optional-dependency skips

All 53 skips are explicit optional-dependency gates for components that are not
claimed as validated in this environment:

- TenSEAL (secure aggregation): 12 tests skip;
- Opacus (production DP-SGD): 1 test skips;
- torch_geometric (TransnnMIL topology/v2): 33 tests skip;
- FAISS (k-NN builder): within the topology suite;
- symlink test: platform limitation.

No failure is hidden by xfail or broad exception handling. No collection error
is skipped.

## Manuscript validator

`validation/validate_manuscript.py` reports **248 checks, 0 failed, status
valid**, covering: every active empirical claim's path/hash/size/commit/
dataset/statistical-unit; architectural source/test bindings; protocol
implemented-vs-specification-only components; placeholder hashes; duplicate
IDs; research-line presence; frozen statuses; status consistency; final review
artifacts; PDF hashes; and boundary wording.
