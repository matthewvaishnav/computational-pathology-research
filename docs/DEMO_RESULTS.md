# Legacy architecture smoke-test results

This page records early software smoke tests. It does **not** describe the data
provenance of the repository's scientific benchmarks.

The old version of this page was misleading because it generalized generated demo
inputs into a statement about the entire repository. That statement was false.

Current scientific evaluations use public datasets derived from real human
histopathology:

- PCam for pathology patch classification;
- PANDA for prostate whole-slide modeling;
- CAMELYON17/WILDS for multi-center lymph-node pathology and external-center
  validation.

Generated inputs used by `run_quick_demo.py`, missing-modality tests, temporal tests,
and similar utilities are test-only fixtures used to validate software behavior.
They are not the source of reported PCam, PANDA, or CAMELYON scientific results.

For current evidence, read:

- `README.md`
- `docs/DATA_PROVENANCE.md`
- `docs/research/camelyon17-external-center-validation-note.md`
- `docs/PCAM_REAL_RESULTS.md`
- `docs/results/panda-centralized-vs-federated.md`

The project remains research-only and is not a clinical diagnostic system.
