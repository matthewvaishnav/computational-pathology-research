# PA-NF shortcut/conflict post-freeze finalization note

**Date:** 2026-09-29  
**Campaign:** `pa-nf-shortcut-conflict-falsification/v1`

The registered runner completed all 25 fold-by-seed evaluation cells (five folds by five frozen projection seeds) and wrote the four registered outcome tables:

- `shortcut_runs.csv`
- `shortcut_slide_metrics.csv`
- `retrieval_query_metrics.csv`
- `retrieval_slide_metrics.csv`

After those tables were written, the process failed during the descriptive balanced-accuracy curve integration because the local NumPy build did not expose `np.trapz`.

This failure occurred after the registered primary contrast and bootstrap computations in the original process and before `primary_summary.json` was written. No model training, projection, pair construction, endpoint definition, comparator, bootstrap design, bootstrap seed, noninferiority margin, or success rule is changed.

Recovery uses `scripts/paired_acquisition/finalize_pa_nf_shortcut_conflict_outcomes.py`. The recovery script:

1. reads only the already-written outcome CSVs;
2. verifies the complete frozen row grids and rejects duplicates or missing registered identities;
3. recomputes the preregistered deterministic bootstrap summaries with the same seeds and rules;
4. computes the same trapezoidal descriptive curve integral using `np.trapezoid`, the current NumPy replacement for the unavailable `np.trapz` call;
5. writes `primary_summary.json` and a separate `postfreeze_finalization.json` provenance record containing hashes of the frozen outcome CSVs and final summary.

The outcome tables are not deleted, modified, regenerated, or selectively filtered. This is a mechanical compatibility recovery after outcome generation, not a methodological amendment.
