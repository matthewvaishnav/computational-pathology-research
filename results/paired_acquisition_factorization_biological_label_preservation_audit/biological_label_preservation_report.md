# Historical biological-label preservation audit — superseded

**Evidence status:** invalid for current claim support  
**Superseded by:** `experiments/paired_acquisition/run_biological_label_preservation_audit_v2.py`

The numerical outputs in this directory are retained only as a historical
record. They must not be cited as validated evidence until the v2 audit has been
run and a new fail-closed result directory has been published.

## Reasons for supersession

1. The linear scanner-subspace baseline fit `StandardScaler` on all rows,
   including held-out test rows.
2. Linear probes were compared across representations without a common
   fit-only standardization pipeline, making the fixed regularization strength
   scale-dependent.
3. Category nearest-neighbour purity searched all rows and excluded only the
   exact query row. Alternate scanner views of the same biological region could
   therefore be counted as category-preserving neighbours.
4. Missing category support was not handled fail-closed.
5. Uncertainty did not resample the biological-sample unit.

## Consequence

The following historical values are not current claim evidence:

- category probe accuracy or macro F1 from this audit;
- category-neighbourhood purity values;
- category/scanner trade-off ratios;
- comparisons against the historical linear scanner-subspace baseline;
- conclusions derived from those quantities in the unified scoreboard.

This supersession does not alter the separately evaluated SCORPION slide-blocked
scanner-probe, paired-cosine, or same-region retrieval results. It specifically
withdraws this canine biological-label audit pending a clean v2 rerun.

## Required replacement

Run:

```bash
python experiments/paired_acquisition/run_biological_label_preservation_audit_v2.py
```

The v2 runner:

- uses fit-only standardization for all probes;
- fits the linear-removal scaler on fit rows only;
- searches only the fit reference pool for category purity;
- excludes same-region and same-sample candidates;
- validates category support for every fold;
- reports biological-sample cluster-bootstrap intervals;
- refuses to publish incomplete metrics.

Raw files in this historical directory remain available for auditability but
must be labelled `superseded` in manuscripts, summaries, and presentations.
