# Historical biological-label preservation audit — superseded

**Evidence status:** invalid for current claim support  
**Current replacement:** `experiments/paired_acquisition/run_biological_label_preservation_fixed_estimand.py`

The numerical outputs in this directory are retained only as a historical
record. They must not be cited as validated evidence until the fixed-estimand
audit has completed and a new fail-closed result directory has been published.

## Reasons for supersession

1. The linear scanner-subspace baseline fit `StandardScaler` on all rows,
   including held-out test rows.
2. Linear probes were compared across representations without a common fit-only
   standardization pipeline, making the fixed regularization strength
   scale-dependent.
3. Category nearest-neighbour purity searched all rows and excluded only the
   exact query row. Alternate scanner views of the same biological region could
   therefore count as category-preserving neighbours.
4. Category support was not handled fail-closed, and the rare-class set could
   differ across sample-blocked test folds.
5. The strongest oldstyle centroid/QR baseline was not evaluated under the same
   corrected category estimand.
6. The reported uncertainty did not respect the complete fold and
   biological-sample structure.

## Consequence

The following historical values are not current claim evidence:

- category probe accuracy or macro F1 from this audit;
- category-neighbourhood purity values;
- category/scanner trade-off ratios;
- bottleneck category-leakage conclusions derived from these values;
- comparisons against the historical linear scanner-subspace baseline;
- conclusions derived from those quantities in the unified scoreboard.

This supersession does not alter the separately evaluated SCORPION
slide-blocked scanner-probe or same-region retrieval results. It specifically
withdraws this canine biological-label audit pending a clean fixed-estimand
rerun.

## Required replacement

```bash
python experiments/paired_acquisition/run_biological_label_preservation_fixed_estimand.py
```

The replacement runner:

- derives one sample-supported category set used in every fold;
- records retained and excluded categories;
- uses fit-only standardization for every probe;
- searches only the fit reference pool for category purity;
- excludes same-region and same-sample neighbours;
- includes oldstyle centroid/QR keep and removed branches;
- averages seeds within fold;
- reports five-fold means and ranges without patch-independent p-values;
- refuses to publish incomplete metrics.

The intermediate v2 runner remains as a regression and implementation record,
but the fixed-estimand runner is the current category-level protocol.

Raw files in this historical directory remain available for auditability and
must be labelled `superseded` in manuscripts, summaries, and presentations.
