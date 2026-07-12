# Candidate Metrics

## Execution status

These metrics are specified but not run. The current feasibility audit inspects only artifact presence, schemas, row alignment, support counts, and claim gates. No representation, probe, residualizer, or projection is trained.

## Notation

- \(z_{r,s}\): fixed embedding for region \(r\) acquired on scanner \(s\).
- \(m(r)\): sample or slide containing region \(r\).
- \(c(r)\): recorded tissue/category label.
- \(p(r)\): explicit preparation/site provenance label, when available.
- `sim`: cosine similarity after a transformation fixed without test-row leakage.

Fold-specific embedding spaces must be evaluated separately. Embeddings from independently fitted fold models must not be pooled into one distance matrix.

## Gate metrics

### G2.1 Existing out-of-fold scanner evidence

Reuse audited scanner-probe summaries only as a premise check. The original frozen canine representation has scanner balanced accuracy near 0.866; the true-pair biological branch is near 0.361; five-class chance is 0.20. The neural branch is therefore scanner-suppressed, not established invariant.

The oldstyle `keep_k4` summary is at 0.20, but a chance linear probe is not proof of invariance and row-level oldstyle features are not currently materialized.

### G2.2 Training-free scanner-neighborhood enrichment

For each test query, exclude all views of the same region and compare observed same-scanner neighbor enrichment with a category- and sample-blocked null. Report by scanner pair, fold, seed, and representation. This is a nonlinear sensitivity check, not proof that all scanner signal is absent.

### G2.3 Conditional scanner distance test

Within category and sample blocks, compare distances among different scanners with independently generated constrained scanner-label shuffles inside each complete region bundle. Each shuffle reassigns the scanner labels among that region's fixed views while the entire region remains the resampling unit. A global relabeling is invalid, and scanner rows are never moved between regions. This is a sensitivity test for detectable scanner organization, not an equivalence test proving invariance.

## Primary candidate-discovery metric

### M1. Matched cross-scanner sample-link AUC

For each eligible anchor \((r,a)\):

- positive: \((r',b)\), where `m(r') == m(r)`, `r' != r`, `c(r') == c(r)`, and `b != a`;
- negative: \((q,b)\), where `m(q) != m(r)`, `c(q) == c(r)`, and the target scanner is the same `b` used for the positive.

Define:

\[
\operatorname{AUC}_{sample}=
P\left[\operatorname{sim}(z_{r,a},z_{r',b})>
\operatorname{sim}(z_{r,a},z_{q,b})\right]
+\tfrac{1}{2}P[\text{tie}].
\]

The null is 0.5. Exact category and target-scanner matching prevents coarse category balance or scanner direction from creating the contrast. Requiring `r' != r` removes exact paired-region retrieval.

This metric detects sample-shared structure beyond the recorded category. It does not determine whether that structure is preparation artifact or unrecorded biology.

Use deterministic all-pairs concordance within every eligible anchor/target-scanner/category stratum: enumerate all eligible positives and all eligible matched negatives, average negative comparisons within each positive, then average positives within anchor. Give eligible anchors equal weight within sample and eligible samples equal weight within fold. Do not let samples with more regions or negatives dominate the estimand. Any sampled approximation must freeze its draws before representation comparison and reproduce this weighting target.

### Eligibility

- Use only `split == test` rows for the corresponding fold archive.
- A candidate-discovery anchor is eligible when its sample-category cell contains at least two distinct regions and the fold/category contains a different sample with at least one region for a matched negative.
- Replicated-anchor confirmatory aggregation additionally requires at least two samples in the fold/category whose sample-category cells each contain at least two distinct regions; only anchors from those replicated cells enter confirmatory aggregation.
- Cell, region, and observation support counts cover anchor-capable cells and exclude rows used only as negatives.
- Both scanners in the directed pair must be present.
- Exclude non-estimable category/fold/scanner-direction cells before observing results.

### Aggregation

1. Average matched comparisons per anchor.
2. Average anchors per sample/category/scanner direction.
3. Average seeds within fold; seeds are repeated fits, not independent datasets.
4. Report fold and sample-blocked uncertainty.
5. Do not treat pairwise or triplet counts as independent observations.

## Provenance metrics after metadata become available

### M2. Cross-sample provenance-link AUC

Replace sample identity with explicit provenance while requiring different declared biological units:

- anchor from sample \(i\);
- positive from different sample \(j\) with the same provenance label and category;
- negative from different sample \(k\) with a different provenance label and the same category;
- exact target-scanner matching;
- no shared region, slide, block, or specimen unless the estimand explicitly targets a higher level.

Different samples do not guarantee independent biology. Known specimen, subject, phenotype, anatomy, and other higher-level biological relationships must be excluded or matched according to the declared hierarchy.

This is the primary attribution metric. It is blocked until valid, crossed provenance labels exist.

### M3. Provenance partial distance association

Estimate a distance-based effect for site/preparation after category, scanner, and biological hierarchy terms. Permutations occur at the independent sample/block unit. Report the design matrix rank and aliasing; do not report a provenance effect when the factor is one-to-one with sample, scanner, fold, or category.

### M4. Held-out-unit provenance retrieval

Measure whether provenance association transfers to biological units not used to define any normalization, centroid, or nuisance adjustment. A positive result confined to samples seen during fitting is compatible with sample memorization.

## Secondary and control metrics

### M5. Different-region same-sample neighborhood enrichment

Restrict the gallery to the same category and a different scanner; exclude the query region. Compare same-sample neighbor rate with its eligible-gallery prevalence and a restricted bundle permutation.

### M6. Same-region cross-scanner link AUC

Use the same region on another scanner as the positive and a different region from the same sample/category as the negative. This is a positive control for paired identity preservation. It must not be used as evidence of non-biological artifact.

### M7. Technical-proxy association

Evaluate association with crop side, padding, affine/registration summaries, annotation rank, image dimensions, or file size after category/scanner matching. Report each proxy separately. A positive association means the representation tracks that proxy; it does not prove the proxy or representation signal is non-biological.

### M8. Pair-condition contrast

Run the identical fixed-embedding metric on true-pair, shuffled-region, shuffled-sample, same-category-different-sample, scanner-balanced-random, and fully-random pairing branches where materialized. A stronger effect under true or within-sample pairing would show that pair construction changes retained sample-shared structure. Its origin would remain unresolved.

### M9. Representation contrast

Compare, with identical eligible anchors and permutation draws:

- original frozen features;
- true-pair biological branch;
- bottleneck biological branches;
- broken-pair biological branches;
- oldstyle `keep_k4` only after row-level features and lineage are available; and
- dimension/covariance-matched random controls.

Use paired contrasts at anchor, sample, and fold level. Do not compare raw metric values computed on different eligible sets.

The current original canine DINOv2 archive contains fold-0 split labels only. For folds 1-4, assign the same fixed rows using the corresponding fold-specific manifest rather than the archive's embedded fold-0 split.

## Restricted nulls

### Sample null

Keep each region's five scanner views atomic. Within fold and category, reassign whole region bundles to pseudo-sample slots while preserving each sample's category counts. Never permute individual scanner rows.

### Provenance null

Permute site/preparation labels at the independent sample or block unit within estimable category/design strata. Recompute every learned normalization or nuisance adjustment inside each permutation.

### Leakage tripwire null

Run the same code with path, filename, row index, and exact region ID as deliberately invalid predictors. Unexpected performance from these controls blocks interpretation until the leak is removed.

## Multiplicity and robustness

- Pre-register one primary representation, one primary metric, and one primary dataset.
- Treat scanner directions, categories, folds, seeds, backbones, and proxies as stratified sensitivity analyses.
- Report complete effect distributions and restricted-null intervals, not only thresholded significance.
- Rare categories with insufficient independent samples are excluded from confirmatory aggregation and shown descriptively only.
- A result confined to one fold, scanner pair, seed, or backbone fails the robustness gate.

## Metrics deliberately rejected

- Same-sample top-1 retrieval that permits another scanner view of the exact same region.
- A scanner probe alone as proof of invariance.
- UMAP or cluster inspection as primary evidence.
- Row-level random train/test splits.
- Pooling fold-specific embedding spaces.
- Unrestricted row permutations.
- Sample or region classification presented as artifact attribution.
- A single composite "non-biological artifact score."

## Interpretation table

| Metric result | Interpretation ceiling |
|---|---|
| M1 above null | Category-adjusted sample-shared structure is detectable |
| M1 above null only when exact region is allowed | Paired-region identity, not sample-level residual structure |
| M7 above null | Association with a measured technical proxy |
| M2/M3/M4 positive under crossed, held-out design | Evidence consistent with a measured non-scanner provenance correlate; add "operationally scanner-invariant" only if G2 passes |
| All metrics at null | No detectable association under tested conditions; artifact absence is not established |
