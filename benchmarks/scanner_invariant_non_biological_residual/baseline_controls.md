# Baselines and Controls

## Principle

The main failure mode is mistaking preserved tissue identity, duplicated paired regions, or metadata leakage for scanner-invariant artifact. Every later metric run must use the same eligible observations and the same leakage guards across candidates and controls.

## Representation baselines

| Baseline | Current availability | Purpose | Failure interpretation |
|---|---|---|---|
| Original frozen DINOv2 features | Row-level available | Shows how much structure predates scanner suppression | If candidate is not different, suppression did not isolate the effect |
| True-pair biological branch | 25 canine fold/seed archives | Primary scanner-suppressed candidate | Scanner probe remains above chance; do not call invariant |
| Bottleneck biological branches | 25 archives per full canine variant | Tests sensitivity to acquisition-branch capacity | A stable biological-branch effect is not provenance evidence |
| Shuffled-region pairing branch | 25 canine archives | Tests dependence on exact region correspondence | Similar effect weakens pair-specific interpretation |
| Shuffled-sample pairing branch | 25 canine archives | Tests dependence on within-sample pairing | Similar effect suggests generic representation structure |
| Same-category-different-sample pairing | 25 canine archives | Controls category while breaking sample identity | If it matches true pairs, sample pairing is unnecessary |
| Scanner-balanced-random and fully-random pairing | Materialized in pairing-ladder artifacts | Tests scanner balance and generic random pairing | Similar effect weakens the paired-acquisition mechanism claim |
| Oldstyle `keep_k4` | Summary only; no row-level archive | Strongest raw scanner-removal reference | Strict residual question remains blocked until materialized |
| PCA removal | Summary evidence; row-level status not primary | Weak scanner-removal control | It may lose category and still retain nuisance structure |
| Dimension/covariance-matched random features | To be generated only in a later metric phase | Controls dimension and anisotropy | Candidate must exceed representation-free geometry |

No baseline is trained or reconstructed in the present feasibility phase.

The materialized original canine DINOv2 archive stores fold-0 split labels. For comparisons in folds 1-4, test membership must come from the corresponding allowlisted fold manifest; reusing the archive's fold-0 split would be leakage. The frozen feature vectors themselves are shared and require no retraining.

## Metadata-only stupid baselines

These are deliberately simple and dangerous:

- category one-hot;
- scanner one-hot;
- category plus scanner;
- sample/category cell prevalence;
- crop side, bounding-box geometry, and region rank;
- padding and inside-image fractions;
- affine/registration summaries;
- source-image dimensions;
- file size and encoding metadata;
- path and filename tokens;
- row index or manifest order; and
- exact region ID.

Path, filename, row order, and exact region ID are leakage tripwires, not valid scientific baselines. If they produce signal in the supposedly leakage-safe implementation, the audit is invalid.

The strongest stupid scientific baseline is category plus low-level geometry/QC. If it matches or exceeds the embedding effect, the result may be a preprocessing or sampling artifact rather than a representation finding.

## Mandatory design controls

### C1. Exact-region exclusion

For sample-level residual metrics, remove every observation with the query `region_id` from the positive and negative gallery. Another scanner view of the same region is not an independent sample-level observation.

### C2. Directed cross-scanner matching

The positive and negative use the same target scanner, and the target scanner differs from the anchor scanner. Report every directed scanner pair before aggregation.

### C3. Exact category matching

Positive and negative regions share the recorded category with the anchor. Cartilage and any fold/category cell without multiple eligible samples are excluded from confirmatory aggregation.

### C4. Fold isolation

Use only `split == test` rows from each fold-specific archive. Do not pool coordinates from different fold models. Any normalization, residualization, PCA, centroid, or threshold is fit on non-test rows only.

### C5. Atomic scanner bundles

All scanner views of a region remain together in splits, resampling units, and permutations.

### C6. Biological-unit aggregation

Aggregate comparisons to anchor and sample/block level before uncertainty estimation. Seeds and millions of pairwise comparisons are not independent replicates.

### C7. Artifact-lineage verification

Require a frozen artifact manifest and checksums. Current internal metadata conflicts with some path-derived dataset/backbone labels; these archives are `manual_review` for confirmatory use.

### C8. Rare-stratum disclosure

Report excluded categories and scanner directions with reasons. Do not silently replace missing matched negatives with unmatched examples.

## Restricted permutation controls

### P1. Region-bundle sample permutation

Within fold and category, reassign whole region bundles to pseudo-sample slots while preserving per-sample category counts. This destroys sample membership while preserving scanner completeness and category composition.

### P2. Provenance permutation

When preparation/site labels exist, permute them at the independent sample or block level within estimable design strata. Never permute scanner rows independently.

### P3. Pair-condition negative controls

Apply identical metric code and permutation draws to true-pair and broken-pair archives. Differences must use the intersection of eligible row keys.

### P4. Label-integrity controls

Verify category is constant across all scanner views of a region and that sample/region keys do not cross folds. Any violation blocks the audit.

## Positive controls

- Same-region cross-scanner retrieval should be strong for representations that preserve paired tissue identity.
- Acquisition branches should retain more scanner organization than biological branches under existing evidence.
- Original frozen features should show stronger scanner recoverability than scanner-suppressed biological branches.

Positive-control success validates sensitivity only. It does not support non-biological attribution.

## Reviewer attack matrix

| Reviewer attack | Required defense |
|---|---|
| "You retrieved the identical tissue patch." | Exact-region exclusion and a separate same-region positive control |
| "Coarse category imbalance explains it." | Exact category matching and restricted within-category null |
| "Scanner direction explains it." | Directed cross-scanner matching with the same target scanner |
| "You leaked samples across folds." | Sample/block split audit and test-only evaluation |
| "Pairwise counts inflate certainty." | Anchor/sample aggregation and blocked resampling |
| "The branch is not invariant." | Scanner-suppressed wording plus explicit scanner gate |
| "Sample identity is biology." | No artifact claim without crossed preparation/site labels |
| "Geometry or filenames explain it." | Metadata-only baselines and leakage tripwires |
| "Artifact provenance is unclear." | Explicit manifest, checksum, vector key, fold, seed, and source commit |

## Stop conditions

Do not advance from feasibility to metric execution if:

- archive and manifest keys do not align exactly;
- lineage metadata remain contradictory for the primary candidate;
- eligible category/sample/scanner strata are insufficient;
- scanner suppression is not characterized;
- exact-region exclusion cannot be enforced;
- the restricted null cannot preserve the paired structure; or
- the intended claim requires site/preparation attribution but those labels remain absent.

Do not advance from candidate detection to non-biological attribution if sample, slide, block, site, or preparation factors are aliased.
