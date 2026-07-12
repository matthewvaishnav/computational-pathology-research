# Scanner-Invariant Residual Provenance Feasibility Audit

## Status and scope

This is a read-only feasibility specification. It does not run representation training, fit new probes, reconstruct oldstyle features, compute the proposed residual metrics, or modify existing experiment outputs.

The research question is:

> Can signal shared across scanner acquisitions remain non-biological even when scanner identity is suppressed?

The question matters because scanner invariance constrains only the scanner variable. It does not, by itself, distinguish tissue biology from upstream variables that are constant across all scans of the same physical slide.

## Research decision protocol

### 1. Strongest version of the idea

Let an observation be generated from:

\[
x_{r,s}=f(B_r, P_r, S_s, E_{r,s}),
\]

where:

- \(B_r\) is biological structure in region \(r\);
- \(P_r\) is scanner-constant provenance such as fixation, staining, sectioning, slide preparation, or annotation history;
- \(S_s\) is scanner acquisition condition;
- \(E_{r,s}\) is observation-specific extraction, registration, or crop variation.

Paired scanner acquisition changes \(S_s\) while holding both \(B_r\) and \(P_r\) fixed. A representation encouraged to agree across scanners can therefore retain \(B_r+P_r\). The strongest research direction is an invariance-blind-spot audit: determine which scanner-shared variables are conserved, and which experimental crossings are needed to identify their provenance.

The general observation that invariance preserves variables not changed by the intervention is not, by itself, a new result. The defensible computational-pathology novelty would be a literature-positioned paired-scanner provenance audit with crossed preparation data, explicit identifiability gates, and leakage-safe estimands. The current M1 feasibility path is exploratory infrastructure, not yet that research result.

### 2. Smallest defensible claim

If the later metric audit is positive, the smallest defensible claim is:

> Existing fixed, scanner-suppressed representations contain category-adjusted structure associated with sample, region, or measured provenance variables under cross-scanner and exact-region-exclusion controls.

That result would establish residual association, not non-biological origin. Non-biological attribution requires explicit provenance labels that vary across independent biological units.

### 3. Most dangerous reviewer objection

Sample and region identity contain fine-grained morphology that seven coarse tissue categories do not explain. Cross-scanner sample or region association is therefore expected from a useful biological representation. When preparation is constant within a slide and unlabelled, biological identity and preparation artifact are not identifiable.

A second objection is that reduced linear scanner recoverability is not scanner invariance. Nonlinear or scanner-pair-specific scanner signal may remain.

### 4. Stupid baseline that could beat it

A category-plus-geometry baseline using crop size, padding, image dimensions, annotation rank, and file-size proxies may reproduce the association. Filename, row-order, or exact-region leakage may also yield apparently strong identity recovery without a meaningful representation result.

### 5. Data required

The metric audit requires row-aligned fixed embeddings, scanner IDs, category labels, sample IDs, region IDs, held-out split labels, and an explicit representation-provenance manifest. Non-biological attribution additionally requires site, laboratory, preparation, block, section, fixation, staining-batch, reagent-lot, or other upstream provenance variables with enough crossing to separate them from sample and category.

### 6. Falsification experiment

The candidate residual hypothesis is not supported if association:

- is at the restricted-permutation null;
- disappears after exact category and target-scanner matching;
- appears only when another scanner view of the exact same region is allowed;
- is reproduced by category-only, geometry-only, path, filename, or row-order controls;
- appears only in one fold, seed, scanner direction, or rare category;
- disappears when evaluated on independent biological units;
- is explained by later fine-grained biological labels; or
- cannot be reproduced after representation lineage and row alignment are verified.

### 7. Research-line placement

This belongs in future work. It should not be inserted into the current Paired-Acquisition Neural Factorization manuscript as evidence, and it should not be presented as a completed Paired Scanner Counterfactual Benchmark result. It may later become a feature-space provenance module adjacent to that benchmark.

### 8. Smallest useful artifact first

The smallest useful artifact is this six-file feasibility package. It records what is measurable from existing fixed artifacts, what metadata are missing, and which controls must precede any new metric run.

## Operational definitions

### Scanner-suppressed representation

A fixed representation with audited scanner recoverability below the original frozen representation. This term does not imply chance-level or universal invariance.

### Strict scanner-invariance candidate

A representation that passes a pre-registered equivalence-style scanner gate across folds, seeds, scanner pairs, and suitable nonlinear sensitivity checks. Failure to reject a scanner probe is not sufficient.

The current neural biological branch is scanner-suppressed, not strictly invariant: its existing linear scanner-probe balanced accuracy is approximately 0.361, versus 0.20 five-class chance. The oldstyle `keep_k4` summary is at 0.20, but no row-level oldstyle embedding archive is materialized in the current workspace.

### Category-unexplained structure

Association measured only among examples with the same recorded category, or after a category adjustment learned without test-row leakage. This means unexplained by the available coarse labels, not non-biological.

### Candidate-discovery eligibility

A held-out sample-category cell is candidate-discovery eligible when it has at least two distinct regions and the same fold/category contains a different sample with at least one region for matched negatives. The reported cell, region, and observation counts cover anchor-capable cells only; negative-only gallery rows are not included.

### Replicated-anchor confirmatory eligibility

A fold/category is replicated-anchor confirmatory eligible only when at least two independent samples are anchor-capable, meaning each sample-category cell has at least two distinct regions. Candidate-discovery support must not be presented as confirmatory support, and scanner views, seeds, or archives are not independent anchor replication.

### Candidate non-biological correlate

Association with an explicit technical or provenance variable after excluding same-region and same-sample explanations where the design permits. Sample ID, slide ID, region ID, crop geometry, and annotation rank are not sufficient by themselves to establish non-biological origin.

## Feasibility gates

| Gate | Requirement | Failure consequence |
|---|---|---|
| G0: lineage | Dataset, backbone, representation, fold, seed, fit splits, vector key, code/commit, and checksum are explicit and internally consistent | `manual_review`; do not run confirmatory metrics |
| G1: alignment | Feature rows join one-to-one to metadata on dataset-appropriate composite keys | `blocked` |
| G2: scanner premise | Scanner suppression or invariance is evaluated out of fold and is not inferred from one weak probe | Interpret only as scanner-suppressed residual structure |
| G3: biological control | Category labels exist; discovery and replicated-anchor confirmatory eligibility are reported separately; confirmatory strata contain at least two anchor-capable samples | Category-unexplained criterion is blocked |
| G4: leakage control | Exact region is excluded; query and comparison scanners differ; fold spaces are not pooled; scanner bundles remain atomic | Identity result is invalid |
| G5: provenance attribution | Provenance labels repeat across independent biological units and are crossed with scanner/category/sample | Non-biological attribution is blocked |

## Existing capability boundary

| Dataset or candidate | Current use | Current boundary |
|---|---|---|
| Canine SCC neural biological branches | Category-conditioned, cross-scanner, different-region sample-structure feasibility | Scanner-suppressed rather than invariant; no site/preparation labels |
| Canine SCC oldstyle `keep_k4` | Strongest strict scanner-removal candidate from existing summary evidence | Row-level residual embeddings are absent; reconstruction is out of scope for this read-only phase |
| SCORPION neural biological branches | Cross-scanner slide/region controls and cross-backbone sensitivity | No category labels; criterion 2 cannot be tested |
| Geometry, registration, crop, and annotation fields | Technical-proxy sensitivity controls | May reflect tissue morphology or scanner; not provenance truth |

## Proposed later audit sequence

No step in this sequence is executed by the current feasibility script.

1. Resolve G0 lineage conflicts and freeze an artifact manifest.
2. Select fixed, already-generated representations; do not refit them.
3. Evaluate only each fold archive's `split == test` rows in that fold's coordinate system.
4. Apply the scanner premise gate using existing evidence plus pre-registered training-free sensitivity metrics.
5. Run the category-matched, cross-scanner, different-region sample-link metric.
6. Compare against raw features, broken-pair branches, metadata-only controls, and oldstyle only when row-level output is available.
7. Use restricted permutations that keep all scanner views of a region atomic.
8. If valid provenance labels are later added, test them across different samples or blocks; do not substitute sample or region identity for provenance.
9. Report each gate separately. Do not collapse results into a single artifact score.

## Decision language

| Evidence state | Allowed conclusion |
|---|---|
| Residual metric not above null | No measurable residual association under this audit; absence of artifact is not established |
| Sample/region metric above controls | Same-category cross-scanner sample association is detectable in the evaluated fixed representation |
| Geometry/QC proxy association | Residual structure is associated with a measured proxy; origin remains ambiguous |
| Site/preparation association across held-out biological units | Evidence is consistent with a measured non-scanner provenance correlate; use "operationally scanner-invariant" only if G2 also passes |
| Sample/region association without provenance metadata | Non-biological attribution is not supported |

## Hard boundaries

- No clinical, diagnostic, deployment, or patient-care claims.
- No claim that scanner bias is solved.
- No claim that scanner invariance implies biological specificity.
- No claim that residual signal is artifact without valid provenance metadata and an identifiable design.
- No causal source claim from association alone.
- No new training in this feasibility phase.
- No pixel-level reconstruction or manipulation.
