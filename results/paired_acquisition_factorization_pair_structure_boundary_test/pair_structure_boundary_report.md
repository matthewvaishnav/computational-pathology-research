# Pair-Structure Boundary Test Report

**Generated:** 2026-07-04 16:39:47
**Runtime:** 1805.6 s

## Scientific question

How exact does the paired-acquisition structure need to be for the method to preserve tissue identity while reducing scanner/acquisition signal?

## Evidence tiers

- **SCORPION_DINOv2**: full (5-fold x 5-seed)
- **canineSCC_DINOv2**: full (5-fold x 5-seed)

## Pairing ladder

| Level | Condition | Description |
|---|---|
| 0 | `true_same_region_pairs` | Same tissue region, different scanners (tested) |
| 1 | `same_slide_different_region_pairs` | Same slide, different tissue region (tested) |
| 2 | `shuffled_sample_pairs` | Different slides (existing falsification condition) (tested) |
| 2 | `same_category_different_sample_pairs` | Same tissue category, different sample (canine SCC only) (tested) |
| 3 | `scanner_balanced_random_pairs` | Random regions preserving scanner assignment (tested) |
| 4 | `fully_random_pairs` | All views randomly assigned, no structure (lower bound) (tested) |

## Datasets and conditions

### SCORPION_DINOv2 (full (5-fold x 5-seed))

- 125 runs across 5 conditions, 5 folds, 5 seeds
  - L0 `true_same_region_pairs` (existing, 25 runs)
  - L1 `same_slide_different_region_pairs` (existing, 25 runs)
  - L2 `shuffled_sample_pairs` (existing, 25 runs)
  - L3 `scanner_balanced_random_pairs` (new, 25 runs)
  - L4 `fully_random_pairs` (new, 25 runs)

### canineSCC_DINOv2 (full (5-fold x 5-seed))

- 150 runs across 6 conditions, 5 folds, 5 seeds
  - L0 `true_same_region_pairs` (existing, 25 runs)
  - L1 `same_slide_different_region_pairs` (existing, 25 runs)
  - L2 `same_category_different_sample_pairs` (new, 25 runs)
  - L2 `shuffled_sample_pairs` (existing, 25 runs)
  - L3 `scanner_balanced_random_pairs` (new, 25 runs)
  - L4 `fully_random_pairs` (new, 25 runs)

## Biological branch: tissue identity preservation

Scanner probe should be low (scanner suppressed); paired cosine and top-1 retrieval should be high (tissue identity preserved).

| Dataset | Level | Condition | Scanner probe | Paired cosine | Top-1 retrieval | Effective rank |
|---|---:|---:|---:|---:|---:|
| SCORPION_DINOv2 | 0 | `true_same_region_pairs` | 0.3998 | 0.8796 | 0.9999 | 54.5 |
| SCORPION_DINOv2 | 1 | `same_slide_different_region_pairs` | 0.3736 | 0.8089 | 0.9954 | 39.1 |
| SCORPION_DINOv2 | 2 | `shuffled_sample_pairs` | 0.3588 | 0.7668 | 0.9792 | 35.2 |
| SCORPION_DINOv2 | 3 | `scanner_balanced_random_pairs` | 0.3899 | 0.7247 | 0.9723 | 41.0 |
| SCORPION_DINOv2 | 4 | `fully_random_pairs` | 0.3877 | 0.7289 | 0.9757 | 41.3 |
| canineSCC_DINOv2 | 0 | `true_same_region_pairs` | 0.3614 | 0.7300 | 0.9334 | 74.0 |
| canineSCC_DINOv2 | 1 | `same_slide_different_region_pairs` | 0.3057 | 0.5422 | 0.7293 | 54.5 |
| canineSCC_DINOv2 | 2 | `shuffled_sample_pairs` | 0.4093 | 0.5849 | 0.7183 | 45.3 |
| canineSCC_DINOv2 | 2 | `same_category_different_sample_pairs` | 0.3517 | 0.5778 | 0.7394 | 49.1 |
| canineSCC_DINOv2 | 3 | `scanner_balanced_random_pairs` | 0.3551 | 0.5445 | 0.7341 | 54.8 |
| canineSCC_DINOv2 | 4 | `fully_random_pairs` | 0.3739 | 0.5462 | 0.7367 | 55.4 |

## Acquisition branch: scanner capture

Scanner probe should be high (scanner captured); paired cosine and top-1 retrieval should be low (tissue identity removed from acquisition branch). Cross-covariance should be low (branches decoupled).

| Dataset | Level | Condition | Scanner probe | Paired cosine | Top-1 retrieval | Cross-cov RMS |
|---|---:|---:|---:|---:|---:|
| SCORPION_DINOv2 | 0 | `true_same_region_pairs` | 0.8582 | 0.2795 | 0.0944 | 0.091709 |
| SCORPION_DINOv2 | 1 | `same_slide_different_region_pairs` | 0.8520 | 0.4328 | 0.3911 | 0.105443 |
| SCORPION_DINOv2 | 2 | `shuffled_sample_pairs` | 0.8558 | 0.4586 | 0.4180 | 0.102893 |
| SCORPION_DINOv2 | 3 | `scanner_balanced_random_pairs` | 0.8420 | 0.4962 | 0.5116 | 0.103242 |
| SCORPION_DINOv2 | 4 | `fully_random_pairs` | 0.8397 | 0.4931 | 0.5252 | 0.103718 |
| canineSCC_DINOv2 | 0 | `true_same_region_pairs` | 0.8651 | 0.4097 | 0.1806 | 0.089831 |
| canineSCC_DINOv2 | 1 | `same_slide_different_region_pairs` | 0.8647 | 0.5343 | 0.4376 | 0.087106 |
| canineSCC_DINOv2 | 2 | `shuffled_sample_pairs` | 0.8302 | 0.5350 | 0.4383 | 0.096097 |
| canineSCC_DINOv2 | 2 | `same_category_different_sample_pairs` | 0.8386 | 0.5445 | 0.4576 | 0.092799 |
| canineSCC_DINOv2 | 3 | `scanner_balanced_random_pairs` | 0.8397 | 0.5510 | 0.4886 | 0.089755 |
| canineSCC_DINOv2 | 4 | `fully_random_pairs` | 0.8319 | 0.5547 | 0.4997 | 0.089983 |

## Interpretation

### SCORPION_DINOv2 (full (5-fold x 5-seed))

**True same-region pairs (L0):** paired cosine = 0.8796, top-1 retrieval = 0.9999, scanner probe = 0.3998

| Level | Condition | Delta cosine vs L0 | Delta retrieval vs L0 | Tissue damage? |
|---|---:|---:|:---:|
| 1 | `same_slide_different_region_pairs` | -0.0706 | -0.0046 | Yes |
| 2 | `shuffled_sample_pairs` | -0.1128 | -0.0207 | Yes |
| 3 | `scanner_balanced_random_pairs` | -0.1549 | -0.0276 | Yes |
| 4 | `fully_random_pairs` | -0.1507 | -0.0242 | Yes |

**Verdict:** SCORPION shows a graded pair-structure boundary. True same-region pairs are strongest (cosine = 0.8796, retrieval = 0.9999). Same-slide-different-region pairs preserve substantial tissue identity (cosine = 0.8089) but are measurably weaker. Shuffled-sample and random pairs degrade further (cosine = 0.7247–0.7668). Paired cosine gap from L0 to best looser condition = 0.0706, retrieval gap = 0.0046. Biological correspondence in the pairing is the active ingredient.

**Scanner suppression:** Biological branch scanner probe is 0.3998 (L0) vs 0.3775 (higher levels, delta = -0.0223).
Scanner suppression is maintained across all pairing conditions — the scanner adversary works regardless of pair quality.

**Acquisition disentanglement:** Acquisition branch paired cosine is 0.2795 (L0) vs 0.4702 (higher levels, delta = +0.1907).
Looser pairing causes the acquisition branch to encode more tissue-level information, reducing disentanglement quality.

### canineSCC_DINOv2 (full (5-fold x 5-seed))

**True same-region pairs (L0):** paired cosine = 0.7300, top-1 retrieval = 0.9334, scanner probe = 0.3614

| Level | Condition | Delta cosine vs L0 | Delta retrieval vs L0 | Tissue damage? |
|---|---:|---:|:---:|
| 1 | `same_slide_different_region_pairs` | -0.1878 | -0.2041 | Yes |
| 2 | `shuffled_sample_pairs` | -0.1451 | -0.2151 | Yes |
| 2 | `same_category_different_sample_pairs` | -0.1522 | -0.1940 | Yes |
| 3 | `scanner_balanced_random_pairs` | -0.1854 | -0.1993 | Yes |
| 4 | `fully_random_pairs` | -0.1838 | -0.1966 | Yes |

**Verdict:** Canine SCC shows true-pair dominance with noisier ordering among looser non-true pairing conditions. True same-region pairs are clearly strongest (cosine = 0.7300, retrieval = 0.9334). All non-true conditions cluster in a lower band (cosine = 0.5422–0.5849) with overlapping confidence intervals, showing no reliable differentiation among same-slide, shuffled-sample, category-matched, or random pairings. Paired cosine gap from L0 = 0.1451, retrieval gap = 0.1940 — larger than the SCORPION gap. Biological correspondence in the pairing is the active ingredient; scanner suppression alone does not preserve tissue identity.

**Scanner suppression:** Biological branch scanner probe is 0.3614 (L0) vs 0.3591 (higher levels, delta = -0.0023).
Scanner suppression is maintained across all pairing conditions — the scanner adversary works regardless of pair quality.

**Acquisition disentanglement:** Acquisition branch paired cosine is 0.4097 (L0) vs 0.5439 (higher levels, delta = +0.1342).
Looser pairing causes the acquisition branch to encode more tissue-level information, reducing disentanglement quality.

## Cross-dataset findings

- **True same-region pairs are strongest in both datasets.** In SCORPION, the L0 biological paired cosine is 0.8796; in canine SCC, 0.7300. Both substantially exceed all looser pairing conditions.
- **Scanner suppression persists across weaker pairings.** The biological branch scanner probe remains low (0.31–0.41) regardless of pairing strictness in both datasets.
- **Weaker and random pairings do not preserve tissue identity as well.** In both datasets, every non-true condition shows measurable degradation in paired cosine and top-1 retrieval relative to L0.
- **Biological correspondence is the active ingredient.** Random pairing and scanner-balanced random pairing perform similarly — scanner balancing alone does not recover tissue identity. The biological structure in the positive pairs (same tissue region, same slide) is what drives tissue-identity preservation in the biological branch.
- **This is not clinical validation, diagnostic evidence, or deployment readiness.** This experiment tests only whether the paired-acquisition factorization effect depends on pair-structure strictness.

## Claim boundaries

- Existing conditions (true_pairs, shuffled_region_pairs, shuffled_sample_pairs) reuse trained models from the pair-integrity falsification experiment. No retraining.
- New conditions (scanner_balanced_random_pairs, fully_random_pairs, same_category_different_sample_pairs) were trained on the same base features with modified pair constructions.
- All metrics computed on held-out test slides only.
- The fully_random_pairs condition is intentionally degraded; it serves as a lower bound, not a recommended training configuration.
- This experiment does not claim clinical validation, diagnostic performance, disease biology discovery, or deployment readiness.

## Output files

| File | Description |
|---|---|
| boundary_raw_metrics.csv | Per-run, per-condition metrics |
| boundary_summary.csv | Aggregated by dataset and condition |
| boundary_condition_contrasts.csv | Level-vs-level contrasts |
| experiment_design.json | Experiment configuration |
| run_log.txt | Timestamped run log |
| pair_structure_boundary_report.md | This report |
