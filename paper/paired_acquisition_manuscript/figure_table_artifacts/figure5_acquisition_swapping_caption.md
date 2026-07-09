# Figure 5: Acquisition Swapping — Dual Evidence

**Intended manuscript location:** Section 8 (Result 5: Decoder-Based Acquisition Swapping)
**Claim:** CLAIM_5_FACTOR_LIKE_SWAPPING
**Type:** Multi-panel main-text figure (4 panels)

## Caption

**Figure 5. Decoder-based acquisition swapping supports factor-like behavior; dual evidence shown — strong decoder-space results and mixed branch-space nearest-neighbor results.** All measurements from canine SCC DINOv2 (5-fold × 5-seed × 150+ swaps per swap type; 5 scanners). **(a)** Swap-type construction diagram. Four swap types: Type A (same sample, different scanner — biology fixed, scanner changes; strongest test), Type B (same category, different sample), Type C (different category, different scanner), Type D (random acquisition source). Biological branch from sample i, acquisition branch from sample j, combined via learned decoder. **(b)** Decoder-space scanner follow rate and category preservation rate for Type A swaps (strongest test), by variant (true_pair, acq_dim8_default, acq_dim16_stronger_xcov). Decoder-space metrics are reported at variant level for Type A only; per-swap-type decoder breakdown not available for Types B/C/D (global decoder aggregate = 0.806). Same-sample swaps (Type A) achieve scanner follow 0.901 and category preservation 0.970–0.992 across variants. Strong evidence for factor-like recombination in the cleanest swap condition. **(c)** Branch-space nearest-neighbor purity (per-variant, invariant across swap types). Bio-space K1 category purity = 0.980 (near-perfect — biological neighbors preserve category). Acq-space K1 scanner purity = 0.880 (decent but clearly weaker — scanner neighbors in acquisition space are less pure than category neighbors in biological space). The asymmetry is key: biological purity near ceiling, acquisition purity decent but notably lower. **(d)** Target-scanner and source-category nearest-neighbor rates under swap (per-variant). Source-category NN rate near-perfect (0.996). Target-scanner NN rate collapses under bottlenecking (0.558 → 0.135) — bottlenecking reduces scanner NN structure while preserving category NN structure. Scanner information follows the acquisition branch through recombination (Panel B), but does not dominate the acquisition branch's nearest-neighbor structure to the same degree that category dominates the biological branch (Panels C–D). **Metric granularity:** Probe-based scanner follow and category preservation are per-swap-type (from acquisition_swapping_summary.csv). Branch-space purity, target-scanner NN rate, and source-category NN rate are per-variant (invariant across swap types within a variant — see aggregate_level column in CSV). Decoder-space metrics are per-variant for Type A only. Factor-like, not factor-proven. Single-dataset, single-backbone limitation (canine SCC DINOv2 only; no SCORPION or cross-backbone swapping). Source: acquisition factor swapping audit (commit aa8d0596).

## Data Source

- **Sole source:** aa8d0596 — acquisition factor swapping audit (5-fold × 5-seed × 4 swap types × 3 variants)

## Key Visual Signals

1. **Panel B (STRONG evidence):** Decoder-space scanner follow (0.901 Type A) and category preservation (0.978+ bottlenecked) — must be shown prominently
2. **Panel C (MIXED evidence):** Bio K1 purity (0.980) must be visually adjacent to acq K1 purity (0.880) — the asymmetry must be visible
3. **Panel D (WEAK evidence for scanner NN):** Target-scanner NN collapses (0.558 → 0.135); source-category NN near-perfect (0.996)
4. **Caption note:** "Panel B shows strong decoder-space factor-like behavior. Panels C–D show mixed branch-space evidence. Factor-like, not factor-proven."
5. **Limitation note visible:** "Single dataset, single backbone (canine SCC DINOv2). SCORPION swapping not run (lacks category labels). Decoder trained for reconstruction, not factor manipulation."

## Forbidden Language

- "Proves perfect causal acquisition factor" → use "factor-like evidence"
- "Proves factorization" → use "supports factor-like behavior"
- "Scanner always follows acquisition" → follow rate is 0.855 avg, not 1.0
- "Proves independence of biological and acquisition factors" → category preservation under swap is ~0.40
- "Scanner information is fully encoded in acquisition branch" → bio_scanner_leakage remains 0.032–0.213
- "Works across all scanners and domains" → single dataset, 5 scanners

## Weakness Visibility

This figure MUST visibly show:
- Strong decoder-space category preservation (0.978+)
- Strong source-category NN preservation (0.996+)
- Scanner following by probe (0.855 avg)
- Weak/mixed target-scanner NN alignment (0.880, collapsing to 0.135 under bottleneck)
- Shuffled not run
- Oldstyle not included (no acquisition branch to swap)
- Single-dataset/single-backbone limitation

## Appendix Trail

Appendix G: Per-swap-type probe metrics, per-variant NN purity tables, decoder-space reconstruction metrics.
