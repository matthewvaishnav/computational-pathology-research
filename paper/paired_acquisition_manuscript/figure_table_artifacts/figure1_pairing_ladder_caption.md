# Figure 1: Pairing Ladder

**Intended manuscript location:** Section 4 (Result 1: Pair Structure Matters)
**Claim:** CLAIM_1_PAIR_STRUCTURE
**Type:** Main-text table with L0 rows visually highlighted

## Caption

**Figure 1. True same-region paired structure is required for tissue-identity preservation.** Paired cosine similarity and top-1 same-region retrieval rate for the biological branch, measured across a five-level pairing ladder (L0: true same-region pairs; L1: same-slide different-region pairs; L2: shuffled-sample pairs; L3: scanner-balanced random pairs; L4: fully random pairs). **(a)** SCORPION DINOv2, Phikon, and ResNet50 (tissue/pair-retrieval metrics). **(b)** Canine SCC DINOv2 (tissue/pair-retrieval plus scanner probe). True same-region pairs (L0, bold) produce the strongest tissue-identity preservation in all datasets and backbones. Tissue preservation degrades monotonically as the pairing is weakened (L0 → L4). Scanner-balanced random pairing (L3) performs similarly to fully random (L4) — scanner balancing alone does not recover tissue identity. Scanner suppression (bio scanner probe) is maintained across all pairing conditions. Error bars: ±1 standard deviation across 25 runs (5-fold × 5-seed). Source: pair-structure boundary test (commit e4819c42) and cross-backbone extension (commit d018c924).

## Data Source

- **Primary:** e4819c42 — pair-structure boundary test (canine SCC DINOv2, SCORPION DINOv2)
- **Cross-backbone:** d018c924 — pair-structure boundary cross-backbone (SCORPION Phikon, SCORPION ResNet50)

## Key Visual Signals

1. L0 row bolded in all datasets — true same-region pairs are the strongest condition
2. L3 and L4 rows visually adjacent — scanner balancing alone does not recover tissue identity
3. L0-to-L1 gap visible — same-region vs same-slide-different-region is a meaningful drop
4. Canine SCC L0-to-L1 gap (0.730 → 0.542, cosine Δ = 0.188) larger than SCORPION DINOv2 (0.880 → 0.809, Δ = 0.071)

## Appendix Trail

Appendix A: Per-condition detail, level-vs-level contrasts, cross-backbone extension tables.
