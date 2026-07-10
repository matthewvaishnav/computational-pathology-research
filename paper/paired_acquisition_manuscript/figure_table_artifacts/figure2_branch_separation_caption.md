# Figure 2: Branch Separation

**Intended manuscript location:** Section 5 (Result 2: Branch Separation)
**Claim:** CLAIM_2_BRANCH_SEPARATION
**Type:** Two-panel main-text figure

## Caption

**Figure 2. Paired-acquisition produces measurable branch separation: biological branch suppresses scanner signal while preserving category structure; acquisition branch captures scanner signal with reduced category structure.** All metrics from canine SCC DINOv2 (5-fold × 5-seed). **(a)** Scanner probe (balanced accuracy) vs category probe (balanced accuracy) for key representations. Biological branch (green) moves to the upper-left quadrant (lower scanner, preserved category). Acquisition branch (red) moves to the lower-right quadrant (high scanner, lower category). Oldstyle centroid/QR (keep_k4, grey) achieves the strongest raw scanner removal (scanner probe 0.200, chance level) but produces no acquisition branch. PCA removal at k=32 (purple) achieves pixel weakness on both axes. Arrow from frozen features to true_pair_biological shows the direction of improvement. **(b)** Neighborhood purity at K=1 and K=5 for biological branch (green bars) and acquisition branch (red bars). Biological branch category purity K1 = 0.973 vs acquisition branch K1 = 0.530. Scanner purity K1 for acquisition branch = 0.880 (measured separately). Separation is partial — biological branch retains residual scanner signal (0.361, above chance 0.20); acquisition branch retains residual category structure (0.346). Cross-covariance RMS between branches = 0.090. Error bars: ±1 standard deviation. Source: biological label preservation audit (commit bec06eb4) and oldstyle residual audit (commit 3450ede2).

## Data Source

- **Primary:** bec06eb4 — biological label preservation audit (frozen, true_pair, shuffled, PCA)
- **Oldstyle:** 3450ede2 — oldstyle residual branch separation audit (oldstyle_keep_k4, oldstyle_removed_k4)
- **Consolidated:** 1c527697 — unified separation scoreboard

## Key Visual Signals

1. Biological branch in upper-left quadrant (low scanner, high category)
2. Acquisition branch in lower-right (high scanner, low category)
3. Oldstyle_keep_k4 highlighted: best raw scanner removal (0.200, 0.400) — no acquisition branch (N/A)
4. True_pair_biological highlighted: structured separation (0.361, 0.386)
5. Arrow from frozen to true_pair_bio: direction of scanner suppression with category preservation
6. Panel B: bio K1 purity (0.973) dramatically higher than acq K1 purity (0.530)
7. Shuffled (broken-pair) controls show degradation on both axes

## Forbidden Language

- "Perfect disentanglement" → use "measurable, partial separation"
- "Scanner-free biological branch" → residual scanner probe is 0.361
- "Category-free acquisition branch" → residual category probe is 0.346

## Appendix Trail

Appendix B: Per-representation per-class recall, PCA k-sweep, linear subspace k-sweep, purity k=1,5,10.
Appendix C: Scanner-heldout per-scanner breakdown.
