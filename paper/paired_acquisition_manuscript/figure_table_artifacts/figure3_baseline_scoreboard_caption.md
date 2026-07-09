# Figure 3: Baseline Scoreboard + Downstream Transfer

**Intended manuscript location:** Section 6 (Result 3: Oldstyle Centroid/QR Is the Strongest Raw Scanner-Removal Baseline)
**Claim:** CLAIM_3_LINEAR_BASELINE_BOUNDARY
**Type:** Main-text table

## Caption

**Figure 3. Baseline scoreboard: oldstyle centroid/QR is the strongest raw scanner-removal baseline; paired-acquisition provides structured separation, not best erasure.** All metrics from canine SCC DINOv2. Scanner probe and category probe are balanced accuracy of linear classifiers (5-fold × 5-seed for neural representations, 5-fold for linear baselines). Scanner-heldout transfer: category probe trained on 4 scanners, tested on the held-out 5th scanner (mean across 5 held-out scanners). Sample-disjoint transfer: additional control — train/test samples originate from disjoint slide subsets. Scanner-confounded: category probe under varying scanner-label correlation strengths (3 strengths × 5 seeds × 5 folds = 375 runs). Oldstyle_keep_k4 (highlighted) achieves chance-level scanner probe (0.200) while preserving category accuracy (0.400) — strictly better raw scanner removal and category preservation than true_pair_biological (scanner 0.361, category 0.386). Paired-acquisition does not claim best raw scanner removal. The contribution is structured separation: an explicit acquisition branch that can be bottlenecked (acq_dim8: category leakage 0.160 vs true_pair 0.346, Δ = −0.186) and swapped (see Figure 5). Lower scanner probe = stronger scanner suppression. Lower category leakage = less biological information in the scanner branch. N/A = not applicable (no separate branch or metric not available). Error bars: ±1 standard deviation. Sources: unified separation scoreboard (commit 1c527697), scanner-heldout audit (535eea18), sample-disjoint audit (0d7cdc92), scanner-confounded audit (b5a9886e), frontier-selected downstream validation (c29a038d).

## Data Source

- **Scoreboard:** 1c527697 — unified separation scoreboard (12 representations)
- **Scanner-heldout transfer:** 535eea18 — scanner-heldout label transfer audit
- **Sample-disjoint transfer:** 0d7cdc92 — sample-disjoint scanner-heldout transfer audit (via c29a038d)
- **Scanner-confounded:** b5a9886e — scanner-confounded label robustness audit (via c29a038d)
- **Downstream validation:** c29a038d — frontier-selected downstream validation

## Key Visual Signals

1. oldstyle_keep_k4 row highlighted — best raw scanner removal (scanner probe 0.200)
2. true_pair_biological row highlighted — structured separation (scanner 0.361, category 0.386)
3. acq_dim8_default row highlighted — reduced leakage (0.160 vs true_pair 0.346) with preserved capture (0.864)
4. Annotation: "Lower scanner probe = stronger scanner suppression. Lower category leakage = less biological information in scanner branch."
5. Oldstyle removes scanner signal; paired-acquisition separates it into an explicit branch

## Forbidden Language

- "Paired-acquisition beats all baselines on scanner removal" → false; oldstyle wins
- "Best scanner erasure" → oldstyle wins
- Any omission of the oldstyle baseline when discussing scanner-removal strength

## Appendix Trail

Appendix E: Consistency audit, oldstyle k=1-4, logistic-SVD.
Appendix H: Full 12-row unified scoreboard.
