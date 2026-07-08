# Figures and Tables Plan — Paired-Acquisition Manuscript

**Branch:** paper/paired-acquisition-manuscript-integration
**Generated:** 2026-07-08
**Purpose:** Main-text figure and table plan. One primary display per claim. Appendix carries detail.

---

## Figure/Table Inventory

### Main-Text Figures (5)

| Figure | Claim | Type | Content |
|---|---|---|---|
| Figure 1 | CLAIM_1 | Table | Pairing ladder |
| Figure 2 | CLAIM_2 | Multi-panel figure | Branch separation |
| Figure 3 | CLAIM_3 | Table | Baseline scoreboard |
| Figure 4 | CLAIM_4 | Multi-panel figure | Bottleneck comparison + cross-backbone |
| Figure 5 | CLAIM_5 | Multi-panel figure | Acquisition swapping (dual evidence) |

### Main-Text Tables (1)

| Table | Claims | Content |
|---|---|---|
| Table 1 | Problem setup | Dataset summary: samples, regions, patches, scanners, categories |

---

## Figure 1: Pairing Ladder Table

**Claim:** CLAIM_1_PAIR_STRUCTURE
**Type:** Main-text table
**Purpose:** Show that true same-region pairs are required for tissue-identity preservation. Scanner balancing alone does not recover it.

**Content:**

```
Dataset          Level  Condition                        Paired cosine  Top-1 retrieval  Bio scanner probe
SCORPION DINOv2  L0     true_same_region_pairs           0.880          1.000            0.400
SCORPION DINOv2  L1     same_slide_different_region      0.809          0.995            0.374
SCORPION DINOv2  L2     shuffled_sample_pairs            0.767          0.979            0.359
SCORPION DINOv2  L3     scanner_balanced_random          0.725          0.972            0.390
SCORPION DINOv2  L4     fully_random                     0.729          0.976            0.388
canine SCC       L0     true_same_region_pairs           0.730          0.933            0.361
canine SCC       L1     same_slide_different_region      0.542          0.729            0.306
canine SCC       L2     shuffled_sample_pairs            0.585          0.718            0.409
canine SCC       L3     scanner_balanced_random          0.545          0.734            0.355
canine SCC       L4     fully_random                     0.546          0.737            0.374
```

**Key visual signal:** L0 row bolded. L3 and L4 close together (scanner balancing doesn't help). L0-to-L1 gap visible.

**Appendix trail:** Appendix A (per-condition detail, level-vs-level contrasts, cross-backbone extension).

---

## Figure 2: Branch Separation

**Claim:** CLAIM_2_BRANCH_SEPARATION
**Type:** Two-panel main-text figure
**Purpose:** Show that branch separation is measurable: biological branch suppresses scanner while preserving category; acquisition branch captures scanner with reduced category.

**Panel A: Scanner probe vs category probe scatter.**
- x-axis: scanner probe (balanced accuracy)
- y-axis: category probe (balanced accuracy)
- Points: original_frozen, true_pair_biological, true_pair_acquisition, shuffled_sample_biological, shuffled_sample_acquisition, pca_removal_k32, linear_projection_k4, oldstyle_keep_k4
- Annotate: "better scanner suppression →" (left), "better category preservation →" (up)
- Oldstyle_keep_k4 highlighted: best raw scanner removal (0.200, 0.400)
- True_pair_biological highlighted: structured separation (0.361, 0.386)
- Arrow from frozen to true_pair_bio showing direction of improvement

**Panel B: Branch purity bar chart.**
- Grouped bars for biological branch and acquisition branch
- Metrics: category purity K1, category purity K5, scanner purity K1, scanner purity K5
- Show: bio K1 = 0.973, acq K1 = 0.530

**Key visual signal:** Biological branch in upper-left quadrant (low scanner, high category). Acquisition branch in lower-right (high scanner, low category). Oldstyle best on scanner but no acquisition branch shown (N/A).

**Appendix trail:** Appendix B (per-representation per-class recall, PCA k-sweep, linear subspace k-sweep, purity k=1,5,10). Appendix C (scanner-heldout per-scanner breakdown).

---

## Figure 3: Baseline Scoreboard Table

**Claim:** CLAIM_3_LINEAR_BASELINE_BOUNDARY
**Type:** Main-text table
**Purpose:** Establish that oldstyle centroid/QR is the strongest raw scanner-removal baseline. Paired-acquisition provides structured separation, not best erasure.

**Content:**

```
Representation                Scanner probe  Category probe  Acq scanner capture  Acq category leakage  Bio scanner leakage  Heldout transfer
original_frozen_features      0.866          0.407           N/A                  N/A                   0.866                 0.845
oldstyle_keep_k4              0.200          0.400           N/A                  N/A                   0.200                 0.835
oldstyle_removed_k4           0.538          0.242           0.538                0.242                 N/A                   N/A
true_pair_biological          0.361          0.386           N/A                  N/A                   0.361                 0.827
true_pair_acquisition         0.865          0.346           0.865                0.346                 N/A                   0.517
acq_dim8_default_biological   0.369          0.385           N/A                  N/A                   0.369                 0.822
acq_dim8_default_acquisition  0.864          0.160           0.864                0.160                 N/A                   0.175
```

**Key visual signal:** oldstyle_keep_k4 row highlighted — best raw scanner removal (0.200). true_pair_acquisition row highlighted — strongest scanner capture with explicit branch (0.865). acq_dim8 row highlighted — reduced leakage (0.160 vs 0.346) with preserved capture (0.864).

**Annotation below table:** "Lower scanner probe = stronger scanner suppression. Lower category leakage = less biological information in scanner branch. N/A = not applicable (no separate branch). Oldstyle removes scanner signal; paired-acquisition separates it into an explicit branch."

**Appendix trail:** Appendix E (consistency audit, oldstyle k=1-4, logistic-SVD). Appendix H (full 12-row scoreboard).

---

## Figure 4: Bottleneck Comparison + Cross-Backbone

**Claim:** CLAIM_4_BOTTLENECK_FRONTIER
**Type:** Multi-panel main-text figure
**Purpose:** Show that bottlenecking reduces acquisition-branch biological leakage while preserving scanner capture. Cross-backbone generalization.

**Panel A: Canine SCC bottleneck comparison.**
- x-axis: acquisition category leakage (lower = better)
- y-axis: acquisition scanner capture (higher = better)
- Points: true_pair (64D), acq_dim16_stronger_xcov (16D), acq_dim8_default (8D)
- Arrow from true_pair to acq_dim8 showing directional improvement: left (less leakage), same height (preserved capture)
- **Label:** "Directional separation-frontier improvement" (not "Pareto front")
- **Note in caption:** "Sparse comparison (4 full-scale variants); supports directional improvement, not dense frontier mapping."

**Panel B-D: SCORPION cross-backbone tissue-retrieval leakage.**
- Three grouped bar charts (one per backbone: DINOv2, Phikon, ResNet50)
- Each chart: two bars — true_pair acquisition leakage, acq_dim8 acquisition leakage
- y-axis: tissue/pair-retrieval leakage in acquisition branch (lower = better)
- **Label:** "Tissue/pair-retrieval leakage" (not "category leakage" — SCORPION lacks labels)

**Key visual signal:** Bottlenecked variants move left (less leakage) without moving down (same capture). Cross-backbone consistency — all three backbones show reduced leakage. SCORPION labeled as "tissue/pair-retrieval," not "category."

**Appendix trail:** Appendix F (smoke + full variant metrics, per-scanner downstream, cross-backbone raw metrics).

---

## Figure 5: Acquisition Swapping — Dual Evidence

**Claim:** CLAIM_5_FACTOR_LIKE_SWAPPING
**Type:** Multi-panel main-text figure
**Purpose:** Show factor-like behavior AND the weaker NN scanner evidence side by side. Do not hide the mixed result.

**REQUIRED: Both strong and weak evidence must appear in this figure.**

**Panel A: Swap-type construction diagram.**
- Schematic of Types A, B, C, D
- Visual: bio branch from sample i, acq branch from sample j, combined via decoder

**Panel B: Decoder-space scanner follow and category preservation (STRONG evidence).**
- Grouped bar chart by variant (true_pair, acq_dim8, acq_dim16_xcov)
- Two grouped metrics: scanner follow rate, category preservation rate
- Show: scanner follow 0.901 (Type A), category preservation 0.978+ (bottlenecked)
- **Label:** "Decoder-reconstructed space"

**Panel C: Branch-space nearest-neighbor purity (MIXED evidence).**
- Grouped bar chart by variant
- Four bars per variant: bio-space K1 category purity, bio-space K5 category purity, acq-space K1 scanner purity, acq-space K5 scanner purity
- **Critical:** bio-space K1 category purity (0.980) must be visually adjacent to acq-space K1 scanner purity (0.880)
- The asymmetry must be visible: biological purity near ceiling, acquisition purity decent but notably lower
- **Label:** "Branch-space nearest-neighbor purity"

**Panel D: Target-scanner and source-category NN rates (WEAK evidence for scanner).**
- Grouped bar chart by variant
- Two metrics: target-scanner NN rate, source-category NN rate
- Show: target-scanner NN collapses under bottleneck (0.558 → 0.135), source-category NN near-perfect (0.996+)
- **Label:** "Nearest-neighbor alignment under swap"

**Figure caption note:** "Panel B shows strong decoder-space factor-like behavior. Panels C-D show mixed branch-space evidence: biological category purity (0.980) exceeds acquisition scanner purity (0.880), and bottlenecking reduces scanner NN alignment. Factor-like, not factor-proven."

**Appendix trail:** Appendix G (per-swap-type probe metrics, per-variant NN purity tables, decoder-space reconstruction metrics).

---

## Table 1: Dataset Summary

**Type:** Main-text table (in Problem Setup, Section 2)
**Purpose:** Characterize the two datasets used.

**Content:**

```
Dataset          Backbone(s)              Samples  Regions  Patches  Scanners  Categories  Labels?
Canine SCC       DINOv2-Base              44       805      4,025    5         7           Yes (expert tissue categories)
SCORPION         DINOv2/Phikon/ResNet50   —        —        —        5         —           No (region/scanner metadata only)
```

---

## Figure/Table to Claim Mapping

| Display | Section | Claim | Appendix |
|---|---|---|---|
| Table 1 | 2. Problem Setup | — | — |
| Figure 1 | 4. Result 1 | CLAIM_1 | Appendix A |
| Figure 2 | 5. Result 2 | CLAIM_2 | Appendix B, C, D |
| Figure 3 | 6. Result 3 | CLAIM_3 | Appendix E, H |
| Figure 4 | 7. Result 4 | CLAIM_4 | Appendix F |
| Figure 5 | 8. Result 5 | CLAIM_5 | Appendix G |

---

## Display Rules

1. **No unqualified "frontier sweep" or "Pareto front"** in any figure caption or label. Use "bottleneck comparison," "capacity-constrained separation," "directional separation-frontier improvement."
2. **SCORPION metrics must be labeled as "tissue/pair-retrieval," not "category."**
3. **Figure 5 must show BOTH strong (Panel B) and weak/mixed (Panels C-D) evidence.** Do not split across main text and appendix.
4. **Figure 3 must highlight oldstyle_keep_k4 as best raw scanner removal.** Do not bury or minimize this result.
5. **All figures must include error bars or confidence intervals** (5-fold × 5-seed standard deviations, or per-fold ranges).
6. **No clinical, diagnostic, or deployment language** in any figure caption.
