# Appendix Plan — Paired-Acquisition Manuscript

**Branch:** paper/paired-acquisition-manuscript-integration
**Generated:** 2026-07-08
**Purpose:** Complete appendix map. Every appendix section maps to a main-text claim and result section. No new claims introduced in appendix.

---

## Appendix A: Pair-Structure Boundary Detail

**Maps to:** Section 4 (Result 1), CLAIM_1
**Main-text display:** Figure 1 (pairing ladder table)

**Content:**
- Per-condition, per-dataset, per-metric full table (paired cosine, top-1 retrieval, scanner probe, effective rank for biological and acquisition branches)
- Level-vs-level contrast tables (Δ cosine, Δ retrieval, tissue damage verdict)
- SCORPION DINOv2: 5 conditions × 25 runs
- Canine SCC DINOv2: 6 conditions × 25 runs (includes same_category_different_sample_pairs)
- Cross-backbone pairing ladder extension (d018c924): SCORPION Phikon, SCORPION ResNet50
- Scanner suppression stability across pairing ladder
- Acquisition disentanglement degradation across pairing ladder

**Source commits:** e4819c42, d018c924
**Result files:** `boundary_raw_metrics.csv`, `boundary_summary.csv`, `boundary_condition_contrasts.csv`

---

## Appendix B: Biological Label Preservation Detail

**Maps to:** Section 5 (Result 2), CLAIM_2
**Main-text display:** Figure 2 (branch separation)

**Content:**
- Full 18-representation summary table (scanner probe, category probe, category F1, purity K1/K5/K10, effective rank)
- Per-representation per-class recall (7 tissue categories)
- PCA component removal k-sweep: k ∈ {1, 2, 4, 8, 16, 32} with scanner and category probe trends
- Linear scanner subspace projection k-sweep: k ∈ {0, 1, 2, 4, 8, 16, 32}
- Scanner/category tradeoff summary for all representations
- Neighborhood purity at k=1, 5, 10 for all representations
- Category/scanner ratio for all representations
- Rare class note: Cartilage (10 patches) and Bone (195 patches) flagged

**Source commits:** bec06eb4
**Result files:** `label_probe_raw_metrics.csv`, `label_probe_summary.csv`, `neighborhood_purity_metrics.csv`, `scanner_label_tradeoff_summary.csv`

---

## Appendix C: Scanner-Heldout Transfer Detail

**Maps to:** Section 5 (Result 2), CLAIM_2
**Main-text display:** Referenced in Figure 2, Figure 3

**Content:**
- Per-representation mean balanced accuracy and macro F1 across 5 held-out scanners
- Per-scanner breakdown for key representations (frozen, true_pair_bio, true_pair_acq, shuffled_bio, linear_projection_k4)
- Per-class recall for all 7 categories across key representations
- Worst-scanner analysis (p1000): the only scanner where true_pair_bio improves over frozen
- Biological vs acquisition gap per scanner
- Biological vs shuffled-sample gap

**Source commits:** 535eea18
**Result files:** `scanner_heldout_summary.csv`, `scanner_heldout_per_scanner.csv`, `scanner_heldout_per_class_recall.csv`

---

## Appendix D: Sample-Disjoint and Scanner-Confounded Audits

**Maps to:** Section 5 (Result 2), CLAIM_1, CLAIM_2
**Main-text display:** Referenced in text, not a standalone figure

**Content:**

### D.1 Sample-Disjoint Scanner-Heldout Transfer
- Sample-subset-disjoint heldout transfer: train on subset of samples, test on held-out scanner with disjoint samples
- Per-representation summary
- Split diagnostics
- Comparison to standard scanner-heldout transfer

### D.2 Scanner-Confounded Label Robustness
- Category probe accuracy when scanner and category are confounded in training
- Per-scanner error patterns
- Split diagnostics
- Confounded vs clean scanner comparisons

**Source commits:** 0d7cdc92, b5a9886e
**Result files:** `sample_disjoint_scanner_heldout_summary.csv`, `scanner_confounded_summary.csv`

---

## Appendix E: Linear Baseline Detail

**Maps to:** Section 6 (Result 3), CLAIM_3
**Main-text display:** Figure 3 (baseline scoreboard)

**Content:**

### E.1 Linear Baseline Consistency Audit
- Oldstyle (centroid/QR) vs newstyle (logistic-SVD) comparison
- Feature-space similarity diagnostics
- Explanation of why oldstyle achieves scanner probe 0.200 and newstyle achieves 0.707
- Implication: the 5 scanner centroids span a 4D affine subspace; QR removes it completely

### E.2 Oldstyle Residual k=1,2,3,4 Full Metrics
- oldstyle_keep_k and oldstyle_removed_k for k ∈ {1, 2, 3, 4}
- Scanner probe, category probe, purity, same-sample retrieval
- Branch contrast: paired vs oldstyle category contrast, scanner contrast
- Leakage comparison: paired acquisition leakage (0.346) vs oldstyle removed leakage (0.242 at k=4)

### E.3 Logistic-SVD Residual Audit
- Original logistic-regression-SVD linear split (ec2a509f)
- Now known to be weaker; included for completeness and audit trail

**Source commits:** a325c009, 3450ede2, ec2a509f
**Result files:** `linear_baseline_consistency_metrics.csv`, `oldstyle_residual_summary.csv`, `oldstyle_residual_branch_contrasts.csv`

---

## Appendix F: Bottleneck Comparison Detail

**Maps to:** Section 7 (Result 4), CLAIM_4
**Main-text display:** Figure 4 (bottleneck comparison + cross-backbone)

**Content:**

### F.1 Bottleneck Comparison: Smoke and Full Variants
- 6 smoke variants: acq_dim ∈ {8, 16, 32, 64} × cross-covariance ∈ {0.05, 0.20}
- 2 full-scale variants: acq_dim8_default, acq_dim16_stronger_xcov
- Full metrics: scanner probe, category probe, purity, cross-covariance, effective rank for biological and acquisition branches
- Branch contrast deltas vs true_pair baseline
- Variant selection log

### F.2 Bottleneck-Selected Downstream Validation
- Per-scanner heldout transfer for bottlenecked variants
- Per-class recall for bottlenecked variants
- Split diagnostics

### F.3 Cross-Backbone SCORPION Raw Metrics
- DINOv2, Phikon, ResNet50: full metrics for true_pair, acq_dim8, acq_dim16_xcov
- Tissue/pair-retrieval leakage in acquisition branch
- Scanner capture in acquisition branch
- Biological branch retrieval preservation
- Cross-covariance comparison

**Source commits:** a89bfb32, c29a038d, 0e2af247
**Result files:** `frontier_variant_summary.csv`, `frontier_full_raw_metrics.csv`, `frontier_downstream_summary.csv`, `frontier_crossbackbone_summary.csv`

---

## Appendix G: Acquisition Swapping Detail

**Maps to:** Section 8 (Result 5), CLAIM_5
**Main-text display:** Figure 5 (dual evidence swapping figure)

**Content:**
- Per-swap-type (A, B, C, D) scanner follow rate and category preservation rate
- Per-variant (true_pair, acq_dim8, acq_dim16_xcov) branch-space NN purity: bio-space K1/K5 category purity, acq-space K1/K5 scanner purity
- Per-variant decoder-space metrics: scanner follow, category preservation
- Per-variant target-scanner NN rate and source-category NN rate
- Bottleneck variant comparison: leakage reduction under swap
- Swap construction validation: all four swap types have examples

**Source commits:** aa8d0596
**Result files:** `acquisition_swapping_summary.csv`, `acquisition_swapping_nearest_neighbor_metrics.csv`, `acquisition_swapping_probe_metrics.csv`

---

## Appendix H: Unified Separation Scoreboard

**Maps to:** Sections 5-7 (Results 2-4), CLAIM_2, CLAIM_3, CLAIM_4
**Main-text display:** Referenced in Figure 3

**Content:**
- Full 12-row scoreboard with all available metrics
- 24 source entries with commit and branch mapping
- Key questions answered: best raw removal, strongest acquisition branch, bottleneck leakage reduction, biological transfer preservation
- Known limitations documented: SCORPION metrics are tissue/pair-retrieval, not category; cross-experiment protocol differences; missing values (NA)

**Source commits:** 1c527697
**Result files:** `unified_separation_scoreboard.csv`, `unified_separation_scoreboard_key_metrics.csv`, `unified_separation_scoreboard_sources.csv`

---

## Appendix I: Acquisition Branch Audit

**Maps to:** Section 5 (Result 2), CLAIM_2
**Main-text display:** Referenced in text

**Content:**
- Branch separation contrasts: scanner and category probe differences between biological and acquisition branches
- Branch audit raw metrics
- Branch audit summary
- Experiment design documentation

**Source commits:** 3e5bf19e
**Result files:** `branch_audit_summary.csv`, `branch_separation_contrasts.csv`

---

## Appendix Rules

1. **No new claims.** Appendix provides detail for claims made in main text. No claim should appear in appendix that is not introduced in main text.
2. **SCORPION scope.** All SCORPION metrics labeled as "tissue/pair-retrieval," not "category." The limitation is stated in every SCORPION appendix section header.
3. **No clinical/diagnostic/deployment language.**
4. **All tables include n (runs), error estimates, and fold/seed documentation.**
5. **Rare class warnings** (Cartilage n=10, Bone n=195) carried through all per-class tables.
6. **Oldstyle baseline** included in every appendix section where scanner removal is compared.
