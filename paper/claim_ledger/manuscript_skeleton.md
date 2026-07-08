# Manuscript Skeleton — Paired-Acquisition Factorization

**Branch:** experiment/claim-ledger-and-paper-skeleton
**Generated:** 2026-07-08
**Purpose:** Paper skeleton mapping sections to claims, commits, figures, and wording constraints.

---

## 1. Abstract Skeleton

### Purpose
State the problem (scanner/acquisition variation confounds biological signal in computational pathology), the method (paired-acquisition factorization with biological and acquisition branches), the contribution (structured separation, not best raw scanner removal), and the key evidence boundary (oldstyle centroid/QR wins raw scanner erasure; paired-acquisition provides an explicit acquisition branch whose biological leakage can be bottlenecked).

### Required Figures/Tables
None (abstract is text-only).

### Exact Result Commits
All five claims summarized. No new results.

### Safe Wording
- "Paired-acquisition factorization learns a structured decomposition..."
- "The biological branch preserves tissue-category structure while reducing scanner recoverability..."
- "The acquisition branch captures scanner signal with biological leakage reducible by bottlenecking..."
- "Decoder-based acquisition swapping supports factor-like behavior..."
- "Oldstyle linear projection remains the strongest raw scanner-removal baseline..."

### Forbidden Wording
- "Best scanner removal method"
- "Solves scanner bias"
- "Perfect factorization"
- "Clinical validation"
- "Diagnostic performance"
- "Deployment-ready"
- Any claim that omits the oldstyle baseline boundary.

---

## 2. Introduction

### Purpose
Motivate the problem: multi-scanner computational pathology datasets contain scanner-specific variation that confounds biological analysis. Existing scanner-removal methods (centroid projection, PCA, adversarial alignment) remove scanner signal but do not produce an explicit, inspectable acquisition representation. Paired-acquisition factorization addresses this gap by learning a structured decomposition from paired same-region cross-scanner samples. State the thesis: the contribution is structured separation, not best raw scanner removal.

### Required Figures/Tables
- Conceptual diagram: paired-acquisition architecture (biological branch, acquisition branch, decoder).
- Optional: motivating example showing scanner variation in t-SNE/UMAP of frozen features.

### Exact Result Commits
None (motivation and background).

### Safe Wording
- "Scanner-specific variation confounds tissue representations..."
- "Existing approaches remove scanner signal but do not produce an explicit acquisition factor..."
- "We propose paired-acquisition factorization, which..."

### Forbidden Wording
- "Scanner bias renders computational pathology unusable"
- "Existing methods fail to address scanner variation"
- "We solve the scanner problem"
- Overstatement of clinical impact.

---

## 3. Problem Setup: Paired Acquisition as Supervision

### Purpose
Define the paired-acquisition supervision signal: same tissue region, imaged on two different scanners, produces a positive pair. Define the training objective: biological branch should preserve tissue identity, acquisition branch should capture scanner identity, branches should be independent. Define the datasets: SCORPION (multi-backbone frozen archives, no category labels) and canine SCC DINOv2 (expert-annotated tissue categories, 5 scanners). Define evaluation protocol: slide-level 5-fold CV, linear probes, neighborhood purity, cross-scanner retrieval, held-out-scanner transfer.

### Required Figures/Tables
- Dataset summary table: samples, regions, patches, scanners, categories.
- Pairing construction diagram.

### Exact Result Commits
- 30abcd39 (final paired-acquisition package audit) — architecture specification
- e4819c42 (pair-structure boundary test) — pairing ladder definitions

### Safe Wording
- "Paired acquisition provides a weak supervision signal..."
- "The same tissue region imaged on two scanners defines a positive pair..."
- "Slide-level cross-validation ensures no test-slide leakage..."

### Forbidden Wording
- "Perfect supervision signal"
- "Ground truth factorization"
- Overclaiming the quality of the pairing signal.

---

## 4. Method: Biological Branch, Acquisition Branch, Bottleneck Variants

### Purpose
Describe the ScorpionProjection architecture: frozen DINOv2 (or Phikon/ResNet50) encoder → biological projector (256D) → acquisition projector (variable D) → decoder (reconstructs original features). Losses: paired cosine reconstruction, scanner adversarial (gradient reversal on acquisition branch), variance regularization, cross-covariance independence. Bottleneck variants: acq_dim ∈ {8, 16, 64} with cross-covariance weights ∈ {0.05, 0.20}. Selected variants: acq_dim8_default, acq_dim16_stronger_xcov.

### Required Figures/Tables
- Architecture diagram with dimensions.

### Exact Result Commits
- 30abcd39 (final paired-acquisition package audit) — architecture
- a89bfb32 (acquisition bottleneck separation frontier sweep) — bottleneck variants

### Safe Wording
- "The acquisition branch dimensionality controls capacity..."
- "Cross-covariance regularization encourages branch independence..."
- "We sweep acquisition dimension and regularization strength..."

### Forbidden Wording
- "Optimal architecture"
- "Theoretically guaranteed independence"
- "Provably complete separation"

---

## 5. Main Result 1: Pair Structure Matters

### Purpose
Present the pairing-ladder experiment. True same-region pairs are measurably better than all broken-pair controls for tissue-identity preservation. Scanner suppression persists across all conditions. Biological correspondence is the active ingredient.

### Required Figures/Tables
- **Figure:** Pairing ladder table — paired cosine, top-1 retrieval, scanner probe for each level in both datasets.
- **Table:** Level-vs-level delta table (cosine gap, retrieval gap).

### Exact Result Commits
- e4819c42 (pair-structure boundary test)
- d018c924 (cross-backbone pair-structure boundary)

### Safe Wording
- "True same-region pairs produce the strongest tissue-identity preservation..."
- "Scanner suppression is maintained across the pairing ladder..."
- "Biological correspondence, not scanner balancing, drives tissue preservation..."

### Forbidden Wording
- "Paired acquisition requires exactly same-region pairs" (same-slide-different-region still preserves substantial identity)
- "Broken pairs destroy the method" (scanner suppression persists)
- "True pairs are necessary for scanner suppression" (false)

---

## 6. Main Result 2: Branch Separation and Biological Preservation

### Purpose
Present the branch separation evidence. Biological branch reduces scanner probe from 0.866 → 0.361 while preserving category probe (0.407 → 0.386) and improving neighborhood purity (0.954 → 0.973). Acquisition branch retains scanner (0.862) with reduced category (0.346). Cross-covariance low. Shuffled-sample controls degrade both branches. Scanner-heldout transfer preserved (0.845 → 0.827).

### Required Figures/Tables
- **Figure:** Scanner probe vs category probe scatter for all representations.
- **Figure:** Biological vs acquisition branch purity bar chart.
- **Table:** Central result table — scanner probe, category probe, category F1, purity K1/K5, heldout transfer accuracy for key representations.

### Exact Result Commits
- bec06eb4 (biological label preservation audit)
- 535eea18 (scanner-heldout label transfer audit)
- 0d7cdc92 (sample-disjoint scanner-heldout transfer audit)
- b5a9886e (scanner-confounded label robustness audit)
- 3450ede2 (oldstyle residual branch separation)

### Safe Wording
- "The biological branch substantially reduces scanner recoverability..."
- "Tissue-category structure is preserved, with neighborhood purity improving..."
- "The acquisition branch captures scanner signal as intended..."
- "Branch separation is measurable and consistent across folds..."

### Forbidden Wording
- "Perfect disentanglement"
- "Complete separation"
- "Scanner-free biological branch" (residual scanner probe 0.361)
- "Category-free acquisition branch" (residual category probe 0.346 in true_pair)

---

## 7. Main Result 3: Strongest Linear Baseline Boundary

### Purpose
Establish the oldstyle centroid/QR baseline as the strongest raw scanner-removal method. Concede that paired-acquisition does not beat it on raw scanner removal. Reframe the contribution as structured separation.

### Required Figures/Tables
- **Table:** Scoreboard summary — scanner probe, category probe, scanner capture for oldstyle_keep_k4, true_pair_biological, true_pair_acquisition, bottlenecked variants.
- Optional: linear baseline consistency comparison (oldstyle vs newstyle).

### Exact Result Commits
- 3450ede2 (oldstyle residual branch separation)
- a325c009 (linear baseline consistency audit)
- 1c527697 (unified separation scoreboard)

### Safe Wording
- "Oldstyle centroid/QR projection achieves stronger raw scanner removal (scanner probe 0.200 at chance) than paired-acquisition (0.361)..."
- "Paired-acquisition should not be claimed as the best scanner-removal method..."
- "The contribution is structured separation: an explicit acquisition branch that can be inspected, bottlenecked, and swapped..."
- "If raw scanner removal is the only goal, oldstyle centroid/QR projection is the stronger choice..."

### Forbidden Wording
- "Paired-acquisition removes scanner bias better than any linear method"
- "Paired-acquisition is the best scanner-removal approach"
- Any statement that omits the oldstyle baseline when discussing scanner-removal strength.
- Any framing of paired-acquisition as best-in-class at scanner erasure.

---

## 8. Main Result 4: Bottlenecked Frontier Improvement

### Purpose
Present the bottleneck frontier: reducing acquisition capacity substantially reduces biological leakage while preserving scanner capture and downstream transfer. Cross-backbone validation in SCORPION confirms generalization for tissue-retrieval leakage reduction.

### Required Figures/Tables
- **Figure:** Frontier plot — acquisition-branch category leakage vs scanner capture.
- **Figure:** Three-panel cross-backbone SCORPION figure — tissue-retrieval leakage for DINOv2, Phikon, ResNet50.
- **Table:** Bottleneck variant metrics — acq_dim8_default, acq_dim16_stronger_xcov vs true_pair.

### Exact Result Commits
- a89bfb32 (acquisition bottleneck separation frontier sweep)
- c29a038d (frontier-selected downstream validation)
- 0e2af247 (frontier-selected cross-backbone validation)

### Safe Wording
- "Bottlenecking the acquisition branch reduces category leakage..."
- "Scanner capture is preserved at bottleneck dimensions as low as 8..."
- "Biological downstream transfer is maintained..."
- "Cross-backbone validation confirms reduced tissue-retrieval leakage..."

### Forbidden Wording
- "Bottlenecking eliminates biological leakage" (residual 0.160 remains)
- "Dimension 8 is optimal" (sparse sweep)
- "SCORPION bottleneck reduces category leakage" (no category labels; tissue-retrieval leakage only)

---

## 9. Main Result 5: Acquisition Swapping / Factor-Like Behavior

### Purpose
Present the acquisition swapping evidence. When acquisition branches are swapped via the decoder, scanner identity follows the acquisition branch. Category structure stays with the biological branch. Bottlenecked variants show improved separation under swap. This supports factor-like behavior but does not prove perfect causal factorization.

### Required Figures/Tables
- **Figure:** Swap-type construction diagram (Types A, B, C, D).
- **Figure:** Bar chart — scanner follow rate and category preservation rate by variant and swap type.
- **Table:** Branch-space and decoder-space swap metrics.

### Exact Result Commits
- aa8d0596 (acquisition factor swapping audit)

### Safe Wording
- "Decoder-based acquisition swapping supports factor-like behavior..."
- "Scanner identity follows the acquisition branch through recombination..."
- "Category structure is preserved in the biological branch under swap..."
- "This is factor-like evidence, not proof of perfect causal factorization..."

### Forbidden Wording
- "Proves perfect causal acquisition factor"
- "Proves independence of biological and acquisition factors"
- "Enables acquisition factor editing for deployment"
- "Scanner information is fully encoded in the acquisition branch"
- "Works across all scanners and domains" (single dataset)

---

## 10. Limitations

### Purpose
Explicitly state all limitations in one consolidated section.

### Content
1. **Single labeled-category dataset.** All category-label claims (CLAIM_2, CLAIM_4 category leakage, CLAIM_5) rest on canine SCC DINOv2 only. SCORPION provides cross-backbone evidence but only for tissue/pair-retrieval metrics.
2. **Partial separation, not perfect.** Biological branch retains scanner signal (probe 0.361). Acquisition branch retains category signal (probe 0.346 in 64D). Separation is measurable and useful, not complete.
3. **Sparse bottleneck frontier.** Only 2 dimensions × 2 regularization strengths tested at full scale. Cannot claim continuous Pareto front or optimal bottleneck.
4. **Linear baselines only.** Strongest scanner-removal baseline is linear centroid/QR projection. Nonlinear baselines (adversarial projection, domain-adversarial feature learning) not compared.
5. **Feature-level, not image-level.** All experiments operate on frozen patch-level features. No end-to-end training, no pixel-level manipulation, no whole-slide-image context.
6. **Five scanners, two datasets.** Scanner diversity is limited. Generalization to unseen scanner types, staining protocols, or tissue types not tested.
7. **Decoder reuse.** Swapping decoder was trained for reconstruction, not factor manipulation. Recombination may introduce artifacts.
8. **NN scanner purity gap.** Nearest-neighbor scanner purity in acquisition space (0.880) is weaker than category purity in biological space (0.980). Acquisition branch carries scanner information but it does not dominate the neighborhood structure to the same degree.
9. **No clinical or diagnostic evidence.** This is a methodological research contribution. No patient-outcome, diagnostic-accuracy, or clinical-utility claims.

### Exact Result Commits
All commits; limitations derived from cross-claim analysis.

### Safe Wording
- As written above.

### Forbidden Wording
- Any minimization of limitations.
- "These limitations are minor" or similar.
- Omission of any limitation listed above.

---

## 11. Appendix Map

### Purpose
Organize appendix material: detailed tables, per-condition breakdowns, additional baselines, cross-backbone detail.

### Content

#### Appendix A: Pairing Ladder Detail
- Per-condition, per-dataset, per-metric tables
- Level-vs-level contrast tables
- Cross-backbone pairing ladder extension (d018c924)

#### Appendix B: Biological Label Preservation Detail
- Per-representation, per-class recall tables
- PCA removal k-sweep full results
- Linear scanner subspace k-sweep full results
- Neighborhood purity at k=1,5,10

#### Appendix C: Scanner-Heldout Transfer Detail
- Per-scanner, per-representation breakdown
- Per-class recall for all representations
- Sample-disjoint scanner-heldout transfer (0d7cdc92)

#### Appendix D: Scanner-Confounded Robustness Detail
- Per-scanner error patterns
- Split diagnostics
- Confounded vs clean scanner comparisons (b5a9886e)

#### Appendix E: Linear Baseline Detail
- Oldstyle vs newstyle consistency comparison (a325c009)
- Oldstyle residual k=1,2,3,4 full metrics (3450ede2)
- Logistic-SVD residual audit (ec2a509f)

#### Appendix F: Bottleneck Frontier Detail
- Smoke and full variant metrics (a89bfb32)
- Frontier-selected downstream per-scanner detail (c29a038d)
- Cross-backbone raw metrics — DINOv2, Phikon, ResNet50 (0e2af247)

#### Appendix G: Acquisition Swapping Detail
- Per-swap-type probe metrics
- Per-variant NN purity tables
- Decoder-space reconstruction metrics

#### Appendix H: Unified Scoreboard
- Full 12-row scoreboard with all metrics
- Source and commit mapping (1c527697)

#### Appendix I: Acquisition Branch Audit
- Branch separation contrasts (3e5bf19e)

### Exact Result Commits
All 15 commits (see `result_to_claim_map.csv`).

### Safe Wording
- Standard appendix labeling.

### Forbidden Wording
- No new claims in appendix that are not supported by main-text evidence.

---

## Global Wording Boundaries

See `wording_boundaries.md` for the complete forbidden-wording list.

### Always Allowed
- "Structured separation"
- "Measurable branch separation"
- "Partial separation"
- "Factor-like behavior"
- "Substantially reduces"
- "Preserves category structure"
- "Supports the interpretation"
- "Research audit"
- "Methodological contribution"

### Always Forbidden
- "Perfect" / "complete" (applied to separation, factorization, disentanglement)
- "Solves" (applied to scanner bias, domain shift)
- "Best" (applied to scanner removal without qualification)
- "Clinical validation" / "diagnostic performance" / "deployment-ready" / "patient-care"
- "Universal" (applied to biological factorization)
- "Proves" (applied to causal factorization)
- "Optimal" (applied to bottleneck dimension, architecture, or hyperparameters)
- Any claim that omits the oldstyle baseline when comparing scanner-removal strength
- "Category leakage eliminated" / "scanner-free" / "category-free" (applied to any branch)
