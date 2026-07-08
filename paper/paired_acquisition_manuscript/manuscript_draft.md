# Paired-Acquisition Factorization for Structured Separation of Scanner and Biological Signal in Multi-Scanner Computational Pathology

**Branch:** paper/paired-acquisition-manuscript-integration
**Generated:** 2026-07-08
**Status:** First draft — not for submission. Converts claim ledger into manuscript form.
**Source:** `paper/claim_ledger/` (PR #18, merged to main @ 57707ad1)

---

## Abstract

**Background.** Multi-scanner computational pathology datasets contain scanner-specific variation that confounds biological signal. Existing scanner-removal methods suppress scanner information but do not produce an explicit, inspectable acquisition representation.

**Method.** We propose paired-acquisition factorization: a neural decomposition trained on same-region cross-scanner image pairs that learns a biological branch (preserving tissue identity) and an acquisition branch (capturing scanner identity), combined through a decoder that reconstructs the original feature representation. The acquisition branch capacity can be constrained via bottlenecking to reduce biological leakage.

**Results.** On canine cutaneous squamous cell carcinoma (SCC) with expert tissue-category annotations across five scanners, and on the SCORPION multi-backbone archive, we find: (1) true same-region paired structure is required for tissue-identity preservation — broken-pair controls degrade performance; (2) the biological branch reduces scanner recoverability from 0.866 to 0.361 while preserving category structure (0.407 → 0.386) and improving neighborhood purity (0.954 → 0.973); (3) oldstyle centroid/QR linear projection achieves stronger raw scanner removal (scanner probe 0.200 at chance level, category 0.400) — paired-acquisition does not claim best raw scanner erasure; (4) bottlenecking the acquisition branch from 64D to 8D reduces category leakage from 0.346 to 0.160 while preserving scanner capture (0.865 → 0.864), with cross-backbone tissue-retrieval leakage reductions in SCORPION (DINOv2: 0.094 → 0.023; Phikon: 0.074 → 0.020; ResNet50: 0.171 → 0.051); (5) decoder-based acquisition swapping shows scanner identity follows the acquisition branch through recombination (decoder-space scanner follow 0.901 for same-sample swaps), supporting factor-like behavior, though nearest-neighbor scanner purity (0.880) is weaker than biological category purity (0.980).

**Conclusion.** Paired-acquisition factorization provides structured separation — an explicit scanner-bearing acquisition branch whose biological leakage can be reduced by bottlenecking — not best-in-class raw scanner removal. If raw scanner erasure is the only goal, oldstyle centroid/QR projection is the stronger choice. If an inspectable, bottleneckable, and swappable acquisition representation is needed, paired-acquisition offers a complementary tool.

---

## 1. Introduction

Multi-scanner computational pathology datasets are increasingly common. Whole-slide images from different institutions are digitized on different scanners, producing systematic variation in color, contrast, and texture that confounds tissue-level analysis. A feature representation that separates scanner-specific variation from biologically meaningful structure would enable more robust cross-scanner analysis, but existing approaches address only part of this problem.

Linear scanner-removal methods — projecting out scanner-centroid directions or removing principal components correlated with scanner identity — suppress scanner signal effectively. In our experiments, a simple centroid/QR linear projection (oldstyle_keep_k4) reduces a linear scanner probe to chance level (0.200, 5-class balanced) while preserving tissue-category accuracy (0.400). These methods are fast, simple, and strong at raw scanner erasure. But they are blind: they remove scanner-discriminative directions without producing an explicit representation of what was removed, cannot be inspected for residual biological content, and cannot be manipulated to test whether scanner information has been cleanly isolated.

**Why not just use the linear baseline?** This question must be answered immediately. If raw scanner removal is the only goal, oldstyle centroid/QR is the stronger choice. Paired-acquisition factorization addresses a different problem: it produces an explicit scanner-bearing acquisition branch that can be inspected, bottlenecked to reduce biological leakage, and swapped through a decoder to test whether scanner information follows the acquisition factor through recombination. The two methods are complementary — centroid/QR for blind removal, paired-acquisition for structured separation.

Paired-acquisition factorization exploits a specific data structure: same tissue region, imaged on two different scanners. This paired supervision provides a weak signal for separating scanner from biology: the biological content is shared across the pair, while scanner characteristics differ. By training a dual-branch architecture with reconstruction, adversarial, and independence losses, the model learns to route biological information through one branch and scanner information through the other.

This paper presents a structured audit of paired-acquisition factorization across two datasets (canine SCC DINOv2 with expert tissue-category annotations; SCORPION with DINOv2, Phikon, and ResNet50 frozen archives) and five core questions:

1. Does true paired structure matter, or would any same-scanner pairing suffice?
2. Is branch separation measurable — does the biological branch suppress scanner signal while preserving category structure?
3. How does paired-acquisition compare to the strongest linear baseline?
4. Can bottlenecking the acquisition branch reduce biological leakage while preserving scanner capture?
5. Does decoder-based acquisition swapping support factor-like behavior?

Our contribution is not a claim of best scanner removal. It is a bounded demonstration that paired supervision can produce structured separation with an explicit, manipulable acquisition branch.

---

## 2. Problem Setup: Paired Acquisition as Supervision

### 2.1 Paired acquisition signal

A positive pair consists of two patch-level feature vectors extracted from the same tissue region imaged on two different scanners. The biological content (tissue type, morphology) is shared; the scanner characteristics differ. This pairing provides a weak supervision signal for decomposition: the model must reconstruct both views from a shared biological code and scanner-specific acquisition codes.

### 2.2 Datasets

**Canine SCC DINOv2.** 4,025 patches from 805 regions across 44 samples, imaged on 5 scanners (cs2, gt450, nz20, nz210, p1000). Expert-annotated tissue categories: Epidermis (1,205), SCC (1,205), Subcutis (510), Dermis (500), Inflamm/Necrosis (400), Bone (195), Cartilage (10). This is the only dataset with biological category labels; it anchors all category-probe claims.

**SCORPION.** Multi-backbone frozen feature archive (DINOv2-Base, Phikon, ResNet50) with paired same-region cross-scanner patches. Contains scanner and region metadata but no biological category labels. Used for cross-backbone tissue/pair-retrieval validation.

### 2.3 Evaluation protocol

All experiments use slide-level 5-fold cross-validation. Probe models (LogisticRegression, balanced class weights, max_iter=5000) are trained on fit-slide features and evaluated on held-out test slides. Key metrics:

- **Scanner probe accuracy.** Balanced accuracy of a linear classifier predicting scanner ID. Lower on the biological branch = stronger scanner suppression. Higher on the acquisition branch = stronger scanner capture.
- **Category probe accuracy.** Balanced accuracy of a linear classifier predicting tissue category. Higher on the biological branch = stronger category preservation. Lower on the acquisition branch = less biological leakage.
- **Neighborhood purity.** Fraction of k-nearest neighbors sharing the same category (biological branch) or same scanner (acquisition branch).
- **Paired cosine / top-1 retrieval.** Cosine similarity and same-region retrieval rate for cross-scanner pairs, measuring tissue-identity preservation.
- **Cross-covariance RMS.** Root-mean-square cross-covariance between biological and acquisition branch features, measuring branch independence.
- **Scanner-heldout transfer.** Train category probe on 4 scanners, test on held-out 5th scanner.

---

## 3. Method: Biological Branch, Acquisition Branch, Bottleneck Variants

### 3.1 Architecture

```
Frozen encoder (DINOv2/Phikon/ResNet50, 768D)
    |
    +---> Biological projector (256D) ---> z_bio
    |
    +---> Acquisition projector (variable D) ---> z_acq
                                                  |
    z_combined = concat(z_bio, z_acq)             |
              |                                   |
              v                                   |
    Decoder (264D -> 512D -> 768D) -----------> reconstruction of original features
```

The biological branch is fixed at 256D. The acquisition branch dimensionality is the primary controllable parameter: 64D (baseline true_pair), 16D (acq_dim16), 8D (acq_dim8).

### 3.2 Training objectives

1. **Paired cosine reconstruction.** The decoder output for each view should match its original feature vector (cosine similarity loss).
2. **Scanner adversarial.** A gradient-reversed scanner classifier operates on the acquisition branch, encouraging it to encode scanner-discriminative information.
3. **Variance regularization.** Each branch's feature variance is regularized to prevent collapse.
4. **Cross-covariance independence.** The cross-covariance between biological and acquisition branch features is penalized, encouraging branch independence. Two strengths: default (0.05) and stronger (0.20).

### 3.3 Bottleneck variants

Six smoke variants were evaluated (acq_dim ∈ {8, 16, 32, 64} × cross-covariance ∈ {0.05, 0.20}). Two were promoted to full-scale 5-fold × 5-seed evaluation:

- **acq_dim8_default.** 8-dimensional acquisition branch, default cross-covariance weight (0.05).
- **acq_dim16_stronger_xcov.** 16-dimensional acquisition branch, stronger cross-covariance weight (0.20).

The baseline for all bottleneck comparisons is true_pair (64D, default cross-covariance).

---

## 4. Result 1: Pair Structure Matters

### 4.1 Experiment

We constructed a pairing ladder with five levels of pair strictness:

| Level | Condition | Description |
|---|---|---|
| L0 | true_same_region_pairs | Same tissue region, different scanners |
| L1 | same_slide_different_region_pairs | Same slide, different tissue region |
| L2 | shuffled_sample_pairs | Different slides, any region |
| L3 | scanner_balanced_random_pairs | Random regions preserving scanner assignment |
| L4 | fully_random_pairs | All views randomly assigned (lower bound) |

All conditions used the same architecture and losses; only pair construction changed. Evaluated on SCORPION DINOv2 and canine SCC DINOv2 (full 5-fold × 5-seed).

### 4.2 Findings

True same-region pairs (L0) are the strongest condition for tissue-identity preservation in both datasets.

**SCORPION DINOv2.** L0 paired cosine = 0.880, top-1 retrieval = 1.000. L1 (same-slide-different-region) drops to cosine 0.809, retrieval 0.995. L2-L4 degrade further (cosine 0.725-0.767). The gap from L0 to the best looser condition (L1) is 0.071 in cosine.

**Canine SCC DINOv2.** L0 paired cosine = 0.730, top-1 retrieval = 0.933. All non-true conditions cluster in a lower band (cosine 0.542-0.585) with overlapping confidence intervals. The gap from L0 is 0.145 in cosine — larger than SCORPION.

**Scanner suppression persists across all conditions.** The biological branch scanner probe remains low (0.36-0.41) regardless of pairing strictness. The scanner adversary works independently of pair quality.

**Scanner balancing is not sufficient.** L3 (scanner_balanced_random) performs similarly to L4 (fully_random) in both datasets. Scanner balancing alone does not recover tissue identity — biological correspondence in the positive pair is the active ingredient.

### 4.3 Interpretation

True same-region paired structure is required for the method to achieve its best tissue-identity preservation. The method does not require exactly same-region pairs — same-slide-different-region (L1) still preserves substantial tissue identity — but tissue preservation degrades as the pairing weakens. This bounds how paired acquisition data must be collected.

---

## 5. Result 2: Branch Separation and Biological Preservation

### 5.1 Experiment

We probed the biological and acquisition branches from true-pair training with linear classifiers for scanner identity and tissue category, and measured neighborhood purity. Baselines: original frozen DINOv2 features, PCA component removal (k ∈ {1,2,4,8,16,32}), linear scanner subspace projection (k ∈ {0,1,2,4,8,16,32}), and shuffled-sample (broken-pair) controls.

### 5.2 Findings

**Biological branch: scanner suppressed, category preserved.** The biological branch reduces scanner probe from 0.866 (frozen) to 0.361 — a reduction of 0.505. Category probe decreases only 0.021 (0.407 → 0.386). The category/scanner ratio more than doubles (0.47 → 1.07). Neighborhood category purity improves (K1: 0.954 → 0.973), suggesting the biological branch may reduce within-category noise while retaining category-separating structure.

**Acquisition branch: scanner captured, category reduced.** Scanner probe is 0.862 — scanner signal is retained. Category probe is 0.346, reduced from 0.407 (frozen) and 0.386 (biological branch). Neighborhood category purity is 0.530 — dramatically lower than the biological branch (0.973).

**Cross-covariance is low.** RMS cross-covariance between branches is 0.092 (SCORPION DINOv2) and 0.090 (canine SCC DINOv2).

**Branch separation contrast.** The scanner contrast between branches (acquisition scanner − biological scanner) is 0.504. The category contrast (biological category − acquisition category) is 0.040. Scanner separation is strong; category separation is modest but directional.

**Shuffled-sample control.** Breaking the pair structure degrades both branches: shuffled biological scanner probe 0.409 (vs 0.361), category probe 0.324 (vs 0.386), purity 0.897 (vs 0.973). True-pair structure is required for the measured separation.

**Comparison with PCA.** PCA removal at k=32 achieves scanner probe 0.649 and category probe 0.289 — worse on both axes than the biological branch. PCA increases effective rank (149.3); factorization reduces it in both branches (biological 74.0, acquisition 13.8).

### 5.3 Scanner-heldout transfer

Under leave-one-scanner-out category classification, the biological branch nearly preserves cross-scanner transfer: balanced accuracy 0.827 vs frozen 0.845 (Δ = −0.018). It improves on the hardest held-out scanner (p1000: 0.755 vs 0.709, Δ = +0.045). The acquisition branch transfers poorly (0.517) as expected — it encodes scanner-specific features. Shuffled-sample biological transfer is substantially worse (0.608). The linear projection baseline (0.835) slightly exceeds the biological branch in mean accuracy, consistent with its stronger raw scanner removal.

### 5.4 Interpretation

Branch separation is measurable and consistent: the biological branch suppresses scanner signal while preserving category structure; the acquisition branch captures scanner signal with reduced category structure. Separation is partial — residual scanner signal remains in the biological branch (0.361, above chance 0.20), and residual category structure remains in the acquisition branch (0.346). This is structured separation, not perfect disentanglement.

---

## 6. Result 3: Oldstyle Centroid/QR Is the Strongest Raw Scanner-Removal Baseline

### 6.1 Experiment

We compared paired-acquisition factorization to oldstyle centroid/QR linear projection: compute per-scanner mean feature vectors, QR-orthonormalize the first k=4 direction rows, and project out the resulting subspace. This is distinct from the logistic-regression-SVD linear projection used in earlier experiments, which was weaker (scanner probe 0.707 vs 0.200). A linear baseline consistency audit documented and reconciled this discrepancy.

### 6.2 Findings

| Representation | Scanner probe | Category probe | Category/scanner ratio |
|---|---|---|---|
| original_frozen_features | 0.866 | 0.407 | 0.47 |
| **oldstyle_keep_k4** | **0.200** | **0.400** | **2.00** |
| true_pair_biological | 0.361 | 0.386 | 1.07 |
| true_pair_acquisition | 0.865 | 0.346 | 0.40 |

Oldstyle_keep_k4 achieves chance-level scanner probe (0.200 for 5 balanced classes) while preserving category accuracy (0.400) — strictly better than paired-acquisition on both raw scanner removal and raw category preservation. The category/scanner ratio (2.00) exceeds the biological branch (1.07).

Oldstyle_removed_k4 (the removed component) carries scanner signal (scanner probe 0.538) but also leaks category signal (0.242) — less category leakage than true_pair_acquisition (0.346).

### 6.3 Interpretation

**Paired-acquisition is not the best raw scanner-removal method.** Oldstyle centroid/QR wins on raw scanner suppression and raw category preservation. Any claim that paired-acquisition outperforms all baselines on scanner removal is false.

**The contribution is structured separation.** The oldstyle baseline removes scanner-centroid directions and produces a cleaned embedding. It does not produce an explicit acquisition branch, cannot be inspected for what scanner information was removed, cannot be bottlenecked to reduce biological leakage, and cannot be swapped to test factor-like behavior. Paired-acquisition solves a different problem: it produces an explicit, manipulable acquisition representation.

If raw scanner removal is the only goal, use oldstyle_keep_k4. If an inspectable acquisition branch is needed, paired-acquisition provides complementary capability.

---

## 7. Result 4: Bottleneck Comparison / Capacity-Constrained Separation

### 7.1 Experiment

We compared two bottlenecked acquisition branch variants against the baseline 64D acquisition branch:

| Variant | Acq dim | Cross-cov weight | Full-scale runs |
|---|---|---|---|
| true_pair (baseline) | 64 | 0.05 | 25 |
| acq_dim16_stronger_xcov | 16 | 0.20 | 25 |
| acq_dim8_default | 8 | 0.05 | 25 |

Evaluated on canine SCC DINOv2 (category leakage, scanner capture, downstream transfer) and SCORPION DINOv2/Phikon/ResNet50 (tissue/pair-retrieval leakage).

### 7.2 Findings

**Canine SCC DINOv2.** Bottlenecking substantially reduces acquisition-branch category leakage while preserving scanner capture:

| Variant | Acq scanner | Acq category leakage | Bio scanner | Bio category |
|---|---|---|---|---|
| true_pair (64D) | 0.865 | 0.346 | 0.361 | 0.386 |
| acq_dim16_stronger_xcov | 0.864 | 0.169 | 0.359 | 0.382 |
| acq_dim8_default | 0.864 | 0.160 | 0.369 | 0.385 |

Category leakage drops from 0.346 to 0.160 (δ = −0.186) while scanner capture is unchanged (0.865 vs 0.864). Biological branch category accuracy and scanner suppression are preserved within narrow bands. Scanner-heldout transfer is maintained: true_pair_bio 0.827, acq_dim8 0.822, acq_dim16_xcov 0.829.

**SCORPION cross-backbone.** Tissue/pair-retrieval leakage in the acquisition branch drops substantially:

| Backbone | true_pair leakage | acq_dim8 leakage | Reduction |
|---|---|---|---|
| DINOv2 | 0.094 | 0.023 | −0.071 |
| Phikon | 0.074 | 0.020 | −0.054 |
| ResNet50 | 0.171 | 0.051 | −0.120 |

Scanner capture is preserved across all three backbones. SCORPION evidence uses tissue/pair-retrieval leakage, not category leakage — SCORPION lacks biological category labels.

### 7.3 Interpretation

Reducing acquisition branch capacity selectively reduces biological leakage while preserving scanner capture. Scanner-discriminative information appears to require lower capacity than the full tissue-category manifold — the bottleneck removes excess capacity that was encoding biological information.

This is a **directional improvement on the separation tradeoff**, not a densely mapped Pareto frontier. Only two bottleneck dimensions (8, 16) with two regularization strengths were tested at full scale. The comparison is sparse; we cannot claim optimal bottleneck size or a continuous Pareto front. We describe the experiment as a **bottleneck comparison** or **capacity-constrained separation audit**, and the result as a **directional separation-frontier improvement**.

---

## 8. Result 5: Decoder-Based Acquisition Swapping / Factor-Like Behavior

### 8.1 Experiment

We tested whether the acquisition branch carries manipulable scanner information by swapping acquisition branches between samples and recombining via the learned decoder. Four swap types:

| Type | Biological source | Acquisition source | Tests |
|---|---|---|---|
| A | Sample i | Sample i, different scanner | Scanner follows acquisition when biology is fixed |
| B | Sample i | Sample j, same category | Scanner follows acquisition across samples |
| C | Sample i | Sample j, different category | Both scanner and category change |
| D | Sample i | Random sample | Unstructured swap (control) |

Evaluated on three variants (true_pair, acq_dim8_default, acq_dim16_stronger_xcov) with 5-fold × 5-seed × 150 swaps per type. Metrics: scanner follow rate (probe on swapped features predicts acq-source scanner), category preservation rate (probe predicts bio-source category), branch-space nearest-neighbor purity, and decoder-space reconstruction metrics.

### 8.2 Findings

**Scanner follows acquisition branch.** Scanner follow rate averages 0.855 across all variants and swap types. For same-sample/different-scanner swaps (Type A, the cleanest test), the rate is 0.871-0.876. Scanner information tracks the acquisition branch through recombination.

**Category preservation is modest.** Category preservation rate averages ~0.40 across variants. Under swap, category structure is partially disrupted — the biological branch carries category information, but recombination through the decoder introduces interference.

**Bottleneck variants show improved separation under swap.** Bottlenecked variants have lower acquisition-branch category leakage under swap (acq_dim16_xcov 0.283, acq_dim8 0.287 vs true_pair 0.296) and comparable scanner follow rates.

**Decoder-space evidence is stronger.** In decoder-reconstructed feature space, same-sample swaps (Type A) achieve scanner follow 0.901 and category preservation 0.978-0.992 for bottlenecked variants. The decoder-space results are stronger than branch-space results, consistent with the decoder having learned to combine the branches into coherent features.

**Nearest-neighbor evidence is mixed — and shown here explicitly.**

| Metric | Value | Assessment |
|---|---|---|
| Bio-space K1 category purity | 0.980 | Near-perfect — biological neighbors preserve category |
| Acq-space K1 scanner purity | 0.880 | Decent but mixed — scanner neighbors in acquisition space are less pure |
| Decoder-space scanner follow (Type A) | 0.901 | Strong — scanner follows acquisition in clean swap |
| Decoder-space category preservation | 0.978+ | Strong — category preserved in bottlenecked variants |
| Target-scanner NN rate (bottlenecked) | 0.135-0.152 | Weak — bottlenecking collapses scanner NN structure |
| Source-category NN rate (bottlenecked) | 0.996 | Near-perfect — category NN structure preserved |

The asymmetry is the key finding: biological category purity (0.980) is near-perfect; acquisition scanner purity (0.880) is decent but clearly weaker. The acquisition branch carries scanner information, but scanner identity does not dominate its nearest-neighbor structure to the same degree that category identity dominates the biological branch.

### 8.3 Interpretation

This is **factor-like behavior**, not proof of perfect causal factorization. Scanner information follows the acquisition branch through decoder recombination — especially in the clean Type A swap (decoder-space scanner follow 0.901). But nearest-neighbor scanner purity (0.880) is weaker than category purity (0.980), and bottlenecking reduces target-scanner NN rates dramatically (0.558 → 0.135).

**This is the weakest of the five claims.** Single-dataset evidence (canine SCC DINOv2 only). No cross-backbone or SCORPION swapping. Decoder trained for reconstruction, not factor manipulation — recombination artifacts may exist. Swap is at feature-representation level, not image-pixel level. The evidence supports "factor-like" but not "factor proven."

---

## 9. Limitations

1. **Single labeled-category dataset.** All category-label claims rest on canine SCC DINOv2 only. SCORPION provides cross-backbone evidence for tissue/pair-retrieval metrics but lacks biological category labels. A second labeled multi-scanner dataset would substantially strengthen claims 2, 4, and 5.

2. **Partial separation, not perfect.** Biological branch retains scanner signal (probe 0.361, above chance 0.20). Acquisition branch retains category signal (probe 0.346 in 64D, 0.160 in 8D). Separation is measurable and useful, not complete.

3. **Sparse bottleneck comparison.** Only two dimensions (8, 16) with two regularization strengths tested at full scale. Cannot claim continuous Pareto front or optimal bottleneck size. Directional improvement, not mapped frontier.

4. **Linear baselines only.** Strongest scanner-removal baseline is linear centroid/QR projection. Nonlinear baselines (adversarial projection, domain-adversarial feature learning) not compared.

5. **Feature-level, not image-level.** All experiments operate on frozen patch-level DINOv2 features. No end-to-end training, no pixel-level manipulation, no whole-slide-image context.

6. **Five scanners, two datasets.** Scanner diversity is limited. Generalization to unseen scanner types, staining protocols, or tissue types not tested.

7. **Decoder reuse.** Swapping decoder was trained for reconstruction, not factor manipulation. Recombination may introduce artifacts.

8. **NN scanner purity gap.** Nearest-neighbor scanner purity in acquisition space (0.880) is weaker than category purity in biological space (0.980). The acquisition branch carries scanner information but does not organize its neighborhood structure around scanner identity to the same degree.

9. **No clinical or diagnostic evidence.** This is a methodological research contribution. No patient-outcome, diagnostic-accuracy, or clinical-utility claims are made or should be inferred.

---

## 10. Appendix Map

See `appendix_plan.md` for the complete appendix structure. Summary:

- **Appendix A.** Pair-structure boundary detail — per-condition per-dataset tables, level-vs-level contrasts, cross-backbone extension.
- **Appendix B.** Biological label preservation detail — per-representation per-class recall, PCA k-sweep, linear scanner subspace k-sweep, neighborhood purity k=1,5,10.
- **Appendix C.** Scanner-heldout transfer detail — per-scanner per-representation breakdown, per-class recall.
- **Appendix D.** Sample-disjoint and scanner-confounded audits.
- **Appendix E.** Linear baseline detail — consistency audit, oldstyle k=1-4, logistic-SVD residual.
- **Appendix F.** Bottleneck comparison detail — smoke and full variant metrics, per-scanner downstream, cross-backbone raw metrics.
- **Appendix G.** Acquisition swapping detail — per-swap-type probe metrics, per-variant NN purity, decoder-space reconstruction.
- **Appendix H.** Unified separation scoreboard — full 12-row table with source and commit mapping.
- **Appendix I.** Cross-backbone SCORPION evidence — DINOv2, Phikon, ResNet50 tissue-retrieval metrics.

---

## References

Result commits (see `result_to_claim_map.csv` for full mapping):

1. e4819c42 — Pair-structure boundary test
2. d018c924 — Cross-backbone pair-structure boundary
3. bec06eb4 — Biological label preservation audit
4. 535eea18 — Scanner-heldout label transfer audit
5. 0d7cdc92 — Sample-disjoint scanner-heldout transfer audit
6. b5a9886e — Scanner-confounded label robustness audit
7. ec2a509f — Linear residual branch separation audit
8. a325c009 — Linear baseline consistency audit
9. 3450ede2 — Oldstyle residual branch separation audit
10. a89bfb32 — Acquisition bottleneck comparison
11. c29a038d — Bottleneck-selected downstream validation
12. 0e2af247 — Bottleneck-selected cross-backbone validation
13. 1c527697 — Unified separation scoreboard
14. 3e5bf19e — Acquisition branch audit
15. aa8d0596 — Acquisition factor swapping audit
