# Paired-Acquisition Factorization — Claim Ledger

**Branch:** experiment/claim-ledger-and-paper-skeleton
**Generated:** 2026-07-08
**Purpose:** Paper-grade claim ledger compressing the experiment stack into five core claims with evidence, baselines, limitations, and allowed wording.
**Upstream:** Built from 15 audit commits across 12 experiment branches (see `result_to_claim_map.csv`).

---

## Core Thesis (compressed)

Paired-acquisition is **not** the best raw scanner-removal method. Oldstyle centroid/QR linear projection wins raw scanner erasure (scanner probe 0.2000 vs paired-acquisition 0.3614).

The contribution is **structured separation**: a biological branch preserving category/transfer behavior and an explicit scanner-bearing acquisition branch whose biological leakage can be reduced by bottlenecking (category leakage 0.3456 → 0.1598, tissue-retrieval leakage 0.0944 → 0.0231 in SCORPION DINOv2). Decoder-based swapping supports factor-like behavior but does not prove perfect causal factorization.

---

## CLAIM_1_PAIR_STRUCTURE

### Claim Title
True paired-acquisition structure is required for tissue/category preservation and downstream transfer.

### Claim Text (safe)
Training paired-acquisition factorization with true same-region cross-scanner pairs produces substantially better biological-branch tissue-identity preservation and category-retrieval performance than any broken-pair or random-pair control. Scanner suppression persists across all pairing conditions, but tissue-identity preservation degrades monotonically as the biological correspondence in the positive pair is weakened. The biological correspondence — not scanner balancing alone — is the active ingredient.

### Strongest Evidence
Pair-structure boundary test (e4819c42): In SCORPION DINOv2, true same-region paired cosine = 0.8796, top-1 retrieval = 0.9999 vs same-slide-different-region = 0.8089, shuffled-sample = 0.7668, random = 0.7247. In canine SCC DINOv2, true-pair paired cosine = 0.7300, retrieval = 0.9334 vs non-true band 0.5422–0.5849. Scanner suppression is maintained across all conditions (bio scanner probe 0.36–0.41), but tissue identity degrades with looser pairing.

### Supporting Commits
- e4819c42 (pair-structure boundary test) — primary
- d018c924 (cross-backbone pair-structure boundary) — generalization
- bec06eb4 (biological label preservation: shuffled vs true-pair contrast)
- 535eea18 (scanner-heldout transfer: shuffled biological 0.6081 vs true-pair biological 0.8273)
- 0d7cdc92 (sample-disjoint scanner-heldout: additional control)

### Primary Result Files
- `results/paired_acquisition_factorization_pair_structure_boundary_test/boundary_summary.csv`
- `results/paired_acquisition_factorization_pair_structure_boundary_test/pair_structure_boundary_report.md`
- `results/paired_acquisition_factorization_pair_structure_boundary_crossbackbone/boundary_summary.csv`

### Strongest Baseline
True same-region pairs (L0) in the pairing ladder — all other ladder rungs are controls. Scanner-balanced random pairing (L3) performs no better than fully random (L4), confirming that scanner balancing alone does not recover tissue identity.

### Limitation
The pairing ladder only controls pair construction, not training dynamics. All conditions use the same architecture and losses. Does not test whether alternative training objectives could recover tissue identity under broken-pair regimes. Cross-backbone evidence (d018c924) extends to SCORPION DINOv2/Phikon/ResNet50 but uses pair/tissue retrieval metrics, not category labels — SCORPION lacks biological category annotations.

### Forbidden Overclaim
- "Paired-acquisition requires exactly same-region pairs" → same-slide-different-region still preserves substantial tissue identity (SCORPION cosine 0.8089, canine SCC 0.5422).
- "Any broken-pair setup destroys the method" → scanner suppression is maintained across all conditions.
- "True pairs are necessary for scanner suppression" → false; scanner adversary works regardless of pair quality.

### Main or Appendix
**Main.** This is the foundational validity claim — without it the method has no mechanism.

### Figure/Table Candidate
Pairing ladder table showing paired cosine, top-1 retrieval, and scanner probe for each level across both datasets. Main-text table + appendix detail.

### Reviewer Objection Answered
- "Isn't scanner balancing across the batch sufficient to learn the factorization?" — Addressed: scanner-balanced random pairing (L3) performs similarly to fully random (L4), and substantially below true same-region pairs (L0). Scanner balancing alone does not recover the biological structure.
- "Maybe the method just needs paired data from the same scanner, not the same region?" — Addressed: same-slide-different-region (L1) is measurably below true same-region (L0) in both datasets.

### Remaining Objection
The pairing ladder controls pair construction but does not disentangle whether the degradation comes from weaker biological correspondence, weaker scanner correspondence, or both. Same-slide-different-region pairs change both the biological content and the scanner content of the positive pair simultaneously.

---

## CLAIM_2_BRANCH_SEPARATION

### Claim Title
Paired-acquisition produces measurable branch separation: biological branch suppresses scanner signal while preserving tissue/category structure; acquisition branch captures scanner signal.

### Claim Text (safe)
Under true-pair training, the biological branch substantially reduces scanner recoverability (scanner probe 0.8656 → 0.3610 in canine SCC DINOv2) while preserving tissue-category structure (category probe 0.4068 → 0.3855, Δ = −0.0213; neighborhood purity improves from 0.9542 → 0.9729). The acquisition branch retains scanner signal (scanner probe 0.8620) while category structure is reduced (category probe 0.3458, purity 0.5295). Cross-covariance between branches is low (RMS < 0.10). This separation is measurable and consistent across folds and seeds.

### Strongest Evidence
Biological label preservation audit (bec06eb4): true_pair_biological achieves category/scanner ratio 1.07 (vs original 0.47), with scanner suppression δ = +0.5046 and category preservation δ = −0.0213. true_pair_acquisition scanner capture 0.8620 with category leakage 0.3456. Biological neighborhood purity K1 = 0.9729 vs acquisition 0.5295.

### Supporting Commits
- bec06eb4 (biological label preservation audit) — primary
- 3450ede2 (oldstyle residual branch separation) — branch contrast quantification
- 3e5bf19e (acquisition branch audit) — branch audit
- e4819c42 (pair-structure boundary test) — cross-covariance and dual-branch metrics
- b5a9886e (scanner-confounded label robustness) — robustness context
- ec2a509f (linear residual branch separation) — linear comparison baseline

### Primary Result Files
- `results/paired_acquisition_factorization_biological_label_preservation_audit/label_probe_summary.csv`
- `results/paired_acquisition_factorization_oldstyle_residual_branch_separation_audit/oldstyle_residual_branch_contrasts.csv`
- `results/paired_acquisition_factorization_oldstyle_residual_branch_separation_audit/oldstyle_residual_summary.csv`

### Strongest Baseline
Original frozen DINOv2 features (scanner probe 0.8656, category probe 0.4068) — the pre-factorization state against which branch separation is measured. Shuffled-sample biological branch (scanner probe 0.4091, category probe 0.3237) — shows that broken-pair factorization cannot achieve the same separation.

### Limitation
Branch separation is measured via linear probe accuracies and neighborhood purity — these are diagnostic metrics, not proof of complete information-theoretic independence. Cross-covariance is low but nonzero. The biological branch still carries residual scanner signal (scanner probe 0.3610, above chance 0.20). The acquisition branch still carries residual category structure (category probe 0.3456). Separation is partial, not perfect. All evidence is from canine SCC DINOv2 only; SCORPION lacks category labels.

### Forbidden Overclaim
- "Perfect disentanglement" or "complete separation" — separation is partial and residual leakage exists in both branches.
- "The biological branch is scanner-free" — scanner probe is 0.3610, well above chance (0.20).
- "The acquisition branch is category-free" — category probe is 0.3456, substantially above its own baseline.

### Main or Appendix
**Main.** This is the core contribution claim — structured separation is the method's reason to exist.

### Figure/Table Candidate
Two-panel figure: (a) scanner probe vs category probe scatter for all representations, (b) biological vs acquisition branch purity bar chart. Central result table with scanner probe, category probe, purity K1/K5 for key representations.

### Reviewer Objection Answered
- "Isn't this just dimensionality reduction? Why is a learned factorization better than PCA?" — Addressed: PCA at k=32 achieves scanner probe 0.6489 and category probe 0.2893 with purity 0.9648 — scanner suppression is weaker and category preservation is worse. PCA increases effective rank (149.3) while factorization reduces it in both branches. PCA is blind to the scanner/category distinction.
- "Is branch separation just a byproduct of the gradient reversal layer?" — Partially, but the reversed gradient only operates on the acquisition branch. The biological branch separation emerges from the paired reconstruction + adversarial + independence loss triplet.

### Remaining Objection
No information-theoretic independence metric exists in the evidence stack. All separation metrics are probe-based and may not detect higher-order dependencies. Would need mutual information or nonlinear independence testing.

---

## CLAIM_3_LINEAR_BASELINE_BOUNDARY

### Claim Title
Oldstyle centroid/QR linear projection is the strongest raw scanner-removal baseline; paired-acquisition should not be framed as best-in-class scanner erasure.

### Claim Text (safe)
Oldstyle centroid/QR linear projection (oldstyle_keep_k4) achieves scanner probe accuracy at chance level (0.2000) while preserving category accuracy (0.4004) — strictly better raw scanner removal and category preservation than true_pair_biological (scanner 0.3614, category 0.3860). The paired-acquisition contribution is not raw scanner erasure but structured separation: an explicit scanner-bearing acquisition branch (scanner capture 0.8651) with biological leakage that can be reduced by bottlenecking. Any claim that paired-acquisition is the best scanner-removal method is false. Claims must be bounded by this baseline boundary.

### Strongest Evidence
Oldstyle residual branch separation audit (3450ede2): oldstyle_keep_k4 scanner probe 0.2000, category probe 0.4004 vs true_pair_biological scanner probe 0.3614, category probe 0.3860. Linear baseline consistency audit (a325c009): documents why the oldstyle (centroid/QR, scanner 0.2000) is stronger than newstyle (logistic-SVD, scanner 0.7071).

### Supporting Commits
- 3450ede2 (oldstyle residual branch separation) — primary
- a325c009 (linear baseline consistency audit) — reconciles oldstyle vs newstyle mismatch
- ec2a509f (linear residual branch separation) — original logistic-SVD baseline; now known to be weaker
- 1c527697 (unified separation scoreboard) — consolidated baseline comparison

### Primary Result Files
- `results/paired_acquisition_factorization_oldstyle_residual_branch_separation_audit/oldstyle_residual_summary.csv`
- `results/paired_acquisition_factorization_linear_baseline_consistency_audit/linear_baseline_consistency_metrics.csv`
- `results/paired_acquisition_factorization_unified_separation_scoreboard/unified_separation_scoreboard_key_metrics.csv`

### Strongest Baseline
oldstyle_keep_k4 (centroid/QR) — the reference baseline that paired-acquisition must be measured against in all scanner-removal comparisons.

### Limitation
The oldstyle baseline is a post-hoc linear operation on frozen DINOv2 embeddings. It does not produce an explicit acquisition branch, cannot be inspected for what scanner information was removed, and cannot be used for acquisition swapping. It is strong at blind removal but provides no structural decomposition. The comparison is fair but the two methods solve different problems (blind removal vs structured separation).

### Forbidden Overclaim
- "Paired-acquisition removes scanner bias better than any linear method" — false; oldstyle_keep_k4 wins raw removal.
- "Paired-acquisition is the best scanner-removal approach" — false for any definition of "best" that means strongest scanner suppression.
- "Paired-acquisition beats all baselines on scanner removal" — false; oldstyle_keep_k4 is a baseline and it wins raw scanner removal.
- Any text that omits the oldstyle baseline when discussing scanner-removal strength.

### Main or Appendix
**Main.** This is a self-imposed boundary that frames the entire contribution. Without it, the paper overclaims. With it, the structured-separation contribution is precisely scoped.

### Figure/Table Candidate
Scoreboard-style summary table (from unified separation scoreboard) showing scanner probe, category probe, and scanner-capture for oldstyle_keep_k4, true_pair_biological, true_pair_acquisition, and bottlenecked variants. Main-text table.

### Reviewer Objection Answered
- "Why not just use a simple linear projection to remove scanner signal? Isn't paired-acquisition over-engineered?" — Addressed head-on: the oldstyle baseline is stronger at raw scanner removal. The paper explicitly concedes this and reframes the contribution as structured separation. The linear baseline removes scanner signal; it does not produce an explicit acquisition branch. If you only want scanner removal, use oldstyle_keep_k4. If you want to inspect, bottleneck, or swap the acquisition factor, use paired-acquisition.
- "Did you cherry-pick a weak linear baseline?" — Addressed by the consistency audit (a325c009): we started with a weaker logistic-SVD baseline, found the discrepancy, and upgraded to the stronger centroid/QR baseline. The strongest linear baseline is now our reference.

### Remaining Objection
Oldstyle_keep_k4 removes scanner-centroid directions but may leave higher-order scanner structure. A nonlinear scanner-removal baseline (e.g., adversarial feature projection) could be even stronger. We have not tested this.

---

## CLAIM_4_BOTTLENECK_FRONTIER

### Claim Title
Bottlenecking the acquisition branch improves the separation frontier by reducing acquisition-branch biological leakage while preserving scanner capture and biological downstream transfer.

### Claim Text (safe)
Reducing the acquisition branch capacity from 64D (true_pair) to 8D or 16D with stronger cross-covariance regularization substantially reduces biological/tissue leakage in the acquisition branch. In canine SCC DINOv2, acq_dim8_default reduces acquisition-branch category probe from 0.3456 (true_pair) to 0.1598 (δ = −0.1858) while maintaining scanner capture (0.8643 vs 0.8651) and preserving biological-branch category accuracy and downstream transfer. In SCORPION DINOv2, acquisition-branch tissue-retrieval leakage drops from 0.0944 (true_pair) to 0.0231 (acq_dim8_default), with similar improvements in Phikon (0.0739 → 0.0204) and ResNet50 (0.1705 → 0.0505). The separation frontier is improved: less biological leakage in the acquisition branch, same scanner capture.

### Strongest Evidence
Acquisition bottleneck separation frontier sweep (a89bfb32): acq_dim8_default achieves acq category 0.1598 (vs true_pair 0.3456), acq scanner 0.8643 (vs true_pair 0.8651), bio category 0.3852 (vs true_pair 0.3860). Cross-backbone validation (0e2af247): SCORPION acq_dim8 retrieval leakage 0.0231 vs true_pair 0.0944 (DINOv2), 0.0204 vs 0.0739 (Phikon), 0.0505 vs 0.1705 (ResNet50).

### Supporting Commits
- a89bfb32 (acquisition bottleneck separation frontier sweep) — primary
- c29a038d (frontier-selected downstream validation) — downstream metrics
- 0e2af247 (frontier-selected cross-backbone validation) — generalization
- 1c527697 (unified separation scoreboard) — consolidated

### Primary Result Files
- `results/paired_acquisition_factorization_acquisition_bottleneck_separation_frontier/frontier_variant_summary.csv`
- `results/paired_acquisition_factorization_frontier_selected_downstream_validation/frontier_downstream_summary.csv`
- `results/paired_acquisition_factorization_frontier_selected_crossbackbone_validation/frontier_crossbackbone_summary.csv`

### Strongest Baseline
true_pair_acquisition (64D, no bottleneck) — the pre-bottleneck acquisition branch. The bottleneck improvement is measured relative to this starting point.

### Limitation
Only two dimensions (8, 16) with two regularization strengths (default, stronger_xcov) were tested at full scale. The frontier is sparse — we cannot claim a continuous Pareto front or optimal bottleneck size. Cross-backbone evidence for SCORPION uses tissue-retrieval leakage (not category leakage) because SCORPION lacks biological labels. The improvement in SCORPION is measured as reduced paired-region retrieval in the acquisition branch, which is a weaker claim than reduced category leakage. Very aggressive bottlenecking (dim < 8) was not tested and may collapse scanner capture.

### Forbidden Overclaim
- "Bottlenecking eliminates biological leakage" — acq_dim8_default still has category probe 0.1598 in canine SCC.
- "The bottleneck is optimal at dimension 8" — only 8 and 16 were tested at full scale; the true optimum is unknown.
- "Bottlenecking solves the separation problem" — residual leakage remains, and biological-branch scanner leakage is unchanged (~0.36).
- "SCORPION bottleneck reduces category leakage" — false; SCORPION has no category labels. It reduces tissue/pair-retrieval leakage.

### Main or Appendix
**Main.** The bottleneck frontier is the primary empirical contribution beyond the baseline separation.

### Figure/Table Candidate
Frontier plot: acquisition-branch category leakage vs scanner capture for true_pair, acq_dim8, acq_dim16. Three-panel cross-backbone figure: SCORPION DINOv2/Phikon/ResNet50 with tissue-retrieval leakage bars.

### Reviewer Objection Answered
- "Doesn't reducing capacity just weaken the acquisition branch uniformly, not selectively reduce biological leakage?" — Addressed: scanner capture is preserved (0.8643 vs 0.8651) while category leakage drops (0.3456 → 0.1598). The bottleneck selectively constrains biological information because scanner-discriminative information requires lower capacity than the full tissue-category manifold.
- "Is 8 dimensions enough to capture all scanner variation?" — Addressed: scanner probe on the acquisition branch remains at 0.8643, nearly identical to the 64D baseline (0.8651). Scanner capture is saturated; the bottleneck removes excess capacity that was encoding biological information.
- "Does this generalize or is it specific to canine SCC DINOv2?" — Addressed by cross-backbone validation across three SCORPION backbones.

### Remaining Objection
No direct causal evidence that the capacity reduction mechanism (bottleneck dimension) is the only driver of reduced leakage. Stronger cross-covariance regularization (0.05 → 0.20) also contributes. Cannot isolate bottleneck-dimension effect from regularization-strength effect with only two selected variants.

---

## CLAIM_5_FACTOR_LIKE_SWAPPING

### Claim Title
Decoder-based acquisition swapping supports factor-like behavior; bottlenecking improves biological/category preservation under swapped acquisition factors.

### Claim Text (safe)
When the acquisition branch from one sample is combined with the biological branch from another sample via the learned decoder, scanner identity follows the acquisition branch: the scanner follow rate (probe prediction matches acq-source scanner) averages 0.855 across all variants and swap types. Category preservation (probe prediction matches bio-source category) averages ~0.40, with bottlenecked variants showing slightly lower acquisition-branch category leakage under swap. Branch-space nearest-neighbor purity confirms the separation: bio-space K1 category purity averages 0.980; acq-space K1 scanner purity averages 0.880. Decoder-reconstructed features show scanner-follow behavior (0.806 average, 0.901 for same-sample/different-scanner swaps) and category preservation (0.978 for bottlenecked variants). This is factor-like behavior — scanner information follows the acquisition branch through recombination — but does not prove perfect causal factorization. Nearest-neighbor scanner alignment in acquisition space (0.880) is mixed relative to biological-space category alignment (0.980).

### Strongest Evidence
Acquisition factor swapping audit (aa8d0596): scanner follow rate 0.855 across all variants/swap types. Bottleneck variants show lower acq_category_leakage (acq_dim16_stronger_xcov 0.283, acq_dim8_default 0.287 vs true_pair 0.296). Decoder-space: scanner follow 0.806 overall (0.901 for same-sample swap type A), category preservation 0.978+ for bottlenecked variants. Bio-space K1 category purity 0.980 vs acq-space K1 scanner purity 0.880.

### Supporting Commits
- aa8d0596 (acquisition factor swapping audit) — sole source

### Primary Result Files
- `results/paired_acquisition_factorization_acquisition_factor_swapping_audit/acquisition_swapping_summary.csv`
- `results/paired_acquisition_factorization_acquisition_factor_swapping_audit/acquisition_swapping_nearest_neighbor_metrics.csv`
- `results/paired_acquisition_factorization_acquisition_factor_swapping_audit/acquisition_swapping_probe_metrics.csv`
- `results/paired_acquisition_factorization_acquisition_factor_swapping_audit/acquisition_swapping_report.md`

### Strongest Baseline
true_pair (64D, no bottleneck) — the baseline acquisition branch against which bottlenecked-swap improvements are measured. Same-sample/different-scanner swap (Type A) is the strongest test: biological content is preserved (same tissue region), only the acquisition factor changes.

### Limitation
Single-dataset evidence (canine SCC DINOv2 only). No SCORPION or cross-backbone swapping because SCORPION lacks biological labels for category preservation measurement. Nearest-neighbor scanner alignment (0.880) is notably weaker than category alignment (0.980) — scanner purity in acquisition space is decent but not as clean as category purity in biological space. Decoder-space results rely on the learned decoder, which was trained for reconstruction, not factor-level manipulation — recombination artifacts may exist. Swap types B/C/D mix category and scanner changes, making interpretation of lowered category preservation ambiguous. Only three variant configurations tested (true_pair, acq_dim8_default, acq_dim16_stronger_xcov). Swap is at feature-representation level, not image-pixel level.

### Forbidden Overclaim
- "Proves perfect causal acquisition factor" — factor-like behavior, not proof. Nearest-neighbor scanner purity is 0.880, not 1.0.
- "Proves biological and acquisition factors are independent" — category preservation under swap is ~0.40, substantially below the non-swapped baseline.
- "Enables acquisition factor editing for deployment" — research audit only; not deployment-ready.
- "Scanner information is fully encoded in the acquisition branch" — bio_scanner_leakage remains 0.032–0.213 depending on swap type.
- "Works across all scanners and domains" — single dataset, 5 scanners.

### Main or Appendix
**Main.** This is the most direct evidence that the acquisition branch carries manipulable acquisition information — factor-like behavior is the strongest interpretability claim. But the mixed NN scanner alignment and single-dataset limitation make it the weakest of the five claims in terms of robustness.

### Figure/Table Candidate
Swap-type diagram showing the four swap constructions. Bar chart: scanner follow rate and category preservation rate by variant and swap type. Table: branch-space NN purity by variant and swap type.

### Reviewer Objection Answered
- "Isn't this just the acquisition branch acting as a domain classifier residual, not a factor?" — Partially addressed by the decoder recombination step: if the acquisition branch were only a discriminative residual, recombining it with a different biological branch via the decoder would not produce features whose scanner identity follows the acquisition source. The decoder-space scanner follow rate (0.806–0.901) supports factor-like recombination.
- "Is the decoder just memorizing training-scanner statistics?" — The swap test uses held-out test samples from all 5 scanners. The same-sample/different-scanner swap (Type A) keeps the biological content identical and only changes the acquisition branch, producing the strongest scanner-follow signal (0.901 in decoder space).

### Remaining Objection
The biggest weakness: nearest-neighbor scanner purity in acquisition space (0.880) is weaker than category purity in biological space (0.980). Scanner information is clearly present in the acquisition branch, but it does not dominate the acquisition branch's nearest-neighbor structure to the same degree that category dominates the biological branch. The acquisition branch may carry additional nuisance variation beyond scanner identity. No SCORPION or cross-backbone swap evidence. Single-dataset, single-backbone. The decoder was trained for reconstruction, not factor manipulation — we are repurposing it post-hoc.

---

## Cross-Claim Consistency Checks

1. **No claim asserts paired-acquisition beats oldstyle on raw scanner removal.** CLAIM_3 explicitly prevents this.
2. **No claim asserts perfect separation or factorization.** All claims use "measurable," "partial," "substantially reduces," "factor-like."
3. **SCORPION evidence is correctly scoped.** CLAIM_4 and CLAIM_1 note that SCORPION uses tissue/retrieval metrics, not category labels.
4. **Canine SCC DINOv2 is the labeled-category anchor.** All category-probe claims rest on this single dataset.
5. **No clinical, diagnostic, or deployment language appears in any claim.**
6. **Every claim has at least one explicit limitation and at least one forbidden overclaim.**
7. **Every claim references at least one commit and at least one result file.**

---

## Main vs Appendix Recommendation

| Claim | Recommendation | Rationale |
|---|---|---|
| CLAIM_1_PAIR_STRUCTURE | Main | Foundational validity; without it the method has no mechanism |
| CLAIM_2_BRANCH_SEPARATION | Main | Core contribution — the method's reason to exist |
| CLAIM_3_LINEAR_BASELINE_BOUNDARY | Main | Self-imposed boundary that frames all other claims |
| CLAIM_4_BOTTLENECK_FRONTIER | Main | Primary empirical contribution beyond baseline separation |
| CLAIM_5_FACTOR_LIKE_SWAPPING | Main | Strongest interpretability claim; single-dataset limitation acknowledged |

All five claims are recommended for main paper. The appendix should carry: detailed pairing-ladder tables, per-scanner breakdowns, per-class recall tables, scanner-confounded robustness details, sample-disjoint transfer details, and the full unified scoreboard.

---

## Biggest Remaining Weakness

The single biggest weakness across the entire claim ledger is **single-dataset, single-backbone anchoring of all category-label claims**. CLAIM_2 (branch separation), CLAIM_4 (bottleneck category-leakage reduction), and CLAIM_5 (factor-like swapping) all rest on canine SCC DINOv2 as the only dataset with biological category labels. SCORPION provides cross-backbone evidence (DINOv2/Phikon/ResNet50) but only for tissue/pair-retrieval metrics, not category labels. The method's category-preservation claim generalizes across backbones for pair retrieval but not across biological classification tasks. A second labeled multi-scanner dataset with different tissue categories would substantially strengthen CLAIM_2, CLAIM_4, and CLAIM_5.

Secondary weakness: nearest-neighbor scanner purity in acquisition space (0.880) is weaker than category purity in biological space (0.980) — the acquisition branch carries scanner information but it does not dominate the branch's neighborhood structure to the same degree.

---

## Validation Checklist

- [x] Every claim references at least one commit/result source
- [x] Every limitation is explicit
- [x] Forbidden wording list exists (see `wording_boundaries.md`)
- [x] No clinical/deployment/diagnostic language
- [x] No previous result files modified
- [x] Branch created from main: `experiment/claim-ledger-and-paper-skeleton`
- [x] All five required files created in `paper/claim_ledger/`
