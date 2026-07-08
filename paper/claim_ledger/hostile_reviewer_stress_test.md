# Hostile Reviewer Stress Test — Paired-Acquisition Claim Ledger

**Branch:** experiment/claim-ledger-and-paper-skeleton
**Generated:** 2026-07-08
**Purpose:** Claim-by-claim adversarial review: can every main-paper claim be defended by one primary table/figure, one strongest baseline, one limitation sentence, and one appendix trail?

---

## CLAIM_1_PAIR_STRUCTURE

### Primary table/figure defense
**PASS.** Pairing ladder table showing paired cosine, top-1 retrieval, and scanner probe across 5 pairing levels × 2 datasets. Single main-text table.

### Strongest baseline
**PASS.** True same-region pairs (L0) in the pairing ladder. Scanner-balanced random pairing (L3) performs no better than fully random (L4), killing the "batch balancing alone is enough" objection.

### One limitation sentence
**PASS.** "The pairing ladder only controls pair construction, not training dynamics; cross-backbone evidence uses tissue-retrieval metrics, not category labels — SCORPION lacks biological category annotations."

### Appendix trail
**PASS.** Appendix A (pairing ladder detail: per-condition, per-metric tables; level-vs-level contrasts; cross-backbone extension). Appendix I (acquisition branch audit: branch separation contrasts).

### Hostile reviewer objection
"Of course true pairs work better — you gave the model more information. This isn't a finding, it's a precondition check."

### Rebuttal
It IS a precondition check — and that's the point. The finding is not "pairs matter" but *how much* they matter: the paired cosine gap from L0 to the non-true band is 0.0706 in SCORPION, 0.1451 in canine SCC. Scanner balancing alone (L3) does not recover tissue identity — L3 ≈ L4 in both datasets. This bounds how strictly paired acquisition must be collected. Without this experiment, reviewers would ask: "Couldn't you just use same-scanner pairs?" Answer: no, scanner balancing alone doesn't work.

### Verdict
**Defensible.** A necessary precondition check that doubles as a mechanism validity result. The cross-backbone extension (d018c924) strengthens generalization.

---

## CLAIM_2_BRANCH_SEPARATION

### Primary table/figure defense
**PASS.** Two-panel figure: (a) scanner probe vs category probe scatter for all representations, (b) biological vs acquisition branch purity bar chart. Central result table with scanner probe, category probe, category F1, purity K1/K5, heldout transfer accuracy for key representations.

### Strongest baseline
**PASS.** Original frozen DINOv2 features (pre-factorization state — scanner probe 0.866, category probe 0.407). Shuffled-sample biological branch (broken-pair control — scanner probe 0.409, category probe 0.324) showing that broken-pair factorization cannot achieve the same separation.

### One limitation sentence
**PASS.** "Branch separation is measured via linear probe accuracies and neighborhood purity — diagnostic metrics, not proof of complete information-theoretic independence; separation is partial, not perfect; all evidence from canine SCC DINOv2 only; SCORPION lacks category labels."

### Appendix trail
**PASS.** Appendix B (biological label preservation detail: per-representation per-class recall, PCA k-sweep, linear scanner subspace k-sweep, neighborhood purity k=1,5,10). Appendix C (scanner-heldout transfer detail: per-scanner, per-class). Appendix D (scanner-confounded robustness detail).

### Hostile reviewer objection
"Scanner probe 0.361 is still 16 points above chance. You reduced scanner signal but didn't remove it. Category probe dropped slightly too (0.407 → 0.386). This is a tradeoff, not a solution."

### Rebuttal
We never claim it's scanner-free. We claim *measurable* separation. The category/scanner ratio goes from 0.47 (original) to 1.07 (biological branch) — more than doubled. Neighborhood purity *improves* (0.954 → 0.973). The category drop is 0.0213 vs the scanner drop of 0.5046 — a 24:1 ratio in the right direction. This is partial but directional separation, clearly quantified.

### Hostile reviewer objection #2
"PCA K32 has purity 0.965 and effective rank 149. You've just done fancier dimensionality reduction."

### Rebuttal
PCA K32 scanner probe is 0.649 (vs our 0.361) and category probe is 0.289 (vs our 0.386). PCA is worse on both axes simultaneously. PCA *increases* effective rank (149.3); factorization reduces it in both branches (biological 74.0, acquisition 13.8). PCA is blind to the scanner/category distinction — that's exactly why paired supervision matters.

### Verdict
**Defensible.** The 24:1 scanner-suppression-to-category-loss ratio is the strongest single number in the paper. PCA comparison is decisive. But the absence of an information-theoretic independence metric (mutual information, HSIC) is a genuine gap.

---

## CLAIM_3_LINEAR_BASELINE_BOUNDARY

### Primary table/figure defense
**PASS.** Scoreboard summary table showing scanner probe, category probe, scanner capture for oldstyle_keep_k4, true_pair_biological, true_pair_acquisition, and bottlenecked variants. Single main-text table.

### Strongest baseline
**PASS.** oldstyle_keep_k4 (centroid/QR) — scanner probe 0.200 (chance), category probe 0.400. The reference baseline that paired-acquisition must be measured against in all scanner-removal comparisons.

### One limitation sentence
**PASS.** "The oldstyle baseline is a post-hoc linear operation on frozen DINOv2 embeddings; it does not produce an explicit acquisition branch, cannot be inspected for what scanner information was removed, and cannot be used for acquisition swapping — the two methods solve different problems (blind removal vs structured separation)."

### Appendix trail
**PASS.** Appendix E (linear baseline detail: consistency audit reconciliation, oldstyle k=1,2,3,4 full metrics, logistic-SVD residual audit). Appendix H (unified scoreboard: full 12-row table).

### Hostile reviewer objection
"If centroid/QR is simpler, faster, and better at scanner removal, why should anyone use your method? You just proved your method is worse."

### Rebuttal
They solve different problems. Centroid/QR removes scanner-centroid directions and produces a cleaned embedding. Paired-acquisition produces two explicit branches with an interpretable acquisition representation. You can bottleneck it to reduce biological leakage (Claim 4), swap it through a decoder (Claim 5), and measure what scanner information looks like decoupled from biology. Centroid/QR cannot do any of that. If you only want scanner removal, use centroid/QR. If you want an explicit acquisition factor you can manipulate, use paired-acquisition.

### Hostile reviewer objection #2
"Isn't this just moving the goalposts? You built a scanner-removal method, found it loses to a linear baseline, and now you're saying 'actually the contribution was structured separation all along.'"

### Rebuttal
The consistency audit (a325c009) documents that we *found* the discrepancy between our initial logistic-SVD baseline and the stronger centroid/QR baseline, and *upgraded* to the strongest baseline rather than hiding the weaker result. This is scientific honesty, not goalpost-moving. The structured-separation framing emerged from the evidence, not despite it. The alternative — hiding the oldstyle baseline and claiming best scanner removal — would be the dishonest move.

### Verdict
**Defensible.** The self-imposed baseline boundary is the paper's strongest honesty signal. But the "why use your method?" question must be answered in the introduction and abstract, not deferred to Section 7. A reviewer who reaches Section 7 without being told the answer will already be drafting their "over-engineered linear baseline" comment.

---

## CLAIM_4_BOTTLENECK_FRONTIER

### Primary table/figure defense
**PASS.** Comparison plot: acquisition-branch category leakage vs scanner capture for true_pair, acq_dim8, acq_dim16. Three-panel cross-backbone SCORPION figure showing tissue-retrieval leakage for DINOv2, Phikon, ResNet50.

### Strongest baseline
**PASS.** true_pair_acquisition (64D, no bottleneck) — the pre-bottleneck acquisition branch against which improvement is measured.

### One limitation sentence
**PASS.** "Only two bottleneck dimensions (8, 16) with two regularization strengths (default, stronger_xcov) were tested at full scale — the comparison is sparse; we cannot claim a continuous Pareto front or optimal bottleneck size; cross-backbone SCORPION evidence uses tissue-retrieval leakage, not category leakage, because SCORPION lacks biological labels."

### Appendix trail
**PASS.** Appendix F (bottleneck comparison detail: smoke and full variant metrics, per-scanner downstream detail, cross-backbone raw metrics).

### Hostile reviewer objection
"Two dimensions and two regularization strengths is four data points at full scale. You cannot call this a 'frontier sweep.' This is a sparse grid search with n=4 full variants."

### Rebuttal
Fair. Six smoke variants were tested, two were promoted to full scale. The claim text says "improves the separation frontier" — a directional claim — not "maps a dense Pareto frontier." But "frontier sweep" in the commit message and report title overstates the density. The manuscript should use "bottleneck comparison" or "capacity-constrained separation audit" and describe the improvement as a "directional separation-frontier improvement," reserving "frontier" for the tradeoff direction, not the sweep density.

### Hostile reviewer objection #2
"The SCORPION cross-backbone evidence uses tissue-retrieval leakage, not category leakage. You cannot claim 'reduced biological leakage' from retrieval metrics alone."

### Rebuttal
Agreed. We bound SCORPION claims to "tissue/pair-retrieval leakage," never "category leakage." The claim text explicitly distinguishes canine SCC (category leakage) from SCORPION (tissue-retrieval leakage). The consistency across three backbones is still evidence — just weaker evidence, correctly scoped.

### Verdict
**Defensible, but language must be softened.** "Frontier sweep" → "bottleneck comparison." "Frontier improvement" → "directional separation-frontier improvement." The sparse full-scale grid (4 points) cannot support frontier-mapping language.

---

## CLAIM_5_FACTOR_LIKE_SWAPPING

### Primary table/figure defense
**PASS.** Swap-type construction diagram (Types A–D). Bar chart showing scanner follow rate and category preservation rate by variant and swap type. Branch-space NN purity table by variant and swap type.

### Strongest baseline
**PASS.** true_pair (64D, no bottleneck). Same-sample/different-scanner swap (Type A) is the cleanest test — biological content identical, only acquisition branch changes.

### One limitation sentence
**PASS.** "Single-dataset evidence (canine SCC DINOv2 only); no SCORPION or cross-backbone swapping; nearest-neighbor scanner purity in acquisition space (0.880) is notably weaker than category purity in biological space (0.980); decoder was trained for reconstruction, not factor manipulation; swap is at feature-representation level, not image-pixel level."

### Appendix trail
**PASS.** Appendix G (acquisition swapping detail: per-swap-type probe metrics, per-variant NN purity tables, decoder-space reconstruction metrics).

### Hostile reviewer objection
"Scanner follow rate 0.855 means 14.5% of the time scanner does NOT follow the acquisition branch. Category preservation 0.40 means 60% of the time category is NOT preserved under swap. NN scanner purity 0.880 vs category purity 0.980 — the acquisition branch is clearly noisier. This is weak factor evidence."

### Rebuttal
The branch-space metrics are mixed — we state this explicitly. But the decoder-space results are stronger: scanner follow 0.901 for same-sample swaps (Type A, the cleanest test), and category preservation 0.978+ in bottlenecked variants. The decoder recombination step is the conceptual key — if the acquisition branch were purely a discriminative residual, recombining it with a different biological branch via the decoder would not produce features whose scanner identity tracks the acquisition source. The fact that it does, especially in the clean Type A swap, is factor-like evidence. We explicitly do not claim proof of perfect causal factorization. We note that NN scanner purity (0.880) is weaker than NN category purity (0.980) and show both results.

### Hostile reviewer objection #2
"Single dataset, single backbone, single decoder. You can't claim factor-like behavior from one experiment on five scanners."

### Rebuttal
This is the claim's single biggest weakness and we state it as the primary limitation in both the claim ledger and the manuscript skeleton. CLAIM_5 is the least robust of the five claims. The evidence supports "factor-like" but not "factor proven." If a reviewer demands cross-dataset swapping, we would need a second labeled multi-scanner dataset — which we acknowledge as the single biggest remaining weakness of the entire paper (see claim ledger Section "Biggest Remaining Weakness").

### Verdict
**Defensible but the weakest of the five claims.** The main-paper placement is correct (it's the strongest interpretability claim), but:
- Both the strong decoder-space signal (0.901 scanner follow) AND the weaker branch-space NN scanner purity (0.880) must appear in the main-text figure — not hide the weak result in appendix.
- "Factor-like" must be defended as "scanner information follows the acquisition branch through decoder recombination," not "the acquisition branch is a clean causal factor."
- Single-dataset limitation must be stated in the main-text claim section, not only in Section 10 (Limitations).

---

## Cross-Claim Coherence

### Main-paper claims only (without appendix detail)

Reading only the five claim titles and one-sentence summaries:

1. **Pair structure matters** — true same-region pairs are required for tissue preservation.
2. **Branch separation is measurable** — biological branch preserves category, acquisition branch captures scanner.
3. **But we don't win raw scanner removal** — centroid/QR does. Our contribution is structured separation.
4. **We can improve the tradeoff** — bottleneck the acquisition branch, reduce biological leakage, keep scanner capture.
5. **The acquisition branch is manipulable** — swap it through the decoder and scanner information follows.

**The arc is coherent.** A reader who only reads main-text sections (no appendix) gets a clear story: we built a method that separates scanner from biology using paired supervision; it's not the best at raw scanner removal; its contribution is an explicit acquisition branch that can be bottlenecked and swapped.

---

## Three Issues Requiring Patches

### Issue 1: CLAIM_4 "frontier" language is overstated
**Severity:** Medium. Four full-scale data points (8D/16D × default/stronger_xcov) cannot support "frontier sweep" claims. Fix: replace with "bottleneck comparison" or "capacity-constrained separation audit." "Frontier" is only allowed as "directional separation-frontier improvement" — a directional claim, not a density claim.

### Issue 2: CLAIM_3 must be answered in introduction, not deferred to Section 7
**Severity:** High. The hostile reviewer's question — "if centroid/QR is better, why use your method?" — is the first question any reader will ask. The answer ("different problems: blind removal vs structured separation") must appear in the abstract and introduction. If deferred to Section 7, reviewers will draft their rejection before reaching the answer.

### Issue 3: CLAIM_5 must show both strong and weak swapping evidence in main text
**Severity:** High. If the main-text figure shows only decoder-space scanner follow (0.901) and hides branch-space NN scanner purity (0.880), a reviewer will find the weaker result in the appendix and accuse selective reporting. The main-text figure for CLAIM_5 must show both results side by side.

---

## Overall Assessment

The claim ledger survives hostile review. All five claims have:
- One primary table/figure (PASS × 5)
- One strongest baseline (PASS × 5)
- One limitation sentence (PASS × 5)
- One appendix trail (PASS × 5)
- Objection addressed with rebuttal (PASS × 5)

No claim makes clinical, diagnostic, or deployment assertions. No claim asserts perfect separation. The oldstyle baseline boundary is explicit. The SCORPION/category-label scope is correctly bounded. The single-dataset limitation is acknowledged.

**The three issues are presentational, not evidentiary.** They would be fixed during manuscript drafting — softening "frontier" language, front-loading the "why use this method" answer, and requiring dual evidence display for CLAIM_5. These patches are applied in this commit to the manuscript skeleton and wording boundaries.

**Ready to commit after patches applied.**
