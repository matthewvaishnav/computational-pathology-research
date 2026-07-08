# Reviewer Risk Register — Paired-Acquisition Manuscript

**Branch:** paper/paired-acquisition-manuscript-integration
**Generated:** 2026-07-08
**Purpose:** Every reviewer objection anticipated, addressed, and the residual risk assessed. Nothing hidden.

---

## Risk 1: Centroid/QR Is Simpler and Better at Raw Scanner Removal

**Severity:** CRITICAL
**Probability:** Certain — every reviewer will ask this.

**Objection:**
"Oldstyle centroid/QR projection achieves scanner probe 0.200 (chance) while preserving category accuracy 0.400. Your biological branch only achieves scanner probe 0.361 with category accuracy 0.386. The linear baseline is simpler, faster, and better at raw scanner removal. Why should anyone use paired-acquisition?"

**Response:**
We agree. Oldstyle centroid/QR is the stronger raw scanner-removal method. We state this in the abstract, introduction, and Section 6. The contribution is not best raw scanner erasure — it is structured separation.

Centroid/QR removes scanner-centroid directions from the embedding. It does not produce an explicit acquisition branch. You cannot inspect what scanner information was removed, you cannot bottleneck the removed component to reduce biological leakage, and you cannot swap it through a decoder to test factor-like behavior.

If raw scanner removal is the only goal, use oldstyle_keep_k4. If you need an explicit, inspectable, bottleneckable, and swappable acquisition representation, paired-acquisition provides complementary capability. The two methods solve different problems.

**Mitigation in manuscript:**
- Abstract and introduction answer this question immediately (not deferred to Section 6).
- Section 6 (Result 3) is titled honestly: "Oldstyle Centroid/QR Is the Strongest Raw Scanner-Removal Baseline."
- Figure 3 (baseline scoreboard) highlights oldstyle_keep_k4 as the best raw removal.
- The thesis statement in the abstract explicitly says "not best-in-class raw scanner removal."

**Residual risk:** MEDIUM. A reviewer who only cares about scanner removal may still find the method over-engineered. The structured-separation framing must be compelling enough to justify the added complexity. The bottleneck and swapping results (Sections 7-8) carry this burden.

---

## Risk 2: Bottleneck Comparison Is Sparse

**Severity:** HIGH
**Probability:** High — quantitative reviewers will notice.

**Objection:**
"You tested two bottleneck dimensions (8, 16) with two regularization strengths at full scale — four data points. This is not a 'frontier.' You cannot claim Pareto optimality or optimal bottleneck size from four points. This is a sparse grid search."

**Response:**
We agree. The manuscript does not claim a densely mapped Pareto frontier, Pareto optimality, or an optimal bottleneck size. We describe the experiment as a "bottleneck comparison" or "capacity-constrained separation audit," and the result as a "directional separation-frontier improvement."

The evidence supports a directional claim: bottlenecking reduces biological leakage while preserving scanner capture. The four full-scale variants consistently support this direction. The six smoke variants provide additional supporting evidence. But the comparison is sparse — we cannot claim a continuous frontier or optimal dimension.

**Mitigation in manuscript:**
- "Frontier sweep" replaced with "bottleneck comparison" throughout.
- Figure 4 labeled as "Directional separation-frontier improvement" (not "Pareto front").
- Section 7 (Result 4) explicitly states the sparsity limitation.
- Limitations section (Section 9) reiterates: "Cannot claim continuous Pareto front or optimal bottleneck size."

**Residual risk:** MEDIUM. The word "frontier" still appears in commit messages and some file paths (historical, cannot change). The directional claim is defensible, but a reviewer may still ask for a denser sweep (dimensions 2, 4, 8, 16, 32, 64) to map the actual tradeoff curve.

---

## Risk 3: Swapping Evidence Is Single-Dataset, Single-Backbone

**Severity:** HIGH
**Probability:** High — this is the most vulnerable claim.

**Objection:**
"The acquisition swapping experiment uses only canine SCC DINOv2 — one dataset, one backbone. There is no SCORPION swapping, no cross-backbone swapping, and no cross-dataset validation. You cannot claim factor-like behavior from a single experiment on five scanners. Category preservation under swap is only ~0.40, and nearest-neighbor scanner purity is 0.880 vs category purity 0.980 — the acquisition branch is clearly noisier."

**Response:**
We agree with all of these points. CLAIM_5 is explicitly identified as the weakest of the five claims. The limitations are stated directly in the claim text, in Section 8 (Result 5), and in Section 9 (Limitations).

The evidence supports "factor-like" but not "factor proven." The decoder-space results (scanner follow 0.901 for Type A swaps, category preservation 0.978+ in bottlenecked variants) are stronger than the branch-space results (NN scanner purity 0.880). Both are shown — the manuscript does not hide the weaker evidence.

The single-dataset limitation is acknowledged as the single biggest remaining weakness of the entire paper. A second labeled multi-scanner dataset would be needed for stronger factor claims.

**Mitigation in manuscript:**
- CLAIM_5 labeled as weakest in the claim ledger and manuscript.
- Figure 5 REQUIRES both strong (decoder-space) and weak (branch-space NN purity) evidence side by side.
- Section 8 states: "This is factor-like behavior, not proof of perfect causal factorization."
- Section 9 lists single-dataset evidence as limitation #1.
- The claim ledger stress test explicitly flags this as the weakest claim.

**Residual risk:** HIGH. This is genuinely the weakest claim. A reviewer who demands cross-dataset swapping evidence has a valid objection that cannot be fully answered with current data. The defense is honesty: we show the weak evidence alongside the strong evidence, we state the limitation, and we do not overclaim.

---

## Risk 4: SCORPION Lacks Biological Category Labels

**Severity:** MEDIUM
**Probability:** Medium — reviewers familiar with SCORPION will know this.

**Objection:**
"SCORPION has no tissue-category labels. You cannot claim 'category leakage reduction' or 'biological label preservation' from SCORPION data. Tissue-retrieval metrics are not a substitute for category-label evidence."

**Response:**
We agree. All SCORPION claims are bounded to "tissue/pair-retrieval" metrics, never "category leakage" or "category preservation." The distinction is made explicitly in every section where SCORPION results appear.

SCORPION provides cross-backbone evidence that the bottleneck reduces tissue-retrieval leakage in the acquisition branch — a weaker but still informative claim. The consistency across three backbones (DINOv2, Phikon, ResNet50) supports generalization of the bottleneck mechanism.

Category-label claims (CLAIM_2 branch separation, CLAIM_4 category leakage reduction, CLAIM_5 factor-like swapping) are restricted to canine SCC DINOv2, which has expert-annotated tissue categories.

**Mitigation in manuscript:**
- Every SCORPION figure and table uses "tissue/pair-retrieval" language, not "category."
- Section 2 (Problem Setup) documents SCORPION's lack of category labels.
- Section 9 (Limitations) lists single-dataset anchoring as limitation #1.
- Wording boundaries file lists SCORPION category claims as forbidden.

**Residual risk:** LOW. The scoping is clear and consistent. A reviewer may still want a second labeled dataset, but they cannot claim we misrepresented SCORPION evidence.

---

## Risk 5: Scanner Signal Remains in the Biological Branch

**Severity:** MEDIUM
**Probability:** Medium — reviewers will notice the residual scanner probe.

**Objection:**
"The biological branch scanner probe is 0.361, which is 16 points above chance (0.20 for 5 balanced classes). You have reduced scanner signal but not removed it. This is partial noise reduction, not scanner removal. Calling this 'separation' when the biological branch still encodes substantial scanner information is misleading."

**Response:**
We never claim the biological branch is scanner-free. We claim "measurable separation" and "partial separation." The scanner probe reduction is δ = +0.505 (0.866 → 0.361), which is substantial. The category/scanner ratio more than doubles (0.47 → 1.07). Neighborhood purity improves despite the residual scanner signal.

The residual scanner signal (0.361) is a limitation we state explicitly. Perfect scanner removal is not the claim — structured separation with an explicit acquisition branch is. The biological branch is not scanner-free; it is scanner-suppressed relative to frozen features, while preserving category structure.

**Mitigation in manuscript:**
- Claim text uses "substantially reduces," never "removes" or "eliminates."
- Section 5 (Result 2) states residual scanner signal explicitly.
- Section 9 (Limitations) lists partial separation as limitation #2.
- Forbidden wording list includes "scanner-free biological branch."
- Figure 2 (scanner vs category scatter) shows the residual position clearly.

**Residual risk:** LOW-MEDIUM. The residual scanner signal is honestly reported. A reviewer who demands below-chance scanner probe may not be satisfied, but the claim is correctly scoped as "partial separation, measured."

---

## Risk 6: No Nonlinear or Learned Baseline Comparison

**Severity:** LOW-MEDIUM
**Probability:** Low-Medium — domain-adversarial baselines are the obvious comparator.

**Objection:**
"You compare against linear centroid projection and PCA. What about domain-adversarial neural networks, gradient-reversal feature learning, or optimal transport-based alignment? A learned nonlinear baseline might achieve both strong scanner removal and an explicit acquisition representation."

**Response:**
True. We do not compare against nonlinear learned baselines. The oldstyle centroid/QR baseline is the strongest linear baseline we are aware of, but a domain-adversarial feature learner could potentially achieve better scanner removal or produce its own acquisition-like representation. This is a genuine gap.

We chose linear baselines because they are simple, reproducible, and represent the most direct comparison for "raw scanner removal." A nonlinear baseline comparison would be valuable but is outside the scope of this audit.

**Mitigation in manuscript:**
- Section 9 (Limitations) lists "linear baselines only" as limitation #4.
- The claim ledger acknowledges this as a remaining objection for CLAIM_3.

**Residual risk:** LOW. This is a legitimate scope limitation, not an error. A reviewer may suggest adding a domain-adversarial baseline as future work, which we would accept.

---

## Risk 7: Overall Contribution Might Be Seen as Incremental

**Severity:** MEDIUM
**Probability:** Medium — depends on reviewer perspective.

**Objection:**
"Paired-acquisition factorization is a straightforward application of paired reconstruction + gradient reversal + independence regularization. The bottleneck finding (reducing capacity reduces leakage) is unsurprising. The swapping experiment is a standard latent-variable-manipulation test. What is the novel contribution beyond combining known techniques?"

**Response:**
The contribution is not a single novel technique but a structured empirical characterization of what paired-acquisition factorization can and cannot do:

1. We establish that true paired structure is required (Claim 1) — a precondition that bounds data collection requirements.
2. We characterize the partial separation achievable (Claim 2) with quantitative leakage measurements.
3. We establish the strongest linear baseline and honestly concede it wins raw removal (Claim 3) — framing the contribution as structured separation, not best erasure.
4. We show that bottlenecking selectively reduces biological leakage while preserving scanner capture (Claim 4) — a directional improvement on an explicit tradeoff.
5. We provide factor-like swapping evidence with honest disclosure of mixed NN results (Claim 5).

The novelty is in the systematic audit, the honest baseline boundary, and the explicit characterization of what the method can and cannot do. This is a contribution of measurement and bounding, not of architecture.

**Mitigation in manuscript:**
- Introduction frames the contribution as structured separation, not a novel architecture.
- The entire paper is organized around questions, not techniques.
- The baseline boundary (Claim 3) demonstrates scientific honesty rather than overclaiming.

**Residual risk:** MEDIUM. A reviewer looking for architectural novelty may be unsatisfied. The contribution must be pitched as empirical characterization, not method invention. The paper's value is in what it measures and bounds, not in what it builds.

---

## Risk 8: p1000 Improvement Might Be a Fluke

**Severity:** LOW
**Probability:** Low — but worth flagging.

**Objection:**
"The biological branch improves cross-scanner transfer on the p1000 scanner (+0.045 over frozen features) but degrades slightly on the other four scanners (−0.015 to −0.058). The mean improvement is negative. Are you cherry-picking the one scanner where it works?"

**Response:**
We report the mean transfer accuracy (0.827 vs 0.845, Δ = −0.018) as the primary result, not the per-scanner breakdown. The p1000 improvement is noted as interesting but not the main finding. The main finding is that biological-branch transfer nearly preserves frozen-feature transfer with a small average decrease, while the acquisition branch and shuffled controls collapse.

The per-scanner breakdown is in Appendix C, not hidden. The p1000 result is flagged as the hardest scanner overall — frozen features achieve only 0.709 on p1000, the lowest of any scanner. The biological branch's improvement on the hardest case is consistent with scanner suppression helping where scanner confounding is strongest.

**Mitigation in manuscript:**
- Mean transfer accuracy is the primary reported metric.
- Per-scanner detail is in appendix, not main text.
- Scanner-heldout section states that biological transfer "nearly preserves" frozen transfer, not "improves."

**Residual risk:** LOW. The claim is correctly scoped. Per-scanner detail is appendix-level.

---

## Risk Summary

| Risk | Severity | Mitigation | Residual |
|---|---|---|---|
| 1. Centroid/QR is better | CRITICAL | Answered in abstract/intro; framed as complementary | MEDIUM |
| 2. Sparse bottleneck comparison | HIGH | "Directional improvement," not "frontier"; limitation explicit | MEDIUM |
| 3. Single-dataset swapping | HIGH | Weakest claim acknowledged; dual evidence shown; limitation explicit | HIGH |
| 4. SCORPION lacks labels | MEDIUM | "Tissue/pair-retrieval" language; scoping consistent | LOW |
| 5. Residual scanner in bio branch | MEDIUM | "Partial separation" framing; residual stated explicitly | LOW-MEDIUM |
| 6. No nonlinear baselines | LOW-MEDIUM | Listed as limitation; scope boundary | LOW |
| 7. Incremental contribution | MEDIUM | Pitched as empirical characterization, not novel architecture | MEDIUM |
| 8. p1000 cherry-picking | LOW | Mean result primary; per-scanner in appendix | LOW |

**Highest residual risk:** Risk 3 (single-dataset swapping). The only fix is a second labeled multi-scanner dataset — which would be a new experiment, not a manuscript revision.
