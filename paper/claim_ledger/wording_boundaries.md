# Wording Boundaries — Paired-Acquisition Factorization

**Branch:** experiment/claim-ledger-and-paper-skeleton
**Generated:** 2026-07-08
**Purpose:** Explicit allowed and forbidden wording for all manuscript text, figures, and claim statements.

---

## Global Rules

### NEVER ALLOWED (anywhere in the paper)

#### Clinical / Diagnostic / Deployment
- "Clinical validation"
- "Clinical utility"
- "Clinical impact"
- "Patient care"
- "Patient outcome"
- "Diagnostic performance"
- "Diagnostic accuracy"
- "Diagnostic-grade"
- "Deployment-ready"
- "Production-ready"
- "Real-world deployment"
- "Translation to clinic"
- "Clinical workflow"
- "Pathologist replacement"
- "Computer-aided diagnosis" (implies clinical use)

#### Absolutist / Overclaim Language
- "Perfect" (applied to separation, factorization, disentanglement, independence)
- "Complete" (applied to separation, factorization, disentanglement, independence)
- "Solved" (applied to scanner bias, domain shift, batch effects)
- "Eliminates" (applied to scanner bias, biological leakage — use "reduces" or "substantially reduces")
- "Optimal" (applied to bottleneck dimension, architecture, hyperparameters — the frontier is sparsely sampled)
- "Best" (applied to scanner removal without qualification — oldstyle wins raw removal)
- "Proves" (applied to causal factorization, independence claims)
- "Guarantees" (any guarantee)
- "Universal" (applied to biological factorization, scanner generalization)
- "State-of-the-art" (on scanner removal — false; oldstyle wins)
- "First" / "Novel" (avoid priority claims; let reviewers decide)

#### Scanner Removal Supremacy
- "Paired-acquisition removes scanner bias better than any linear method" (false)
- "Paired-acquisition is the best scanner-removal approach" (false)
- "Paired-acquisition beats all baselines on scanner removal" (false)
- "Best scanner erasure" (oldstyle wins)
- "Strongest scanner suppression" (oldstyle wins)
- Any sentence comparing paired-acquisition to baselines on scanner removal that omits the oldstyle centroid/QR result

#### Branch Purity Absolutism
- "Scanner-free biological branch" (residual scanner probe 0.361)
- "Category-free acquisition branch" (residual category probe 0.346 in 64D)
- "Biological leakage eliminated" (residual 0.160 in acq_dim8_default)
- "No scanner information in biological branch"
- "No biological information in acquisition branch"

#### Category Label Scope (for SCORPION)
- "SCORPION category leakage" (SCORPION has no category labels)
- "Category preservation in SCORPION" (use "tissue-retrieval preservation")
- "Biological label transfer in SCORPION" (no labels)
- Only allowed: "tissue/pair-retrieval," "same-region retrieval," "tissue-identity preservation via paired retrieval"

#### Frontier / Bottleneck Language
- "Frontier sweep" (unqualified — implies dense Pareto mapping; the full-scale comparison has 4 variants)
- "Pareto front" or "Pareto optimal" (sparse data cannot support this)
- "Mapped the separation frontier" (use "directional separation-frontier improvement")
- "Dense frontier" or "continuous frontier" (only 2 dimensions × 2 regularization strengths tested at full scale)

#### Acquisition Swapping Absolutism
- "Perfect causal acquisition factor" (use "factor-like evidence")
- "Proves factorization" (use "supports factor-like behavior")
- "Scanner always follows acquisition" (follow rate is 0.855, not 1.0)
- "Proves independence of biological and acquisition factors" (category preservation under swap is ~0.40)
- "Scanner information is fully encoded in the acquisition branch" (bio_scanner_leakage remains 0.032–0.213)
- "Works across all scanners and domains" (single dataset, 5 scanners)
- "Enables acquisition factor editing for deployment" (research audit only; not deployment-ready)

---

### ALWAYS ALLOWED (safe vocabulary)

#### Contribution Framing
- "Structured separation"
- "Branch separation"
- "Measurable separation"
- "Partial separation"
- "Learned decomposition"
- "Explicit acquisition branch"
- "Inspectable acquisition representation"

#### Evidential Strength
- "Supports the interpretation that..."
- "Provides evidence for..."
- "Is consistent with..."
- "Suggests that..."
- "Factor-like behavior"
- "Substantially reduces"
- "Preserves"
- "Maintains"
- "Improves the separation frontier"

#### Claim Scoping
- "Under the tested conditions..."
- "In canine SCC DINOv2..."
- "In SCORPION [backbone]..."
- "Research audit"
- "Methodological contribution"

#### Frontier / Bottleneck (allowed)
- "Bottleneck comparison" (preferred term for the experiment)
- "Capacity-constrained separation audit" (alternative experiment name)
- "Directional separation-frontier improvement" (allowed with qualification)
- "Dimensionality-constrained acquisition branch"
- "Reduced-capacity acquisition branch"

#### Acquisition Swapping (allowed)
- "Factor-like evidence"
- "Factor-like behavior"
- "Probe-supported scanner following"
- "Mixed nearest-neighbor scanner alignment"
- "Decoder-based acquisition swapping"
- "Scanner information follows the acquisition branch through recombination"
- "Supports the interpretation of factor-like structure"
- "Controlled multi-scanner setting"
- "Feature-level evidence"

#### Baseline Honesty
- "Oldstyle centroid/QR projection is the strongest raw scanner-removal baseline"
- "Paired-acquisition does not claim best raw scanner removal"
- "If raw scanner removal is the only goal, oldstyle centroid/QR is the stronger choice"

---

## Section-Specific Rules

### Abstract
- **FORBIDDEN:** Any claim of best scanner removal. Any omission of the oldstyle baseline boundary. Any clinical/diagnostic/deployment language.
- **REQUIRED:** Statement that oldstyle centroid/QR is the strongest raw scanner-removal baseline. Statement that the contribution is structured separation.

### Introduction
- **FORBIDDEN:** "Scanner bias renders computational pathology unusable." "Existing methods fail." "We solve the scanner problem."
- **ALLOWED:** "Scanner-specific variation confounds tissue representations." "Existing approaches remove scanner signal but do not produce an explicit acquisition factor."

### Method
- **FORBIDDEN:** "Optimal architecture." "Theoretically guaranteed independence." "Provably complete separation."
- **ALLOWED:** "The acquisition branch dimensionality controls capacity." "Cross-covariance regularization encourages branch independence."

### Main Result 1 (Pair Structure)
- **FORBIDDEN:** "Requires exactly same-region pairs." "Broken pairs destroy the method." "True pairs are necessary for scanner suppression."
- **ALLOWED:** "True same-region pairs produce the strongest tissue-identity preservation." "Scanner suppression is maintained across the pairing ladder."

### Main Result 2 (Branch Separation)
- **FORBIDDEN:** "Perfect disentanglement." "Complete separation." "Scanner-free." "Category-free."
- **ALLOWED:** "The biological branch substantially reduces scanner recoverability while preserving tissue-category structure."

### Main Result 3 (Linear Baseline Boundary)
- **FORBIDDEN:** Any claim that paired-acquisition beats baselines on raw scanner removal. Omission of oldstyle result.
- **ALLOWED:** "Oldstyle centroid/QR projection achieves stronger raw scanner removal." "The contribution is structured separation."

### Main Result 4 (Bottleneck Frontier)
- **FORBIDDEN:** "Eliminates biological leakage." "Dimension 8 is optimal." "SCORPION category leakage reduced" (no category labels).
- **ALLOWED:** "Bottlenecking reduces category leakage in the acquisition branch." "Scanner capture is preserved."

### Main Result 5 (Factor Swapping)
- **FORBIDDEN:** "Proves perfect causal acquisition factor." "Proves independence." "Enables acquisition factor editing for deployment." "Scanner information fully encoded." "Works across all scanners/domains."
- **ALLOWED:** "Decoder-based acquisition swapping supports factor-like behavior." "Scanner identity follows the acquisition branch through recombination."

### Limitations
- **FORBIDDEN:** Minimization ("These limitations are minor"). Omission of any documented limitation.
- **REQUIRED:** All nine limitations from manuscript skeleton Section 10.

---

## Per-Claim Forbidden Overclaim Summary

| Claim | Forbidden |
|---|---|
| CLAIM_1 | "Requires exactly same-region pairs"; "Broken pairs destroy the method"; "True pairs necessary for scanner suppression" |
| CLAIM_2 | "Perfect disentanglement"; "Scanner-free biological branch"; "Category-free acquisition branch" |
| CLAIM_3 | "Best scanner removal"; any framing omitting oldstyle baseline; "Beats all baselines on scanner removal" |
| CLAIM_4 | "Eliminates leakage"; "Dimension 8 is optimal"; "SCORPION category leakage reduced" |
| CLAIM_5 | "Proves perfect causal factor"; "Proves independence"; "Enables deployment editing"; "Works across all domains" |

---

## Reviewer Response Templates

### If a reviewer says: "Isn't this just linear projection? Why the complexity?"
> We agree that oldstyle centroid/QR linear projection is the stronger raw scanner-removal method (scanner probe 0.200 vs our 0.361). Our contribution is not raw scanner removal but structured separation: paired-acquisition produces an explicit acquisition branch that can be inspected, bottlenecked to reduce biological leakage (category probe 0.346 → 0.160), and swapped to test factor-like behavior. The linear baseline removes scanner signal but provides no structured decomposition.

### If a reviewer says: "This doesn't prove factorization — the branches still leak."
> We do not claim perfect factorization. We claim measurable, partial separation. The biological branch reduces scanner probe from 0.866 to 0.361 while preserving category structure. The acquisition branch captures scanner signal (0.862) with reduced category structure (0.346). Cross-covariance is low (RMS < 0.10). This is structured separation, not perfect independence. We explicitly acknowledge residual leakage in both branches.

### If a reviewer says: "SCORPION doesn't have labels — you can't claim category preservation."
> We agree. All SCORPION claims are bounded to tissue/pair-retrieval metrics (paired cosine, top-1 retrieval, acquisition-branch retrieval leakage). Category-label claims are restricted to canine SCC DINOv2, which has expert-annotated tissue categories. We state this boundary explicitly.

### If a reviewer says: "This is just one dataset."
> We agree that canine SCC DINOv2 is the only dataset with biological category labels. We provide cross-backbone evidence in SCORPION (DINOv2, Phikon, ResNet50) for tissue-retrieval metrics, and we list single-dataset anchoring as a primary limitation. A second labeled multi-scanner dataset would strengthen the category-label claims.
