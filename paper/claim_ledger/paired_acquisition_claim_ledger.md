# Paired-Acquisition Neural Factorization — audit-aware claim ledger

**Status:** corrected evidence promoted on 2026-07-26

**Authority:** subordinate to `CLAIM_BOUNDARY.md`

This ledger replaces the previous numerical claim ledger. Historical details
remain available in Git history, but claims that relied on contaminated category
metrics, unmatched protocols, or defective TransnnMIL execution paths are not
promoted here.

## Active thesis

Paired-Acquisition Neural Factorization is a neural representation-auditing
method that uses matched acquisitions of the same tissue to encourage partial
separation between tissue-oriented information and acquisition information.

Under the tested paired-scanner protocols:

- the tissue-oriented branch contains substantially less linearly recoverable
  scanner identity than paired-consistency references;
- an explicit acquisition branch retains strong scanner information;
- corrected canine tissue-category balanced accuracy and fit-pool category
  purity remain close to the frozen representation;
- same-region retrieval is largely preserved and SCORPION cross-view cosine
  agreement improves geometrically;
- true same-region correspondence in the training pairs improves same-region
  identity preservation relative to weakened or shuffled pair controls.

The contribution is **partial structured separation**, not complete
factorization, causal biology, or best-in-class scanner erasure.

## CLAIM 1 — Pair correspondence supports tissue-identity retention

### Active wording

> True same-region cross-scanner pairs produce stronger same-region identity
> retention than weakened or shuffled pair controls under the tested SCORPION and
> canine SCC protocols. Scanner suppression alone persists under broken pairs,
> so pair correspondence is primarily supported as a tissue-identity retention
> signal rather than as the sole cause of scanner suppression.

### Active evidence

- SCORPION pair-structure boundary experiments using paired cosine and same-region
  retrieval.
- Canine SCC pair-integrity controls using paired cosine and same-region
  retrieval.
- Cross-backbone SCORPION pair-structure tests where available.

### Boundary

These metrics identify tissue regions, not pathology categories, diagnoses, or
patient outcomes. Near-saturated retrieval is a weak preservation endpoint and
must be complemented by harder outcomes.

## CLAIM 2 — Partial scanner/acquisition branch allocation

### Active wording

> Under the tested protocols, the tissue-oriented branch has reduced linearly
> recoverable scanner identity, while the acquisition branch retains scanner
> identity. Residual leakage remains in both branches.

### Active evidence

- Held-out linear and stronger scanner probes.
- Acquisition-branch scanner capture.
- Cross-covariance and rank diagnostics.
- Pair-integrity and acquisition-swapping mechanism audits under their explicit
  boundaries.

### Corrected canine evidence

The active replacement uses one fixed five-category estimand in all five
biological-sample-blocked folds, fit-only probe standardization, fit-pool
neighbours, same-region and same-sample exclusions, and seed averaging within
fold. Exact five-fold means are:

| Representation | Scanner probe balanced accuracy | Category probe balanced accuracy | Fit-pool purity k=5 |
|---|---:|---:|---:|
| Original frozen features | 0.8628184955462632 | 0.44086590895818567 | 0.42409036761242336 |
| True-pair tissue-oriented | 0.3614076415619065 | 0.4353482507842298 | 0.429310749459204 |
| True-pair acquisition | 0.865097576168538 | 0.39818864063789994 | 0.3063124063151831 |
| Linear centroid/QR k=4 | 0.2 | 0.44245014748183487 | 0.4377408893348 |

These are descriptive representation metrics. Historical canine category
metrics remain withdrawn and are not combined with the corrected values.

### Forbidden wording

- scanner-free biological branch;
- category-free acquisition branch;
- complete disentanglement;
- proof that the branch is disease biology.

## CLAIM 3 — Raw scanner-removal baseline boundary

### Active wording

> Under the corrected fixed-estimand canine audit, the linear centroid/QR
> baseline removes linearly recoverable scanner information more aggressively
> than the neural tissue-oriented branch while retaining similar descriptive
> category structure. The neural method additionally retains an explicit
> acquisition branch with strong scanner information.

The corrected k=4 linear projection has scanner balanced accuracy `0.2`,
category balanced accuracy `0.44245014748183487`, and fit-pool purity k=5
`0.4377408893348`. It must not be converted into a claim of biological purity.
Cosine differences across differently transformed feature spaces must not be
called biological preservation or tissue damage.

### Forbidden wording

- best scanner removal;
- beats all linear baselines;
- PCA damages biology based on cosine alone;
- best scanner-suppression/tissue-preservation trade-off.

## CLAIM 4 — Acquisition bottlenecking

### Active wording

> Capacity constraints can change scanner capture, acquisition-branch retrieval
> leakage, rank, and cross-covariance under the tested protocols.

### Active evidence

SCORPION cross-backbone same-region retrieval leakage may be reported by
backbone, provided it is not relabelled as category leakage.

### Boundary

The corrected true-pair acquisition branch retains strong scanner information
(`0.865097576168538` scanner balanced accuracy). Its descriptive category
balanced accuracy (`0.39818864063789994`) and fit-pool purity k=5
(`0.3063124063151831`) do not establish category absence or biological purity.
Historical bottleneck category-leakage numbers remain withdrawn. No acquisition
dimension may be described as optimal; the locked dimension-by-cross-covariance
factorial remains the appropriate confirmatory experiment.

## CLAIM 5 — Acquisition swapping is factor-like evidence

### Active wording

> Decoder-based acquisition swapping provides probe-supported evidence that
> scanner information can follow the acquisition branch through recombination
> under a controlled five-scanner feature-level experiment.

### Boundary

This does not prove causal factors, independence, complete scanner transfer,
pixel-level realism, or deployment-ready editing.

## Required controls before stronger promotion

1. Capacity-matched SCORPION objective ablations.
2. Completion and preregistered aggregate analysis of the locked
   dimension-by-cross-covariance factorial.
3. Repaired TransnnMIL matched reruns.
4. Harder retention tests beyond near-saturated same-region retrieval.
5. Forward-valid provenance releases for every additional numerical claim.
6. A second labeled multi-scanner dataset and stronger independent human-tissue
   validation.

## TransnnMIL boundary

Historical PANDA QWK values do not demonstrate successful TransMIL–nnMIL fusion.
The historical canonical fusion was query-invariant, and historical topology
mode used an unregistered random projection. The repaired implementation is a
new model for evidentiary purposes and requires new comparisons against
AttentionMIL, TransMIL, nnMIL, concat, gate, and learned branch-attention
controls.

## Statistical boundary

The historical 48-slide SCORPION bootstrap and sign-flip p-values treated slide
summaries as more independent than the five-fold training design supports and
remain withdrawn. Current fold-aware contrasts for
`pathoalign_dep20_minus_paired_reference` are:

| Metric | Mean difference | 95% fold/slide bootstrap interval |
|---|---:|---:|
| Scanner probe accuracy | -0.38541666666666674 | [-0.42064000000000007, -0.34643478260869565] |
| Pair cosine average | 0.031035241278509318 | [0.026858407069387497, 0.035674580187052544] |
| Pair cosine worst | 0.03109579056501389 | [0.025433909586680174, 0.03806131886798837] |
| Retrieval top-1 average | -0.0000833333333333311 | [-0.0003829787234042509, 0.00020833333333334278] |
| Retrieval top-1 worst | -0.0004166666666666624 | [-0.002083333333333321, 0.0014893617021276633] |

The scanner and cosine contrasts support substantial scanner-recoverability
reduction and improved representation geometry. Both retrieval intervals include
zero, so same-region retrieval is effectively unchanged and retrieval
improvement is not supported. Five training clusters remain a small number.

## Clinical boundary

No claim in this ledger supports diagnosis, patient care, clinical utility,
clinical readiness, improved outcomes, regulatory readiness, or deployment.
