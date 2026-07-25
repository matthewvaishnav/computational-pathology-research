# Paired-Acquisition Neural Factorization — audit-aware claim ledger

**Status:** revised after scientific audit on 2026-07-25  
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
- same-region retrieval and cross-scanner agreement are largely preserved;
- true biological correspondence in the training pairs improves same-region
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

### Pending evidence

Canine biological-category probe, category-purity, and category/scanner-ratio
numbers from the historical audit are withdrawn. They may return only after the
corrected audit defines a fixed sample-supported category estimand, uses fit-only
preprocessing, excludes same-region and same-sample neighbours, and reports
sample-aware uncertainty.

### Forbidden wording

- scanner-free biological branch;
- category-free acquisition branch;
- complete disentanglement;
- proof that the branch is disease biology.

## CLAIM 3 — Raw scanner-removal baseline boundary

### Active wording

> Historical oldstyle centroid/QR projection produced the strongest raw linear
> scanner-removal result. Paired-Acquisition Neural Factorization therefore must
> not be presented as the best scanner-erasure method.

### Pending comparison

The historical category-preservation comparison against oldstyle projection is
not active evidence until it is regenerated under the corrected category audit.
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

### Pending evidence

Historical canine acquisition-branch category-leakage numbers are withdrawn
pending the corrected category audit. No acquisition dimension may be described
as optimal. The locked dimension-by-cross-covariance factorial remains the
appropriate confirmatory experiment.

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
2. Corrected canine biological-label audit with a fixed category estimand.
3. Fold-aware SCORPION inference.
4. Repaired TransnnMIL matched reruns.
5. Harder retention tests that exclude same-region and same-sample shortcuts.
6. Forward-valid provenance releases.
7. A second labeled multi-scanner dataset and stronger independent human-tissue
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
summaries as more independent than the five-fold training design supports.
Current inference must use fold-aware analysis and retain the limitation that
five training clusters are few. Large descriptive effects may be reported
separately from formal inferential precision.

## Clinical boundary

No claim in this ledger supports diagnosis, patient care, clinical utility,
clinical readiness, improved outcomes, regulatory readiness, or deployment.
