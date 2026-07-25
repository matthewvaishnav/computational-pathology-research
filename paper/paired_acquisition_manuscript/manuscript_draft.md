# Paired-Acquisition Neural Factorization manuscript — scientific audit hold

**Original draft:** 2026-07-08  
**Audit hold applied:** 2026-07-25  
**Submission status:** not submission-ready

The previous manuscript draft has been withdrawn from current claim use. It
combined valid paired-scanner representation results with several numerical
claims that require corrected analysis or complete reruns. The historical text
is preserved in Git history and must not be copied into a submission,
presentation, outreach package, or funding application without checking the
current `CLAIM_BOUNDARY.md`.

## Why the draft was withdrawn

1. Canine biological-category probe and neighbourhood-purity claims came from an
   audit with scale-incomparable probes, test-set preprocessing leakage in one
   baseline, same-region nearest-neighbour leakage, and a category estimand that
   could vary across sample-blocked folds.
2. The draft treated cosine changes across differently transformed feature
   spaces as evidence of biological preservation or tissue damage.
3. The simple-baseline section omitted the strongest historical oldstyle
   centroid/QR scanner-removal result in some comparisons.
4. The paired-consistency and factorized models were not capacity matched, so the
   original comparison did not isolate which branch or objective produced the
   scanner-probe change.
5. Historical slide-level confidence intervals and sign-flip p-values did not
   fully account for dependence induced by the five-fold training design.
6. Bottleneck category-leakage claims depended on the withdrawn category audit.

## Active manuscript center

The next manuscript may presently defend this bounded result:

> Paired-Acquisition Neural Factorization uses matched acquisitions of the same
> tissue to learn a tissue-oriented branch and an explicit acquisition branch.
> Across the tested SCORPION and canine paired-scanner protocols, the
> tissue-oriented branch contains substantially less linearly recoverable scanner
> identity, the acquisition branch retains strong scanner information, and
> same-region tissue retrieval is largely preserved. These findings support
> partial structured separation under the tested conditions.

The manuscript must not call either branch biologically pure, claim complete
factorization, infer disease biology, or claim best scanner removal.

## Evidence currently permitted in a rebuilt draft

### SCORPION

- 48 original H&E slides;
- 480 aligned tissue regions;
- five scanners and 2,400 image patches;
- original-slide-blocked rotating folds;
- frozen objective transfer across DINOv2, Phikon, and ResNet50;
- reduced held-out linear scanner recoverability in the tissue-oriented branch;
- strong scanner capture in the acquisition branch;
- same-region retrieval and cross-scanner agreement reported as representation
  metrics, not biological endpoints.

Formal uncertainty should be regenerated with the fold-aware v2 analyzer.

### External canine SCC

Permitted evidence is restricted to:

- five-scanner matched-acquisition geometry;
- biological-sample-blocked folds;
- scanner probes;
- cross-scanner agreement;
- same-region retrieval;
- pair-integrity controls;
- acquisition-branch scanner capture;
- explicit resolution and geometry limitations.

Biological-category numbers are pending the corrected audit.

### Synthetic studies

Ground-truth latent-factor recovery, paired exposure, unique-anchor allocation,
and pair-repeat experiments may be included under their own generator,
identifiability, and external-validity limitations.

## Required evidence before reconstruction

1. Run `run_biological_label_preservation_audit_v2.py` after finalizing a fixed,
   sample-supported category estimand and including the strongest oldstyle
   baseline.
2. Run the 175-fit capacity-matched SCORPION objective-ablation study.
3. Generate fold-aware SCORPION intervals.
4. Complete the locked dimension-by-cross-covariance factorial.
5. Publish new outputs with forward-valid provenance.
6. Rebuild every table and figure from the corrected result directories.

## Required manuscript structure

A replacement draft should separate:

1. the identification problem;
2. negative results from unconditional nuisance removal;
3. paired-acquisition method and objectives;
4. SCORPION representation-level evidence;
5. external canine paired-scanner evidence;
6. capacity-matched controls;
7. corrected downstream-label evidence, when available;
8. limitations and alternative explanations.

The entire broader research program—TransnnMIL, institutional weighting,
CAMELYON17, PCam, systems, and privacy work—should be described as related work
or a separate research-program appendix. It must not be used to inflate the
central paired-acquisition claim.

## Authority

For all current wording, use:

- `CLAIM_BOUNDARY.md`
- `paper/claim_ledger/paired_acquisition_claim_ledger.md`
- `docs/research/scientific-audit-remediation-20260725.md`

This file is a hold notice and reconstruction plan, not a finished manuscript.
