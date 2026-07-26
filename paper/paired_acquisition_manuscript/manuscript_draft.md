# Paired-Acquisition Neural Factorization manuscript — scientific audit hold

**Original draft:** 2026-07-08  
**Audit hold applied:** 2026-07-25  
**Corrected evidence record:** 2026-07-26

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

> Under corrected biological-sample-blocked and fold-aware evaluation,
> Paired-Acquisition Neural Factorization substantially reduces linearly
> recoverable scanner identity in its tissue-oriented representation while
> preserving descriptive tissue-category structure and same-region retrieval.
> The acquisition branch retains strong scanner information. These findings
> support partial structured separation under the tested conditions.

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
  metrics, not biological endpoints;
- fold-aware `pathoalign_dep20_minus_paired_reference` scanner-probe contrast
  `-0.38541666666666674`, with 95% fold/slide bootstrap interval
  `[-0.42064000000000007, -0.34643478260869565]`;
- positive cross-view cosine contrasts, alongside retrieval contrasts whose
  intervals include zero and therefore do not support retrieval improvement.

The fold-aware v2 analysis is the current inference record. Historical
slide-independent sign-flip p-values remain withdrawn.

### External canine SCC

Permitted evidence includes:

- five-scanner matched-acquisition geometry;
- biological-sample-blocked folds;
- scanner probes;
- cross-scanner agreement;
- same-region retrieval;
- pair-integrity controls;
- acquisition-branch scanner capture;
- the corrected fixed five-category estimand with fit-only probe
  standardization and leakage-safe fit-pool neighbours;
- explicit resolution and geometry limitations.

For the original frozen, true-pair tissue-oriented, true-pair acquisition, and
linear centroid/QR k=4 representations, the exact
scanner/category/purity-k=5 five-fold means are respectively:

- `0.8628184955462632 / 0.44086590895818567 / 0.42409036761242336`;
- `0.3614076415619065 / 0.4353482507842298 / 0.429310749459204`;
- `0.865097576168538 / 0.39818864063789994 / 0.3063124063151831`;
- `0.2 / 0.44245014748183487 / 0.4377408893348`.

These are descriptive tissue-category representation metrics, not diagnostic,
clinical, patient-level, causal, or biological-purity evidence.

### Synthetic studies

Ground-truth latent-factor recovery, paired exposure, unique-anchor allocation,
and pair-repeat experiments may be included under their own generator,
identifiability, and external-validity limitations.

## Required evidence before reconstruction

1. Run the 175-fit capacity-matched SCORPION objective-ablation study.
2. Complete and aggregate the locked dimension-by-cross-covariance factorial.
3. Run repaired TransnnMIL matched controls before restoring any historical
   fusion or topology statement.
4. Obtain stronger independent human-tissue and second labeled multi-scanner
   confirmation before broader biological claims.
5. Rebuild every table and figure from corrected, provenance-bound result
   directories.

The fixed-estimand canine audit, fold-aware SCORPION intervals, and their
forward-valid corrected-evidence record are complete. The public PDF remains on
audit hold.

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
