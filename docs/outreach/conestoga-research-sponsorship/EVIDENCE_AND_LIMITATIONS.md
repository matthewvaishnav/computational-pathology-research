# Evidence, limitations, and current development status

This document separates completed evidence from active development so an institutional reviewer can assess the project without interpreting planned work as completed work.

## Completed evidence suitable for preliminary review

### 1. SCORPION human H&E paired-scanner study

- 48 original slides.
- 480 aligned tissue regions.
- Five scanners and 2,400 real-tissue patches.
- Five rotating original-slide test folds.
- Frozen objective transferred across DINOv2-Base, Phikon, and ImageNet ResNet50.
- Reduced linearly recoverable scanner identity in the tissue factor while preserving retrieval and improving cross-scanner cosine agreement.
- Compact acquisition factor retained strong scanner information, supporting factor separation rather than simple representational erasure.

Primary record: [SCORPION results](../../research/paired-acquisition-factorization-scorpion-results.md)

### 2. Independent canine SCC paired-scanner validation

- Public external dataset with 44 biological samples and five scanners.
- Geometry-qualified subset of 805 complete five-view regions and 4,025 scanner views.
- Five sample-blocked external folds and five seeds.
- Hyperparameters locked from SCORPION.
- Scanner-probe accuracy reduced by approximately 0.39 absolute while retrieval was preserved and cross-scanner cosine agreement improved.

Primary record: [external canine SCC results](../../research/paired-acquisition-factorization-caninescc-results.md)

### 3. Pair-repeat allocation controls

- Matched total pair-presentation budgets.
- Direct comparison between greater unique biological-pair diversity and repeated presentation of fewer anchors.
- Current evidence favors unique biological-pair diversity for biological consistency and factor separation.

Primary record: [pair-repeat allocation study package](https://github.com/matthewvaishnav/paired-acquisition-factorization-allocation)

### 4. Forward-valid evidence infrastructure

- Immutable run and release identifiers.
- Dataset, split, configuration, environment, command, commit, seed, parent, and artifact bindings.
- SHA-256 validation and fail-closed publication.
- Corruption tests and independent validators.
- Legacy artifacts with unresolved lineage remain explicitly excluded from new claim evidence.

This infrastructure is stronger than the historical artifact record; it is forward-valid rather than a claim that every older run has been retroactively reconstructed.

## Active development

### Locked capacity by regularization factorial

The complete design crosses:

- bottleneck dimensions: 2, 4, 8, 16, 32, and 64;
- cross-covariance weights: 0.00, 0.05, and 0.20;
- five folds;
- five seeds;
- 75 epochs per cell.

This produces 450 provenance-bound runs. Gate 1 passed and the resumable fail-closed Gate 2 runner is implemented. The full run and preregistered aggregate analysis remain active work.

The purpose is to determine whether the observed factor separation is stable across capacity and regularization choices, and whether those factors interact. Until that analysis is complete, the repository does not claim a general capacity or regularization law.

Primary record: [Gate 2 execution boundary](../../research/paired-acquisition-factorial-gate2-execution.md)

## Material limitations

1. **Research-only:** no patient-care use, clinical validation, regulatory review, or diagnostic-performance claim.
2. **Representation-level outcome:** the central result concerns scanner identifiability, cross-scanner agreement, and tissue-region retrieval—not a demonstrated improvement in a clinical endpoint.
3. **Finite paired benchmarks:** SCORPION is a 48-slide benchmark, and the external validation is canine rather than an independent human clinical cohort.
4. **Acquisition scope:** current evidence does not cover every scanner, laboratory, staining process, preparation workflow, compression pipeline, or tissue type.
5. **No perfect disentanglement claim:** reduced linear scanner recoverability does not prove that every nonlinear acquisition signal has been removed.
6. **External canine resolution boundary:** the public release is downsampled to approximately 4 micrometres per pixel, so it is evidence at the released resolution rather than cellular-scale full-resolution validation.
7. **Historical provenance boundary:** unresolved historical local artifacts are preserved as limitations and are not promoted into the forward-valid evidence set.
8. **Independent expert review pending:** the work has not yet received formal peer review from a computational pathology faculty member, pathologist, or journal/conference review process.

## Highest-value next steps

1. Complete the locked 450-run factorial and preregistered aggregate analysis.
2. Obtain faculty and pathology-domain review of the estimands, controls, and biological interpretation.
3. Add independent human-tissue external validation with patient- or specimen-grouped splits where metadata permits.
4. Evaluate whether scanner suppression preserves meaningful downstream labels under domain transfer.
5. Consolidate the central study into a publication-ready manuscript and archive.
6. Identify a pathology, scanner, hospital, or medical-technology partner with a real paired-acquisition problem.

## Questions for a Conestoga reviewer

- Is the central representation-identifiability question scientifically well framed?
- Which additional control or external cohort would most improve publication readiness?
- Is there a suitable Conestoga faculty supervisor or research centre for this work?
- Can the project receive Digital Research Alliance sponsorship?
- Could the next stage fit a paid student-research role, Mitacs project, NSERC College and Community Innovation proposal, OCI program, or partner-funded applied-research project?
- What institutional, ethics, data-governance, or commercialization review would be needed before working with non-public pathology data?

## Public claim boundary

The authoritative wording boundary is maintained in [CLAIM_BOUNDARY.md](../../../CLAIM_BOUNDARY.md). When this package and a result document appear to differ, the narrower claim must be used.
