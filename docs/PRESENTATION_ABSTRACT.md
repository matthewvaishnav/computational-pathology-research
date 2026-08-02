# Paired-Acquisition Neural Factorization: A Research-Engineering Audit of Scanner Structure in Pathology Embeddings

## Abstract

Digital-pathology representations can encode scanner optics, colour response,
compression, staining, and workflow effects in addition to tissue-associated
structure. This project studies whether matched acquisitions of the same tissue
region can support a more interpretable separation of these signals in frozen
pathology embeddings.

Paired-Acquisition Neural Factorization maps each embedding into a tissue-oriented
branch and an explicit acquisition branch, then reconstructs the original
embedding from both. Training uses matched scanner views, scanner-suppression and
scanner-retention objectives, pair agreement, variance controls, covariance
regularization, and capacity-matched controls. Evaluation is blocked by the
biological sampling unit rather than by patch.

On SCORPION, the promoted capacity-matched campaign completed 175 registered
fits across seven variants, five original-slide-blocked folds, and five seeds.
Relative to an equal-capacity two-branch control, the full model reduced
linearly recoverable scanner identity in the tissue-oriented branch while
preserving same-region retrieval within a preregistered noninferiority margin.
The acquisition branch retained strong scanner information. A separate 450-cell
canine SCC factorial found no universal bottleneck dimension or
cross-covariance setting, preventing a single configuration from being presented
as generally optimal.

The project also documents negative results and corrections: simpler linear
baselines can remove scanner information more aggressively, several historical
estimands contained leakage or pseudoreplication, and repaired whole-slide
fusion models require new matched reruns. Current paired-affine and crossed-target
synthetic studies remain prospective or exploratory and are not promoted
pathology-domain evidence.

The bounded contribution is a reproducible research-engineering framework for
studying partial structured separation under paired acquisition. The work does
not establish pure biological factors, complete disentanglement, novelty or
priority, diagnostic improvement, clinical utility, regulatory compliance, or
deployment readiness.

## Current presentation boundary

Any presentation using this abstract should also link to:

- the repository-root `CLAIM_BOUNDARY.md`;
- `docs/CURRENT_STATUS.md`; and
- `docs/research/scientific-audit-remediation-20260725.md`.

Older production-platform, clinical-deployment, benchmark-superiority, and
HIPAA-compliance descriptions are obsolete.
