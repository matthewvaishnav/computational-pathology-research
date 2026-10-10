# PA-NF v5 — RT1–RT3 root-cause and falsification ledger

**Date:** 2026-10-10  
**Status:** post-outcome investigation, not a newly frozen training experiment  
**Dataset:** SCORPION, 48 source slides / 480 regions / five scanner acquisitions  
**Claim boundary:** RT1 failed, RT2 mechanism localization supported, RT3 failed its frozen head-to-head. No code or metric described here reclassifies these results. SCORPION is already burned development data.

## Observed constraints

1. RT1's inverse canonicalization had positive operator-only transport and known-scanner probe reduction, yet worse held-out alignment than its no-inverse control. Matched arm held-out calibration and operator-only transport were exactly equal.
2. RT2, on fresh seeds, showed candidate-minus-control same-input post-transport residual sensitivity +0.0067524 [0.0058437, 0.0077397]; an additional residual-direction-specific Jacobian gain component +0.0122778 [0.0098967, 0.0145957]. This supports an encoder-response explanation but does NOT prove the residual is pure nuisance.
3. RT3's inverse-residual-tangent arm improved alignment over both original inverse and no-inverse by +0.07022 and +0.06310 MSE, respectively, while isotropic smoothing aligned still more (isotropic-minus-targeted alignment=-0.00154695). Frozen RT3 development pass = FALSE.
4. Read-only RT3 retention-risk audit: no-inverse alignment 0.071953, top1 0.950000, probe 0.278819, reconstruction 0.689982; isotropic 0.007306 / 0.913889 / 0.289627 / 0.744385; residual-targeted 0.008853 / 0.955903 / 0.265017 / 0.725620. Isotropic's stronger alignment coincided with worse within-slide retrieval and reconstruction. This is a tradeoff, not proof of collapse.
5. Frozen RT3 result raw SHA-256: `021435970716e1a13a26ba3b8b4d16788453f328abef3737fe6bbf19a1b3dd25`.

## Structural blind spots identified in code

### A. Coordinate-wise variance does not certify eight-dimensional biology

`rt1._latent_variance_penalty` penalizes `relu(0.25 - per-coordinate std)^2` but never tests off-diagonal covariance or singular spectra. The counterexample `z_i=t_i(1,1,...,1)` with `std(t)>0.25` has zero variance-floor penalty but rank ONE.

*Falsifier*: verified unchanged RT3 replay recovers eight-dimensional test embeddings; effective rank, participation ratio and eigenspectrum remain comparably broad for regularized and unregularized arms. High effective rank alone does not establish diagnostic information.

### B. Reported scanner probe is not scale-robust

`rt1.fit_scanner_probe` uses `v4._ridge_fit(..., lam=1e-2)` with no train-only standardization of 8-D encoded features. Scaling all representation coordinates by a positive scalar `a` changes the effective regularization to `lambda/a^2`; probe changes can be partly scale artifacts.

*Falsifier*: after standardizing encoded features using only known-scanner TRAIN representations, the scanner-probe reduction versus no-inverse persists over a predefined regularization sweep (lambda 1e-4, 1e-2, 1.0). Such a result establishes stronger linear probe robustness, not absence of all nonlinear scanner information.

### C. Within-slide gallery can obscure global identity loss

`rt1.evaluate_model` calculates same-region cosine top-1 separately among the 10 regions from each slide. This cannot test confusion among regions on DIFFERENT test slides, nor clinical-label retention.

*Falsifier*: verified unchanged RT3 replay exhibits comparable scanner→AT2 top-1 retrieval against the *entire held-out fold* gallery (90–100 paired regions rather than 10), without dropping significantly for the regularized arm. Clinical biology still unproven.

### D. Tangent regularization can erase information aligned with scanner residuals

`rt3._sensitivity_penalty` penalizes `MSE[E(x0+r),E(x0)] / MSE[r,0]`, with `r=T_s^-1(x_s)-x0` detached and no guarantee that r is orthogonal to tissue-discriminative directions. Registration errors, scanner×morphology interactions and nonlinear feature response may create biological components of r. The regularizer can therefore penalize biology alongside nuisance.

*Falsifier*: independent same-physical-slide acquisitions with verified registration and biological labels show that the correction suppresses scanner variation without degrading relevant tissue-class or morphological tasks. Paired embedding distance by itself cannot prove this.

### E. Current embedding alignment is scale-sensitive

`heldout_alignment_mse = mean((u_h-u_0)^2)` is reduced by coordinate contraction. Reference reconstruction loss and rank-aware / normalized retrieval / external biological discrimination must remain separately adequate; an alignment-only gate is insufficient.

## Exact RT3 replay audit (in progress, no original result modified)

Branch `research/panf-v5-rt3-geometry-replay-audit-20261010`. Script `experiments/scorpion/audit_pa_nf_v5_rt3_exact_replay_geometry.py`.

1. Check SHA-256 of the ORIGINAL frozen RT3 JSON (fail closed).
2. Check frozen RT3 run-key completeness, feature SHA, split semantic identities, preprocessor hashes.
3. Reexecute the original RT3 training, held-out calibration, and evaluation with the SAME seeds `4901..4903`, same four arms, all folds and holdouts. **This is a replay on burned data**, not a new experiment.
4. For every fit, verify each original slide metric, training-final scalar and held-out calibration loss numerically before allowing further conclusions.
5. Measure *test* effective latent rank, participation ratio, top singular-value share, per-coordinate standard deviation, whole-test-fold retrieval versus within-slide retrieval, and train-standardized scanner probe for lambdas 1e-4, 1e-2 and 1.
6. Write results only if all 240 fits and original metrics replicate within declared tolerance. Store results in a NEW sidecar file; no existing outputs overwritten.

The replay cannot prove biological sufficiency without independent labels or outcomes. A changed result is replication failure, not a tunable tolerance to hide.

## Precommitted interpretation — not promotion gates

- **If isotropic loses rank/global retrieval while residual-targeted does not:** generic smoothing contracts discriminative geometry. Investigate a future covariance/eigenspectrum protection and explicit cross-slide discrimination anchor, while keeping the RT3 targeted-vs-isotropic frozen failure intact.
- **If both regularizers lose rank/global retrieval:** both are over-smoothed; investigate an information-preserving encoder constraint before further scanner modeling.
- **If normalized scanner probe removes targeted advantage:** scanner-probe suppression was at least partly scale-dependent; require normalization-robust probe comparisons and stronger scanner classifiers in future studies.
- **If normalized probe remains lower, rank remains broad, and global retrieval is retained:** targeted model is the stronger *development candidate*, but needs unseen paired-scanner and independent biological-label tests, not a retrospective RT3 victory.
- **If all arms retain rank but registration/tissue-conditioned scanner residuals predict failures:** model scanner×tissue interaction or local alignment quality explicitly; do not assume residual direction equals acquisition-only tangent.

## Scientific precedents (external, not PA-NF evidence)

- Bardes, Ponce & LeCun, *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*, ICLR 2022: per-coordinate variance plus covariance decorrelation; https://arxiv.org/abs/2105.04906
- Zbontar et al., *Barlow Twins: Self-Supervised Learning via Redundancy Reduction*, ICML 2021: cross-view correlation and redundancy reduction; https://proceedings.mlr.press/v139/zbontar21a.html
- Schieb et al., *UWarp: A Whole Slide Image Registration Pipeline to Characterize Scanner-Induced Local Domain Shift*, 2025: local scanner effects and registration; https://arxiv.org/abs/2503.20653
- Thiringer et al., *Scanner-Induced Domain Shifts Undermine the Robustness of Pathology Foundation Models*, 2026: matched scanner acquisitions and biology-task controls; https://arxiv.org/abs/2601.04163
- FEATMAP (2026), bioRxiv https://www.biorxiv.org/content/10.64898/2026.07.02.736184v1.full : matched-tile embedding harmonization and external reference comparator. Compare on identical data/splits, not on unmatched headline metrics.

## Next design work, explicitly NOT yet a frozen RT4 claim

Potential mechanism-specific successors after replay diagnosis:

1. Scanner inverse + rank-aware biological encoder: variance **and covariance/eigenspectrum protection**, while preserving matched tissue identity; compare to unchanged no-inverse, unchanged inverse, generic smoothing, and residual-targeted smoothing with matched capacity.
2. Geometry-constrained nuisance removal using training-derived scanner residual subspaces plus a *protected tissue manifold*; must prevent acquisition correction from erasing independently measured tissue-diagnostic directions.
3. Biological sufficiency validated on independent task labels with slide/patient identity legality, alongside full-fold retrieval, rank, standardized/nonlinear scanner probes, operator transport and reconstruction.

No tuning on the test slides. The same physical-slide multiple-scanner dataset must be independent of SCORPION to make external confirmation claims, and the external evaluation rules must freeze before inspection. ComBat, FEATMAP and straightforward alignment baselines deserve fair matched-split comparisons, not rhetorical claims of supremacy.
