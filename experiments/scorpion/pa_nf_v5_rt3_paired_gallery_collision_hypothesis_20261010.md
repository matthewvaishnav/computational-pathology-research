# PA-NF v5 RT3: paired-gallery collision falsification plan

**Date:** 2026-10-10  
**Status:** frozen RT3 failed; post-outcome replay investigation only. No new training hypothesis promoted.  
**Branch:** `research/panf-v5-rt3-paired-gallery-collision-audit-20261010`

## What exact RT3 replay establishes

Every one of the 240 original RT3 fits regenerated with matching training, calibration and slide-level evaluation metrics (within the replay's fixed numerical tolerance). The frozen outcome is still **FAIL**.

| Measured mean | No inverse | Inverse baseline | Inverse isotropic | Inverse residual-tangent |
|---|---:|---:|---:|---:|
| Effective rank, heldout | 6.4132 | 6.4587 | 6.7152 | 6.1956 |
| Top singular variance share, heldout | .3106 | .3048 | .2905 | .3288 |
| Full heldout-fold region retrieval top1 | .8799 | .8692 | .7999 | .9018 |
| Within-slide region retrieval top1 | .9495 | .9441 | .9139 | .9557 |
| Known-scanner linear probe accuracy | .2789 | .2709 | .2897 | .2649 |
| Train-standardized known-scanner probe at lambda .01 | .2789 | .2709 | .2897 | .2649 |

All these are average descriptive outcomes from user-reported complete replay, not an independent dataset or clinical biology readout.

**What is NOT supported:** The claim that isotropic lost retrieval mainly because its biological latent rank collapsed. It had the highest average effective rank of four arms. Effective rank alone does not identify what latent directions encode.

**What remains plausible:** Both types of regularization suppress the within-region scanner displacement, but generic isotropic smoothing weakens cosine discrimination against other region identities; the residual-tangent objective may better preserve those discriminative angular directions. This is a hypothesis, not an established root cause.

At lambda 0.01 the normalized known-scanner probe and original probe are equal to four decimal places. A large scale artifact at that one probe setting is not supported by the displayed data, but probe-lambda dependence and nonlinear leakage remain unexamined.

## Falsifier: test-slide impostor decomposition

Instrument unchanged RT3 replay for **only two originally frozen arms** (`inverse_isotropic` and `inverse_residual_tangent`). Replay 120 original fits: five folds × four heldout scanner systems × three frozen seeds × two arms. Every replay must match saved per-slide original metrics plus final training and calibration losses before any new interpretation.

For every heldout region in a test fold, compute the cosine similarities to:
- its *true paired AT2* region, using existing row identity;
- the hardest *other* region on its own source slide;
- the hardest region from *any other* test slide within the same fold.

Derive cross-slide-only error (correct among own-slide gallery but incorrect among full fold), within-slide error, true-match cosine, hardest-impostor cosine, positive/negative margins and global rank. Average ten regions per slide, three model seeds, four scanners; compare arms using **48 source slides** as the sole inference unit. Avoid pseudo-replication of regions, folds, or seeds. Descriptive post-outcome CIs can characterize effects, not retrospectively promote RT3.

### Interpretation tree (no new gates)

1. **Isotropic produces higher cross-slide-only error and lower cross-slide cosine margin**, especially while residual-tangent remains stable: identity crowding across source slides is supported as a mechanism of retrieval loss. A later *development* candidate could couple residual-tangent nuisance suppression to a source-slide-legal supervised paired-identity contrastive anchor or an independent tissue-task label. This requires fresh prospective rules and data/seed separation.
2. **Isotropic loses primarily inside the source-slide gallery**: local region discrimination has been impaired, not only across-slide crowding; a same-slide hard-negative signal may be necessary, evaluated without using heldout labels to train.
3. **Both same- and cross-slide margins deteriorate, with broad latent rank**: global cosine/angular organization, not dimensional rank collapse, is likely compromised. Test norm/cosine geometry and feature angle separation explicitly.
4. **Cosine margins and confusion are similar despite different accuracy**: inspect ties, gallery composition, normalization and retrieval implementation; do not claim a biological mechanism.
5. **Scanner probe outcomes remain strong only for the targeted model while retrieval margins survive**: still development evidence and only a known-scanner linear probe, not complete scanner information removal.

**Critical limitation:** Same-tissue region ID is a geometric identity, not an independent diagnosis or biological-label endpoint. Even a perfect identity-margin audit cannot prove clinical biological retention. Independent tissue labels, rank-sensitive controls, transparent scanner leakage probes and unseen paired acquisitions remain necessary for any PA-NF confirmation claim.

## Data and safeguards

- Only input evidence: frozen RT3 JSON SHA256 `021435970716e1a13a26ba3b8b4d16788453f328abef3737fe6bbf19a1b3dd25`, frozen DINOv2 feature SHA and semantic slide folds.
- No modifications to RT1–RT3 model architecture, optimizer, seed sequence, frozen gates or test splits.
- Replay fails closed on original metrics before reporting new diagnostics.
- Output sidecar: `results/pa_nf_v5_rt3_residual_tangent_regularization_20261006/pa_nf_v5_rt3_paired_gallery_collision_audit_20261010.json`.
- Do not overwrite any existing artifact.
- Preserve data provenance and any external scanner collaborator's confidentiality and background-IP boundaries.

**Strategic criterion:** An architectural successor should demonstrate scanner/acquisition suppression *and* preservation of region identity and independent biological tasks under fair paired controls. Superior alignment alone cannot establish scientific supremacy.
