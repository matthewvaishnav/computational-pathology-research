# PA-NF — Post-RT3 scientific review and RT4 development plan

**Prepared:** 2026-10-11  
**Status:** PROPOSED / NOT A FROZEN EXPERIMENT. No RT4 results, seed choice, success gates, or model promotion authorized by this planning record.  
**Parent evidence:** frozen RT1 failure, RT2 supported residual-direction amplification, RT3 frozen FAIL, exact RT3 geometry replay, post-RT3 paired-gallery collision replay.  
**Source code:** `experiments/scorpion/run_pa_nf_v5_rt3_residual_tangent_regularization.py`, `run_pa_nf_v5_rt1_scorpion_translation.py`, `audit_pa_nf_v5_rt3_paired_gallery_collisions.py`  
**Source:** user-provided console from 2026-10-11; inspect original stored JSON before finalizing a protocol.  
**Original RT3 raw JSON SHA256:** `021435970716e1a13a26ba3b8b4d16788453f328abef3737fe6bbf19a1b3dd25`.

## Evidence and falsifiable question

The corrected collision-audit suite passed 10/10 tests. The audit then replayed 120/120 frozen RT3 isotropic and residual-tangent fits and verified original metrics before inspecting new margins. These are **post-outcome descriptive** results on already-studied SCORPION, not a new independent experiment.

| Read-only descriptive measure | Isotropic | Residual tangent | Isotropic minus tangent (95% source-slide bootstrap CI) |
|---|---:|---:|---:|
| Whole-fold paired-region top-1 | .799306 | .901736 | -.102431 [-.117361, -.088194] |
| Within-slide paired-region top-1 | .913889 | .955903 | not primary for causal conclusion |
| Cross-slide-only error | .114583 | .054167 | +.060417 [+.046875, +.073611] |
| Cross-slide hard-negative cosine margin | +.060261 | +.097700 | -.037438 [-.042966, -.031980] |
| Within-slide hard-negative cosine margin | +.114759 | +.150857 | -.036098 [-.045601, -.027060] |
| Equal-size (nine-negative) cross-slide strict top-1 | .957948 | .985091 | -.027143 [-.031245, -.023090] |
| Correct-pair cosine difference | — | — | -.007975 [-.010781, -.005217] |
| Hardest cross-slide impostor cosine difference | — | — | +.029463 [+.023183, +.035819] |

No near-ties were reported. Isotropic had a higher effective latent rank in the earlier exact replay; a simple rank-collapse account is not supported. The mechanism that is actually measured is poorer angular separation of correct tissue-region identity from impostors under generic smoothing, even at equal gallery size.

Do **not** infer pure biological loss, clinically significant information, causality of any specific latent direction, scanner-independence, or superior external-site performance solely from these retrieval metrics.

**Primary RT4 question:** can preservation of **correct paired-region identity margins** restore/reinforce scanner-robust tissue discrimination without sacrificing genuine scanner invariance, transport correctness, or meaningful biology?

## What the code implies

- The RT3 encoder maps reference-preprocessed 32D frozen DINOv2 features to 8D biological codes; downstream decoder reconstructs the AT2 reference feature.
- RT3 inverse residual sensitivity: `||E(x_0+r_s)-E(x_0)||^2 / ||r_s||^2`, with `r_s=T_s^{-1}(x_s)-x_0` detached and estimated only on known training scanners.
- Isotropic comparison replaces `r_s` with an RMS-matched random direction. It suppresses generic local sensitivity; loss of identity margin is therefore plausible, but not causally isolated by an architecture comparison alone.
- Neither RT3 sensitivity loss explicitly enforces positive paired-region similarity relative to distinct-region negatives.
- The coordinate-wise variance floor cannot guarantee a full-rank 8D code, but higher observed isotropic effective rank means **rank collapse is not the leading explanation of its retrieval loss**.
- Known-scanner linear probe accuracy is approximately 25–29% across four classes (chance ~25% when balanced); the small differences should not be described as absence of scanner information. Add train-only standardized and nonlinear scanner-leakage probes.
- The heldout-scanner operator receives **separate calibration** on validation acquisitions; call this calibrated scanner transfer, not zero-shot generalization.
- Slide-level bootstrap treats 48 slides as biological units; if patients/specimens overlap, patient-level independence and inference are **unverified**.

## Phase A — provenance and integrity gate, no new training

1. Validate existing original frozen RT3 hash, replay sidecar hashes, complete source-slide/region/scanner keys, fold training/validation/test separation, paired registration and operator calibration scope.
2. Export per-slide/per-scanner signed effect tables (alignment, within/full-fold retrieval, cross-slide error, margins, reconstruction, scanner probe); preserve all individual failures, not only means.
3. Classify likely shortcut classes from available metadata: slide identity, region adjacency, near-duplicate tiles, registration/misalignment, tissue compartment and preparation. Avoid claiming patient-level separation without patient IDs.
4. Inspect whether multiple regions from the same slide are near-duplicate biological content; if so, distinguish paired identity retrieval from morphology generalization.

Proceed to RT4 development only with explicit train-only negative sampling and a frozen design.

## Phase B — minimal mechanism-specific prospective development

**Single intervention family:** add a training-only paired-region discrimination loss to RT3 without changing operator construction, encoder/decoder parameter count, preprocessing or calibrated scanner protocol.

Let `u_{i,s}=normalize(E(T_s^{-1}(x_{i,s})))`, including `T_AT2=I`. Form same-region multi-scanner positives from known training scanners and balanced distinct-region negatives from both same and different *training* slides. A contrastive/margin term must favor correct correspondence over hardest or log-sum-exp training negatives. Calibration and test regions are NEVER eligible as positives, negatives, a hard-negative mining index, or an adaptive loss-weight source.

Do not assert that region-specific discrimination equals clinical biology: a low-level region fingerprint can satisfy the loss. Complement with independent biological task and cross-slide transfer tests.

**Factorial arms, matched seeds and capacity:**
- original inverse + isotropic sensitivity
- original inverse + residual-tangent sensitivity
- isotropic + real identity-margin anchor
- residual-tangent + real identity-margin anchor
- residual-tangent + within-slide/per-scanner matched **sham identity** anchor (incorrect positive mapping) to test generic regularization effects
- original no-inverse control under matched evaluation

Retain the untouched original inverse baseline as a separate registered comparator if compute permits. A full six-arm three-seed five-fold four-heldout development matrix is 360 fits; this count is a **planning budget**, not a frozen run.

Use paired initialization where feasible, fixed train-only negative pools, identical hyperparameter search resources per intervention, and report training cost. Parameter counts are equal because the experiment adds losses, not layers. Factorial interaction tests answer whether a real identity signal benefits residual-direction regularization differently than isotropic.

**Selection policy:** select any temperature/margin/loss-weight from **training slides only** through nested development splits, not RT3 heldout slides, RT3 diagnostic outcomes, or heldout-scanner calibration data. Burn selected development seeds and freeze a final protocol before evaluating any new external collection.

## Phase C — discriminating endpoints and explicit failure conditions

Primary mechanism endpoint in development: slide-clustered **full-fold scanner-to-AT2 paired-region retrieval** under fixed galleries. Alongside it, report matched-size cross-slide accuracy, cross-slide/within-slide cosine margins, top-5 and margin distributions, and per-scanner worst-case performance. Do not replace the original RT3 gate retrospectively.

Critical guardrails:
1. **Scanner robustness:** train-standardized ridge probes across a fixed lambda sweep plus a stronger trained-on-training-only nonlinear probe; report accuracy with permutation/null baselines. Not just a 4-class linear probe near chance.
2. **Biological retention:** morphology/pathology label prediction with independently sourced labels and slide/patient-grouped splits where provenance permits. If unavailable, no biological-sufficiency claim.
3. **Reconstruction and identity:** AT2-reference reconstruction versus no-inverse and residual controls; do not accept lower alignment if recon/biological labels degrade beyond prespecified noninferiority bounds.
4. **Latent geometry:** effective rank and participation ratio, coordinate covariance, angular crowding, feature norms. Rank alone is descriptive.
5. **Transport integrity:** heldout calibrated transport gain, known scanner gain, numerical inverse/roundtrip, frozen operator calibration scope.
6. **Statistical legality:** predeclare primary contrast, source-slide or patient hierarchy, clustering/bootstrapping, multiple comparisons, missing-data rules, and what constitutes an inconclusive result.

A candidate may only advance if its primary identity improvement is positive with an interval that excludes zero **and** all predeclared guardrails pass. Choose practical effect thresholds and noninferiority margins **before running RT4**, based on an independently justified tolerance; do not mine them from the already-consumed RT3 test results. If a check is missing, mark the candidate incomplete, not passed.

## Phase D — external result that could justify a stronger PA-NF claim

An independent multi-scanner acquisition collection (e.g., the possible McGill collaboration) must establish whether **the exact same physical tissue slides or regions** were scanned on multiple systems. Merely having Leica/Aperio, Hamamatsu and Zeiss data in distinct patient cohorts is NOT sufficient for paired scanner identity estimands.

Before reading external outcomes: freeze the selected architecture, weights/protocol, reference scanner, calibration allowance, site/scanner holdouts, eligibility and registration tolerances, unit hierarchy, endpoints and honest competitor matrix (raw encoder, matched-capacity controls, strong normalization/ComBat/FEATMAP-type baselines where lawful, and downstream WSI MIL tasks where labels support them). Zero-shot and calibrated performance must be separately labeled.

External primary endpoint must test biological task retention (if labels available) and acquisition robustness jointly. If matched scans but no biological labels exist, only paired tissue-identity and scanner acquisition robustness claims are admissible.

## Stop rules

- RT3 stays **FAIL** regardless of RT4 outcome.
- RT4 identity gain with lost biological labels/reconstruction => not a success; revisit the anchor.
- RT4 identity gain with worse scanner recoverability => not a successful factorization.
- No difference from sham anchor => no evidence that correct paired correspondence was necessary.
- No independent matched-scanner acquisitions => development candidate only.
- No verified independent biological labels => no diagnostic-biological sufficiency claim.
- Avoid additional post-outcome audit loops without a specific branch-changing falsifier.

**Immediate action:** implement *one* RT4 pre-outcome validator and matrix on a dedicated development branch only after the design is frozen; keep all RT3 files untouched. In parallel, ask any external partner for a de-identified manifest of physical-slide pairing, scanner assignments, permitted biology labels, and patient/specimen grouping without sharing protected research code or data outside agreed terms.
