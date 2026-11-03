# PA-NF canine SCC — scanner-only A development and purity gate (2026-11-03)

**Status: DEVELOPMENT / DO NOT PROMOTE AS BIOLOGICAL PURITY.** This is a separate follow-up to the original frozen canine SCC validation and does not revise its source checkpoints, model weights, evaluation gates, or published numbers. No other dataset was used.

## Motivation

The earlier author-provided 44-sample/805-region/five-scanner canine audit found that an image-conditioned acquisition encoder A(x) retained tissue-category and region-specific information. Simply erasing category directions did not remove the region signal. The issue is architectural: a freely encoded A(x) may learn arbitrary region context.

## Restricted acquisition architecture

A scanner-only acquisition code is determined *exclusively* by the measured scanner identity, not by image content:

```text
mu_s         = mean(x_region,s) over fold-training biological samples only
A(s)         = one_hot(scanner_id(s))
B(x_r,s, s)  = x_r,s - mu_s
T_t(B)       = B + mu_t
x_hat_r,t    = T_t(B(x_r,s,s))
```

For every region r and r' scanned by device s, **A(r,s)=A(r',s)** by construction. Target observations are used only for evaluation, never to construct test predictions. The five-scanner completeness of the dataset means the scanner label A(s) cannot discriminate its six supported region categories on this cohort; this does *not* prove biology is exclusively contained in B. Morphology-dependent scanner interactions can still remain in B. This is a simple scanner-centroid calibration baseline, not a novel neural mechanism.

## Verified read-only development results

Input: owner-uploaded `PA_NF_Canine_20261103_094701.zip`, SHA256 `78a736bd8c672d9fedb2719db89cc9045567efaff726d67af5029c6d72511e25`. Dataset: 4,025 original DINOv2 observations, 805 five-view regions, 44 sample-blocked units, five historical folds. Cartilage excluded from six-class tissue readouts because it has only two regions.

| Measurement | Raw DINOv2 | Scanner-centered B |
|---|---:|---:|
| Sample-blocked linear scanner balanced accuracy | 0.86793 | **0.21672** |
| Sample-blocked linear tissue-category balanced accuracy | 0.45077 | **0.45235** |
| Fixed ExtraTrees nonlinear scanner balanced accuracy | — | **0.32212** |

The fixed nonlinear scanner classifier uses 24 trees, max depth 12, minimum leaf 10, five folds, constant random seed. Five-class chance = 0.20. **The strict scanner invariance gate FAILS.** The linear decrease is not a purity certificate.

Source-only target-feature reconstruction: 805 regions × 20 ordered scanner directions = 16,100 held-out predictions:

| Method | MSE (lower better) |
|---|---:|
| Scanner-centered B + target scanner mean, A scanner-only | **0.432823** |
| Direct pairwise 768D raw Ridge (train-sample fitted) | 0.442694 |
| Uncorrected raw source | 0.490242 |
| Frozen PA-NF B (seed 911, training-only four scanner directions removed) with target-conditioned Ridge decoder | **0.497372 — failed** |

The more complex B-only decoder **does not beat** simple scanner-centroid correction and should not replace the canonical neural architecture. These comparisons are exploratory, without a new independent TEST cohort or inferential improvement claim. No frozen neural checkpoint was retrained.

## Gate decisions

- **PASS by construction:** no per-image/per-region acquisition vector A(x) in this baseline.
- **Descriptive positive:** scanner-centered B retains the six-category linear readout while suppressing linear scanner recoverability.
- **FAIL:** nonlinear scanner recoverability in B remains above chance (0.322 vs 0.20).
- **FAIL:** the more complex frozen-B scanner-conditioned decoder underperforms both mean shift and raw Ridge.
- **BLOCKED:** distinguishing B biology from scanner-constant staining/fixation/slide preparation; canine SCC lacks independently crossed preparation metadata.
- **UNCHANGED:** October 19 frozen independent-human joint confirmation G0; the original canine cohorts have previously been exposed.

## Reproduce

Use Python/NumPy/Pandas/scikit-learn (CPU). Save the user's original ZIP locally; the ZIP and all raw imagery are **not** published in GitHub.

```powershell
python experiments/paired_acquisition/canine_scanner_only_a_baseline.py \
  --source-zip PA_NF_Canine_20261103_094701.zip \
  --out scanner_only_canine_report
```

This script fails closed on unexpected source hashes, missing or corrupt members, row-key mismatch, scanner-view gaps, or biological unit split violations. Output is descriptive fold scores and a JSON gate receipt. The longer experimental decoder comparison, scripts, and per-pair predictions are archived in the local downloadable `PA_NF_Canine_Scanner_Only_v1.zip` package; the canonical original source archive is not redistributed.

**Not for clinical diagnostic inference, perfect separation claims, or publication promotion.** A future candidate must pass nonlinear scanner probes, tissue retention, cross-scanner region fidelity, and stronger matched controls on a genuinely untouched cohort with independent preparation and biological reference interventions.
