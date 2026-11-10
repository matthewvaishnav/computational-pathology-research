# ScanComp hmi825 — external scanner registration diagnostic

## Evidence that motivated the stage

The 10 November 2026 read-only `hmi825_repo.zip` contains `SCANCOMP_PIXEL_DECODE_RECEIPT.json` and four 5×5 sparse contact sheets. All four TIFF files expose the same 23,337×21,006 first-IFD coordinate dimensions and each passes 25/25 sampled patch decodes. **These are not registered coordinates yet.**

On **seven tissue-bearing positions** in the 160×160 PNG contact-sheet cells, match the Hamamatsu 80×80 cell centre within the other scanner's 160×160 same-location cell, using `cv2.TM_CCOEFF_NORMED` on grayscale. Previews are a downsampled 224×224 source patch; offsets below are in the preview, **not** level-0 coordinates:

| Comparison against Hamamatsu | True-location NCC range | Wrong-location max NCC (42 controls) | Median offset in 160px preview |
|---|---:|---:|---:|
| 3DHISTECH P1000 | 0.948–0.979 | 0.590 | (+8, −2) px |
| Leica GT450 | 0.943–0.984 | 0.552 | (+2, +3) px |
| Philips UFS | 0.914–0.972 | 0.600 | (−11, −18) px |

NCC is **in-sample optimization** for that template and should **not** be used as a registration accuracy estimate. The contact-sheet cells are sparse samples, not whole-slide thumbnails, and their resized visual data cannot certify full-resolution correspondence. In particular, Philips horizontal offset varies from −3 to −17 preview pixels across these seven patches, suggesting a position-dependent transform. The wrong-region controls use seven other tissue-bearing positions within each target scanner.

## Current code

This script uses `src.data.wsi_pipeline.reader.WSIReader`; the reader extension from draft [PR #103](https://github.com/matthewvaishnav/computational-pathology-research/pull/103) is required. Nothing changes the source TIFFs. Every sampled patch is bounded. The script computes only level-0 scanner-to-scanner geometric diagnostics, **not** PA-NF evaluation.

From your separate PR #103 worktree after fetching the new stacked branch:

```powershell
cd "$env:USERPROFILE\scancomp-pr103"
python -m pip install opencv-python-headless
python -m pytest tests/test_scancomp_registration_qc.py -q -o addopts=""
python -m scripts.external_multiscanner.scancomp_registration_qc `
  --data-dir "$env:USERPROFILE\Downloads\hmi" `
  --out-dir "$env:USERPROFILE\Documents\PA_NF_ScanComp_Intake\hmi825_registration" `
  --slide-id hmi825
```

Use the stacked registration branch's checkout, not the original draft PR #103 checkout, before running. Command above assumes this checkout has been switched onto that branch.

## Diagnostic protocol

- Hamamatsu is an arbitrary **coordinate reference**, not ground-truth scanner biology.
- Probe 9×9 reproducibly positioned level-0 sites. Require reference RGB tissue fraction ≥0.16 and grayscale SD ≥12 to match (both **exploratory** filters; reference scanner only).
- At each reference tissue site read 256×256 pixels; for each target scanner match within a 448×448 full-resolution window (±96px displacement), using grayscale normalized cross correlation. Record all rejected/failed sites.
- Assign `fit`/`heldout` via deterministic hash of slide alias + grid position **before reading target data**. Require ≥5 accepted fit sites and ≥3 accepted held-out sites for each scanner.
- Fit translation, RANSAC similarity, and RANSAC full-affine transformations from fit sites only. Compare the held-out displacement prediction residuals (in **source level-0 pixels**) against identity and constant translation. Save complete transforms, per-site scores, and failed matches to JSON/CSV.
- **No `registration PASS` flag** is assigned: a small number of exploratory tissue locations and matching to the same tissue do not establish histology-grade registration. The final geometry gate still requires held-out landmark/patch overlays, location-specific residual errors, and potentially deformable refinement with explicit constraints.

## Important prohibitions

Do **not** treat the resulting data as independent clinical labels, scanner-agnostic frozen PA-NF intervention validation, 4× independent human specimens, or proof of biological purity. There is only one physical H&E slide in this pilot. The G0 patient/case identity and independently adjudicated clinical labels are missing. No training, checkpoint changes, or protected G0 test evaluation is performed here.
