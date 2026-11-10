# ScanComp hmi825 FIT-only visual QC of held-out patches

This is a read-only image export for human review, **not** another registration algorithm, no model retraining, and no claim of independent anatomical certification. It only uses reference-scanner HELDOUT site positions and the previously fitted predeclared affine matrices from `SCANCOMP_REGISTRATION_DIAGNOSTIC.json` to resample 256x256 target tissue patches. The target-scanner HELDOUT `dx`, `dy`, NCC, and labels from the CSV are **not used for alignment or model selection**.

## Run on Windows in the PR #104 worktree

```powershell
cd "$env:USERPROFILE\scancomp-pr104"
python -m pytest tests/test_scancomp_visual_holdout_qc.py -q -o addopts=
python -m scripts.external_multiscanner.scancomp_visual_holdout_qc `
  --receipt "$env:USERPROFILE\Documents\PA_NF_ScanComp_Intake\hmi825_registration_v2\SCANCOMP_REGISTRATION_DIAGNOSTIC.json" `
  --landmarks "$env:USERPROFILE\Documents\PA_NF_ScanComp_Intake\hmi825_registration_v2\SCANCOMP_REGISTRATION_LANDMARKS.csv" `
  --data-dir "$env:USERPROFILE\Downloads\hmi" `
  --out-dir "$env:USERPROFILE\Documents\PA_NF_ScanComp_Intake\hmi825_visual_holdout_qc"
```

The script streams and compares all original TIFF SHA-256 values against the supplied receipt. It refuses to run if the inputs changed. Outputs are three PNG montages, each containing 8 held-out location panels showing **reference**, **FIT-affine-predicted target**, **50/50 color overlay**, and **checkerboard**. It also writes `SCANCOMP_BLINDED_VISUAL_QC_RECEIPT.json` with source and input hashes and descriptive fixed-transform (not optimized) grayscale NCC / RGB MAE. No best-match search, iterative warping, or adjustment of the affine occurs; the old receipt remains unmodified.

Humans should assess corresponding nuclei/glands/crypts and structures, patch clipping, and registration interpolation artifacts. It is possible to see artifacts even if coarse NCC scores are high. Do not interpret attractive overlays as clinical label retention or as an independent human source-cohort confirmation. The original G0 remains DATA_BLOCKED.

**Artifact status**: hmi825 is development-exposed. This provides technical eligibility to plan frozen-feature extraction only after independently reviewed overlays. Confirmation requires other untouched slides and a frozen fit/validation/test plan.