# SCANCOMP–PA-NF–FROZEN–001R: corrected two-track protocol

**Status:** preflight implementation draft; synthetic tests are subject to CI. **No inference authorized by this document alone.** Supersedes the prior mixed physical/pixel-scale 001R draft but **does not alter** the completed 256-pixel development pilot.

**Scope:** public ScanComp H&E slide alias hmi825, four acquisitions (hms360 reference, hstp1000, lcap450, phufs). The wider PA-NF program is not reducible to scanner invariance. These experiments measure representation geometry and scanner-consistency *proxies*, not biological sufficiency, clinical validity, or a G0 admission result.

## 0. Existing evidence and boundaries

- Prior frozen development pilot: 8 held-out reference locations × 4 scanners = 32 images, **256 × 256 level-0 source pixels**, 25 original checkpoints. Raw DINOv2 X768 vs B256 median same-region cosine **0.9314 vs 0.8764**; hard-negative margin **0.0976 vs 0.2001**; top-1 retrieval **97.92% vs 100%**; scanner-associated *fractions* **6.72% vs 6.01%**, A64 **21.45%**. These are observations at the tested pixel context, not physical-field-matched or independently replicated results. Absolute scanner variance cannot be inferred from fractions.
- Historical SCORPION input is reported as an **800 × 800 µm tissue field** presented as **1024 × 1024 RGB pixels**, followed by the pinned Hugging Face DINOv2 image processor. Verify that definition from original extraction code before a physically matched run. A 1024-pixel *presentation* is not a 1024-pixel ScanComp source crop.
- All four hmi825 BigTIFF first IFDs are **23337 × 21006**. TIFF density candidates (x/y µm/px): hms360 **0.230107/0.230123**, hstp1000 **0.242535/0.242647**, lcap450 **0.263062/0.263062**, phufs **~1,000,000/~1,000,000** (implausible). The first three OME fragments contain similar candidates; Philips lacks a useful candidate. Descriptions identify **pyvips 2.2.2** and XML parsing fails with an unclosed token. These are *candidate tags*, **not** authenticated final registered-export scale.
- Exploratory FIT-only affine registration diagnostics and independent 256-pixel held-out visual QC exist. Fitted affine singular values near one and identical dimensions are compatible with approximately aligned pixel geometry at sampled sites. They **do not prove** a common physical µm/px, publisher VALIS export resampling policy, or large-field non-rigid registration.
- FIT affine convention: **target_pixel = affine(reference_pixel)**. Never refit on held-out matches, use held-out target dx/dy to adjust transforms, or infer physical scale from near-identity matrices.
- Historical 256-pixel coordinates are **top-left**, not centers: the continuous-coordinate center is **(x+128, y+128)**. Keep original eight sites and prior local pilot commit/results unchanged.
- Original frozen DINOv2: **facebook/dinov2-base** revision **f9e44c814b77203eaa57a6bdbbd535f21ede1415**. Original PA-NF release: **MatthewVaishnav/paired-acquisition-neural-factorization** revision **b5de3cf9c062cb0d5623628165098c26923309c7**, 25 checkpoints (5 folds × 5 seeds) and five fold standardizers. Verify original source/model parity and SHA256 before inference. The previous pilot local commit **9bfcbbf3** is not in this PR's remote base.

## 1. Two separate experiments

| | Track A — registered-pixel context sensitivity | Track B — historical physical-field matching |
|---|---|---|
| Question | Does frozen PA-NF behavior depend on *source pixel context*? | Does it persist for fields physically comparable to SCORPION? |
| Units | Reference level-0 **pixels**, never µm | Authenticated **µm**, then reference pixels |
| Sizes | Predeclare 256, 512, 1024 square reference pixels at the same centers | 800 × 800 µm, x/y source extents calculated from authenticated final registered-export scale |
| Presentation | Exact historical DINOv2 processor; log any added resampling | Entire physical field resized to 1024 × 1024 RGB **before** historical processor |
| Sites | Original eight centers; retrieval only if no overlap at that size; otherwise separately frozen pixel gallery | **New** nonoverlapping reference-only tissue gallery, frozen after scale authentication |
| Physical scale required? | **No**, forbid physical-equivalence language | **Yes**, fail closed |
| Current status | Preflight available; enlarged-field tiled warping and QC **pending** | **BLOCKED** pending authoritative registered-export scale |

Track A cannot be used to bypass Track B's physical calibration gate or to support G0.

## 2. Admission gates

**S0 — source integrity (both):** Verify filenames, dimensions, byte counts and SHA256 against the original registration receipt. Preserve source hashes and hash of the receipt. The preflight has optional streaming SHA verification; a size-only match is not source authentication.

**G0 — fixed pixel-coordinate mapping (both):** Read the original FIT-only affine matrices and FIT/HELDOUT provenance. Record singular values, source hashes and QC receipt. Near-one singular values are a diagnostic screen, not certification of a common physical grid. Reject source mismatch, unbounded sampling, or use of held-out offsets for fitting.

**P0 — final registered-export physical scale (Track B only):** Obtain an authoritative chain for **reference-grid x/y µm per pixel**, not just original optical scanner resolution. Suitable evidence could be publisher export/resampling configuration and code, original reference calibration, transformation order and resulting output pixel extent, or an independently calibrated physical landmark with uncertainty and provenance. Record version, reference image, transformations, anisotropy, input/output dimensions and hashes. Require independent review. **No user-supplied numeric mpp override may auto-PASS.** The current TIFF/OME candidates, including Philips, **do not authenticate P0**. Until resolved: **STOP Track B**.

**P1 — physical crop geometry (Track B only, after P0):** With authenticated reference mx,my µm/px, calculate width=round(800/mx), height=round(800/my), record physical rounding error, and do not assume isotropy or use target TIFF legacy mpp separately. Apply fixed reference->target transform; present each entire matched field as 1024 × 1024 RGB before the historical HF processor. Check channel order, interpolation, boundaries and provenance.

## 3. Sites and leakage

**Legacy eight-site manifest:** Preserve original CSV byte-for-byte. Convert top-left 256-pixel x/y to centers x+128,y+128. At each Track A size, calculate half-open pixel boxes and all pairwise overlaps. If overlapping, same-region similarity can be descriptive with a dependence caveat, but **hard-negative and top-1 retrieval for that scale are inadmissible** until a separately locked nonoverlapping gallery exists. Never select sites after seeing model results.

**Track A pixel gallery if needed:** CSV columns **site_id,center_x_px,center_y_px**. Minimum five for a geometry-only check, preferably at least eight for retrieval. Require every 1024-pixel box in reference bounds and at least **64 reference pixels** of axis-aligned clearance between boxes; unique IDs and centers. This is an engineering guard, **not** proof of biological independence. Choose using *reference-image* tissue occupancy, texture, focus and bounds only, not PA-NF output or target-scanner similarity. Freeze SHA and candidate/exclusion audit before inference.

**Track B physical gallery after P0:** Select a *new* reference-only tissue gallery of nonoverlapping 800-µm rectangles with predeclared physical clearance, bounds, and non-rigid sampling support. Never relabel the pixel gallery as physical. If adequate fields cannot be found, STOP. Previous six-pair overlap estimates are **conditional on unverified metadata**, not observed physical measurements.

**Counts:** For N sites, observation count is **4N**, not automatically 32. Freeze any exclusions *before* inference. Overlapping fields cannot serve as independent wrong-region negatives.

## 4. Enlarged-field extraction and QC

**E0 — bounded tiled extraction:** Current first-IFD WSIReader has a **4,194,304-pixel bounded read cap**, and the existing FIT-affine warp helper allows at most **512 output pixels**. Do **not** pass 1024 or 3000+-pixel fields to it. Implement a deterministic tiled extractor with fixed output reference grid, bounded target sampling and interpolation guard per tile, synthetic oracle comparison, seam checks, coverage and out-of-bounds rejection, and per-tile source-window manifest. Never change the FIT transform silently.

**E1 — large-field correspondence:** Predeclare distributed subfield QC (e.g. 3×3 positions and borders) for every region/scanner. Evaluate fixed-transform overlays/checkerboards and local correspondence without optimizing on held-out data. Reject/record local non-rigid mismatch and coverage failure. The existing 256-pixel held-out QC does **not** certify 512-, 1024-pixel or physical fields. This gate is mandatory for Track A enlarged contexts as well.

**E2 — frozen encoder parity:** Use exact pinned processor and DINOv2 CLS last_hidden_state[:,0,:] 768-D, original RGB order, normalization and library revisions. Validate historical feature parity on a known patch. Hash presentation images and record source hash, center, dimensions, affine, tile manifest, resize and processor version. Do not run models until E0–E2 pass.

## 5. Frozen representation/metrics contract

- Compute raw **X768** once per RGB presentation and preserve it as a baseline.
- For each fold f compute **X768_standardized_f = (X768−mean_f)/std_f** with the original fold standardizer. Feed that exact tensor to all five checkpoints of fold f, unless original inference code wraps standardization internally (then verify parity and do **not** standardize twice). **Never** feed unstandardized raw X768 to a model requiring standardized input.
- Run **all 25 frozen** fold×seed checkpoints, no retraining, tuning, checkpoint selection, threshold search or re-registration. Record B256 and A64. A64 is an acquisition branch, not a scanner classifier or proof of factorization.
- Report raw X768, fold-standardized X768, B256 and A64. Different feature dimensions and scales prohibit naive comparisons of absolute Euclidean variance. Record L2 norms, **SS/(R×S×D)**, and a separate unit-L2-normalized analysis. Optional predeclared 256/64-dimensional random-projection controls must be label-blind and reported separately.
- Per checkpoint, scanner direction and site: same-region cosine; nearest *nonoverlapping* wrong-region cosine; hard-negative margin; top-1 retrieval; count of worsened directional queries and adverse cases. No self-pairing or mixing field sizes.
- For balanced R regions × S=4 scanners × D features, with grand mean g, region mean r_i, scanner mean s_j, compute:
  - **SS_region = S Σ_i ||r_i−g||²**
  - **SS_scanner_main = R Σ_j ||s_j−g||²**
  - **SS_interaction_plus_error = Σ_ij ||z_ij−r_i−s_j+g||²**
  - **SS_total = sum of the three components**
  Report **absolute SS**, **SS/(R×S×D)** and **SS/SS_total** independently. The last term is interaction **plus unreplicated error**, not pure acquisition. Within-region dispersion and scanner main effect are different estimands. Fractions can improve because region spread increases. The contract module computes this decomposition for synthetic inputs.
- One slide is **one biological sampling unit**. No population confidence intervals or claims of biological sufficiency. Twenty-five correlated checkpoints are **not 25 independent patients**. Additional independently sampled slides/patients are required for generalization.

## 6. Stop matrix and receipts

| Gate | Track A | Track B | If failed |
|---|---|---|---|
| Source SHA, dimensions, FIT receipt | Required | Required | STOP both |
| Final registered-export µm/px | Not applicable | **Required; blocked** | STOP B |
| Frozen nonoverlapping gallery | Required for retrieval | Required for retrieval | STOP retrieval |
| Tiled extraction and large-field QC | Required at 512/1024 | Required at physical size | STOP affected inference |
| Exact encoder parity and 25 checkpoints | Required | Required | STOP affected inference |
| Independent slides and labels | Not available | Not available | No G0 or biological/clinical claim |

Produce only after execution: metadata receipt, source-hash manifest, immutable sites and overlaps, tiled extraction synthetic/coverage receipts, fixed-transform large-field QC, presentation-image hashes, encoder parity, per-checkpoint arrays, directional retrieval tables, absolute/fractional variance decomposition, adverse cases and interpretation. Each receipt must identify the exact gates **actually** passed; never invent completed work.

## 7. What is implemented now

- **Pure contracts:** scripts/external_multiscanner/scancomp_panf_001r_contract.py — candidate TIFF density, malformed OME fragment values, top-left-to-center conversion, half-open box overlap and clearance, affine singular values, unit L2 normalization, balanced variance decomposition.
- **Read-only preflight:** scripts/external_multiscanner/scancomp_panf_001r_preflight.py — first-IFD metadata only, optional SHA256 verification, FIT receipt check, legacy-site overlap scenarios and optional pixel-gallery geometry gate. **Always reports physical calibration unauthenticated**.
- **Windows launcher:** scripts/external_multiscanner/Run-ScanCompPANF001RPreflight.ps1.
- **Synthetic tests:** tests/test_scancomp_panf_001r_contract.py; CI includes Linux Python compile/unittest and Windows PowerShell parser.
- **Not implemented or executed:** local pilot commit/results, publisher scale authentication, tissue-only gallery selection, tiled extraction, large-field QC, 25-checkpoint inference, or real ScanComp model runs. CI tests are **synthetic contracts**, not data-level validation.

### PowerShell — no ZIP download

Run from the existing local Git checkout, without touching the pilot worktree:

~~~powershell
Set-Location "$env:USERPROFILE\scancomp-pr104"
git fetch origin scancomp/panf-001r-two-track-protocol-20261110
git worktree add --detach "$env:USERPROFILE\scancomp-panf-001r" origin/scancomp/panf-001r-two-track-protocol-20261110
Set-Location "$env:USERPROFILE\scancomp-panf-001r"
python -m unittest -v tests.test_scancomp_panf_001r_contract
& .\scripts\external_multiscanner\Run-ScanCompPANF001RPreflight.ps1 -VerifyHashes
~~~

Writes only a JSON preflight receipt to Documents\PA_NF_ScanComp_Intake\hmi825_panf_001r_preflight.json. No TIFF pixel loading or model execution. For legacy overlaps add **-LegacySites 'C:\path\to\LOCKED_HMI825_SITES.csv'** if the original file exists. Do **not** reconstruct an unverified locked site list. For a new frozen pixel-only gallery use **-PixelGallery 'C:\path\to\sites.csv'**; this checks geometry only.

### Evidence and sources

- User-supplied 2026-11-10 first-IFD TIFF audit and OME description output; existing local registration and held-out QC receipts, to be source-hash verified.
- [VALIS documentation](https://valis.readthedocs.io/en/latest/) describes conversion and registration capabilities but **does not** establish this publisher's final export calibration.
- [PA-NF project source](https://github.com/matthewvaishnav/computational-pathology-research) and [PA-NF pipeline report](https://matthewvaishnav.github.io/computational-pathology-research/paired-acquisition-neural-factorization-pipeline.pdf). Verify historical SCORPION image extraction and preprocessing against the immutable release, not this narrative alone.

**Decision:** Track A **preflight implementation available, inference blocked pending enlarged-field extraction/QC and manifest**. Track B **blocked at physical-scale provenance**. Research-only.
