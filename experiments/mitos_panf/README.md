# PA-NF MITOS-ATYPIA-14 two-slide localization pilot

**Status: code prepared; real all-96 feature extraction and PA-NF model fitting NOT executed.**
This directory is development-only and does not supersede CLAIM_BOUNDARY.md,
the registered SCORPION capacity-matched result, or the negative crossed-intervention gates.

## New empirical findings (user Windows host, 21 Oct 2026)

- A07/H07 and A17/H17, two distinct physical slides, 144 registered fields.
- Original 224px paired-patch corpus: 1,614 pairs (693/921), 21 flagged.
- Frozen DINOv2 Base (HF revision f9e44c814b77203eaa57a6bdbbd535f21ede1415), 3,228 CLS embeddings.
- Raw 224px within-field exact pair retrieval A->H 86.25%, H->A 87.30%;
  fit-only scanner mean-shift improved retrieval to about 89-91% on held-out slides.
- Original 224px single-slide-trained Aperio mitosis probes on the OTHER physical slide:
  AUROC ~0.485 and ~0.510.
- Retrospectively selected 96px central annotated crops, 392 paired candidate patches:
  52 mitosis / 340 non-mitosis, 26 positives on each slide.
  Slide 07->17: Aperio AUROC 0.650148, Hamamatsu AUROC 0.710059,
  Hamamatsu FIT mean-shift 0.708950.
  Slide 17->07: Aperio 0.667832, Hamamatsu 0.667541,
  Hamamatsu FIT mean-shift 0.666084.
- Cell-centered report SHA-256:
  BDDDB702BC1EB739B41B40EC172E954D0925266108134EE5E0C19E04B3CEB882.

The change of crop size was chosen *after* seeing the 224px biological-label
failure, so its improvement is **exploratory outcome-adaptive analysis**, not a
locked confirmation. All labels refer to A-annotated central objects, not whole
patches; H correspondence is assumed, not independently re-annotated.
The 392 candidates are not independent tissue observations and only TWO
physical slides exist. Strong clinical, patient, biomarker, or PA-NF superiority
claims are invalid here. PA-NF has **not** been evaluated on MITOS yet.

## Reproducible next actions

From the repository root on the **dedicated research branch**, run:

    python -m pytest -q tests/test_mitos_panf_pilot.py
    python experiments/mitos_panf/extract_all96_features.py

This extracts *all* 1,614 pairs as 96x96 center crops (not only the 392
annotated pairs) with the original pinned encoder, source SHA checks, and
mandatory feature parity against the previously materialized 392 crops.
Do **not** relabel the other 1,222 unannotated grid crops as non-mitotic.
Output remains at
PA-NF-MITOS-ALL96-DINOV2-v1.incomplete until the Python process exits;
on Windows, verify the report before renaming to PA-NF-MITOS-ALL96-DINOV2-v1.
No prior output is overwritten. Real weights may download from HF if uncached.

After verifying complete extraction, run a one-epoch execution test:

    python experiments/mitos_panf/run_two_slide_pilot.py --smoke

A smoke run is not evidence. If and only if smoke outputs are valid,
the frozen exploratory comparison can be run as:

    python experiments/mitos_panf/run_two_slide_pilot.py

Again, a staged .incomplete output is finalized only *after* Python exits.

## Frozen experimental design (not confirmatory)

Two folds: FIT slide 07 -> TEST 17, and FIT 17 -> TEST 07.
Every paired 96px sample from the FIT slide enters *unsupervised* training,
regardless of annotation status. The TEST slide is not used for training,
preprocessing fit, hyperparameter selection, or model selection.
Fixed 75 epochs, seeds 801/802/803, pair-region batch size 32,
AdamW learning rate 3e-4, weight decay 1e-4, no early stopping.
All models receive train-slide-only 768-D standardization and exactly
two registered scanner views per region.

Neural methods use the same 256-D biological and 64-D acquisition branches,
hidden width 512, and joint decoder, with the identical two-scanner
architecture. The neutral two-branch control switches OFF scanner
objectives, scanner-dependence, cross-covariance and gradient reversal.
The PA-NF variant uses the SCORPION frozen full objective:
scanner adversary 0.5, scanner acquisition 0.5, scanner dependence 20.0,
cross-covariance 0.05, gradient reversal 1.0. Core n_scanners=2 adaptation
must not alter default n_scanners=5 SCORPION behavior.

Non-neural references: raw DINOv2 and FIT-only scanner centroid mean-shift
of unit features. Fixed FIT-only balanced logistic probes L2=0.01 on the
Aperio annotations, then apply frozen probes to held-out Aperio/Hamamatsu;
report AUROC/AP (including annotation correspondence subset), scanner
probe balanced accuracy from FIT to held-out TEST, acquisition-branch
scanner capture, within-field exact-row retrieval and collapse checks.

This is NOT patient-level inference: no case mapping, only one slide per fold,
correlated overlapping samples and outcome-adaptive crop decision. Report
descriptive per-slide/per-seed results only; no pooled confidence interval
using patch rows as if independent. Do not promote PA-NF if scanner
suppression improves while held-out mitosis discrimination degrades, or
if simple correction does as well or better.

## Source locations

Main code: src/models/scorpion_pathoalign.py
Original runner: experiments/scorpion/run_pathoalign_projection.py
Capacity-matched SCORPION variant: experiments/scorpion/run_pathoalign_capacity_matched_ablations.py
Scientific boundary: CLAIM_BOUNDARY.md

Host-only frozen input data in %USERPROFILE%\Downloads.
No original TIFFs, patch arrays, patient information or private model
artifacts are uploaded by this research branch.
