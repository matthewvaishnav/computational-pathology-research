# Future Work Plan

## Phase A: Benchmark v0

Use existing feature and decoder artifacts to implement the benchmark v0:

- dataset capability matrix;
- feature-space scanner/category/pair metrics;
- oldstyle centroid/QR baseline;
- decoder-space swap metrics;
- shuffled-pair and scanner-balanced controls;
- bounded report language.

Expected claim scope: observed scanner counterfactuals, paired acquisition as
supervision, feature-space counterfactual audit, and decoder-space factor-like
evidence.

## Phase B: Paired patch registration audit

If raw image data is available, audit whether paired patch or WSI data can
support pixel-space reconstruction:

- verify patch paths or WSI paths;
- verify same-region scanner pairs;
- verify coordinates and coordinate frames;
- compute or ingest registration confidence;
- define QC rejection rules;
- stratify by scanner pair and tissue/category labels when available.

Expected claim scope: pixel-space readiness audit only.

## Phase C: Pixel-level reconstruction baseline suite

After Phase B passes, implement pixel baselines:

- raw source patch;
- stain/color normalization;
- scanner-pair color transform;
- paired image translation without factorization;
- unpaired image translation;
- explicit biological/acquisition factor model.

Expected claim scope: pixel-space future work until validated against real
registered paired patches.

## Phase D: Cross-dataset paired scanner replication

Replicate feature-space and decoder-space audits across independently collected
paired scanner datasets:

- shared metric definitions;
- per-dataset capability matrices;
- scanner-pair stratification;
- category-label anchors separated from pair-retrieval-only anchors;
- conditional cross-dataset evidence.

Expected claim scope: conditional cross-dataset evidence, not universal claims.

## Forbidden claim boundaries

Do not claim universal disentanglement proven, pixel-level acquisition modeling
proven, clinical validation, diagnostic performance, deployment readiness,
patient-care readiness, FDA readiness, HIPAA readiness, scanner bias solved,
scanner-free representation, perfect causal factorization, breakthrough status,
or solves scanner bias.
