# PathoAlign Oncology Identity Benchmark v0

## Core question

Has a pathology representation learned disease biology, or has it learned institutional and acquisition identity?

Pathology AI systems often receive whole-slide images that contain both biological signal and non-biological shortcut signal. Biological signal includes tissue architecture, tumor morphology, cellular patterns, disease state, and clinically relevant spatial organization. Shortcut signal includes scanner fingerprint, stain chemistry, laboratory protocol, hospital identity, cohort composition, annotation-source identity, and federated-client identity.

The benchmark asks whether a representation preserves the biological signal while making institutional and acquisition identity measurable, separable, and less dominant.

## Benchmark objective

Given pathology representations from whole-slide images, patches, matched tissue regions, or federated clients, evaluate whether biological identity is preserved while scanner, stain, site, hospital, cohort, annotation-source, and federated-client identity become less predictive.

This benchmark is intended to evaluate representation identifiability and robustness under institutional and acquisition confounding. It is not a clinical diagnostic validation by itself.

## Required measurements

| Measurement | Direction | Meaning |
|---|---:|---|
| Biological preservation | Higher | Same tissue, sample, disease, or morphology identity remains recoverable. |
| Same-region retrieval | Higher or preserved | Matched tissue regions remain close across acquisition conditions. |
| Cross-acquisition consistency | Higher | Paired or matched biological units become more consistent across scanner, stain, or site. |
| Scanner/site/stain probe | Lower | Acquisition identity is less decodable from the biological representation. |
| Federated client probe | Lower | Client or hospital identity is less decodable from the biological representation. |
| WSI task utility | Higher or preserved | Slide-level diagnostic or biological task signal is not destroyed. |
| Calibration shift | Lower | Confidence and calibration are less site-dependent. |
| Biological-acquisition separation | Higher | Biological and acquisition branches carry distinct, auditable information. |

## Required benchmark outputs

Each benchmark run should produce a compact evidence table with the following structure:

| Method | Biological preservation ↑ | Retrieval ↑/≈ | Cross-acquisition consistency ↑ | Scanner/site/client probe ↓ | WSI utility ↑/≈ | Calibration shift ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Raw frozen encoder | TBD | TBD | TBD | TBD | TBD | TBD |
| Baseline alignment method | TBD | TBD | TBD | TBD | TBD | TBD |
| PathoAlign ablation | TBD | TBD | TBD | TBD | TBD | TBD |
| PathoAlign full model | TBD | TBD | TBD | TBD | TBD | TBD |

The benchmark should also report confidence intervals, sample-blocked or client-blocked contrasts, and the number of biological units supporting each claim.

## Required baselines

PathoAlign should be evaluated against simple and strong alternatives, including:

- Raw frozen encoder features
- Stain normalization
- Color and stain augmentation
- CORAL, MMD, or optimal-transport feature alignment
- Domain-adversarial representation learning
- Contrastive scanner, site, or acquisition invariance
- Federated baselines such as FedAvg-style training where applicable
- PathoAlign ablations
- PathoAlign full model

The purpose is not only to improve downstream performance. The purpose is to test whether the representation has learned biological structure instead of institutional shortcuts.

## Initial evidence packages

The first version of the benchmark is supported by focused PathoAlign evidence packages:

| Evidence package | Role |
|---|---|
| External paired-scanner canine SCC validation | Tests acquisition-identity reduction and biological preservation under paired scanner acquisition. |
| Matched-budget biological-pair allocation study | Tests whether representation alignment benefits more from broader biological pair diversity than repeated anchors. |
| TransnnMIL slide-level utility experiments | Tests whether biological representations remain useful for whole-slide learning. |
| PathologyFL client/site identity experiments | Tests whether institution identity remains controlled under decentralized multi-site training. |

## Four-pillar mapping

| Pillar | Benchmark role |
|---|---|
| PathoAlign | Representation-identifiability and biological/acquisition separation layer. |
| TransnnMIL | Whole-slide utility layer for biological representations. |
| PathologyFL | Federated multi-site training layer under client and institution shift. |
| Evidence infrastructure | Frozen tables, verification scripts, child repositories, confidence intervals, and claim boundaries. |

## Claim boundary

This benchmark evaluates whether pathology representations encode disease-relevant biology rather than institutional or acquisition shortcuts. It supports representation-level and robustness claims. It does not, by itself, establish clinical diagnostic safety, regulatory readiness, or prospective patient benefit.

## Short benchmark thesis

PathoAlign asks whether pathology AI has learned disease biology or merely learned the institution that produced the slide.
