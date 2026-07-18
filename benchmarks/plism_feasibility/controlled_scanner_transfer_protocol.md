# Controlled scanner-robustness transfer protocol

Status: draft research artifact for PR #28. No results are claimed.

## Scientific question

Do robustness rankings measured on heterogeneous multi-center pathology datasets transfer to a controlled same-tissue, multi-scanner setting?

This question is intentionally narrower than showing that scanner bias exists. Recent work already demonstrates scanner-specific embedding shifts, calibration failures despite stable AUC, and broad robustness deficits across pathology foundation models. The unresolved question is whether robustness metrics and model rankings remain valid when technical variation is isolated more cleanly.

## Datasets and roles

### SCORPION: primary controlled benchmark

Use the five-scanner, spatially aligned same-tissue images as the primary scanner-only benchmark. Preserve the physical hierarchy:

- 48 physical slides;
- 10 regions per slide;
- 480 regions total;
- five scanner images per region;
- 2,400 images total.

All splitting, confidence intervals, permutation tests, and bootstrap resampling must operate at the physical-slide level. Region-level resampling is reported only as a pseudoreplication sensitivity analysis and must not support the main claims.

### PathoROB: external ranking source

Reuse published model-level robustness results where definitions and model checkpoints are compatible. Do not rerun the complete benchmark merely to reproduce its primary finding. Treat its center factor as a mixture of staining, preparation, scanner, section thickness, and institution-specific effects rather than a scanner-only label.

### PLISM: external replication

Use same-stain, same-section scanner comparisons only for scanner claims. Cross-stain comparisons remain preparation-plus-section comparisons because each staining condition is applied to a different serial section.

## Locked primary endpoints

For each accessible overlapping encoder:

1. Paired cosine distance between scanner views of the same region.
2. Same-region retrieval accuracy across scanners.
3. Scanner-identification accuracy from frozen embeddings.
4. Biological-content retention using tissue or slide identity where labels permit.
5. Worst scanner-pair consistency across all ten scanner pairs.
6. Downstream prediction instability, label-flip rate, Brier score, and calibration error when a valid supervised endpoint is available.

No single metric is designated as a universal robustness score before analysis.

## Primary hypotheses

- H1: Published multi-center robustness rankings will correlate only imperfectly with controlled scanner-only rankings.
- H2: At least one material rank reversal will occur between PathoROB and SCORPION.
- H3: Average consistency will conceal a worse scanner-pair failure for at least one model.
- H4: Scanner identifiability and paired embedding instability will not be interchangeable metrics.
- H5: Post-hoc robustification may improve scanner invariance while reducing biological-content retention; both must be measured.

## Statistical plan

- Use Spearman rho and Kendall tau for cross-benchmark ranking agreement.
- Compute uncertainty using a cluster bootstrap over the 48 physical slides.
- Report all ten scanner-pair estimates, not only a pooled mean.
- Quantify rank uncertainty by recording each model's bootstrap rank distribution.
- Report the difference between naive region bootstrap and slide-cluster bootstrap intervals.
- Use paired permutation tests that swap scanner labels within matched regions while preserving slide clusters.
- Correct secondary pairwise comparisons using a declared false-discovery-rate procedure.
- Treat effect sizes and uncertainty as primary; p-values are supporting evidence only.

## Leakage and preprocessing controls

- Never split regions from the same physical slide across train and test folds.
- Lock encoder checkpoint, input size, magnification interpretation, normalization, and feature layer before outcome analysis.
- Do not tune normalization or robustification hyperparameters on the final SCORPION test set.
- Evaluate raw images and each normalization method as separate declared conditions.
- Record interpolation, color conversion, crop policy, tissue masking, and scanner-specific missingness.
- Do not infer native-resolution nuclear robustness from heavily downsampled public derivatives.

## Novelty boundaries

This study must not claim novelty for establishing that:

- pathology foundation models encode scanner or center information;
- AUC can remain stable while calibration changes;
- stain normalization or representation correction can improve robustness;
- scanner-aware consistency training can improve scanner generalization.

The proposed contribution is instead a validity study of robustness measurement: whether metrics and model rankings transfer from heterogeneous center variation to controlled paired scanner variation, and where they fail.

## Publication-ready minimum result

A publishable result requires at least one of the following with slide-clustered uncertainty:

1. weak or unstable rank agreement between PathoROB and SCORPION;
2. a reproducible rank reversal involving a commonly used accessible model;
3. disagreement between scanner-identification, paired-distance, retrieval, and calibration metrics;
4. a robustification method that improves one axis while harming biological-content retention;
5. replication of the same ordering failure on PLISM's same-section scanner pairs.

A null result is still informative only if confidence intervals exclude practically important disagreement and the overlapping model set is sufficiently broad.

## Stop conditions

Do not escalate to full-image download or large-scale model extraction if:

- fewer than four accessible overlapping models can be evaluated reproducibly;
- physical slide identifiers cannot be recovered reliably;
- scanner pairing cannot be verified from public metadata;
- preprocessing differences across models cannot be standardized or transparently stratified;
- the controlled benchmark duplicates a recently published experiment without a distinct endpoint.

## Required manifest fields

Every observation record must include:

- dataset_id;
- physical_slide_id;
- region_id;
- scanner_id;
- stain_id where applicable;
- tissue or biological label where available;
- image path and checksum;
- native and delivered microns-per-pixel;
- crop dimensions;
- encoder checkpoint identifier;
- preprocessing fingerprint;
- split assignment;
- exclusion reason if omitted.

## Evidence informing this protocol

- PathoROB, Nature Communications, published 11 June 2026: broad multi-center robustness benchmark across 20 foundation models.
- Scanner-Induced Domain Shifts Undermine the Robustness of Pathology Foundation Models, 2026: scanner-specific embeddings and calibration changes can persist despite stable AUC.
- Reliability of foundation models for image retrieval in histopathology, npj Imaging, published 27 May 2026: co-registered same-slide scanner analysis for retrieval.
- SCORPION, 2025: five-scanner same-tissue benchmark and scanner-consistency framework.
- Low-Rank Adaptations for increased Generalization in Foundation Model features, PMLR 2026: external robustness evaluation and adaptation overlap that narrows the novelty claim.
