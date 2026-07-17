# PLISM bounded feasibility experiment protocol

Status: pre-analysis design draft

## Research question

After holding registered field of view and tissue label as constant as the public PLISM metadata allows, how much of a pathology foundation model embedding is associated with scanner domain, staining condition, and their interaction?

This is an exploratory representation study. It is not a causal estimate of preparation or scanner effects because PLISM does not expose complete preparation-batch, scan-batch, acquisition-order, physical-device, exact section-distance, or immutable source-event provenance.

## Unit structure

The analysis must preserve four distinct units:

- **registered image group**: common field-of-view correspondence supplied by PLISM;
- **staining condition**: one of the public PLISM stain labels;
- **scanner domain**: one of the public WSI scanner labels;
- **tissue category**: one of the 46 reported tissue types.

Different staining conditions come from serial sections and must never be represented as the same physical section. Scanner comparisons within a staining condition are the closest available same-section contrast.

## Stage 0: manifest and contradiction gate

Before image download or embedding extraction:

1. fetch both Figshare article manifests;
2. retain article version, file identifier, advertised byte size, download URL, and supplied checksum;
3. parse the registered-tile list and filenames into explicit tissue, stain, scanner, coordinate, and image-path fields;
4. reject duplicate image paths and duplicate identities;
5. reject groups with more than one image for the same stain-scanner identity;
6. reject scanner comparisons that do not share stain and registered group;
7. label stain comparisons as serial-section correspondences rather than same-section pairs;
8. print the executed crossing matrix and all missing cells.

No embeddings are generated until this gate passes.

## Stage 1: storage-safe smoke subset

Use the registered PLISM-wsi tiles rather than original WSIs.

Select deterministically:

- three routine-like stains: GV, GVH, and GMH, matching the restricted external-validation subset reported by FEATMAP;
- two scanners with complete overlap for the selected groups;
- a balanced tissue subset selected only after the manifest reveals actual availability;
- at most 25 registered groups per tissue for the first smoke run.

The selected group IDs must be derived from a fixed lexical ordering or a recorded random seed. No manual cherry-picking after viewing embeddings.

## Stage 2: frozen feature extraction

Begin with one accessible frozen encoder already supported by the repository. Do not train or fine-tune the encoder during the feasibility stage.

For each image, record:

- encoder name and exact checkpoint identifier;
- preprocessing transform and input resolution;
- source file identity and checksum;
- tissue, stain, scanner, and registered group;
- embedding dimension;
- software and hardware environment.

Embeddings must be written once and treated as immutable inputs to subsequent analyses.

## Stage 3: paired scanner analysis

Within each staining condition and registered group, compare embeddings across scanners.

Primary descriptive quantities:

- paired cosine distance;
- Euclidean distance after a declared normalization rule;
- top-k cross-scanner retrieval of the matching registered group;
- scanner classification accuracy under group-aware cross-validation;
- tissue classification transfer from one scanner to another.

The split unit must be registered group or a higher biological grouping where available. Tiles from one registered group cannot be split across train and test.

## Stage 4: serial-section stain analysis

Across staining conditions, compare only registered correspondences and preserve the serial-section limitation.

Primary descriptive quantities:

- within-scanner cross-stain embedding distance;
- cross-stain retrieval of corresponding fields;
- stain classification accuracy under group-aware cross-validation;
- tissue classification transfer across stains;
- sensitivity of conclusions to registration quality and tissue category.

These quantities measure association with public stain labels under serial-section correspondence. They do not isolate chemistry from section variation, batch, order, or handling.

## Stage 5: crossed decomposition

For groups with adequate stain × scanner coverage, fit a descriptive crossed model or variance decomposition with terms for:

- registered group;
- tissue category;
- staining condition;
- scanner domain;
- stain × scanner interaction.

The registered-group term is essential. A model that treats all tiles as independent is invalid for this design.

Report effect sizes and uncertainty. Do not rely on UMAP separation or p-values alone.

## Stage 6: harmonization comparison

The novelty target is not to repeat FEATMAP. FEATMAP already demonstrates an embedding-space affine correction trained on paired acquisition domains and externally evaluates scanner harmonization on a restricted PLISM subset.

The initial comparison should instead ask:

1. whether scanner suppression changes stain-associated structure;
2. whether stain suppression changes scanner-associated structure;
3. whether either correction preserves tissue retrieval and classification;
4. whether a transformation learned on one stain transfers to other stains;
5. whether transformations are stable across tissue categories.

Candidate baselines:

- uncorrected embeddings;
- per-domain centering and scaling;
- ComBat where assumptions are explicitly satisfied;
- orthogonal Procrustes on paired scanner observations;
- affine mapping comparable in scope to FEATMAP;
- the repository's existing scanner-residual method, if its input contract matches the executed matrix.

## Failure criteria

Stop or narrow claims when any of the following occurs:

- insufficient complete stain × scanner crossing;
- registered-group leakage across train and test;
- conclusions depend on one tissue category;
- scanner reduction is accompanied by major tissue-information loss;
- serial-section registration error dominates stain comparisons;
- transformations fail to transfer across held-out stains or tissues;
- metadata cannot unambiguously reconstruct the selected observations.

## Required outputs

- normalized source manifest;
- executed crossing matrix;
- exclusion ledger;
- immutable embedding manifest;
- paired-distance tables with group-aware uncertainty;
- retrieval and transfer results;
- scanner/stain predictability before and after correction;
- tissue preservation results;
- explicit limitations tied to missing provenance;
- a publication decision stating whether the result supports only tooling, a dataset note, a methods paper, or a larger prospective study.
