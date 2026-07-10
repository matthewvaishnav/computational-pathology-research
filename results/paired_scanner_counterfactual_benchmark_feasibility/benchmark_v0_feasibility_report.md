# Paired Scanner Counterfactual Benchmark v0 Feasibility Report

Generated: 2026-07-10T10:34:06

## Benchmark layers feasible now

- Feature-space: external_multiscanner_caninescc, scorpion.
- Decoder-space: external_multiscanner_caninescc, scorpion.
- Pixel-space: no dataset is marked ready by this audit; candidates require explicit registration/QC review before reconstruction metrics.

## Existing artifact support

### external_multiscanner_caninescc

- Scanner IDs: available (data/external_multiscanner_caninescc/manifest.csv columns=scanner_id; data/external_multiscanner_caninescc/scanner_geometry_quantiles.csv columns=scanner_id; +240 more).
- Paired region IDs: available (data/external_multiscanner_caninescc/manifest.csv columns=region_id,sample_id; data/external_multiscanner_caninescc/sample_folds.csv columns=sample_id; +254 more).
- Category labels: available (data/external_multiscanner_caninescc/manifest.csv columns=category_id,category_name; data/external_multiscanner_caninescc/patch_manifests/full_patch_manifest.csv columns=category_name; +22 more).
- Frozen feature arrays: available (results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz; results/external_multiscanner_caninescc/features/fold_0_phikon.npz; +9 more).
- Branch embeddings: biological=available, acquisition=available.
- Decoder/composition weights: available (results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold/fold_0/runs/paired_reference_seed_911/checkpoint.pt (decoder keys found); results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold/fold_0/runs/paired_reference_seed_912/checkpoint.pt (decoder keys found); +3 more).
- Patch images: available (data/external_multiscanner_caninescc/patches/cs2/scc_01__region_0002__cs2__ad44284b82.jpg; data/external_multiscanner_caninescc/patches/cs2/scc_01__region_0007__cs2__af123fe3ee.jpg; +14 more).
- WSI paths: metadata_only (metadata_only: data/external_multiscanner_caninescc/manifest.csv file_name=scc_01_cs2.tif; metadata_only: data/external_multiscanner_caninescc/splits/fold_0_manifest.csv file_name=scc_01_cs2.tif; +6 more).
- Coordinates: available (data/external_multiscanner_caninescc/manifest.csv columns=bbox,bbox_center_x,bbox_center_y,bbox_height,bbox_width,bbox_x,bbox_y; data/external_multiscanner_caninescc/splits/fold_0_manifest.csv columns=bbox,bbox_center_x,bbox_center_y,bbox_height,bbox_width,bbox_x,bbox_y; +13 more).
- Registration metadata: available (data/external_multiscanner_caninescc/manifest.csv columns=correspondence_basis; data/external_multiscanner_caninescc/patch_manifests/full_patch_manifest.csv columns=orientation_normalization_degrees; +31 more).
- Registration confidence/QC: available (results/external_multiscanner_caninescc/adaptive_crop_audit/adaptive_crop_plan.csv columns=inside_image_fraction,padding_fraction; results/external_multiscanner_caninescc/geometry_qualified/excluded_regions.csv columns=inside_image_fraction,padding_fraction,region_max_padding_fraction; +3 more).
- Pixel-space status: candidate_requires_registration_qc_review.

### scorpion

- Scanner IDs: available (data/scorpion/manifest.csv columns=scanner_id; data/scorpion/splits/fold_0_manifest.csv columns=scanner_id; +620 more).
- Paired region IDs: available (data/scorpion/manifest.csv columns=region_id; data/scorpion/splits/fold_0_manifest.csv columns=region_id; +661 more).
- Category labels: not_found (not_found).
- Frozen feature arrays: available (results/scorpion/features/fold_0_dinov2_base.npz; results/scorpion/features/fold_0_phikon.npz; +9 more).
- Branch embeddings: biological=available, acquisition=available.
- Decoder/composition weights: available (results/scorpion/pathoalign_crossbackbone_transfer/phikon/fold_0/runs/paired_reference_seed_701/checkpoint.pt (decoder keys found); results/scorpion/pathoalign_crossbackbone_transfer/phikon/fold_0/runs/paired_reference_seed_702/checkpoint.pt (decoder keys found); +3 more).
- Patch images: available (data/scorpion/manifest.csv path=slide_1/sample_1/AT2.jpg; data/scorpion/splits/fold_0_manifest.csv path=slide_1/sample_1/AT2.jpg; +5 more).
- WSI paths: not_found (not_found).
- Coordinates: not_found (not_found).
- Registration metadata: not_found (not_found).
- Registration confidence/QC: not_found (not_found).
- Pixel-space status: future_only.

## Pixel-space requirements

Pixel-space reconstruction requires paired image paths, scanner-specific paired acquisitions, local region correspondence, patch coordinates, and registration confidence/QC rules. The audit treats patch files or path metadata as insufficient by themselves; registration/QC evidence is required before pixel metrics are reported.

## Category-label anchors

- external_multiscanner_caninescc.

## Pair/tissue-retrieval-only anchors

- scorpion.

## Explicitly unsupported claims

This audit does not support clinical validation, diagnostic performance, deployment, patient-care readiness, FDA readiness, HIPAA readiness, scanner bias solved, universal disentanglement proven, pixel-level acquisition modeling proven, factorization proven, scanner-free representation, breakthrough claims, perfect causal factorization, or solves scanner bias claims.

## Bottom line

Feature-space and decoder-space benchmark v0 are feasible for datasets with available artifacts in the capability matrix. Pixel-space reconstruction remains future work until registration/QC readiness is explicitly validated.
