# Table 1: Dataset and Evidence Boundary

**Intended manuscript location:** Section 2 (Problem Setup)
**Source commits:** e4819c42, d018c924, bec06eb4, 3450ede2, aa8d0596
**Source files:** result_to_claim_map.csv, paired_acquisition_claim_ledger.md, manuscript_draft.md

## Dataset Summary

| Dataset | Backbone(s) | Samples | Regions | Patches | Scanners | Biological Category Labels? |
|---|---|---|---|---|---|---|
| Canine SCC | DINOv2-Base | 44 | 805 | 4,025 | 5 | **Yes** — 7 expert-annotated tissue categories |
| SCORPION | DINOv2-Base, Phikon, ResNet50 | — | — | — | 5 | **No** — scanner/region metadata only |

## Evidence Boundary

| Evidence Type | Canine SCC DINOv2 | SCORPION DINOv2 | SCORPION Phikon | SCORPION ResNet50 |
|---|---|---|---|---|
| Category probe (scanner + biological) | ✓ CLAIM_2, CLAIM_3, CLAIM_4 | ✗ (no labels) | ✗ (no labels) | ✗ (no labels) |
| Scanner probe | ✓ | ✓ | ✓ | ✓ |
| Pair/tissue retrieval (paired cosine, top-1) | ✓ | ✓ CLAIM_1 | ✓ CLAIM_1 (cross-backbone) | ✓ CLAIM_1 (cross-backbone) |
| Acquisition branch retrieval leakage | ✓ | ✓ CLAIM_4 | ✓ CLAIM_4 (cross-backbone) | ✓ CLAIM_4 (cross-backbone) |
| Branch separation (purity, cross-cov) | ✓ CLAIM_2 | ✓ (scanner purity only) | ✓ (scanner purity only) | ✓ (scanner purity only) |
| Scanner-heldout transfer | ✓ CLAIM_2 | ✗ (no labels) | ✗ (no labels) | ✗ (no labels) |
| Sample-disjoint transfer | ✓ CLAIM_2 | ✗ (no labels) | ✗ (no labels) | ✗ (no labels) |
| Scanner-confounded robustness | ✓ CLAIM_2 | ✗ (no labels) | ✗ (no labels) | ✗ (no labels) |
| Acquisition swapping | ✓ CLAIM_5 | ✗ (no labels) | ✗ (no labels) | ✗ (no labels) |

## Baseline Anchoring

| Baseline | Method | Role | Evidence Availability |
|---|---|---|---|
| oldstyle_keep_k4 | Centroid/QR linear projection | **Strongest raw scanner-removal baseline** | Scanner probe 0.200 (chance), category probe 0.400. No acquisition branch produced. |
| oldstyle_removed_k4 | Centroid/QR removed component | Reference for structured decomposition comparison | Scanner capture 0.538, category leakage 0.242 |
| pca_removal_k32 | PCA scanner-subspace removal | PCA comparison baseline | Scanner probe 0.649, category probe 0.289 |
| original_frozen_features | Frozen DINOv2 | Pre-factorization reference | Scanner probe 0.866, category probe 0.407 |

## Key Rules

1. **SCORPION category claims forbidden.** Use "tissue/pair-retrieval" only.
2. **Canine SCC DINOv2 = labeled category anchor.** All category-probe, branch-separation, bottleneck-category-leakage, and swapping claims rest on this single dataset.
3. **Oldstyle centroid/QR = strongest raw scanner-removal baseline.** All scanner-removal comparisons must reference this baseline.
4. **Acquisition swapping = canine SCC DINOv2 only.** Single-dataset, single-backbone. SCORPION has no category labels.
5. **No clinical, diagnostic, or deployment language.** Research audit only.
