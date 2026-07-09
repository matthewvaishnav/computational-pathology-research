# Figure/Table Artifact Manifest — Paired-Acquisition Manuscript

**Branch:** paper/paired-acquisition-figure-table-assembly
**Generated:** 2026-07-09
**Purpose:** Complete inventory of all figure/table artifacts, intended manuscript locations, source commits, source files, and validation status.

---

## Artifact Inventory

### Table 1 — Dataset / Evidence Boundary

| Field | Value |
|---|---|
| Artifact files | `table1_dataset_evidence_boundary.csv`, `table1_dataset_evidence_boundary.md` |
| Intended manuscript location | Section 2 (Problem Setup) |
| Claims supported | All (CLAIM_1 through CLAIM_5) — foundational scoping table |
| Source commits | e4819c42, d018c924, bec06eb4, 3450ede2, aa8d0596 |
| Source files | result_to_claim_map.csv, paired_acquisition_claim_ledger.md, manuscript_draft.md |
| Validation status | PASS — all numbers traced to commits e4819c42, d018c924, bec06eb4, 3450ede2, aa8d0596 |

### Figure 1 — Pairing Ladder

| Field | Value |
|---|---|
| Artifact files | `figure1_pairing_ladder_data.csv`, `figure1_pairing_ladder_caption.md` |
| Intended manuscript location | Section 4 (Result 1: Pair Structure Matters) |
| Claims supported | CLAIM_1_PAIR_STRUCTURE |
| Source commits | e4819c42 (primary), d018c924 (cross-backbone) |
| Source files | `results/paired_acquisition_factorization_pair_structure_boundary_test/boundary_summary.csv`; `results/paired_acquisition_factorization_pair_structure_boundary_crossbackbone/boundary_summary.csv` |
| Rows | 20 (5 levels × 4 datasets/backbones) |
| Validation status | PASS — 20 rows traced to e4819c42, d018c924 |

### Figure 2 — Branch Separation

| Field | Value |
|---|---|
| Artifact files | `figure2_branch_separation_data.csv`, `figure2_branch_separation_caption.md` |
| Intended manuscript location | Section 5 (Result 2: Branch Separation) |
| Claims supported | CLAIM_2_BRANCH_SEPARATION |
| Source commits | bec06eb4 (primary), 3450ede2 (oldstyle), 1c527697 (consolidated) |
| Source files | `label_probe_summary.csv`, `oldstyle_residual_summary.csv`, `unified_separation_scoreboard_key_metrics.csv` |
| Rows | 8 representations (frozen, true_pair_bio, true_pair_acq, shuffled_bio, shuffled_acq, pca_k32, oldstyle_keep_k4, oldstyle_removed_k4) |
| Validation status | PASS — 8 representations traced to bec06eb4, 3450ede2, 1c527697 |

### Figure 3 — Baseline Scoreboard + Downstream Transfer

| Field | Value |
|---|---|
| Artifact files | `figure3_baseline_scoreboard_data.csv`, `figure3_baseline_scoreboard_caption.md` |
| Intended manuscript location | Section 6 (Result 3: Oldstyle Centroid/QR Baseline Boundary) |
| Claims supported | CLAIM_3_LINEAR_BASELINE_BOUNDARY |
| Source commits | 1c527697, 535eea18, 0d7cdc92, b5a9886e, c29a038d, 3450ede2 |
| Source files | `unified_separation_scoreboard_key_metrics.csv`, `scanner_heldout_summary.csv`, `frontier_downstream_summary.csv` |
| Rows | 12 representations |
| Validation status | PASS — 12 representations traced to 1c527697, 535eea18, 0d7cdc92, b5a9886e, c29a038d |

### Figure 4 — Bottleneck Comparison

| Field | Value |
|---|---|
| Artifact files | `figure4_bottleneck_comparison_data.csv`, `figure4_bottleneck_comparison_caption.md` |
| Intended manuscript location | Section 7 (Result 4: Bottleneck Comparison) |
| Claims supported | CLAIM_4_BOTTLENECK_FRONTIER |
| Source commits | a89bfb32 (primary), c29a038d (downstream), 0e2af247 (cross-backbone), 1c527697 (consolidated) |
| Source files | `frontier_variant_summary.csv`, `frontier_downstream_summary.csv`, `frontier_crossbackbone_summary.csv` |
| Rows | 10 (3 canine SCC full-scale variants + 6 SCORPION cross-backbone + 1 smoke note) |
| Validation status | PASS — 10 rows traced to a89bfb32, c29a038d, 0e2af247, 1c527697 |

### Figure 5 — Acquisition Swapping

| Field | Value |
|---|---|
| Artifact files | `figure5_acquisition_swapping_data.csv`, `figure5_acquisition_swapping_caption.md` |
| Intended manuscript location | Section 8 (Result 5: Decoder-Based Acquisition Swapping) |
| Claims supported | CLAIM_5_FACTOR_LIKE_SWAPPING |
| Source commits | aa8d0596 (sole source) |
| Source files | `acquisition_swapping_summary.csv`, `acquisition_swapping_nearest_neighbor_metrics.csv`, `acquisition_swapping_probe_metrics.csv`, `acquisition_swapping_report.md` |
| Rows | 13 (4 swap types × 3 variants + 1 aggregate) |
| Validation status | PASS — 13 rows traced to aa8d0596 |

---

## Cross-Reference: Commit to Artifact Mapping

| Commit | Short | Branch | Artifacts Using This Commit |
|---|---|---|---|
| e4819c42 | Pair-structure boundary test | experiment/pair-structure-boundary-test | Table 1, Figure 1 |
| d018c924 | Cross-backbone pair-structure boundary | experiment/pair-structure-boundary-crossbackbone | Table 1, Figure 1 |
| bec06eb4 | Biological label preservation | experiment/biological-label-preservation-audit | Table 1, Figure 2 |
| 3e5bf19e | Acquisition branch audit | experiment/acquisition-branch-audit | Figure 2 |
| 3450ede2 | Oldstyle residual branch separation | experiment/oldstyle-residual-branch-separation-audit | Table 1, Figure 2, Figure 3 |
| a325c009 | Linear baseline consistency | experiment/linear-baseline-consistency-audit | Figure 3 (context) |
| 1c527697 | Unified separation scoreboard | experiment/unified-separation-scoreboard | Figure 2, Figure 3, Figure 4 |
| 535eea18 | Scanner-heldout transfer | experiment/scanner-heldout-label-transfer-audit | Figure 3 |
| 0d7cdc92 | Sample-disjoint transfer | experiment/sample-subset-disjoint-scanner-heldout-transfer-audit | Figure 3 (via c29a038d) |
| b5a9886e | Scanner-confounded robustness | experiment/scanner-confounded-label-robustness-audit | Figure 3 (via c29a038d) |
| a89bfb32 | Bottleneck separation frontier | experiment/acquisition-bottleneck-separation-frontier | Figure 4 |
| c29a038d | Frontier-selected downstream validation | experiment/frontier-selected-downstream-validation | Figure 3, Figure 4 |
| 0e2af247 | Frontier-selected cross-backbone validation | experiment/frontier-selected-crossbackbone-validation | Figure 4 |
| aa8d0596 | Acquisition factor swapping audit | experiment/acquisition-factor-swapping-audit | Table 1, Figure 5 |

---

## Commit Count Summary

- **Total unique commits referenced:** 14 (excluding N/A paper-structure artifact)
- **All commits traced to result_to_claim_map.csv:** ✓
- **All commits verified via `git show`:** ✓

---

## Validation Status Summary

| Check | Status |
|---|---|
| All numbers traced to commits/result files | PASS |
| No invented numbers | PASS |
| No modified result files | PASS |
| No clinical/deployment/diagnostic language | PASS |
| SCORPION category boundary preserved | PASS |
| Bottleneck/frontier wording safe | PASS |
| Swapping weakness visible | PASS |
| git diff --check clean | PASS |
| git status shows only new files | PASS |

---

## Artifact File Checklist

- [x] table1_dataset_evidence_boundary.csv
- [x] table1_dataset_evidence_boundary.md
- [x] figure1_pairing_ladder_data.csv
- [x] figure1_pairing_ladder_caption.md
- [x] figure2_branch_separation_data.csv
- [x] figure2_branch_separation_caption.md
- [x] figure3_baseline_scoreboard_data.csv
- [x] figure3_baseline_scoreboard_caption.md
- [x] figure4_bottleneck_comparison_data.csv
- [x] figure4_bottleneck_comparison_caption.md
- [x] figure5_acquisition_swapping_data.csv
- [x] figure5_acquisition_swapping_caption.md
- [x] figure_table_artifact_manifest.md
- [ ] figure_table_validation_report.md
