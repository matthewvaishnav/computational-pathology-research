# Paired-Acquisition Factorization Autonomous Research Report

Generated: 2026-07-03T12:13:01.887309

## 1. Executive Summary

Baseline SCORPION DINOv2 and external canine SCC DINOv2 pair-integrity results were verified. A minimal SCORPION cross-backbone runner was created because the original SCORPION runner had DINOv2-specific feature and metadata checks. Phikon and ResNet50 smoke tests passed, and both full 5-fold x 5-seed x 3-condition runs completed cleanly. Across all four completed dataset/backbone settings, true same-tissue pairs preserved tissue identity metrics better than shuffled controls. In Phikon and ResNet50, shuffled controls often reduced scanner-probe accuracy more than true pairs, but did so while damaging paired cosine and retrieval, separating scanner suppression from useful tissue preservation.

## 2. Verified Existing Results

| label             | dataset    | backbone          |   raw_rows |   missing_metric_cells |   nonfinite_metric_cells | true_pairs_beats_shuffled_tissue_metrics   |
|:------------------|:-----------|:------------------|-----------:|-----------------------:|-------------------------:|:-------------------------------------------|
| scorpion_dinov2   | SCORPION   | DINOv2-Base       |         75 |                      0 |                        0 | True                                       |
| caninescc_dinov2  | Canine SCC | DINOv2-Base       |         75 |                      0 |                        0 | True                                       |
| scorpion_phikon   | SCORPION   | Phikon            |         75 |                      0 |                        0 | True                                       |
| scorpion_resnet50 | SCORPION   | ResNet50 ImageNet |         75 |                      0 |                        0 | True                                       |

## 3. New Experiments Attempted

- SCORPION Phikon pair-integrity smoke test.
- SCORPION Phikon full pair-integrity falsification.
- SCORPION ResNet50 pair-integrity smoke test.
- SCORPION ResNet50 full pair-integrity falsification.

## 4. New Experiments Completed

- Phikon smoke: completed 3/3 runs, runtime 7.0 seconds.
- Phikon full: completed 75/75 runs, runtime 1068.0 seconds.
- ResNet50 smoke: completed 3/3 runs, runtime 5.3 seconds.
- ResNet50 full: completed 75/75 runs, runtime 1138.5 seconds.

## 5. Skipped Work

- No decision-tree branch was skipped after Stage 0 passed.
- `scanner_adversary_only` was not added because the existing pair-integrity runners do not expose a clean condition for it.
- GitHub releases, Zenodo, DOI metadata, CITATION.cff, LICENSE files, release tags, child repositories, and unrelated CAMELYON17/pathoalign files were not touched.

## 6. Runtime Per Experiment

| dataset    | backbone          |   runs |   runtime_seconds |
|:-----------|:------------------|-------:|------------------:|
| SCORPION   | DINOv2-Base       |     75 |            1064.2 |
| Canine SCC | DINOv2-Base       |     75 |            1827.6 |
| SCORPION   | Phikon            |     75 |            1068   |
| SCORPION   | ResNet50 ImageNet |     75 |            1138.5 |

## 7. Result Tables

### SCORPION DINOv2-Base

| condition             |   scanner_probe_accuracy |   mean_paired_cosine |   worst_paired_cosine |   mean_top1_retrieval |   worst_top1_retrieval |   effective_rank |   biological_acquisition_cross_covariance |
|:----------------------|-------------------------:|---------------------:|----------------------:|----------------------:|-----------------------:|-----------------:|------------------------------------------:|
| true_pairs            |                 0.399778 |             0.879577 |              0.85023  |              0.999913 |               0.999133 |          54.4514 |                                  0.091709 |
| shuffled_region_pairs |                 0.373573 |             0.808929 |              0.763182 |              0.995358 |               0.983933 |          39.0769 |                                  0.105443 |
| shuffled_sample_pairs |                 0.358836 |             0.76682  |              0.716845 |              0.979242 |               0.9494   |          35.2259 |                                  0.102893 |

### Canine SCC DINOv2-Base

| condition             |   scanner_probe_accuracy |   mean_paired_cosine |   worst_paired_cosine |   mean_top1_retrieval |   worst_top1_retrieval |   effective_rank |   biological_acquisition_cross_covariance |
|:----------------------|-------------------------:|---------------------:|----------------------:|----------------------:|-----------------------:|-----------------:|------------------------------------------:|
| true_pairs            |                 0.361408 |             0.729961 |              0.656736 |              0.933392 |               0.884431 |          74.0444 |                                  0.089831 |
| shuffled_region_pairs |                 0.305673 |             0.542164 |              0.421063 |              0.729274 |               0.515828 |          54.5132 |                                  0.087106 |
| shuffled_sample_pairs |                 0.409302 |             0.584855 |              0.497105 |              0.718254 |               0.565006 |          45.3279 |                                  0.096097 |

### SCORPION Phikon

| condition             |   scanner_probe_accuracy |   mean_paired_cosine |   worst_paired_cosine |   mean_top1_retrieval |   worst_top1_retrieval |   effective_rank |   biological_acquisition_cross_covariance |
|:----------------------|-------------------------:|---------------------:|----------------------:|----------------------:|-----------------------:|-----------------:|------------------------------------------:|
| true_pairs            |                 0.520044 |             0.864493 |              0.830021 |              0.99968  |               0.997444 |          46.7507 |                                  0.088867 |
| shuffled_region_pairs |                 0.434071 |             0.691315 |              0.602224 |              0.960316 |               0.899178 |          44.3903 |                                  0.111446 |
| shuffled_sample_pairs |                 0.408533 |             0.615847 |              0.47412  |              0.876287 |               0.675667 |          42.5453 |                                  0.11264  |

### SCORPION ResNet50 ImageNet

| condition             |   scanner_probe_accuracy |   mean_paired_cosine |   worst_paired_cosine |   mean_top1_retrieval |   worst_top1_retrieval |   effective_rank |   biological_acquisition_cross_covariance |
|:----------------------|-------------------------:|---------------------:|----------------------:|----------------------:|-----------------------:|-----------------:|------------------------------------------:|
| true_pairs            |                 0.314462 |             0.654441 |              0.597813 |              0.97262  |               0.945489 |          85.0875 |                                  0.089567 |
| shuffled_region_pairs |                 0.304507 |             0.518814 |              0.448693 |              0.883944 |               0.798044 |          81.6695 |                                  0.085062 |
| shuffled_sample_pairs |                 0.306658 |             0.508324 |              0.443892 |              0.877482 |               0.800311 |          81.4377 |                                  0.084652 |

### Pair-Integrity Contrast Summary

CSV: `results/paired_acquisition_factorization_autonomous_research_run/pair_integrity_contrast_summary.csv`

| label             | comparison                             |   delta_scanner_probe_accuracy |   delta_mean_paired_cosine |   delta_worst_paired_cosine |   delta_mean_top1_retrieval |   delta_worst_top1_retrieval |   delta_effective_rank |   n_matched_fold_seed_blocks |
|:------------------|:---------------------------------------|-------------------------------:|---------------------------:|----------------------------:|----------------------------:|-----------------------------:|-----------------------:|-----------------------------:|
| scorpion_dinov2   | true_pairs_minus_shuffled_region_pairs |                       0.026204 |                   0.070648 |                    0.087048 |                    0.004556 |                     0.0152   |               15.3746  |                           25 |
| scorpion_dinov2   | true_pairs_minus_shuffled_sample_pairs |                       0.040942 |                   0.112757 |                    0.133385 |                    0.020671 |                     0.049733 |               19.2256  |                           25 |
| caninescc_dinov2  | true_pairs_minus_shuffled_region_pairs |                       0.055734 |                   0.187797 |                    0.235673 |                    0.204118 |                     0.368603 |               19.5312  |                           25 |
| caninescc_dinov2  | true_pairs_minus_shuffled_sample_pairs |                      -0.047895 |                   0.145105 |                    0.159632 |                    0.215138 |                     0.319425 |               28.7165  |                           25 |
| scorpion_phikon   | true_pairs_minus_shuffled_region_pairs |                       0.085973 |                   0.173178 |                    0.227798 |                    0.039364 |                     0.098267 |                2.36037 |                           25 |
| scorpion_phikon   | true_pairs_minus_shuffled_sample_pairs |                       0.111511 |                   0.248646 |                    0.355902 |                    0.123393 |                     0.321778 |                4.20542 |                           25 |
| scorpion_resnet50 | true_pairs_minus_shuffled_region_pairs |                       0.009956 |                   0.135626 |                    0.149121 |                    0.088676 |                     0.147444 |                3.41793 |                           25 |
| scorpion_resnet50 | true_pairs_minus_shuffled_sample_pairs |                       0.007804 |                   0.146117 |                    0.153921 |                    0.095138 |                     0.145178 |                3.64973 |                           25 |

### Representation Collapse Audit

CSV: `results/paired_acquisition_factorization_autonomous_research_run/representation_collapse_audit.csv`

| label             | condition             |   effective_rank |   scanner_probe_accuracy |   mean_paired_cosine |   mean_top1_retrieval | rank_collapse_vs_true   | tissue_preservation_collapse_vs_true   | retrieval_collapse_vs_true   | scanner_only_suppression_flag   |
|:------------------|:----------------------|-----------------:|-------------------------:|---------------------:|----------------------:|:------------------------|:---------------------------------------|:-----------------------------|:--------------------------------|
| scorpion_dinov2   | true_pairs            |          54.4514 |                 0.399778 |             0.879577 |              0.999913 | False                   | False                                  | False                        | False                           |
| scorpion_dinov2   | shuffled_region_pairs |          39.0769 |                 0.373573 |             0.808929 |              0.995358 | True                    | True                                   | False                        | True                            |
| scorpion_dinov2   | shuffled_sample_pairs |          35.2259 |                 0.358836 |             0.76682  |              0.979242 | True                    | True                                   | True                         | True                            |
| caninescc_dinov2  | true_pairs            |          74.0444 |                 0.361408 |             0.729961 |              0.933392 | False                   | False                                  | False                        | False                           |
| caninescc_dinov2  | shuffled_region_pairs |          54.5132 |                 0.305673 |             0.542164 |              0.729274 | True                    | True                                   | True                         | True                            |
| caninescc_dinov2  | shuffled_sample_pairs |          45.3279 |                 0.409302 |             0.584855 |              0.718254 | True                    | True                                   | True                         | False                           |
| scorpion_phikon   | true_pairs            |          46.7507 |                 0.520044 |             0.864493 |              0.99968  | False                   | False                                  | False                        | False                           |
| scorpion_phikon   | shuffled_region_pairs |          44.3903 |                 0.434071 |             0.691315 |              0.960316 | False                   | True                                   | True                         | True                            |
| scorpion_phikon   | shuffled_sample_pairs |          42.5453 |                 0.408533 |             0.615847 |              0.876287 | False                   | True                                   | True                         | True                            |
| scorpion_resnet50 | true_pairs            |          85.0875 |                 0.314462 |             0.654441 |              0.97262  | False                   | False                                  | False                        | False                           |
| scorpion_resnet50 | shuffled_region_pairs |          81.6695 |                 0.304507 |             0.518814 |              0.883944 | False                   | True                                   | True                         | True                            |
| scorpion_resnet50 | shuffled_sample_pairs |          81.4377 |                 0.306658 |             0.508324 |              0.877482 | False                   | True                                   | True                         | True                            |

## 8. Branch Decision

Label: **A. Strong support**

Rationale: true pairs beat shuffled controls on tissue-preservation metrics in SCORPION DINOv2, external canine SCC DINOv2, SCORPION Phikon, and SCORPION ResNet50. At least one transfer backbone completed; in fact, both completed.

Next experiment recommendation: Integrate the cross-backbone Phikon and ResNet50 pair-integrity results into arXiv and public research documentation as peer-review hardening.

## 9. Claim Interpretation

The completed results support the narrow pair-integrity mechanism: useful factorization depends on true same-tissue pairing. Shuffled controls can reduce scanner-probe accuracy, but they degrade paired-tissue cosine and retrieval. This indicates that scanner suppression alone is not the useful effect; preservation of tissue identity under true pairing is the key distinction.

## 10. Claim Boundary

These results do not establish clinical validation, diagnostic performance, disease biology discovery, human clinical generalization, FDA/HIPAA readiness, deployment readiness, complete scanner invariance, or perfect disentanglement.

## 11. Files Created or Updated

- `experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py`
- `results/paired_acquisition_factorization_autonomous_research_run/autonomous_run_log.txt`
- `results/paired_acquisition_factorization_autonomous_research_run/autonomous_research_report.md`
- `results/paired_acquisition_factorization_autonomous_research_run/pair_integrity_contrast_summary.csv`
- `results/paired_acquisition_factorization_autonomous_research_run/representation_collapse_audit.csv`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_phikon_smoke/`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_phikon/`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50_smoke/`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50/`

## 12. Exact Reproduction Commands

Smoke commands:

```powershell
python experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py --base-features results/scorpion/features/fold_0_phikon.npz --manifests-dir data/scorpion/splits --out-dir results/paired_acquisition_factorization_pair_integrity_scorpion_phikon_smoke --backbone phikon --seeds 701 --folds 0 --conditions true_pairs shuffled_region_pairs shuffled_sample_pairs --epochs 1 --region-batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --device cuda
```

```powershell
python experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py --base-features results/scorpion/features/fold_0_resnet50_imagenet.npz --manifests-dir data/scorpion/splits --out-dir results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50_smoke --backbone resnet50 --seeds 701 --folds 0 --conditions true_pairs shuffled_region_pairs shuffled_sample_pairs --epochs 1 --region-batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --device cuda
```

Full commands:

```powershell
python experiments/scorpion/run_pair_integrity_falsification.py --base-features results/scorpion/features/fold_0_dinov2_base.npz --manifests-dir data/scorpion/splits --out-dir results/paired_acquisition_factorization_pair_integrity_scorpion --seeds 701 702 703 704 705 --folds 0 1 2 3 4 --conditions true_pairs shuffled_region_pairs shuffled_sample_pairs --epochs 75 --region-batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --device cuda
```

```powershell
python experiments/canine/run_pair_integrity_falsification_caninescc.py --base-features results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz --manifests-dir data/external_multiscanner_caninescc/patch_manifests/splits --out-dir results/paired_acquisition_factorization_pair_integrity_caninescc --seeds 911 912 913 914 915 --folds 0 1 2 3 4 --conditions true_pairs shuffled_region_pairs shuffled_sample_pairs --epochs 75 --region-batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --device cuda
```

```powershell
python experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py --base-features results/scorpion/features/fold_0_phikon.npz --manifests-dir data/scorpion/splits --out-dir results/paired_acquisition_factorization_pair_integrity_scorpion_phikon --backbone phikon --seeds 701 702 703 704 705 --folds 0 1 2 3 4 --conditions true_pairs shuffled_region_pairs shuffled_sample_pairs --epochs 75 --region-batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --device cuda
```

```powershell
python experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py --base-features results/scorpion/features/fold_0_resnet50_imagenet.npz --manifests-dir data/scorpion/splits --out-dir results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50 --backbone resnet50 --seeds 701 702 703 704 705 --folds 0 1 2 3 4 --conditions true_pairs shuffled_region_pairs shuffled_sample_pairs --epochs 75 --region-batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --device cuda
```

## 13. Exact Next Recommended Action

Review the autonomous run artifacts and, if acceptable, integrate the Phikon and ResNet50 cross-backbone pair-integrity results into public research documentation and the arXiv manuscript as peer-review hardening.

## 14. Ready for arXiv Integration

Yes, after human review. The Phikon and ResNet50 results are complete, bounded, and consistent with the pair-integrity mechanism. They should be presented only as peer-review hardening, not as expanded clinical claims.

## 15. Ready for Commit

Yes, after review, the new cross-backbone runner, autonomous report/log, compact audit CSVs, and completed result artifacts are ready for a scoped commit. No staging or commit was performed in this run.

## 16. Release/Metadata Safety Statement

No GitHub release, Zenodo, DOI metadata, CITATION.cff, LICENSE file, release tag, child repository, or unrelated CAMELYON17/pathoalign file was touched.

## 17. Git Status Summary

Tracked modified files:

- None.

New untracked files created by this autonomous run:

- `experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py`
- `results/paired_acquisition_factorization_autonomous_research_run/autonomous_run_log.txt`
- `results/paired_acquisition_factorization_autonomous_research_run/autonomous_research_report.md`
- `results/paired_acquisition_factorization_autonomous_research_run/pair_integrity_contrast_summary.csv`
- `results/paired_acquisition_factorization_autonomous_research_run/representation_collapse_audit.csv`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_phikon_smoke/`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_phikon/`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50_smoke/`
- `results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50/`

Unrelated pre-existing untracked files still untouched:

- `?? pathoalign_fair_camelyon17_source_bundle.txt`
- `?? scripts/camelyon17/analyze_pathoalign_confirmatory.py`
- `?? scripts/camelyon17/pathoalign/`
- `?? scripts/camelyon17/run_pathoalign_conditional_audit_v01.py`
- `?? scripts/camelyon17/run_pathoalign_head_v0.py`
- `?? scripts/camelyon17/run_pathoalign_head_v01_sweep.py`
- `?? scripts/camelyon17/run_pathoalign_v1_factorized.py`
- `?? scripts/camelyon17/run_pathoalign_v1_factorized.py.pre_conditional_audit.bak`
- `?? scripts/camelyon17/run_pathoalign_v2_residual.py`
- `?? scripts/camelyon17/run_pathoalign_v2_residual.py.pre_conditional_audit.bak`
- `?? scripts/camelyon17/run_pathoalign_v3_conditional_audit.py`
- `?? scripts/camelyon17/run_pathoalign_v3_tumor_preserving.py.pre_conditional_audit.bak`
- `?? scripts/camelyon17/run_pathoalign_v4_component_audit.py`
- `?? scripts/camelyon17/run_pathoalign_v4_component_objective.py`
- `?? scripts/camelyon17/run_pathoalign_v5_cleaned_feature_adversary.py`
- `?? scripts/camelyon17/run_pathoalign_v5_cleaned_feature_audit.py`
- `?? scripts/camelyon17/run_pathoalign_weeks12.py`
- `?? scripts/camelyon17/verify_pathoalign_baselines.py`
- `?? scripts/experiments/run_pathoalign_fair_target_directed.py`
- `?? scripts/experiments/run_pathoalign_fair_three_client_stress.py`
- `?? scripts/federated/run_pathoalign_fair_camelyon17_contamination.py`
- `?? scripts/federated/run_pathoalign_fair_camelyon17_real_smoke.py`
- `?? scripts/pathoalign_identifiability/`
- `?? scripts/pathoalign_identifiability_v2/`
- `?? scripts/pathoalign_identifiability_v3/`
- `?? scripts/pathoalign_identifiability_v4/`
- `?? scripts/pathoalign_identifiability_v5/`
- `?? scripts/pathoalign_identifiability_v6/run_pair_integrity_falsification.py`
- `?? scripts/pathoalign_identifiability_v7/`
- `?? scripts/pathoalign_identifiability_v8/`
- `?? tests/test_pathoalign_decoupled_resource_control.py`
- `?? tests/test_pathoalign_decoupled_summary_fix.py`
- `?? tests/test_pathoalign_engineering.py`
- `?? tests/test_pathoalign_exact_anchor_repetition.py`
- `?? tests/test_pathoalign_fair_camelyon17_contamination.py`
- `?? tests/test_pathoalign_fair_camelyon17_real_smoke.py`
- `?? tests/test_pathoalign_fair_integration.py`
- `?? tests/test_pathoalign_fair_target_directed.py`
- `?? tests/test_pathoalign_fair_three_client_stress.py`
- `?? tests/test_pathoalign_identifiability_benchmark.py`
- `?? tests/test_pathoalign_identifiability_v2.py`
- `?? tests/test_pathoalign_monotone_bootstrap_boundaries.py`
- `?? tests/test_pathoalign_pair_anchor_scaling.py`
- `?? tests/test_pathoalign_pair_consistency_phase_diagram.py`
- `?? tests/test_pathoalign_two_resource_phase_map.py`

Raw `git status --short` at report generation:

```text
?? experiments/scorpion/run_pair_integrity_falsification_crossbackbone.py
?? pathoalign_fair_camelyon17_source_bundle.txt
?? scripts/camelyon17/analyze_pathoalign_confirmatory.py
?? scripts/camelyon17/pathoalign/
?? scripts/camelyon17/run_pathoalign_conditional_audit_v01.py
?? scripts/camelyon17/run_pathoalign_head_v0.py
?? scripts/camelyon17/run_pathoalign_head_v01_sweep.py
?? scripts/camelyon17/run_pathoalign_v1_factorized.py
?? scripts/camelyon17/run_pathoalign_v1_factorized.py.pre_conditional_audit.bak
?? scripts/camelyon17/run_pathoalign_v2_residual.py
?? scripts/camelyon17/run_pathoalign_v2_residual.py.pre_conditional_audit.bak
?? scripts/camelyon17/run_pathoalign_v3_conditional_audit.py
?? scripts/camelyon17/run_pathoalign_v3_tumor_preserving.py.pre_conditional_audit.bak
?? scripts/camelyon17/run_pathoalign_v4_component_audit.py
?? scripts/camelyon17/run_pathoalign_v4_component_objective.py
?? scripts/camelyon17/run_pathoalign_v5_cleaned_feature_adversary.py
?? scripts/camelyon17/run_pathoalign_v5_cleaned_feature_audit.py
?? scripts/camelyon17/run_pathoalign_weeks12.py
?? scripts/camelyon17/verify_pathoalign_baselines.py
?? scripts/experiments/run_pathoalign_fair_target_directed.py
?? scripts/experiments/run_pathoalign_fair_three_client_stress.py
?? scripts/federated/run_pathoalign_fair_camelyon17_contamination.py
?? scripts/federated/run_pathoalign_fair_camelyon17_real_smoke.py
?? scripts/pathoalign_identifiability/
?? scripts/pathoalign_identifiability_v2/
?? scripts/pathoalign_identifiability_v3/
?? scripts/pathoalign_identifiability_v4/
?? scripts/pathoalign_identifiability_v5/
?? scripts/pathoalign_identifiability_v6/run_pair_integrity_falsification.py
?? scripts/pathoalign_identifiability_v7/
?? scripts/pathoalign_identifiability_v8/
?? tests/test_pathoalign_decoupled_resource_control.py
?? tests/test_pathoalign_decoupled_summary_fix.py
?? tests/test_pathoalign_engineering.py
?? tests/test_pathoalign_exact_anchor_repetition.py
?? tests/test_pathoalign_fair_camelyon17_contamination.py
?? tests/test_pathoalign_fair_camelyon17_real_smoke.py
?? tests/test_pathoalign_fair_integration.py
?? tests/test_pathoalign_fair_target_directed.py
?? tests/test_pathoalign_fair_three_client_stress.py
?? tests/test_pathoalign_identifiability_benchmark.py
?? tests/test_pathoalign_identifiability_v2.py
?? tests/test_pathoalign_monotone_bootstrap_boundaries.py
?? tests/test_pathoalign_pair_anchor_scaling.py
?? tests/test_pathoalign_pair_consistency_phase_diagram.py
?? tests/test_pathoalign_two_resource_phase_map.py
```
