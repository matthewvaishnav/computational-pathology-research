# Unified Separation Scoreboard

## Purpose

One table showing the entire paired-acquisition contribution, the strongest
linear baseline boundary, and the bottleneck frontier improvement across
all completed audits.

## Scoreboard

                     representation        method_family branch_type dimensionality scanner_probe_balanced_acc category_probe_balanced_acc category_macro_f1 category_weighted_f1 category_purity_k1 category_purity_k5 category_purity_k10 acquisition_scanner_capture acquisition_category_leakage biological_scanner_leakage biological_category_preservation scanner_heldout_balanced_acc scanner_heldout_macro_f1 sample_disjoint_scanner_heldout_balanced_acc sample_disjoint_scanner_heldout_macro_f1 scanner_confounded_balanced_acc scanner_confounded_macro_f1 scorpion_dinov2_acquisition_pair_retrieval_leakage scorpion_dinov2_acquisition_scanner_capture scorpion_phikon_acquisition_pair_retrieval_leakage scorpion_phikon_acquisition_scanner_capture scorpion_resnet50_acquisition_pair_retrieval_leakage scorpion_resnet50_acquisition_scanner_capture
           original_frozen_features               frozen    original            NaN                     0.8638                      0.4003            0.3579               0.4270             0.9602             0.8717              0.7306                          NA                           NA                     0.8638                           0.4003                           NA                       NA                                           NA                                       NA                              NA                          NA                                                 NA                                          NA                                                 NA                                          NA                                                   NA                                            NA
                   oldstyle_keep_k4      oldstyle_linear        keep            NaN                     0.2000                      0.4004            0.3485               0.4250             0.9678             0.8895              0.7456                          NA                           NA                     0.2000                           0.4004                           NA                       NA                                           NA                                       NA                              NA                          NA                                                 NA                                          NA                                                 NA                                          NA                                                   NA                                            NA
                oldstyle_removed_k4      oldstyle_linear     removed            NaN                     0.5384                      0.2421            0.1746               0.2313             0.3464             0.3387              0.3341                      0.5384                       0.2421                         NA                               NA                           NA                       NA                                           NA                                       NA                              NA                          NA                                                 NA                                          NA                                                 NA                                          NA                                                   NA                                            NA
               true_pair_biological neural_factorization  biological            NaN                     0.3614                      0.3860            0.3389               0.4174             0.9729             0.8973              0.7509                          NA                           NA                     0.3614                           0.3860                       0.8273                   0.7887                                       0.3260                                   0.3013                          0.3791                      0.3475                                                 NA                                          NA                                                 NA                                          NA                                                   NA                                            NA
              true_pair_acquisition neural_factorization acquisition            NaN                     0.8651                      0.3456            0.2736               0.3246             0.5736             0.4712              0.4183                      0.8651                       0.3456                         NA                               NA                       0.5153                   0.4147                                       0.2654                                   0.2165                          0.3072                      0.2489                                             0.0944                                      0.8582                                             0.0739                                      0.9711                                               0.1705                                        0.7845
        acq_dim8_default_biological  bottlenecked_neural  biological      acq_dim=8                     0.3691                      0.3852            0.3386               0.4117             0.9730             0.8971              0.7509                          NA                           NA                     0.3691                           0.3852                       0.8221                   0.7852                                       0.3295                                   0.2960                          0.3794                      0.3480                                                 NA                                          NA                                                 NA                                          NA                                                   NA                                            NA
       acq_dim8_default_acquisition  bottlenecked_neural acquisition      acq_dim=8                     0.8643                      0.1598            0.1044               0.1087             0.3265             0.3060              0.2924                      0.8643                       0.1598                         NA                               NA                       0.1751                   0.0565                                       0.1304                                   0.0494                          0.1623                      0.1251                                             0.0231                                      0.8508                                             0.0204                                      0.9733                                               0.0505                                        0.7701
 acq_dim16_stronger_xcov_biological  bottlenecked_neural  biological     acq_dim=16                     0.3593                      0.3824            0.3394               0.4135             0.9730             0.8976              0.7519                          NA                           NA                     0.3593                           0.3824                       0.8292                   0.7911                                       0.3371                                   0.3108                          0.3761                      0.3454                                                 NA                                          NA                                                 NA                                          NA                                                   NA                                            NA
acq_dim16_stronger_xcov_acquisition  bottlenecked_neural acquisition     acq_dim=16                     0.8638                      0.1689            0.1177               0.1381             0.3806             0.3376              0.3165                      0.8638                       0.1689                         NA                               NA                       0.2037                   0.0869                                       0.1404                                   0.0676                          0.1784                      0.1351                                             0.0253                                      0.8565                                             0.0219                                      0.9772                                               0.0646                                        0.7767
         shuffled_sample_biological     shuffled_control  biological            NaN                     0.4093                      0.3228            0.2752               0.3386             0.9273             0.7801              0.6355                          NA                           NA                     0.4093                           0.3228                           NA                       NA                                           NA                                       NA                              NA                          NA                                                 NA                                          NA                                                 NA                                          NA                                                   NA                                            NA
        shuffled_sample_acquisition     shuffled_control acquisition            NaN                     0.8302                      0.3871            0.3150               0.3727             0.7309             0.6086              0.5338                      0.8302                       0.3871                         NA                               NA                           NA                       NA                                           NA                                       NA                              NA                          NA                                                 NA                                          NA                                                 NA                                          NA                                                   NA                                            NA
                    pca_removal_k32          pca_removal     control            NaN                     0.6489                      0.2893            0.2754                   NA                 NA                 NA                  NA                          NA                           NA                         NA                               NA                           NA                       NA                                           NA                                       NA                              NA                          NA                                                 NA                                          NA                                                 NA                                          NA                                                   NA                                            NA


## Key Questions Answered

### 1. What wins raw scanner removal?
oldstyle_keep_k4 scanner probe: 0.2000
true_pair_biological scanner probe: 0.3614
Answer: oldstyle_keep_k4 (centroid/QR linear projection) removes scanner most completely.

### 2. What gives the strongest explicit scanner/acquisition branch?
oldstyle_removed_k4 scanner capture: 0.5384
true_pair_acquisition scanner capture: 0.8651
Answer: Both capture scanner strongly. Bottlenecked variants add lower category/tissue leakage.

### 3. Does bottlenecking reduce acquisition leakage?
true_pair_acquisition category leakage: 0.3456
acq_dim8_default_acquisition: 0.1598
acq_dim16_stronger_xcov_acquisition: 0.1689
Answer: Yes. Category leakage drops from ~0.35 to ~0.16-0.17 in canine SCC.
SCORPION cross-backbone pair retrieval leakage also drops substantially with bottlenecking.

### 4. Does bottlenecking preserve biological downstream transfer?
true_pair_biological scanner-heldout: 0.8273
acq_dim8_default_biological: 0.8221
acq_dim16_stronger_xcov_biological: 0.8292
Answer: Yes. Biological downstream transfer stays within a narrow band and sometimes improves slightly.

### 5. Is paired-acquisition the best raw scanner remover?
Answer: No. oldstyle_keep_k4 (centroid/QR) is stronger at raw scanner removal.

### 6. What is the contribution?
Answer: Structured separation. Paired-acquisition produces an explicit scanner-bearing acquisition branch with reduced biological leakage (especially when bottlenecked), while the biological branch preserves category signal and downstream transfer. Bottlenecking trades a small amount of scanner capture for substantially lower category/tissue leakage in the acquisition branch. Cross-backbone validation confirms this generalizes across DINOv2, Phikon, and ResNet50.


## Limitations

- SCORPION cross-backbone values measure tissue/pair-retrieval leakage,
  not category-label leakage. SCORPION has no biological category labels.
- Canine SCC DINOv2 is the only labeled-category anchor.
- Oldstyle centroid/QR linear projection is the strongest raw scanner-removal baseline.
- Cross-experiment comparisons may use slightly different evaluation protocols.
- Not all metrics are available for all representations; missing values shown as NA.

## Validation

- Scoreboard rows: 12
- Source entries: 24
- Validation issues: 0
  - No validation issues.

## Data Sources

- acq_dim16_stronger_xcov_acquisition: acquisition_bottleneck_frontier (experiment/acquisition-bottleneck-separation-frontier @ a89bfb32)
- acq_dim16_stronger_xcov_acquisition: frontier_crossbackbone (experiment/frontier-selected-crossbackbone-validation @ 0e2af247)
- acq_dim16_stronger_xcov_acquisition: frontier_downstream (experiment/frontier-selected-downstream-validation @ c29a038d)
- acq_dim16_stronger_xcov_biological: acquisition_bottleneck_frontier (experiment/acquisition-bottleneck-separation-frontier @ a89bfb32)
- acq_dim16_stronger_xcov_biological: frontier_crossbackbone (experiment/frontier-selected-crossbackbone-validation @ 0e2af247)
- acq_dim16_stronger_xcov_biological: frontier_downstream (experiment/frontier-selected-downstream-validation @ c29a038d)
- acq_dim8_default_acquisition: acquisition_bottleneck_frontier (experiment/acquisition-bottleneck-separation-frontier @ a89bfb32)
- acq_dim8_default_acquisition: frontier_crossbackbone (experiment/frontier-selected-crossbackbone-validation @ 0e2af247)
- acq_dim8_default_acquisition: frontier_downstream (experiment/frontier-selected-downstream-validation @ c29a038d)
- acq_dim8_default_biological: acquisition_bottleneck_frontier (experiment/acquisition-bottleneck-separation-frontier @ a89bfb32)
- acq_dim8_default_biological: frontier_crossbackbone (experiment/frontier-selected-crossbackbone-validation @ 0e2af247)
- acq_dim8_default_biological: frontier_downstream (experiment/frontier-selected-downstream-validation @ c29a038d)
- oldstyle_keep_k4: oldstyle_residual (experiment/oldstyle-residual-branch-separation-audit @ 3450ede2)
- oldstyle_removed_k4: oldstyle_residual (experiment/oldstyle-residual-branch-separation-audit @ 3450ede2)
- original_frozen_features: oldstyle_residual (experiment/oldstyle-residual-branch-separation-audit @ 3450ede2)
- pca_removal_k32: biological_label_preservation (experiment/biological-label-preservation-audit @ bec06eb4)
- shuffled_sample_acquisition: oldstyle_residual (experiment/oldstyle-residual-branch-separation-audit @ 3450ede2)
- shuffled_sample_biological: oldstyle_residual (experiment/oldstyle-residual-branch-separation-audit @ 3450ede2)
- true_pair_acquisition: frontier_crossbackbone (experiment/frontier-selected-crossbackbone-validation @ 0e2af247)
- true_pair_acquisition: frontier_downstream (experiment/frontier-selected-downstream-validation @ c29a038d)
- true_pair_acquisition: oldstyle_residual (experiment/oldstyle-residual-branch-separation-audit @ 3450ede2)
- true_pair_biological: frontier_crossbackbone (experiment/frontier-selected-crossbackbone-validation @ 0e2af247)
- true_pair_biological: frontier_downstream (experiment/frontier-selected-downstream-validation @ c29a038d)
- true_pair_biological: oldstyle_residual (experiment/oldstyle-residual-branch-separation-audit @ 3450ede2)

## Output Files

- unified_separation_scoreboard.csv
- unified_separation_scoreboard_key_metrics.csv
- unified_separation_scoreboard_sources.csv
- unified_separation_scoreboard_report.md
- experiment_design.json
- run_log.txt
