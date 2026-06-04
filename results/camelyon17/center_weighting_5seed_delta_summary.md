# Camelyon17 5-seed center-weighting delta summary

## Purpose

This report compares alternative source-center weighting policies against FedAvg-style equal-patch weighting.

Positive values mean the alternative policy improved over FedAvg-style equal-patch weighting.

## Delta summary

| split   |   seed_mean |   seed_std |   accuracy_equal_minus_fedavg_mean |   accuracy_equal_minus_fedavg_std |   accuracy_downweight_minus_fedavg_mean |   accuracy_downweight_minus_fedavg_std |   balanced_accuracy_equal_minus_fedavg_mean |   balanced_accuracy_equal_minus_fedavg_std |   balanced_accuracy_downweight_minus_fedavg_mean |   balanced_accuracy_downweight_minus_fedavg_std |   macro_f1_equal_minus_fedavg_mean |   macro_f1_equal_minus_fedavg_std |   macro_f1_downweight_minus_fedavg_mean |   macro_f1_downweight_minus_fedavg_std |   auc_equal_minus_fedavg_mean |   auc_equal_minus_fedavg_std |   auc_downweight_minus_fedavg_mean |   auc_downweight_minus_fedavg_std |
|:--------|------------:|-----------:|-----------------------------------:|----------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|-------------------------------------------:|-------------------------------------------------:|------------------------------------------------:|-----------------------------------:|----------------------------------:|----------------------------------------:|---------------------------------------:|------------------------------:|-----------------------------:|-----------------------------------:|----------------------------------:|
| id_val  |           3 |     1.5811 |                            -0.0154 |                            0.0012 |                                 -0.0179 |                                 0.0027 |                                     -0.0154 |                                     0.0012 |                                          -0.0179 |                                          0.0027 |                            -0.0154 |                            0.0012 |                                 -0.0179 |                                 0.0027 |                       -0.0108 |                       0.001  |                            -0.012  |                            0.0013 |
| test    |           3 |     1.5811 |                             0.082  |                            0.0135 |                                  0.0782 |                                 0.0122 |                                      0.082  |                                     0.0135 |                                           0.0782 |                                          0.0122 |                             0.0848 |                            0.0141 |                                  0.081  |                                 0.0128 |                        0.0083 |                       0.0055 |                             0.0075 |                            0.0051 |
| val     |           3 |     1.5811 |                            -0.041  |                            0.0144 |                                 -0.0422 |                                 0.0123 |                                     -0.041  |                                     0.0144 |                                          -0.0422 |                                          0.0123 |                            -0.0435 |                            0.0147 |                                 -0.0448 |                                 0.0126 |                       -0.0016 |                       0.0089 |                            -0.0045 |                            0.0096 |

## Main result

On the held-out test center, equal-client weighting improves accuracy over FedAvg-style equal-patch weighting by approximately +0.0820, or +8.20 percentage points.

Downweighting the dominant source center improves held-out test-center accuracy by approximately +0.0782, or +7.82 percentage points.

## Interpretation

FedAvg-style equal-patch weighting performs better on source-like validation, but equal-client and downweighted-dominant-center policies perform much better on the held-out test center. This supports further detector-switch experiments where source-domain diagnostics decide when to reduce sample-volume dominance.
