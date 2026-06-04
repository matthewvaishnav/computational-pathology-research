# Camelyon17 validation-aware detector-switch analysis

## Purpose

This analysis tests a simple detector-switch rule using only validation diagnostics.

Decision inputs:

- `id_val`: source-domain validation centers
- `val`: OOD validation center

Held-out evaluation:

- `test`: held-out OOD test center

The detector does not use test performance when choosing a policy.

## Rule

Switch from FedAvg-style equal-patch weighting to the alternative policy when:

    alternative val accuracy - FedAvg val accuracy >= min_val_gain

and

    FedAvg id_val accuracy - alternative id_val accuracy <= max_id_val_cost

Otherwise keep FedAvg-style equal-patch weighting.

## Per-seed decisions

| alternative                | chosen_policy              |   switched |   val_gain |   id_val_cost |   test_accuracy_chosen |   test_accuracy_fedavg |   test_accuracy_chosen_minus_fedavg |   test_macro_f1_chosen_minus_fedavg |   test_auc_chosen_minus_fedavg |
|:---------------------------|:---------------------------|-----------:|-----------:|--------------:|-----------------------:|-----------------------:|------------------------------------:|------------------------------------:|-------------------------------:|
| equal_client               | equal_client               |          1 |     -0.045 |        0.0147 |                  0.895 |                  0.813 |                               0.082 |                              0.0855 |                         0.0133 |
| equal_client               | equal_client               |          1 |     -0.021 |        0.0163 |                  0.92  |                  0.816 |                               0.104 |                              0.1077 |                         0.0059 |
| equal_client               | equal_client               |          1 |     -0.04  |        0.0137 |                  0.927 |                  0.852 |                               0.075 |                              0.0768 |                         0.0034 |
| equal_client               | equal_client               |          1 |     -0.038 |        0.0163 |                  0.917 |                  0.849 |                               0.068 |                              0.0704 |                         0.0039 |
| equal_client               | fedavg_equal_patch         |          0 |     -0.061 |        0.016  |                  0.826 |                  0.826 |                               0     |                              0      |                         0      |
| downweight_dominant_center | downweight_dominant_center |          1 |     -0.049 |        0.0153 |                  0.889 |                  0.813 |                               0.076 |                              0.0795 |                         0.0117 |
| downweight_dominant_center | downweight_dominant_center |          1 |     -0.025 |        0.0193 |                  0.915 |                  0.816 |                               0.099 |                              0.1027 |                         0.005  |
| downweight_dominant_center | downweight_dominant_center |          1 |     -0.04  |        0.0147 |                  0.923 |                  0.852 |                               0.071 |                              0.0728 |                         0.0034 |
| downweight_dominant_center | downweight_dominant_center |          1 |     -0.039 |        0.019  |                  0.917 |                  0.849 |                               0.068 |                              0.0704 |                         0.0031 |
| downweight_dominant_center | fedavg_equal_patch         |          0 |     -0.058 |        0.021  |                  0.826 |                  0.826 |                               0     |                              0      |                         0      |

## Summary

| alternative                |   min_val_gain |   max_id_val_cost |   switched_mean |   switched_std |   id_val_accuracy_chosen_minus_fedavg_mean |   id_val_accuracy_chosen_minus_fedavg_std |   id_val_balanced_accuracy_chosen_minus_fedavg_mean |   id_val_balanced_accuracy_chosen_minus_fedavg_std |   id_val_macro_f1_chosen_minus_fedavg_mean |   id_val_macro_f1_chosen_minus_fedavg_std |   id_val_auc_chosen_minus_fedavg_mean |   id_val_auc_chosen_minus_fedavg_std |   val_accuracy_chosen_minus_fedavg_mean |   val_accuracy_chosen_minus_fedavg_std |   val_balanced_accuracy_chosen_minus_fedavg_mean |   val_balanced_accuracy_chosen_minus_fedavg_std |   val_macro_f1_chosen_minus_fedavg_mean |   val_macro_f1_chosen_minus_fedavg_std |   val_auc_chosen_minus_fedavg_mean |   val_auc_chosen_minus_fedavg_std |   test_accuracy_chosen_minus_fedavg_mean |   test_accuracy_chosen_minus_fedavg_std |   test_balanced_accuracy_chosen_minus_fedavg_mean |   test_balanced_accuracy_chosen_minus_fedavg_std |   test_macro_f1_chosen_minus_fedavg_mean |   test_macro_f1_chosen_minus_fedavg_std |   test_auc_chosen_minus_fedavg_mean |   test_auc_chosen_minus_fedavg_std |
|:---------------------------|---------------:|------------------:|----------------:|---------------:|-------------------------------------------:|------------------------------------------:|----------------------------------------------------:|---------------------------------------------------:|-------------------------------------------:|------------------------------------------:|--------------------------------------:|-------------------------------------:|----------------------------------------:|---------------------------------------:|-------------------------------------------------:|------------------------------------------------:|----------------------------------------:|---------------------------------------:|-----------------------------------:|----------------------------------:|-----------------------------------------:|----------------------------------------:|--------------------------------------------------:|-------------------------------------------------:|-----------------------------------------:|----------------------------------------:|------------------------------------:|-----------------------------------:|
| downweight_dominant_center |          -0.05 |              0.03 |             0.8 |         0.4472 |                                    -0.0137 |                                    0.0079 |                                             -0.0137 |                                             0.0079 |                                    -0.0137 |                                    0.0079 |                               -0.0099 |                               0.0056 |                                 -0.0306 |                                 0.0191 |                                          -0.0306 |                                          0.0191 |                                 -0.0325 |                                 0.0201 |                            -0.0026 |                            0.0092 |                                   0.0628 |                                  0.0372 |                                            0.0628 |                                           0.0372 |                                   0.0651 |                                  0.0385 |                              0.0047 |                             0.0044 |
| equal_client               |          -0.05 |              0.03 |             0.8 |         0.4472 |                                    -0.0122 |                                    0.0069 |                                             -0.0122 |                                             0.0069 |                                    -0.0122 |                                    0.0069 |                               -0.0089 |                               0.005  |                                 -0.0288 |                                 0.0185 |                                          -0.0288 |                                          0.0185 |                                 -0.0306 |                                 0.0193 |                            -0.0004 |                            0.0086 |                                   0.0658 |                                  0.0392 |                                            0.0658 |                                           0.0392 |                                   0.0681 |                                  0.0406 |                              0.0053 |                             0.0049 |

## Conservative interpretation

This is a first validation-aware detector-switch analysis, not a final detector. It asks whether source-domain and OOD-validation diagnostics can choose when to reduce sample-volume dominance without looking at the held-out test center.
