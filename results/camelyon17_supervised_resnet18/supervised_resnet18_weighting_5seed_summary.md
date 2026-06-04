# Camelyon17 supervised ResNet18 feature 5-seed center-weighting summary

## Purpose

This repeats the center-weighting analysis using features extracted from a Camelyon17 source-trained ResNet18 checkpoint rather than frozen ImageNet ResNet18 features.

Checkpoint selection was based on source-domain validation, not held-out test performance.

## Summary table

| policy                     | split   |   accuracy_mean |   accuracy_std |   balanced_accuracy_mean |   balanced_accuracy_std |   macro_f1_mean |   macro_f1_std |   auc_mean |   auc_std |
|:---------------------------|:--------|----------------:|---------------:|-------------------------:|------------------------:|----------------:|---------------:|-----------:|----------:|
| downweight_dominant_center | id_val  |          0.9692 |         0.002  |                   0.9692 |                  0.002  |          0.9692 |         0.002  |     0.9949 |    0.0006 |
| downweight_dominant_center | test    |          0.9322 |         0.0053 |                   0.9322 |                  0.0053 |          0.9321 |         0.0053 |     0.9783 |    0.0026 |
| downweight_dominant_center | train   |          0.9681 |         0.0041 |                   0.9681 |                  0.0041 |          0.9681 |         0.0041 |     0.9946 |    0.0008 |
| downweight_dominant_center | val     |          0.9228 |         0.0027 |                   0.9228 |                  0.0027 |          0.9228 |         0.0027 |     0.9788 |    0.0017 |
| equal_client               | id_val  |          0.9698 |         0.0018 |                   0.9698 |                  0.0018 |          0.9698 |         0.0018 |     0.995  |    0.0006 |
| equal_client               | test    |          0.9318 |         0.0048 |                   0.9318 |                  0.0048 |          0.9317 |         0.0048 |     0.9788 |    0.0026 |
| equal_client               | train   |          0.9683 |         0.0034 |                   0.9683 |                  0.0034 |          0.9683 |         0.0034 |     0.9948 |    0.0008 |
| equal_client               | val     |          0.9232 |         0.0031 |                   0.9232 |                  0.0031 |          0.9232 |         0.0031 |     0.9794 |    0.0017 |
| fedavg_equal_patch         | id_val  |          0.9641 |         0.0045 |                   0.9641 |                  0.0045 |          0.9641 |         0.0045 |     0.9923 |    0.0014 |
| fedavg_equal_patch         | test    |          0.9052 |         0.0359 |                   0.9052 |                  0.0359 |          0.9045 |         0.0372 |     0.966  |    0.0105 |
| fedavg_equal_patch         | train   |          0.9991 |         0.0007 |                   0.9991 |                  0.0007 |          0.9991 |         0.0007 |     1      |    0      |
| fedavg_equal_patch         | val     |          0.8986 |         0.0143 |                   0.8986 |                  0.0143 |          0.8984 |         0.0144 |     0.9667 |    0.0108 |

## Key question

Does the source-center weighting effect survive with Camelyon17-trained features?

Check the held-out `test` split and compare:

- `equal_client` vs `fedavg_equal_patch`
- `downweight_dominant_center` vs `fedavg_equal_patch`

## Conservative interpretation

This is still a sampled feature-level baseline, not full iterative federated learning. However, if the equal-client or downweight-dominant policies improve held-out test performance across seeds, the Camelyon17 evidence becomes stronger than the frozen ImageNet feature result alone.
