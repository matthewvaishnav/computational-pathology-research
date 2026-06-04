# Camelyon17 supervised ResNet18 feature weighting result

## Purpose

This repeats the Camelyon17 center-weighting analysis using features from a Camelyon17 source-trained ResNet18 checkpoint instead of frozen ImageNet ResNet18 features.

The checkpoint was selected by source-domain validation behavior, not held-out test performance.

## 5-seed summary

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

## Key held-out test comparison

FedAvg-style equal-patch test accuracy:

    0.9052

Equal-client test accuracy:

    0.9318

Downweight-dominant-center test accuracy:

    0.9322

Equal-client held-out test gain:

    +0.0266 accuracy = +2.66 percentage points

Downweight-dominant held-out test gain:

    +0.0270 accuracy = +2.70 percentage points

## Generalization pattern

FedAvg-style equal-patch weighting nearly saturates source training accuracy:

    train accuracy = 0.9991

But it performs worse on validation and held-out test than equal-client or downweight-dominant weighting:

    id_val equal-client delta = +0.0057
    val equal-client delta    = +0.0246
    test equal-client delta   = +0.0266

    id_val downweight delta   = +0.0051
    val downweight delta      = +0.0242
    test downweight delta     = +0.0270

## Conservative interpretation

This result strengthens the Camelyon17 external-center evidence because the weighting effect survives when moving from frozen ImageNet ResNet18 features to Camelyon17-trained ResNet18 features.

The gain is smaller than the frozen ImageNet feature result, but the pattern is cleaner: FedAvg-style equal-patch weighting fits the source training set almost perfectly, while equal-client and downweight-dominant policies improve id_val, OOD validation, and held-out OOD test performance.
