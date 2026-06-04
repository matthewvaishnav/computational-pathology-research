# Camelyon17 accuracy-per-communication analysis

## Purpose

This report connects Pillar 2 communication accounting to empirical held-out test performance.

It uses the Camelyon17-trained ResNet18 feature weighting results and compares feature/head communication against a full ResNet18 federation communication proxy.

## Communication anchor

100-round fp32 full ResNet18 federation across 3 source clients:

    24.9837 GB

100-round fp32 feature/head federation across 3 source clients:

    2.3483 MB

Full ResNet18 communication is approximately:

    10,894x larger

than feature/head communication under the same round/client assumptions.

## Accuracy per communication

| method                                         | feature_source              | policy                     |   test_accuracy |   test_macro_f1 |   test_auc | communication_regime                          |   communication_gb |   communication_mb |   accuracy_per_gb |   macro_f1_per_gb |   auc_per_gb |
|:-----------------------------------------------|:----------------------------|:---------------------------|----------------:|----------------:|-----------:|:----------------------------------------------|-------------------:|-------------------:|------------------:|------------------:|-------------:|
| feature_head_downweight_dominant_center        | camelyon17_trained_resnet18 | downweight_dominant_center |          0.9322 |          0.9321 |     0.9783 | feature_head_federation_100_round_fp32        |           0.002293 |            2.34833 |        406.49     |        406.447    |   426.593    |
| full_resnet18_proxy_downweight_dominant_center | camelyon17_trained_resnet18 | downweight_dominant_center |          0.9322 |          0.9321 |     0.9783 | full_resnet18_federation_100_round_fp32_proxy |          24.9837   |        25583.4     |          0.037312 |          0.037308 |     0.039157 |
| feature_head_equal_client                      | camelyon17_trained_resnet18 | equal_client               |          0.9318 |          0.9317 |     0.9788 | feature_head_federation_100_round_fp32        |           0.002293 |            2.34833 |        406.316    |        406.272    |   426.811    |
| full_resnet18_proxy_equal_client               | camelyon17_trained_resnet18 | equal_client               |          0.9318 |          0.9317 |     0.9788 | full_resnet18_federation_100_round_fp32_proxy |          24.9837   |        25583.4     |          0.037296 |          0.037292 |     0.039177 |
| feature_head_fedavg_equal_patch                | camelyon17_trained_resnet18 | fedavg_equal_patch         |          0.9052 |          0.9045 |     0.966  | feature_head_federation_100_round_fp32        |           0.002293 |            2.34833 |        394.717    |        394.412    |   421.229    |
| full_resnet18_proxy_fedavg_equal_patch         | camelyon17_trained_resnet18 | fedavg_equal_patch         |          0.9052 |          0.9045 |     0.966  | full_resnet18_federation_100_round_fp32_proxy |          24.9837   |        25583.4     |          0.036232 |          0.036204 |     0.038665 |

## Feature/head policy deltas

| policy                     |   test_accuracy |   test_accuracy_delta_vs_fedavg |   communication_mb |   communication_gb |   accuracy_per_gb |   accuracy_per_gb_delta_vs_fedavg |
|:---------------------------|----------------:|--------------------------------:|-------------------:|-------------------:|------------------:|----------------------------------:|
| downweight_dominant_center |          0.9322 |                          0.027  |            2.34833 |           0.002293 |           406.49  |                           11.7735 |
| equal_client               |          0.9318 |                          0.0266 |            2.34833 |           0.002293 |           406.316 |                           11.5991 |
| fedavg_equal_patch         |          0.9052 |                          0      |            2.34833 |           0.002293 |           394.717 |                            0      |

## Key result

Using Camelyon17-trained ResNet18 features, feature/head federation has extremely low communication cost under this accounting model. Within that feature/head regime:

- Equal-client weighting improves held-out test accuracy over FedAvg-style equal-patch weighting by +2.66 percentage points.
- Downweight-dominant-center weighting improves held-out test accuracy over FedAvg-style equal-patch weighting by +2.70 percentage points.
- All three feature/head policies use the same communication budget, so the gain is not purchased by additional communication.

## Conservative interpretation

This still does not prove real deployment communication efficiency. It is an accounting-plus-performance proxy.

However, it reframes Pillar 2 as an auditable optimization target: held-out external-center performance per GB communicated. The next experiment should replace the full-model proxy with actual iterative FL runs and measured wall-clock/network costs.
