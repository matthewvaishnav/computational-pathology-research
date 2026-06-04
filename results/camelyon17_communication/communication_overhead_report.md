# Camelyon17 communication-overhead analysis

## Purpose

This is the first Pillar 2 analysis: communication overhead.

It estimates how much traffic is required for different federated-learning regimes in the Camelyon17 external-center setup.

This is an accounting analysis, not a deployment benchmark.

## Assumptions

- Source clients: 3
- Full model: binary ResNet18
- ResNet18 parameters: 11,177,538
- Feature/head model: 512-to-2 logistic head
- Logistic-head parameters: 1,026
- Communication model: each round includes one model download and one update upload per client.

## Main communication table

| regime                   |   clients |   rounds | precision   |   params_transmitted |     total_mb |   total_gb |
|:-------------------------|----------:|---------:|:------------|---------------------:|-------------:|-----------:|
| full_resnet18_federation |         3 |        5 | fp32        |             11177538 |  1279.17     |   1.24919  |
| feature_head_federation  |         3 |        5 | fp32        |                 1026 |     0.117416 |   0.000115 |
| full_resnet18_federation |         3 |        5 | fp16        |             11177538 |   639.584    |   0.624594 |
| feature_head_federation  |         3 |        5 | fp16        |                 1026 |     0.058708 |   5.7e-05  |
| full_resnet18_federation |         3 |        5 | int8        |             11177538 |   319.792    |   0.312297 |
| feature_head_federation  |         3 |        5 | int8        |                 1026 |     0.029354 |   2.9e-05  |
| full_resnet18_federation |         3 |       10 | fp32        |             11177538 |  2558.34     |   2.49837  |
| feature_head_federation  |         3 |       10 | fp32        |                 1026 |     0.234833 |   0.000229 |
| full_resnet18_federation |         3 |       10 | fp16        |             11177538 |  1279.17     |   1.24919  |
| feature_head_federation  |         3 |       10 | fp16        |                 1026 |     0.117416 |   0.000115 |
| full_resnet18_federation |         3 |       10 | int8        |             11177538 |   639.584    |   0.624594 |
| feature_head_federation  |         3 |       10 | int8        |                 1026 |     0.058708 |   5.7e-05  |
| full_resnet18_federation |         3 |       25 | fp32        |             11177538 |  6395.84     |   6.24594  |
| feature_head_federation  |         3 |       25 | fp32        |                 1026 |     0.587082 |   0.000573 |
| full_resnet18_federation |         3 |       25 | fp16        |             11177538 |  3197.92     |   3.12297  |
| feature_head_federation  |         3 |       25 | fp16        |                 1026 |     0.293541 |   0.000287 |
| full_resnet18_federation |         3 |       25 | int8        |             11177538 |  1598.96     |   1.56148  |
| feature_head_federation  |         3 |       25 | int8        |                 1026 |     0.14677  |   0.000143 |
| full_resnet18_federation |         3 |       50 | fp32        |             11177538 | 12791.7      |  12.4919   |
| feature_head_federation  |         3 |       50 | fp32        |                 1026 |     1.17416  |   0.001147 |
| full_resnet18_federation |         3 |       50 | fp16        |             11177538 |  6395.84     |   6.24594  |
| feature_head_federation  |         3 |       50 | fp16        |                 1026 |     0.587082 |   0.000573 |
| full_resnet18_federation |         3 |       50 | int8        |             11177538 |  3197.92     |   3.12297  |
| feature_head_federation  |         3 |       50 | int8        |                 1026 |     0.293541 |   0.000287 |
| full_resnet18_federation |         3 |      100 | fp32        |             11177538 | 25583.4      |  24.9837   |
| feature_head_federation  |         3 |      100 | fp32        |                 1026 |     2.34833  |   0.002293 |
| full_resnet18_federation |         3 |      100 | fp16        |             11177538 | 12791.7      |  12.4919   |
| feature_head_federation  |         3 |      100 | fp16        |                 1026 |     1.17416  |   0.001147 |
| full_resnet18_federation |         3 |      100 | int8        |             11177538 |  6395.84     |   6.24594  |
| feature_head_federation  |         3 |      100 | int8        |                 1026 |     0.587082 |   0.000573 |

## Detector reduced-round accounting

| baseline                 | detector_regime                    |   clients | precision   |   baseline_rounds |   detector_rounds |   baseline_gb |   detector_gb |   gb_saved |   relative_reduction |
|:-------------------------|:-----------------------------------|----------:|:------------|------------------:|------------------:|--------------:|--------------:|-----------:|---------------------:|
| full_resnet18_100_rounds | diagnose_or_switch_after_5_rounds  |         3 | fp32        |               100 |                 5 |      24.9837  |      1.24919  |   23.7346  |                 0.95 |
| full_resnet18_100_rounds | diagnose_or_switch_after_5_rounds  |         3 | fp16        |               100 |                 5 |      12.4919  |      0.624594 |   11.8673  |                 0.95 |
| full_resnet18_100_rounds | diagnose_or_switch_after_5_rounds  |         3 | int8        |               100 |                 5 |       6.24594 |      0.312297 |    5.93364 |                 0.95 |
| full_resnet18_100_rounds | diagnose_or_switch_after_10_rounds |         3 | fp32        |               100 |                10 |      24.9837  |      2.49837  |   22.4854  |                 0.9  |
| full_resnet18_100_rounds | diagnose_or_switch_after_10_rounds |         3 | fp16        |               100 |                10 |      12.4919  |      1.24919  |   11.2427  |                 0.9  |
| full_resnet18_100_rounds | diagnose_or_switch_after_10_rounds |         3 | int8        |               100 |                10 |       6.24594 |      0.624594 |    5.62134 |                 0.9  |
| full_resnet18_100_rounds | diagnose_or_switch_after_25_rounds |         3 | fp32        |               100 |                25 |      24.9837  |      6.24594  |   18.7378  |                 0.75 |
| full_resnet18_100_rounds | diagnose_or_switch_after_25_rounds |         3 | fp16        |               100 |                25 |      12.4919  |      3.12297  |    9.3689  |                 0.75 |
| full_resnet18_100_rounds | diagnose_or_switch_after_25_rounds |         3 | int8        |               100 |                25 |       6.24594 |      1.56148  |    4.68445 |                 0.75 |

## Key comparison

100-round fp32 full ResNet18 federation:

    24.9837 GB

100-round fp32 feature/head federation:

    2.3483 MB

Full ResNet18 traffic is approximately:

    10,894x larger

than feature/head federation under the same client/round assumptions.

## Conservative interpretation

This does not solve communication overhead yet. It quantifies the communication problem and shows why full-model FL is expensive in pathology-style models.

The current Camelyon17 weighting experiments are feature-level baselines, so they avoid repeated full-model communication. The next Pillar 2 experiment should connect this accounting to empirical accuracy by comparing accuracy-per-GB under full-model, compressed, feature/head, and detector-reduced-round regimes.
