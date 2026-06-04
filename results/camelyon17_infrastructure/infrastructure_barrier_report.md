# Camelyon17 infrastructure-barrier simulation

## Purpose

This is the first Pillar 4 analysis: implementation and infrastructure barriers.

It simulates infrastructure friction in a Camelyon17-style federated pathology setup:

- heterogeneous client compute speeds
- client dropout
- synchronous straggler delay
- communication cost
- detector-style reduced-round operation

This is not a real hospital deployment benchmark. It is a reproducible infrastructure-friction accounting simulation.

## Assumed source clients

| Client | Mean local update time |
|---|---:|
| center_0_fast | 4 minutes |
| center_3_medium | 7 minutes |
| center_4_slow | 13 minutes |

## Summary table

| regime                               | model                          |   rounds | mode        |   dropout_prob |   communication_gb_mean |   communication_gb_std |   total_hours_mean |   total_hours_std |   failed_rounds_mean |   failed_rounds_std |   mean_active_clients_mean |   mean_active_clients_std |   min_active_clients_mean |   min_active_clients_std |
|:-------------------------------------|:-------------------------------|---------:|:------------|---------------:|------------------------:|-----------------------:|-------------------:|------------------:|---------------------:|--------------------:|---------------------------:|--------------------------:|--------------------------:|-------------------------:|
| detector_switch_after_10_rounds      | full_resnet18_detector_limited |       10 | sync        |           0    |                2.49837  |                      0 |            2.26507 |          0.153302 |                 0    |            0        |                     3      |                  0        |                      3    |                 0        |
| detector_switch_after_10_rounds      | full_resnet18_detector_limited |       10 | sync        |           0.05 |                2.49837  |                      0 |            2.18613 |          0.206256 |                 0    |            0        |                     2.854  |                  0.123436 |                      2.15 |                 0.538891 |
| detector_switch_after_10_rounds      | full_resnet18_detector_limited |       10 | sync        |           0.1  |                2.49837  |                      0 |            2.11633 |          0.185288 |                 0.02 |            0.140705 |                     2.693  |                  0.164074 |                      1.85 |                 0.51981  |
| detector_switch_after_10_rounds      | full_resnet18_detector_limited |       10 | sync        |           0.2  |                2.49837  |                      0 |            2.03978 |          0.208647 |                 0.06 |            0.238683 |                     2.402  |                  0.216949 |                      1.27 |                 0.565953 |
| feature_head_sync_100_rounds         | feature_head                   |      100 | sync        |           0    |                0.002293 |                      0 |           22.436   |          0.543983 |                 0    |            0        |                     3      |                  0        |                      3    |                 0        |
| feature_head_sync_100_rounds         | feature_head                   |      100 | sync        |           0.05 |                0.002293 |                      0 |           21.9846  |          0.646209 |                 0.03 |            0.171447 |                     2.8491 |                  0.037392 |                      1.49 |                 0.559491 |
| feature_head_sync_100_rounds         | feature_head                   |      100 | sync        |           0.1  |                0.002293 |                      0 |           21.4192  |          0.567848 |                 0.09 |            0.287623 |                     2.6953 |                  0.050661 |                      0.94 |                 0.342893 |
| feature_head_sync_100_rounds         | feature_head                   |      100 | sync        |           0.2  |                0.002293 |                      0 |           20.2564  |          0.706378 |                 0.76 |            0.877554 |                     2.409  |                  0.078785 |                      0.47 |                 0.501614 |
| feature_head_sync_25_rounds          | feature_head                   |       25 | sync        |           0    |                0.000573 |                      0 |            5.62176 |          0.266412 |                 0    |            0        |                     3      |                  0        |                      3    |                 0        |
| feature_head_sync_25_rounds          | feature_head                   |       25 | sync        |           0.05 |                0.000573 |                      0 |            5.43586 |          0.330978 |                 0    |            0        |                     2.8392 |                  0.078363 |                      1.84 |                 0.419716 |
| feature_head_sync_25_rounds          | feature_head                   |       25 | sync        |           0.1  |                0.000573 |                      0 |            5.35046 |          0.279404 |                 0    |            0        |                     2.7152 |                  0.091193 |                      1.49 |                 0.502418 |
| feature_head_sync_25_rounds          | feature_head                   |       25 | sync        |           0.2  |                0.000573 |                      0 |            5.04087 |          0.33115  |                 0.19 |            0.394277 |                     2.4096 |                  0.124055 |                      0.88 |                 0.498077 |
| full_resnet18_async_proxy_100_rounds | full_resnet18                  |      100 | async_proxy |           0    |               24.9837   |                      0 |           12.0042  |          0.285964 |                 0    |            0        |                     3      |                  0        |                      3    |                 0        |
| full_resnet18_async_proxy_100_rounds | full_resnet18                  |      100 | async_proxy |           0.05 |               24.9837   |                      0 |           12.2728  |          0.303716 |                 0    |            0        |                     2.8503 |                  0.032643 |                      1.6  |                 0.492366 |
| full_resnet18_async_proxy_100_rounds | full_resnet18                  |      100 | async_proxy |           0.1  |               24.9837   |                      0 |           12.4894  |          0.348736 |                 0.06 |            0.238683 |                     2.7088 |                  0.05677  |                      1    |                 0.348155 |
| full_resnet18_async_proxy_100_rounds | full_resnet18                  |      100 | async_proxy |           0.2  |               24.9837   |                      0 |           12.8885  |          0.439038 |                 0.93 |            0.95616  |                     2.3877 |                  0.074615 |                      0.36 |                 0.482418 |
| full_resnet18_sync_100_rounds        | full_resnet18                  |      100 | sync        |           0    |               24.9837   |                      0 |           22.4438  |          0.59496  |                 0    |            0        |                     3      |                  0        |                      3    |                 0        |
| full_resnet18_sync_100_rounds        | full_resnet18                  |      100 | sync        |           0.05 |               24.9837   |                      0 |           21.8807  |          0.57711  |                 0.03 |            0.171447 |                     2.8512 |                  0.038932 |                      1.44 |                 0.556323 |
| full_resnet18_sync_100_rounds        | full_resnet18                  |      100 | sync        |           0.1  |               24.9837   |                      0 |           21.4708  |          0.646809 |                 0.14 |            0.376588 |                     2.7113 |                  0.047219 |                      0.93 |                 0.4324   |
| full_resnet18_sync_100_rounds        | full_resnet18                  |      100 | sync        |           0.2  |               24.9837   |                      0 |           20.2907  |          0.682885 |                 0.81 |            0.906709 |                     2.4149 |                  0.070661 |                      0.43 |                 0.49757  |

## Key comparison at 10% dropout

Full ResNet18 synchronous FL, 100 rounds:

- Communication: 24.9837 GB
- Mean wall-clock time: 21.47 hours
- Mean failed rounds: 0.14

Full ResNet18 async proxy, 100 rounds:

- Communication: 24.9837 GB
- Mean wall-clock time: 12.49 hours
- Mean failed rounds: 0.06

Feature/head synchronous federation, 100 rounds:

- Communication: 0.002293 GB
- Mean wall-clock time: 21.42 hours
- Mean failed rounds: 0.09

Detector-style 10-round full-model diagnostic/switch regime:

- Communication: 2.4984 GB
- Mean wall-clock time: 2.12 hours
- Mean failed rounds: 0.02

## Interpretation

This does not solve hospital deployment. It makes the infrastructure barrier measurable.

The simulation shows why full synchronous FL is sensitive to slow clients and dropouts: round time is governed by the slowest active client. Feature/head federation does not remove straggler delay by itself, but it massively reduces communication. Detector-style reduced-round operation reduces both communication and exposure to repeated straggler rounds.

## Conservative claim

Pillar 4 is not solved. This is a deployment-friction simulation, not a real hospital network experiment.

The useful contribution is a reproducible accounting framework for infrastructure burden: communication cost, wall-clock delay, failed rounds, active-client count, and sensitivity to dropout.
