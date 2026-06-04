# Camelyon17 5-seed center-weighting summary

## Headline

A 5-seed frozen ResNet18 feature baseline shows a strong external-center generalization tradeoff:

- FedAvg-style equal-patch weighting performs best on source-like validation.
- Equal-client weighting performs much better on the held-out test center.

## Summary table

| policy                     | split   |   accuracy_mean |   accuracy_std |   balanced_accuracy_mean |   balanced_accuracy_std |   macro_f1_mean |   macro_f1_std |   auc_mean |   auc_std |
|:---------------------------|:--------|----------------:|---------------:|-------------------------:|------------------------:|----------------:|---------------:|-----------:|----------:|
| downweight_dominant_center | id_val  |          0.9125 |         0.0013 |                   0.9125 |                  0.0013 |          0.9125 |         0.0013 |     0.9653 |    0.0011 |
| downweight_dominant_center | test    |          0.9094 |         0.0135 |                   0.9094 |                  0.0135 |          0.9094 |         0.0136 |     0.9629 |    0.0046 |
| downweight_dominant_center | val     |          0.8246 |         0.0051 |                   0.8246 |                  0.0051 |          0.8218 |         0.0055 |     0.9224 |    0.0041 |
| equal_client               | id_val  |          0.9149 |         0.002  |                   0.9149 |                  0.002  |          0.9149 |         0.002  |     0.9665 |    0.0012 |
| equal_client               | test    |          0.9132 |         0.0125 |                   0.9132 |                  0.0125 |          0.9132 |         0.0125 |     0.9637 |    0.0042 |
| equal_client               | val     |          0.8258 |         0.0066 |                   0.8258 |                  0.0066 |          0.8232 |         0.007  |     0.9253 |    0.0035 |
| fedavg_equal_patch         | id_val  |          0.9303 |         0.003  |                   0.9303 |                  0.003  |          0.9303 |         0.003  |     0.9773 |    0.0016 |
| fedavg_equal_patch         | test    |          0.8312 |         0.0183 |                   0.8312 |                  0.0183 |          0.8284 |         0.019  |     0.9554 |    0.0086 |
| fedavg_equal_patch         | val     |          0.8668 |         0.0091 |                   0.8668 |                  0.0091 |          0.8666 |         0.0092 |     0.9269 |    0.0057 |

## Key comparison

FedAvg-style equal-patch weighting:

- id_val accuracy mean: 0.9303
- val accuracy mean: 0.8668
- test accuracy mean: 0.8312

Equal-client weighting:

- id_val accuracy mean: 0.9149
- val accuracy mean: 0.8258
- test accuracy mean: 0.9132

Held-out test-center difference:

    equal_client - fedavg_equal_patch = +0.0820 accuracy = +8.20 percentage points

Downweighted-dominant-center test difference:

    downweight_dominant_center - fedavg_equal_patch = +0.0782 accuracy = +7.82 percentage points

## Conservative interpretation

This does not prove final federated clinical performance. It is a fast feature-level baseline using frozen ImageNet ResNet18 features and logistic regression. However, it demonstrates that aggregation-style source-center weighting can materially change natural held-out-center performance on Camelyon17/WILDS.
