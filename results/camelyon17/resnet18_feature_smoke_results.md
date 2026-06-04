# Camelyon17 ResNet18 feature smoke baseline

- Sample size: 8,000
- Device: cuda
- Max per split/center/class: 500

## Results

| eval_group   | split   | center   |    n |   accuracy |   balanced_accuracy |   macro_f1 |      auc |
|:-------------|:--------|:---------|-----:|-----------:|--------------------:|-----------:|---------:|
| split        | id_val  | all      | 3000 |   0.938333 |            0.938333 |   0.938333 | 0.980504 |
| split        | test    | all      | 1000 |   0.821    |            0.821    |   0.817263 | 0.949664 |
| split        | train   | all      | 3000 |   0.982333 |            0.982333 |   0.982332 | 0.998414 |
| split        | val     | all      | 1000 |   0.882    |            0.882    |   0.881998 | 0.93974  |
| split_center | id_val  | 0        | 1000 |   0.936    |            0.936    |   0.935991 | 0.979496 |
| split_center | id_val  | 3        | 1000 |   0.931    |            0.931    |   0.930999 | 0.975836 |
| split_center | id_val  | 4        | 1000 |   0.948    |            0.948    |   0.94799  | 0.986548 |
| split_center | test    | 2        | 1000 |   0.821    |            0.821    |   0.817263 | 0.949664 |
| split_center | train   | 0        | 1000 |   0.983    |            0.983    |   0.982999 | 0.998236 |
| split_center | train   | 3        | 1000 |   0.982    |            0.982    |   0.981999 | 0.998008 |
| split_center | train   | 4        | 1000 |   0.982    |            0.982    |   0.981999 | 0.99898  |
| split_center | val     | 1        | 1000 |   0.882    |            0.882    |   0.881998 | 0.93974  |

## Interpretation

This is a frozen ImageNet ResNet18 feature sanity check, not the final computational pathology model. It verifies that image loading, center splits, source-domain training, and OOD evaluation are wired correctly.
