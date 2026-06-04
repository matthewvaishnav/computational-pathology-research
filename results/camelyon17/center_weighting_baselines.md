# Camelyon17 center-weighting baselines

- Sample size: 8,000
- Device: cuda
- Max per split/center/class: 500

## Results

| policy                     | eval_group   | split   | center   |    n |   accuracy |   balanced_accuracy |   macro_f1 |      auc |
|:---------------------------|:-------------|:--------|:---------|-----:|-----------:|--------------------:|-----------:|---------:|
| downweight_dominant_center | split        | id_val  | all      | 3000 |   0.908667 |            0.908667 |   0.908666 | 0.960246 |
| downweight_dominant_center | split        | test    | all      | 1000 |   0.924    |            0.924    |   0.924    | 0.962096 |
| downweight_dominant_center | split        | train   | all      | 3000 |   0.927333 |            0.927333 |   0.927333 | 0.974386 |
| downweight_dominant_center | split        | val     | all      | 1000 |   0.798    |            0.798    |   0.793093 | 0.90622  |
| downweight_dominant_center | split_center | id_val  | 0        | 1000 |   0.889    |            0.889    |   0.88896  | 0.95688  |
| downweight_dominant_center | split_center | id_val  | 3        | 1000 |   0.908    |            0.908    |   0.907917 | 0.967596 |
| downweight_dominant_center | split_center | id_val  | 4        | 1000 |   0.929    |            0.929    |   0.928998 | 0.96814  |
| downweight_dominant_center | split_center | test    | 2        | 1000 |   0.924    |            0.924    |   0.924    | 0.962096 |
| downweight_dominant_center | split_center | train   | 0        | 1000 |   0.914    |            0.914    |   0.913997 | 0.968308 |
| downweight_dominant_center | split_center | train   | 3        | 1000 |   0.923    |            0.923    |   0.922972 | 0.978276 |
| downweight_dominant_center | split_center | train   | 4        | 1000 |   0.945    |            0.945    |   0.944997 | 0.984192 |
| downweight_dominant_center | split_center | val     | 1        | 1000 |   0.798    |            0.798    |   0.793093 | 0.90622  |
| equal_client               | split        | id_val  | all      | 3000 |   0.909667 |            0.909667 |   0.909664 | 0.961701 |
| equal_client               | split        | test    | all      | 1000 |   0.928    |            0.928    |   0.927995 | 0.962904 |
| equal_client               | split        | train   | all      | 3000 |   0.929333 |            0.929333 |   0.92933  | 0.975368 |
| equal_client               | split        | val     | all      | 1000 |   0.798    |            0.798    |   0.793476 | 0.909924 |
| equal_client               | split_center | id_val  | 0        | 1000 |   0.895    |            0.895    |   0.894982 | 0.96016  |
| equal_client               | split_center | id_val  | 3        | 1000 |   0.906    |            0.906    |   0.905904 | 0.968052 |
| equal_client               | split_center | id_val  | 4        | 1000 |   0.928    |            0.928    |   0.928    | 0.968948 |
| equal_client               | split_center | test    | 2        | 1000 |   0.928    |            0.928    |   0.927995 | 0.962904 |
| equal_client               | split_center | train   | 0        | 1000 |   0.918    |            0.918    |   0.918    | 0.97118  |
| equal_client               | split_center | train   | 3        | 1000 |   0.923    |            0.923    |   0.922944 | 0.978784 |
| equal_client               | split_center | train   | 4        | 1000 |   0.947    |            0.947    |   0.946999 | 0.9837   |
| equal_client               | split_center | val     | 1        | 1000 |   0.798    |            0.798    |   0.793476 | 0.909924 |
| fedavg_equal_patch         | split        | id_val  | all      | 3000 |   0.926333 |            0.926333 |   0.926333 | 0.976275 |
| fedavg_equal_patch         | split        | test    | all      | 1000 |   0.866    |            0.866    |   0.864954 | 0.959012 |
| fedavg_equal_patch         | split        | train   | all      | 3000 |   0.982667 |            0.982667 |   0.982666 | 0.998426 |
| fedavg_equal_patch         | split        | val     | all      | 1000 |   0.84     |            0.84     |   0.839744 | 0.906028 |
| fedavg_equal_patch         | split_center | id_val  | 0        | 1000 |   0.911    |            0.911    |   0.910953 | 0.972812 |
| fedavg_equal_patch         | split_center | id_val  | 3        | 1000 |   0.925    |            0.925    |   0.924967 | 0.970936 |
| fedavg_equal_patch         | split_center | id_val  | 4        | 1000 |   0.943    |            0.943    |   0.942999 | 0.986972 |
| fedavg_equal_patch         | split_center | test    | 2        | 1000 |   0.866    |            0.866    |   0.864954 | 0.959012 |
| fedavg_equal_patch         | split_center | train   | 0        | 1000 |   0.983    |            0.983    |   0.982999 | 0.996748 |
| fedavg_equal_patch         | split_center | train   | 3        | 1000 |   0.977    |            0.977    |   0.976999 | 0.998656 |
| fedavg_equal_patch         | split_center | train   | 4        | 1000 |   0.988    |            0.988    |   0.988    | 0.999368 |
| fedavg_equal_patch         | split_center | val     | 1        | 1000 |   0.84     |            0.84     |   0.839744 | 0.906028 |

## Interpretation

These are frozen ResNet18 feature baselines using logistic regression with different source-center weighting policies. They are a fast proxy for testing whether aggregation-style weighting changes source-domain and OOD-center performance before full iterative FL.
