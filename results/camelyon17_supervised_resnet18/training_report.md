# Camelyon17 supervised ResNet18 source-domain training

- Device: cuda
- Source train examples: 3,000
- Eval examples: 5,000
- Max per split/center/class: 500
- Epochs: 2
- Best epoch by id_val AUC: 2

## Training history

|   epoch |   train_loss |   id_val_n |   id_val_accuracy |   id_val_balanced_accuracy |   id_val_macro_f1 |   id_val_auc |   test_n |   test_accuracy |   test_balanced_accuracy |   test_macro_f1 |   test_auc |   val_n |   val_accuracy |   val_balanced_accuracy |   val_macro_f1 |   val_auc |
|--------:|-------------:|-----------:|------------------:|---------------------------:|------------------:|-------------:|---------:|----------------:|-------------------------:|----------------:|-----------:|--------:|---------------:|------------------------:|---------------:|----------:|
|       1 |    0.182431  |       3000 |          0.963667 |                   0.963667 |          0.963658 |     0.991118 |     1000 |           0.924 |                    0.924 |        0.923794 |   0.96882  |    1000 |          0.906 |                   0.906 |       0.905998 |  0.962052 |
|       2 |    0.0717044 |       3000 |          0.962    |                   0.962    |          0.961974 |     0.99382  |     1000 |           0.901 |                    0.901 |        0.90028  |   0.964828 |    1000 |          0.905 |                   0.905 |       0.904973 |  0.972148 |

## Interpretation

This supervised source-domain model is intended as a Camelyon17-trained feature extractor for the next center-weighting and detector-switch experiments. It is not a final clinical model.
