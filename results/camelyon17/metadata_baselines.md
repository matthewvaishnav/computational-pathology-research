# Camelyon17 metadata-only baseline sanity check

These baselines intentionally do not use image pixels. They verify split/client accounting before expensive feature extraction or FL training.

## Results

| baseline                           | split   | client_id   |      n |   predicted_label |   accuracy |   balanced_accuracy |   macro_f1 |
|:-----------------------------------|:--------|:------------|-------:|------------------:|-----------:|--------------------:|-----------:|
| global_train_majority              | id_val  | all         |  33560 |                 1 |   0.494875 |                 0.5 |   0.331048 |
| global_train_majority              | test    | all         |  85054 |                 1 |   0.5      |                 0.5 |   0.333333 |
| global_train_majority              | train   | all         | 302436 |                 1 |   0.500569 |                 0.5 |   0.333586 |
| global_train_majority              | val     | all         |  34904 |                 1 |   0.5      |                 0.5 |   0.333333 |
| client_majority_or_global_fallback | id_val  | 0           |   6011 |                 1 |   0.495591 |                 0.5 |   0.331368 |
| client_majority_or_global_fallback | id_val  | 3           |  12879 |                 1 |   0.496622 |                 0.5 |   0.331829 |
| client_majority_or_global_fallback | id_val  | 4           |  14670 |                 1 |   0.493047 |                 0.5 |   0.330229 |
| client_majority_or_global_fallback | test    | 2           |  85054 |                 1 |   0.5      |                 0.5 |   0.333333 |
| client_majority_or_global_fallback | train   | 0           |  53425 |                 1 |   0.500496 |                 0.5 |   0.333554 |
| client_majority_or_global_fallback | train   | 3           | 116959 |                 1 |   0.500372 |                 0.5 |   0.333499 |
| client_majority_or_global_fallback | train   | 4           | 132052 |                 1 |   0.500772 |                 0.5 |   0.333676 |
| client_majority_or_global_fallback | val     | 1           |  34904 |                 1 |   0.5      |                 0.5 |   0.333333 |

## Interpretation

Because Camelyon17 is balanced by construction in this local audit, majority-class baselines should be weak. If a later image/feature model cannot beat these baselines, the training or split logic is broken.
