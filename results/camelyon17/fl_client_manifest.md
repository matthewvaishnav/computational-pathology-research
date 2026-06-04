# Camelyon17 FL client manifest

## Purpose

Convert the Camelyon17/WILDS metadata audit into explicit federated-learning client roles for natural multi-center external validation.

## Split counts

```json
{
  "id_val": 33560,
  "test": 85054,
  "train": 302436,
  "val": 34904
}
```

## FL role counts

```json
{
  "federated_train": 302436,
  "ood_test": 85054,
  "ood_validation": 34904,
  "source_domain_validation": 33560
}
```

## Training-client aggregation weights

|   client_id |   train_n |   fedavg_weight |   equal_client_weight |
|------------:|----------:|----------------:|----------------------:|
|           4 |    132052 |        0.436628 |              0.333333 |
|           3 |    116959 |        0.386723 |              0.333333 |
|           0 |     53425 |        0.176649 |              0.333333 |

## Dominant training client

```json
{
  "client_id": 4.0,
  "equal_client_weight": 0.3333333333333333,
  "fedavg_weight": 0.43662791466624346,
  "train_n": 132052.0
}
```

## Client / split / class summary

|   client_id | split   | fl_role                  |     0 |     1 |   total_n |
|------------:|:--------|:-------------------------|------:|------:|----------:|
|           0 | id_val  | source_domain_validation |  3032 |  2979 |      6011 |
|           3 | id_val  | source_domain_validation |  6483 |  6396 |     12879 |
|           4 | id_val  | source_domain_validation |  7437 |  7233 |     14670 |
|           2 | test    | ood_test                 | 42527 | 42527 |     85054 |
|           0 | train   | federated_train          | 26686 | 26739 |     53425 |
|           3 | train   | federated_train          | 58436 | 58523 |    116959 |
|           4 | train   | federated_train          | 65924 | 66128 |    132052 |
|           1 | val     | ood_validation           | 17452 | 17452 |     34904 |

## Next experiment

Train source-domain clients on centers that appear in the train split, evaluate on source-domain validation id_val and OOD centers val/test, then compare FedAvg against equal-client weighting, FedProx, and detector-switch logic.
