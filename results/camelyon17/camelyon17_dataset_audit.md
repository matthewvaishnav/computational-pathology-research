# Camelyon17/WILDS dataset audit

## Purpose

Prepare Camelyon17 as the first natural multi-site external validation target for the dominant-site detector-switch hypothesis.

## Dataset metadata

- Total examples: 455,954
- Metadata fields: ['hospital', 'slide', 'y', 'from_source_domain']
- Available columns: ['index', 'y', 'patient', 'node', 'x_coord', 'y_coord', 'tumor', 'slide', 'center', 'split']
- Selected FL client column: `center`

## Split counts

```json
{
  "id_val": 33560,
  "test": 85054,
  "train": 302436,
  "val": 34904
}
```

## Class counts

```json
{
  "0": 227977,
  "1": 227977
}
```

## Client summary

|   client_id |   total_n |   id_val__0 |   id_val__1 |   test__0 |   test__1 |   train__0 |   train__1 |   val__0 |   val__1 |
|------------:|----------:|------------:|------------:|----------:|----------:|-----------:|-----------:|---------:|---------:|
|           4 |    146722 |        7437 |        7233 |         0 |         0 |      65924 |      66128 |        0 |        0 |
|           3 |    129838 |        6483 |        6396 |         0 |         0 |      58436 |      58523 |        0 |        0 |
|           2 |     85054 |           0 |           0 |     42527 |     42527 |          0 |          0 |        0 |        0 |
|           0 |     59436 |        3032 |        2979 |         0 |         0 |      26686 |      26739 |        0 |        0 |
|           1 |     34904 |           0 |           0 |         0 |         0 |          0 |          0 |    17452 |    17452 |

## Next decision

Use the selected client column as the simulated FL `client_id`, then compare FedAvg, equal-client weighting, FedProx, and detector-switch logic on worst-site and global metrics.
