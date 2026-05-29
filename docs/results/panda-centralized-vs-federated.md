# PANDA Centralized vs Federated Benchmark

**Status:** completed 1,000-slide and 3,000-slide PANDA-derived benchmark tiers  
**Dataset source:** PANDA-derived Phikon slide feature cache  
**Clinical status:** simulated-federation benchmark only; not real multi-center clinical validation; not diagnostic software

---

## Research question

How much performance is lost when PANDA-derived pathology features are trained through simulated federated learning instead of centralized training?

This benchmark compares three regimes on the same cached PANDA Phikon feature split:

| Regime | Description |
|---|---|
| `centralized_all` | One model trained on the union of all simulated-site training data |
| `local_site_k` | One model trained only on a single simulated site's local data |
| `fedavg` | Standard sample-size-weighted federated averaging across simulated sites |

The purpose is to establish the trunk benchmark for PathologyFL before evaluating experimental aggregation variants such as FAIR-WEIGHTS-H.

---

## Input data

The benchmark uses cached subsets of PANDA-derived Phikon features.

### 1,000-slide cache

| Property | Value |
|---|---:|
| Cached slides | 1,000 |
| Feature dimension | 768 |
| Labels | ISUP grade 0–5 |
| Cache file | `C:/panda_cache/panda_phikon_mean_features_1000.npz` |
| Feature pooling | mean pooling over per-slide patch features |
| Seeds | `42`, `123`, `2025`, `7`, `99` |

Label distribution in the 1,000-slide cache:

| ISUP grade | Count |
|---:|---:|
| 0 | 169 |
| 1 | 167 |
| 2 | 166 |
| 3 | 166 |
| 4 | 166 |
| 5 | 166 |

### 3,000-slide cache

| Property | Value |
|---|---:|
| Attempted slides | 3,000 |
| Cached readable slides | 2,999 |
| Bad feature files | 1 |
| Feature dimension | 768 |
| Labels | ISUP grade 0–5 |
| Cache file | `C:/panda_cache/panda_phikon_mean_features_3000.npz` |
| Feature pooling | mean pooling over per-slide patch features |
| Seeds | `42`, `123`, `2025`, `7`, `99` |

Label distribution in the 3,000-slide cache:

| ISUP grade | Count |
|---:|---:|
| 0 | 500 |
| 1 | 499 |
| 2 | 500 |
| 3 | 500 |
| 4 | 500 |
| 5 | 500 |

One HDF5 feature file failed during cache creation and was skipped:

```text
0032bfa835ce0f43a92ae0bbab6871cb.h5
error: Can't synchronously read data (filter returned failure during read)
```

The simulated federation used five sites with approximate proportions:

```text
0.45, 0.15, 0.15, 0.125, 0.125
```

This creates one larger simulated institution and four smaller institutions. No label noise was injected in this benchmark; this is the clean centralized/local/FedAvg baseline.

---

## Reproduction commands

Create a cache:

```powershell
python -u scripts\data\cache_panda_pooled_features.py `
  --manifest results\panda_manifest\panda_phikon_manifest.csv `
  --output C:\panda_cache\panda_phikon_mean_features_3000.npz `
  --pool mean `
  --limit 3000 `
  --progress-every 50
```

Run the five-seed benchmark:

```powershell
foreach ($s in 42,123,2025,7,99) {
  python scripts\experiments\run_panda_centralized_vs_federated.py `
    --feature-cache C:\panda_cache\panda_phikon_mean_features_3000.npz `
    --output-dir "results\panda_centralized_vs_federated_3000_seed_$s" `
    --rounds 5 `
    --local-epochs 1 `
    --epochs 10 `
    --seed $s `
    --device cuda
}
```

Aggregate the results:

```powershell
python scripts\experiments\aggregate_panda_centralized_vs_federated.py `
  --pattern "results\panda_centralized_vs_federated_3000_seed_*\summary.csv" `
  --output-dir results\panda_centralized_vs_federated_3000_aggregate `
  --baseline centralized_all
```

---

## Aggregate result across five seeds

### 3,000-slide PANDA-derived benchmark

| Regime family | Mean global QWK | Mean accuracy | Mean macro F1 | Mean worst-site QWK | Interpretation |
|---|---:|---:|---:|---:|---|
| `centralized_all` | **0.6949** | **0.5208** | **0.5174** | **0.6064** | strongest overall baseline |
| `fedavg` | 0.6659 | 0.4829 | 0.4709 | 0.5769 | improves substantially over average isolated local training, but still trails centralized training |
| local-only family mean | 0.6075 | 0.4280 | 0.4239 | 0.5062 | weakest average regime |

The 3,000-slide aggregator reported:

```text
Best global_qwk: centralized_all mean=0.6949
Best global_accuracy: centralized_all mean=0.5208
Best macro_f1: centralized_all mean=0.5174
Best worst_site_qwk: centralized_all mean=0.6064
Best mean_site_qwk: centralized_all mean=0.6981
Local-only family mean global_qwk=0.6075, global_accuracy=0.4280, macro_f1=0.4239
```

### 1,000-slide PANDA-derived benchmark

| Regime family | Mean global QWK | Mean accuracy | Mean macro F1 | Mean worst-site QWK | Interpretation |
|---|---:|---:|---:|---:|---|
| `centralized_all` | **0.6425** | **0.4946** | **0.4894** | **0.4275** | strongest overall baseline |
| `fedavg` | 0.5550 | 0.4374 | 0.4232 | 0.3690 | improves over average isolated local training, but trails centralized training |
| local-only family mean | 0.5075 | 0.3687 | 0.3616 | 0.3246 | weakest average regime |

The 1,000-slide aggregator reported:

```text
Best global_qwk: centralized_all mean=0.6425
Best global_accuracy: centralized_all mean=0.4946
Best macro_f1: centralized_all mean=0.4894
Best worst_site_qwk: centralized_all mean=0.4275
Best mean_site_qwk: centralized_all mean=0.6450
Local-only family mean global_qwk=0.5075, global_accuracy=0.3687, macro_f1=0.3616
```

---

## Main finding

The clean benchmark pattern is stable across both the 1,000-slide and 3,000-slide cached PANDA-derived feature tiers:

```text
centralized_all > fedavg > average local-only
```

FedAvg does not match centralized training, but it closes part of the gap between isolated local training and centralized learning.

### 3,000-slide benchmark: FedAvg vs local-only family mean

| Metric | FedAvg | Local-only family mean | Difference |
|---|---:|---:|---:|
| Global QWK | 0.6659 | 0.6075 | +0.0584 |
| Accuracy | 0.4829 | 0.4280 | +0.0549 |
| Macro F1 | 0.4709 | 0.4239 | +0.0470 |
| Worst-site QWK | 0.5769 | 0.5062 | +0.0707 |

### 3,000-slide benchmark: FedAvg vs centralized training

| Metric | FedAvg | Centralized | Difference |
|---|---:|---:|---:|
| Global QWK | 0.6659 | 0.6949 | -0.0290 |
| Accuracy | 0.4829 | 0.5208 | -0.0379 |
| Macro F1 | 0.4709 | 0.5174 | -0.0465 |
| Worst-site QWK | 0.5769 | 0.6064 | -0.0295 |

### 1,000-slide benchmark: FedAvg vs local-only family mean

| Metric | FedAvg | Local-only family mean | Difference |
|---|---:|---:|---:|
| Global QWK | 0.5550 | 0.5075 | +0.0475 |
| Accuracy | 0.4374 | 0.3687 | +0.0687 |
| Macro F1 | 0.4232 | 0.3616 | +0.0616 |
| Worst-site QWK | 0.3690 | 0.3246 | +0.0444 |

### 1,000-slide benchmark: FedAvg vs centralized training

| Metric | FedAvg | Centralized | Difference |
|---|---:|---:|---:|
| Global QWK | 0.5550 | 0.6425 | -0.0875 |
| Accuracy | 0.4374 | 0.4946 | -0.0572 |
| Macro F1 | 0.4232 | 0.4894 | -0.0662 |
| Worst-site QWK | 0.3690 | 0.4275 | -0.0585 |

---

## Scaling observation

Increasing the cached PANDA-derived feature subset from 1,000 to 3,000 slides improved all regimes and narrowed the performance gap between FedAvg and centralized training.

| Metric | 1,000-slide centralized - FedAvg gap | 3,000-slide centralized - FedAvg gap |
|---|---:|---:|
| Global QWK | 0.0875 | 0.0290 |
| Accuracy | 0.0572 | 0.0379 |
| Macro F1 | 0.0662 | 0.0465 |
| Worst-site QWK | 0.0585 | 0.0295 |

This suggests that FedAvg becomes more competitive as the simulated federation has more slide-level feature data available, while still retaining a measurable gap relative to centralized training.

---

## Interpretation

This result supports a narrow and useful baseline claim:

> On cached PANDA-derived Phikon slide features across five seeds, FedAvg improves over average isolated local-only training but remains below centralized training across global QWK, accuracy, macro F1, and worst-site QWK.

This is the expected and scientifically useful result: federation provides a practical middle ground between isolated institutional training and full centralization, but it does not remove the performance cost of distributed training.

---

## Relationship to FAIR-WEIGHTS-H

This benchmark is the trunk result. FAIR-WEIGHTS-H and other weighting methods should be evaluated against this baseline, not treated as the central project claim.

Current interpretation:

- `centralized_all` is the upper-bound baseline for this cached feature setup.
- `local_site_k` models represent isolated institutional training.
- `fedavg` is the standard federated baseline.
- FAIR-WEIGHTS-H / contribution-aware methods are experimental extensions that should be judged by whether they improve on FedAvg without eroding worst-site robustness.

---

## Claim boundary

This benchmark does **not** establish clinical utility, regulatory readiness, or real-world multi-hospital performance.

Limitations:

1. Sites are simulated from cached PANDA-derived feature subsets.
2. The benchmark uses mean-pooled Phikon features rather than full MIL over all patches.
3. The largest completed benchmark uses 2,999 readable cached slides, not the full PANDA feature set.
4. The site split is synthetic and does not correspond to a real hospital federation.
5. No external clinical validation was performed.

Supported claim:

> This is PANDA-derived simulated-federation benchmark evidence for comparing centralized, local-only, and FedAvg training regimes on cached pathology foundation-model features.

Unsupported claims:

- This is not a clinically validated prostate cancer diagnostic model.
- This is not proof of real multi-hospital deployment readiness.
- This is not evidence that any experimental weighting rule universally improves federated pathology learning.

---

## Next validation steps

1. Repeat on the full readable PANDA feature cache if storage throughput permits.
2. Add confidence intervals across seeds and site splits.
3. Compare FedAvg against contribution-aware blends and robust aggregation methods using the 3,000-slide cache.
4. Replace mean-pooled features with MIL over variable-length Phikon bags.
5. Move from simulated PANDA sites toward real multi-center datasets such as Camelyon17.
