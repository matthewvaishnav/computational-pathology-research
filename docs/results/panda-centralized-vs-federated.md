# PANDA Centralized vs Federated Benchmark

**Status:** completed 1,000-slide PANDA-derived benchmark tier  
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

The benchmark used a cached subset of PANDA-derived Phikon features:

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

The simulated federation used five sites with approximate proportions:

```text
0.45, 0.15, 0.15, 0.125, 0.125
```

This creates one larger simulated institution and four smaller institutions. No label noise was injected in this benchmark; this is the clean centralized/local/FedAvg baseline.

---

## Reproduction commands

Create the 1,000-slide cache:

```powershell
python -u scripts\data\cache_panda_pooled_features.py `
  --manifest results\panda_manifest\panda_phikon_manifest.csv `
  --output C:\panda_cache\panda_phikon_mean_features_1000.npz `
  --pool mean `
  --limit 1000 `
  --progress-every 25
```

Run the five-seed benchmark:

```powershell
foreach ($s in 42,123,2025,7,99) {
  python scripts\experiments\run_panda_centralized_vs_federated.py `
    --feature-cache C:\panda_cache\panda_phikon_mean_features_1000.npz `
    --output-dir "results\panda_centralized_vs_federated_1000_seed_$s" `
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
  --pattern "results\panda_centralized_vs_federated_1000_seed_*\summary.csv" `
  --output-dir results\panda_centralized_vs_federated_1000_aggregate `
  --baseline centralized_all
```

---

## Aggregate result across five seeds

| Regime family | Mean global QWK | Mean accuracy | Mean macro F1 | Mean worst-site QWK | Interpretation |
|---|---:|---:|---:|---:|---|
| `centralized_all` | **0.6425** | **0.4946** | **0.4894** | **0.4275** | strongest overall baseline |
| `fedavg` | 0.5550 | 0.4374 | 0.4232 | 0.3690 | improves over average isolated local training, but trails centralized training |
| local-only family mean | 0.5075 | 0.3687 | 0.3616 | 0.3246 | weakest average regime |

The aggregator reported:

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

The clean benchmark pattern is:

```text
centralized_all > fedavg > average local-only
```

FedAvg does not match centralized training, but it closes part of the gap between isolated local training and centralized learning.

Relative to the local-only family mean:

| Metric | FedAvg | Local-only family mean | Difference |
|---|---:|---:|---:|
| Global QWK | 0.5550 | 0.5075 | +0.0475 |
| Accuracy | 0.4374 | 0.3687 | +0.0687 |
| Macro F1 | 0.4232 | 0.3616 | +0.0616 |
| Worst-site QWK | 0.3690 | 0.3246 | +0.0444 |

Relative to centralized training:

| Metric | FedAvg | Centralized | Difference |
|---|---:|---:|---:|
| Global QWK | 0.5550 | 0.6425 | -0.0875 |
| Accuracy | 0.4374 | 0.4946 | -0.0572 |
| Macro F1 | 0.4232 | 0.4894 | -0.0662 |
| Worst-site QWK | 0.3690 | 0.4275 | -0.0585 |

---

## Interpretation

This result supports a narrow and useful baseline claim:

> On 1,000 cached PANDA-derived Phikon slide features across five seeds, FedAvg improves over average isolated local-only training but remains below centralized training across global QWK, accuracy, macro F1, and worst-site QWK.

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

1. Sites are simulated from a cached PANDA-derived feature subset.
2. The benchmark uses mean-pooled Phikon features rather than full MIL over all patches.
3. The result uses 1,000 cached slides, not the full PANDA feature set.
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

1. Repeat the benchmark on the 3,000-slide cache once feature caching completes.
2. Repeat on the full readable PANDA feature cache if storage throughput permits.
3. Add confidence intervals across seeds and site splits.
4. Compare FedAvg against contribution-aware blends and robust aggregation methods.
5. Move from simulated PANDA sites toward real multi-center datasets such as Camelyon17.
