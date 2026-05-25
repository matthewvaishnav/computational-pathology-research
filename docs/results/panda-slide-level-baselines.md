# PANDA Slide-Level Baselines

**Status:** first slide-level PANDA baseline completed  
**Dataset:** PANDA prostate cancer pathology  
**Feature source:** Phikon patch features  
**Clinical status:** research-only, not clinically validated

---

## Summary

This page tracks slide-level PANDA experiments using extracted Phikon feature files. These experiments move the project beyond patch-level PCam validation toward whole-slide prostate pathology modeling.

The current baseline uses simple mean pooling over patch-level Phikon features followed by a small MLP classifier for ISUP grade prediction.

---

## Data integrity status

The PANDA manifest was built from:

- labels: `D:\panda\train.csv`
- features: `D:\panda\features_phikon`
- manifest: `results/panda_manifest/panda_phikon_manifest.csv`

Current validated feature status:

| Item | Count |
|---|---:|
| PANDA labels | 10,616 |
| HDF5 feature files | 10,615 |
| Missing feature files | 1 |
| Feature files selected after manifest filtering | 10,614 |
| Additional unreadable compressed HDF5 files dropped during read verification | 3 |
| Readable slides used for full baseline | 10,611 |

Unreadable HDF5 files are recorded in:

```text
results/panda_mean_pooling_baseline/unreadable_features.csv
```

---

## Baseline 1: mean-pooled Phikon features

**Script:**

```text
scripts/training/train_panda_mean_pooling_baseline.py
```

**Command:**

```powershell
python scripts\training\train_panda_mean_pooling_baseline.py --epochs 20 --batch-size 64 --device cuda --verify-read
```

**Model:**

```text
Phikon patch features -> mean pooling -> MLP -> ISUP grade
```

**Split:**

| Split | Slides |
|---|---:|
| Train | 8,488 |
| Validation | 2,123 |

**Class counts after readability verification:**

| ISUP grade | Slides |
|---:|---:|
| 0 | 2,890 |
| 1 | 2,663 |
| 2 | 1,343 |
| 3 | 1,242 |
| 4 | 1,249 |
| 5 | 1,224 |

**Validation result:**

| Metric | Value |
|---|---:|
| Best validation QWK | 0.7274 |
| Final validation QWK | 0.7274 |
| Final validation accuracy | 0.5874 |
| Final macro F1 | 0.5563 |

Outputs:

```text
results/panda_mean_pooling_baseline/metrics.json
results/panda_mean_pooling_baseline/val_predictions.csv
results/panda_mean_pooling_baseline/unreadable_features.csv
```

The checkpoint `mean_pooling_mlp.pt` is intentionally not tracked.

---

## Baseline 2: AttentionMIL

**Status:** trainer added, experiment pending.

**Script:**

```text
scripts/training/train_panda_attention_mil_baseline.py
```

**Smoke test:**

```powershell
python scripts\training\train_panda_attention_mil_baseline.py --limit 500 --epochs 2 --batch-size 16 --device cuda --verify-read
```

**Full run:**

```powershell
python scripts\training\train_panda_attention_mil_baseline.py --epochs 20 --batch-size 16 --device cuda --verify-read
```

**Model:**

```text
Phikon patch features -> gated attention pooling -> classifier -> ISUP grade
```

The AttentionMIL result should be compared against the mean-pooling QWK baseline of 0.7274 before evaluating custom TransnnMIL variants.

---

## Interpretation

The mean-pooling baseline establishes that the PANDA Phikon feature pipeline is trainable end to end and provides a first slide-level benchmark. It is not an optimized prostate pathology model and should not be interpreted as clinical validation.

The next experimental question is whether learned attention pooling improves over simple mean pooling on the same manifest, split strategy, and metrics.

---

## Claim boundary

Use:

> PANDA slide-level baseline training has started. The first mean-pooled Phikon baseline achieved validation QWK 0.7274 on a held-out split after readability verification.

Do not use:

> The project is clinically validated for prostate cancer grading.

Do not use:

> The model is ready for deployment.
