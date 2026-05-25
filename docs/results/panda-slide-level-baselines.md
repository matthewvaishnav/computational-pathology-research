# PANDA Slide-Level Baselines

**Status:** mean-pooling, AttentionMIL, and tuned TransnnMIL slide-level baselines completed  
**Dataset:** PANDA prostate cancer pathology  
**Feature source:** Phikon patch features  
**Clinical status:** research-only, not clinically validated

---

## Summary

This page tracks slide-level PANDA experiments using extracted Phikon feature files. These experiments move the project beyond patch-level PCam validation toward whole-slide prostate pathology modeling.

Three slide-level baselines are now complete:

1. mean pooling over patch-level Phikon features followed by a small MLP classifier,
2. gated AttentionMIL over patch-level Phikon feature bags,
3. tuned TransnnMIL over capped Phikon feature bags.

AttentionMIL substantially improves over mean pooling. Tuned TransnnMIL slightly exceeds AttentionMIL under the current held-out validation split, using a lower learning rate, lower dropout, larger patch cap, and smaller batch size.

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
| Readable slides used for full baselines | 10,611 |

Unreadable HDF5 files are recorded in:

```text
results/panda_mean_pooling_baseline/unreadable_features.csv
results/panda_attention_mil_baseline/unreadable_features.csv
results/panda_transnnmil_baseline/unreadable_features.csv
```

---

## Baseline comparison

| Model | Readable slides | Train | Validation | Best validation QWK | Final validation QWK | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean-pooled Phikon + MLP | 10,611 | 8,488 | 2,123 | 0.7274 | 0.7274 | 0.5874 | 0.5563 |
| Gated AttentionMIL | 10,611 | 8,488 | 2,123 | 0.8100 | 0.8100 | 0.6821 | 0.6389 |
| Tuned TransnnMIL | 10,611 | 8,488 | 2,123 | 0.8155 | 0.8155 | 0.6528 | See metrics JSON |

Compared with the initial baselines:

- AttentionMIL improves validation QWK by +0.0826 over mean pooling.
- Tuned TransnnMIL improves validation QWK by +0.0881 over mean pooling.
- Tuned TransnnMIL improves validation QWK by +0.0055 over AttentionMIL.

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

## Baseline 2: gated AttentionMIL

**Script:**

```text
scripts/training/train_panda_attention_mil_baseline.py
```

**Command:**

```powershell
python scripts\training\train_panda_attention_mil_baseline.py --epochs 20 --batch-size 16 --device cuda --verify-read
```

**Model:**

```text
Phikon patch features -> gated attention pooling -> classifier -> ISUP grade
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
| Best validation QWK | 0.8100 |
| Final validation QWK | 0.8100 |
| Final validation accuracy | 0.6821 |
| Final macro F1 | 0.6389 |

Outputs:

```text
results/panda_attention_mil_baseline/metrics.json
results/panda_attention_mil_baseline/val_predictions.csv
results/panda_attention_mil_baseline/unreadable_features.csv
```

The checkpoint `attention_mil.pt` is intentionally not tracked.

---

## Baseline 3: tuned TransnnMIL

**Script:**

```text
scripts/training/train_panda_transnnmil_baseline.py
```

**Command:**

```powershell
python scripts\training\train_panda_transnnmil_baseline.py --epochs 30 --batch-size 2 --device cuda --verify-read --max-patches 1200 --lr 3e-4 --dropout 0.15
```

**Model:**

```text
Phikon patch features -> capped variable-length bags -> TransnnMIL fusion model -> ISUP grade
```

The tuned TransnnMIL run used a conservative RTX 3060-compatible configuration:

| Setting | Value |
|---|---:|
| Epochs | 30 |
| Batch size | 2 |
| Max patches per slide | 1,200 |
| Learning rate | 3e-4 |
| Dropout | 0.15 |

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
| Best validation QWK | 0.8155 |
| Final validation QWK | 0.8155 |
| Final validation accuracy | 0.6528 |

Outputs:

```text
results/panda_transnnmil_baseline/metrics.json
results/panda_transnnmil_baseline/val_predictions.csv
results/panda_transnnmil_baseline/unreadable_features.csv
```

The checkpoint `transnnmil.pt` is intentionally not tracked.

---

## Interpretation

The mean-pooling baseline establishes that the PANDA Phikon feature pipeline is trainable end to end and provides a first slide-level benchmark. AttentionMIL improves over this baseline by learning patch-level importance weights before slide-level classification.

The first conservative TransnnMIL configuration only slightly exceeded mean pooling and underperformed AttentionMIL. A tuned TransnnMIL configuration using a larger patch cap, lower learning rate, lower dropout, smaller batch size, and longer training improved to validation QWK 0.8155, slightly exceeding the AttentionMIL baseline of 0.8100 on the current split.

This supports TransnnMIL as a competitive PANDA slide-level model under tuned settings, but the margin over AttentionMIL is small. The next step should be controlled ablation rather than stronger claims.

---

## Claim boundary

Use:

> PANDA slide-level baseline training has started. Mean-pooled Phikon features achieved validation QWK 0.7274, gated AttentionMIL improved to validation QWK 0.8100, and tuned TransnnMIL reached validation QWK 0.8155 on the current held-out split after HDF5 readability verification.

Do not use:

> The project is clinically validated for prostate cancer grading.

Do not use:

> The model is ready for deployment.

Do not use:

> TransnnMIL is conclusively superior to AttentionMIL.

Use instead:

> Tuned TransnnMIL slightly exceeded AttentionMIL on the current PANDA validation split; controlled ablations and repeated splits are needed to establish robustness.
