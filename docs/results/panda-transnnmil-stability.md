# PANDA TransnnMIL Stabilization Results

**Status:** completed repeated-seed optimizer-stability summary  
**Task:** PANDA slide-level ISUP grading  
**Features:** Phikon patch feature bags  
**Model:** TransnnMIL baseline  
**Clinical status:** research-only; not clinically validated; not diagnostic software

---

## Why this experiment was run

Earlier PANDA TransnnMIL ablations showed strong optimizer sensitivity. In particular, the model could reach competitive validation QWK under a tuned learning rate, but a higher learning rate degraded performance substantially.

The goal of this stabilization pass was therefore not only to maximize QWK, but to test whether a more conservative training recipe widens the stable learning-rate regime.

The stabilized recipe used:

- AdamW optimizer
- warmup-cosine learning-rate scheduling
- 2 warmup epochs
- gradient clipping at norm 1.0
- early stopping patience of 6 epochs
- repeated seeds: 42, 123, 2025
- learning rates: 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3

---

## Dataset / run setup

The full PANDA Phikon feature manifest was verified for readability before training.

```text
Readable slide feature files: 10,611
Dropped unreadable files: 3
Feature dimension: 768
Train rows: 8,488
Validation rows: 2,123
```

The same unreadable HDF5 files were consistently dropped during verification, which indicates the training split was being built from the stable readable feature subset rather than failing during dataloading.

---

## Aggregated result

| Learning rate | Runs | Seeds | Mean best val QWK | Std | Min | Max | Mean best epoch |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1e-4 | 3 | 42, 123, 2025 | **0.8257** | 0.0169 | 0.8087 | 0.8425 | 17.67 |
| 2e-4 | 3 | 42, 123, 2025 | **0.8245** | 0.0192 | 0.8077 | 0.8455 | 14.67 |
| 3e-4 | 3 | 42, 123, 2025 | **0.8238** | 0.0160 | 0.8127 | 0.8422 | 18.67 |
| 1e-3 | 3 | 42, 123, 2025 | **0.8160** | 0.0170 | 0.7998 | 0.8337 | 15.00 |
| 5e-4 | 3 | 42, 123, 2025 | **0.8158** | 0.0144 | 0.8042 | 0.8319 | 16.67 |
| 7e-4 | 3 | 42, 123, 2025 | **0.8117** | 0.0163 | 0.7994 | 0.8301 | 16.00 |

Best mean best-validation QWK was observed at **1e-4** with mean QWK **0.8257** across three seeds.

---

## Interpretation

The main result is that stabilization widened the usable learning-rate regime.

Earlier ablations suggested that TransnnMIL was highly optimization-sensitive in this PANDA setup. After adding warmup-cosine scheduling, AdamW, gradient clipping, and early stopping, all six learning rates remained competitive across repeated seeds. Even the 1e-3 condition, previously associated with severe degradation, reached mean best validation QWK of approximately 0.816.

This supports the more careful claim:

> TransnnMIL was optimizer-sensitive in the initial PANDA setup, but a stabilized training recipe substantially reduced learning-rate sensitivity and kept performance competitive across a broad LR grid.

It does **not** prove that TransnnMIL is conclusively superior to gated AttentionMIL. The margin remains small and would require stronger controlled validation, additional splits, and/or external data before making architecture-superiority claims.

---

## Recommended claim boundary

Safe wording:

> On PANDA Phikon feature bags, stabilized TransnnMIL remained competitive across 18 full-PANDA runs spanning six learning rates and three seeds, with mean best validation QWK ranging from approximately 0.812 to 0.826. This suggests the model is not inherently unstable, but requires a careful training recipe.

Avoid wording:

> TransnnMIL is fixed.

> TransnnMIL is definitively better than AttentionMIL.

> This is clinically validated.

---

## Reproduction

Training grid:

```powershell
$seeds = 42,123,2025
$lrs = "1e-4","2e-4","3e-4","5e-4","7e-4","1e-3"

foreach ($seed in $seeds) {
  foreach ($lr in $lrs) {
    python scripts\training\train_panda_transnnmil_baseline.py `
      --epochs 20 `
      --batch-size 8 `
      --lr $lr `
      --scheduler warmup_cosine `
      --warmup-epochs 2 `
      --grad-clip-norm 1.0 `
      --early-stopping-patience 6 `
      --seed $seed `
      --device cuda `
      --verify-read `
      --out-dir "results\transnnmil_stability_warmup_cosine_clip_lr_${lr}_seed_${seed}"
  }
}
```

Aggregation:

```powershell
python scripts\experiments\aggregate_transnnmil_stability.py `
  --results-dir results `
  --pattern "transnnmil_stability_warmup_cosine_clip_lr_*_seed_*" `
  --out-dir results\transnnmil_stability_summary
```
