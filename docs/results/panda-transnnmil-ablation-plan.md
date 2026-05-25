# PANDA TransnnMIL Ablation Plan

**Status:** planned controlled validation  
**Dataset:** PANDA prostate cancer pathology  
**Feature source:** Phikon patch features  
**Clinical status:** research-only, not clinically validated

---

## Purpose

The tuned TransnnMIL run reached validation QWK 0.8155 on the current PANDA split, slightly exceeding the gated AttentionMIL baseline QWK 0.8100. Because the margin is small, the next step is controlled ablation and repeated-split validation rather than stronger claims.

This plan is designed to answer three questions:

1. Is the TransnnMIL improvement robust across random seeds?
2. Which tuning changes mattered most?
3. Does TransnnMIL beat AttentionMIL because of architecture or because of training/patch-budget differences?

---

## Current scoreboard

| Model | Configuration | Best validation QWK |
|---|---|---:|
| Mean-pooled Phikon + MLP | 20 epochs, batch 64, full mean-pooled bags | 0.7274 |
| Gated AttentionMIL | 20 epochs, batch 16, full bags | 0.8100 |
| Initial TransnnMIL | 20 epochs, batch 4, max 600 patches, lr 1e-3, dropout 0.25 | 0.7326 |
| Tuned TransnnMIL | 30 epochs, batch 2, max 1200 patches, lr 3e-4, dropout 0.15 | 0.8155 |

---

## Fixed protocol

All controlled runs should use:

- same PANDA manifest: `results/panda_manifest/panda_phikon_manifest.csv`
- same readability verification: `--verify-read`
- same validation fraction: `0.2`
- same metric priority: quadratic weighted kappa (QWK)
- same output pattern: `results/panda_transnnmil_ablation/<run_name>/`
- no tracked checkpoints: delete `transnnmil.pt` before committing results

The known unreadable HDF5 files should be dropped during verification and recorded in each run's `unreadable_features.csv`.

---

## Phase 1: confirm tuned TransnnMIL across repeated seeds

These runs test whether the QWK 0.8155 result survives different train/validation splits and patch subsampling seeds.

```powershell
python scripts\training\train_panda_transnnmil_baseline.py --out-dir results\panda_transnnmil_ablation\tuned_seed_42 --epochs 30 --batch-size 2 --device cuda --verify-read --max-patches 1200 --lr 3e-4 --dropout 0.15 --seed 42
python scripts\training\train_panda_transnnmil_baseline.py --out-dir results\panda_transnnmil_ablation\tuned_seed_123 --epochs 30 --batch-size 2 --device cuda --verify-read --max-patches 1200 --lr 3e-4 --dropout 0.15 --seed 123
python scripts\training\train_panda_transnnmil_baseline.py --out-dir results\panda_transnnmil_ablation\tuned_seed_2025 --epochs 30 --batch-size 2 --device cuda --verify-read --max-patches 1200 --lr 3e-4 --dropout 0.15 --seed 2025
```

Success criterion:

- mean QWK exceeds 0.8100, or
- at least 2/3 repeated seeds exceed the AttentionMIL baseline.

If not, the claim should remain: tuned TransnnMIL slightly exceeded AttentionMIL on one split, but robustness is unproven.

---

## Phase 2: isolate tuning effects

Run one-factor changes from the tuned configuration.

| Run name | Purpose | Command change |
|---|---|---|
| `cap_600` | Test patch budget effect | `--max-patches 600` |
| `lr_1e3` | Test learning-rate effect | `--lr 1e-3` |
| `dropout_25` | Test dropout effect | `--dropout 0.25` |
| `epochs_20` | Test longer training effect | `--epochs 20` |
| `batch_4_cap_1200` | Test batch-size effect if VRAM allows | `--batch-size 4 --max-patches 1200` |

Suggested commands:

```powershell
python scripts\training\train_panda_transnnmil_baseline.py --out-dir results\panda_transnnmil_ablation\cap_600 --epochs 30 --batch-size 2 --device cuda --verify-read --max-patches 600 --lr 3e-4 --dropout 0.15 --seed 42
python scripts\training\train_panda_transnnmil_baseline.py --out-dir results\panda_transnnmil_ablation\lr_1e3 --epochs 30 --batch-size 2 --device cuda --verify-read --max-patches 1200 --lr 1e-3 --dropout 0.15 --seed 42
python scripts\training\train_panda_transnnmil_baseline.py --out-dir results\panda_transnnmil_ablation\dropout_25 --epochs 30 --batch-size 2 --device cuda --verify-read --max-patches 1200 --lr 3e-4 --dropout 0.25 --seed 42
python scripts\training\train_panda_transnnmil_baseline.py --out-dir results\panda_transnnmil_ablation\epochs_20 --epochs 20 --batch-size 2 --device cuda --verify-read --max-patches 1200 --lr 3e-4 --dropout 0.15 --seed 42
python scripts\training\train_panda_transnnmil_baseline.py --out-dir results\panda_transnnmil_ablation\batch_4_cap_1200 --epochs 30 --batch-size 4 --device cuda --verify-read --max-patches 1200 --lr 3e-4 --dropout 0.15 --seed 42
```

Interpretation:

- If `cap_600` collapses, patch budget is a major driver.
- If `lr_1e3` collapses, optimization stability is a major driver.
- If `dropout_25` collapses, regularization was too strong.
- If `epochs_20` is close to tuned, 30 epochs may not be necessary.
- If `batch_4_cap_1200` OOMs or underperforms, batch size 2 remains the RTX 3060-safe setting.

---

## Phase 3: compare against AttentionMIL under similar budget

The current AttentionMIL result used full bags. If TransnnMIL is capped at 1200 patches for memory reasons, a fairer comparison should also run AttentionMIL with the same cap and seed protocol.

Suggested comparison:

```powershell
python scripts\training\train_panda_attention_mil_baseline.py --out-dir results\panda_attention_mil_ablation\cap_1200_seed_42 --epochs 30 --batch-size 16 --device cuda --verify-read --max-patches 1200 --lr 3e-4 --dropout 0.15 --seed 42
```

If AttentionMIL with the same patch cap and tuning also improves beyond 0.8155, then the improvement is likely training-budget/configuration driven rather than TransnnMIL-specific.

---

## Reporting table template

| Run | Seed | Epochs | Batch | Max patches | LR | Dropout | Best QWK | Accuracy | Macro F1 | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| tuned_seed_42 | 42 | 30 | 2 | 1200 | 3e-4 | 0.15 | 0.8155 | 0.6528 | TBD | completed |
| tuned_seed_123 | 123 | 30 | 2 | 1200 | 3e-4 | 0.15 | TBD | TBD | TBD | planned |
| tuned_seed_2025 | 2025 | 30 | 2 | 1200 | 3e-4 | 0.15 | TBD | TBD | TBD | planned |
| cap_600 | 42 | 30 | 2 | 600 | 3e-4 | 0.15 | TBD | TBD | TBD | planned |
| lr_1e3 | 42 | 30 | 2 | 1200 | 1e-3 | 0.15 | TBD | TBD | TBD | planned |
| dropout_25 | 42 | 30 | 2 | 1200 | 3e-4 | 0.25 | TBD | TBD | TBD | planned |
| epochs_20 | 42 | 20 | 2 | 1200 | 3e-4 | 0.15 | TBD | TBD | TBD | planned |

---

## Claim boundary

Use:

> Tuned TransnnMIL slightly exceeded AttentionMIL on one PANDA held-out split, reaching validation QWK 0.8155 versus 0.8100.

Do not use:

> TransnnMIL is conclusively superior to AttentionMIL.

Use after repeated-seed confirmation only if supported:

> Tuned TransnnMIL showed a small but repeated QWK improvement over the current AttentionMIL baseline across repeated splits.

---

## Next deliverable

The next deliverable is a compact ablation summary file:

```text
results/panda_transnnmil_ablation/ablation_summary.csv
```

It should aggregate each run's `metrics.json` into one table with best QWK, final QWK, accuracy, macro F1, seed, epochs, learning rate, dropout, max patch cap, and runtime.
