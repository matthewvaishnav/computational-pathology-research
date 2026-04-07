# Benchmark Results Manifest

## pcam_baseline

- **Dataset**: PatchCamelyon (PCam)
- **Subset Size**: 700
- **Status**: COMPLETE
- **Date**: 2026-04-07

### Metrics
- **test_accuracy**: 0.94
- **test_auc**: 1.0
- **precision_macro**: 0.951
- **recall_macro**: 0.933
- **f1_macro**: 0.938
- **training_time_seconds**: 40
- **inference_time_seconds**: 0.81
- **throughput_samples_per_sec**: 123.5
- **final_epoch**: 8
- **best_epoch**: 3

### Commands

**Train**:
```bash
python experiments/train_pcam.py --config experiments/configs/pcam.yaml
```

**Eval**:
```bash
python experiments/evaluate_pcam.py --checkpoint checkpoints/pcam/best_model.pth --data-root data/pcam --output-dir results/pcam --batch-size 64 --num-workers 0
```

### Caveats
- Synthetic data: Not real PCam samples, generated for testing
- Tiny scale: 500 train / 100 test vs 262K train / 32K test in full PCam
- No distribution shift: Train/test from same synthetic generation
- Not comparable to published baselines: Different dataset scale

---

