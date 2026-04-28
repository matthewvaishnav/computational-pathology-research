# Checkpoint Validation Report

**Date**: 2026-04-28 08:58:24

---

## 1. Checkpoint Loading

✓ **Status**: Success

- Device: cpu
- Epoch: 2

**Training Metrics**:
- val_loss: 0.3476
- val_accuracy: 0.8786
- val_f1: 0.8677
- val_auc: 0.9537

**Model Parameters**:
- CNN Encoder: 11,176,512
- Attention Model: 6,733,569
- Total: 17,910,081

## 2. Model Inference (Synthetic Data)

✓ **Status**: Success

- Device: cpu
- Samples tested: 100
- Avg inference time: 13.20ms ± 1.91ms
- Min/Max time: 10.43ms / 21.88ms
- Throughput: 75.8 samples/sec

## 3. Batch Processing Benchmark

✓ **Status**: Success

- Device: cpu
- Optimal batch size: 64

| Batch Size | Avg Time (ms) | Time/Sample (ms) | Throughput (samples/sec) |
|------------|---------------|------------------|-------------------------|
| 1 | 13.25 | 13.25 | 75.5 |
| 4 | 25.33 | 6.33 | 157.9 |
| 16 | 59.00 | 3.69 | 271.2 |
| 32 | 96.36 | 3.01 | 332.1 |
| 64 | 136.22 | 2.13 | 469.8 |

## Summary

✓ **All validation tests passed successfully!**

The trained models loaded correctly and are ready for use in the streaming pipeline.

**Key Metrics**:
- Total parameters: 17,910,081
- Inference time: 13.20ms per sample
- Throughput: 75.8 samples/sec
- Optimal batch size: 64
- Validation AUC: 0.9537
- Validation Accuracy: 0.8786
