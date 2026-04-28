# Checkpoint Validation Report

**Date**: 2026-04-28 09:19:16

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
- Samples tested: 1000
- Avg inference time: 7.27ms ± 1.24ms
- Min/Max time: 5.18ms / 17.31ms
- Throughput: 137.5 samples/sec

## 3. Batch Processing Benchmark

✓ **Status**: Success

- Device: cpu
- Optimal batch size: 64

| Batch Size | Avg Time (ms) | Time/Sample (ms) | Throughput (samples/sec) |
|------------|---------------|------------------|-------------------------|
| 1 | 7.82 | 7.82 | 127.8 |
| 4 | 14.39 | 3.60 | 277.9 |
| 16 | 31.19 | 1.95 | 513.0 |
| 32 | 56.38 | 1.76 | 567.6 |
| 64 | 107.67 | 1.68 | 594.4 |

## Summary

✓ **All validation tests passed successfully!**

The trained models loaded correctly and are ready for use in the streaming pipeline.

**Key Metrics**:
- Total parameters: 17,910,081
- Inference time: 7.27ms per sample
- Throughput: 137.5 samples/sec
- Optimal batch size: 64
- Validation AUC: 0.9537
- Validation Accuracy: 0.8786
