# Checkpoint Validation Report

**Date**: 2026-04-28 09:52:12

---

## 1. Checkpoint Loading

✓ **Status**: Success

- Device: cuda
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

- Device: cuda
- Samples tested: 1000
- Avg inference time: 3.37ms ± 11.40ms
- Min/Max time: 1.43ms / 362.86ms
- Throughput: 296.8 samples/sec

## 3. Batch Processing Benchmark

✓ **Status**: Success

- Device: cuda
- Optimal batch size: 64

| Batch Size | Avg Time (ms) | Time/Sample (ms) | Throughput (samples/sec) |
|------------|---------------|------------------|-------------------------|
| 1 | 2.95 | 2.95 | 338.8 |
| 4 | 2.97 | 0.74 | 1347.2 |
| 16 | 2.88 | 0.18 | 5554.8 |
| 32 | 3.55 | 0.11 | 9020.8 |
| 64 | 6.75 | 0.11 | 9482.7 |

## Summary

✓ **All validation tests passed successfully!**

The trained models loaded correctly and are ready for use in the streaming pipeline.

**Key Metrics**:
- Total parameters: 17,910,081
- Inference time: 3.37ms per sample
- Throughput: 296.8 samples/sec
- Optimal batch size: 64
- Validation AUC: 0.9537
- Validation Accuracy: 0.8786
