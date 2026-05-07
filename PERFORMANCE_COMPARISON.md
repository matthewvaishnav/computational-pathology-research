# Performance Comparison: HistoCore vs Competitors

**Last Updated**: 2026-05-06 16:33:01

Real benchmark results from identical training tasks on NVIDIA GeForce RTX 4070 Laptop GPU. 
All frameworks used identical datasets, hyperparameters, and random seeds for fair comparison.

## Performance Summary

**Hardware**: NVIDIA GeForce RTX 4070 Laptop GPU (8188 MB)  
**Dataset**: PatchCamelyon (262,144 train / 32,768 val / 32,768 test)  
**Configuration**: 10 epochs, batch size 32, AdamW optimizer, ResNet18 CNN

| Framework   |   Accuracy |   AUC |     F1 |   Training Time (s) |   Peak GPU Memory (MB) |   Model Parameters |
|:------------|-----------:|------:|-------:|--------------------:|-----------------------:|-------------------:|
| PyTorch     |     0.7917 | 0.854 | 0.7884 |              2257.3 |                  226.7 |          4,812,610 |
| HistoCore   |     0.7917 | 0.854 | 0.7884 |              1630.2 |                  227.6 |          4,812,610 |

**Key Finding**: HistoCore **1.38x faster** training time vs baseline PyTorch (1630s vs 2257s = 627s saved per 10 epochs). Identical accuracy.

## Statistical Significance

Comparison of HistoCore against competitors using t-tests and Cohen's d effect size:

### PyTorch - Accuracy

- **HistoCore**: 0.7917
- **PyTorch**: 0.7917
- **Improvement**: +0.0000 (+0.00%)
- **Cohen's d**: 0.000 (No Effect)
- **p-value**: 1.0000
- **Statistically Significant**: No
- **CI Overlap**: Yes

### PyTorch - Auc

- **HistoCore**: 0.8540
- **PyTorch**: 0.8540
- **Improvement**: +0.0000 (+0.00%)
- **Cohen's d**: 0.000 (No Effect)
- **p-value**: 1.0000
- **Statistically Significant**: No
- **CI Overlap**: Yes

### PyTorch - F1

- **HistoCore**: 0.7884
- **PyTorch**: 0.7884
- **Improvement**: +0.0000 (+0.00%)
- **Cohen's d**: 0.000 (No Effect)
- **p-value**: 1.0000
- **Statistically Significant**: No
- **CI Overlap**: Yes

## Detailed Metrics

Complete performance metrics for all frameworks:

| Framework   |   Accuracy |   AUC |     F1 |   Precision |   Recall |   Training Time (s) |   Samples/sec |   Inference Time (ms) |   Peak GPU Memory (MB) |   Avg GPU Util (%) |   Peak GPU Temp (°C) |   Model Parameters |   Epochs |   Final Train Loss |   Final Val Loss |   Accuracy CI Lower |   Accuracy CI Upper |   AUC CI Lower |   AUC CI Upper |   F1 CI Lower |   F1 CI Upper | Status   |
|:------------|-----------:|------:|-------:|------------:|---------:|--------------------:|--------------:|----------------------:|-----------------------:|-------------------:|---------------------:|-------------------:|---------:|-------------------:|-----------------:|--------------------:|--------------------:|---------------:|---------------:|--------------:|--------------:|:---------|
| PyTorch     |     0.7917 | 0.854 | 0.7884 |       0.811 |   0.7917 |              2257.3 |        1161.3 |                0.0613 |                226.71  |                  0 |                    0 |          4,812,610 |       10 |             0.0787 |           0.5322 |              0.7873 |              0.7962 |         0.8498 |         0.8583 |        0.7838 |        0.7931 | success  |
| HistoCore   |     0.7917 | 0.854 | 0.7884 |       0.811 |   0.7917 |              1630.2 |        1608   |                0.0664 |                227.585 |                  0 |                    0 |          4,812,610 |       10 |             0.0787 |           0.5322 |              0.7873 |              0.7962 |         0.8498 |         0.8583 |        0.7838 |        0.7931 | success  |

## Quality Assurance Flags

The following frameworks have QA flags indicating potential issues:

### PyTorch

**Flags**:
- `low_gpu_utilization`

**Issues**:
- [WARNING] Low GPU utilization (0.0%) may indicate inefficient training

**Manual Review Required**: Yes

### HistoCore

**Flags**:
- `low_gpu_utilization`

**Issues**:
- [WARNING] Low GPU utilization (0.0%) may indicate inefficient training

**Manual Review Required**: Yes

## Reproducibility

### Environment Details

- **Dataset**: PatchCamelyon
- **Model Architecture**: resnet18_transformer
- **Epochs**: 10
- **Batch Size**: 32
- **Learning Rate**: 0.0001
- **Optimizer**: AdamW
- **Random Seed**: 42

### System Information

- **Platform**: Windows 10
- **Python Version**: 3.11.15
- **Processor**: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel

### GPU Information

- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU
- **Memory**: 8188 MB
- **Peak Memory Usage**: 227.6 MB
- **Peak Temperature**: 0.0°C

## Next Steps

### Competitor Framework Integration

1. **PathML**: Java JDK installed, build fixes needed
2. **CLAM**: Package structure fixes needed
3. **Full Comparison**: Run all frameworks on same hardware with same config

### Future Benchmarks

- **Newer Datasets**: SPIDER 2025, HISTAI 2025
- **Larger Models**: Vision Transformers, Foundation Models
- **Multi-GPU**: Distributed training benchmarks
