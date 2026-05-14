# TransnnMIL v2.0: Hierarchical Pooling Ablations

## Overview

Ablation studies for hierarchical pooling module (Phase 1, Week 4).

**Status:** Partial completion
- ✅ Pooling methods ablation (4.4)
- ⏳ Num regions ablation (4.2) - pending
- ⏳ Clustering methods ablation (4.3) - pending
- ⏳ TCGA-BRCA benchmark (4.5) - pending real data

---

## Pooling Methods Ablation (Task 4.4)

**Date:** 2026-05-14

**Setup:**
- Dataset: Synthetic MIL (500 samples, 200 patches/bag)
- Feature dim: 512
- Num regions: 16
- Num classes: 2 (binary)
- Epochs: 20
- Batch size: 8
- Device: CPU

**Methods Compared:**
1. **Attention pooling** - Learned attention weights per region
2. **Mean pooling** - Weighted average (baseline)
3. **Max pooling** - Max feature per region (baseline)

### Results

| Method    | Val AUC | Val Acc | Val F1 | Params    | Train Time (s/epoch) | Memory (MB) |
|-----------|---------|---------|--------|-----------|----------------------|-------------|
| Attention | 1.0000  | 1.0000  | 1.0000 | 2,163,747 | 0.98                 | 0.0         |
| Mean      | 1.0000  | 0.8900  | 0.8881 | 2,097,954 | 0.42                 | 0.0         |
| Max       | 0.5918  | 0.5100  | 0.3377 | 2,097,954 | 0.77                 | 0.0         |

### Analysis

**Attention Pooling (Winner):**
- Best performance: AUC 1.0, Acc 1.0, F1 1.0
- +65,793 params vs baselines (+3.1%)
- 2.3x slower than mean pooling
- Perfect classification on synthetic data
- Learns region-specific importance

**Mean Pooling (Strong Baseline):**
- Excellent AUC (1.0), good accuracy (0.89)
- Fastest training (0.42s/epoch)
- Fewest parameters (2.1M)
- Simple, efficient, no learned weights
- Good tradeoff: speed vs performance

**Max Pooling (Failed):**
- Poor performance: AUC 0.59, Acc 0.51 (random)
- Slower than mean (0.77s/epoch)
- Hard assignment → gradient issues
- Not suitable for soft region assignments

### Recommendations

1. **Default:** Attention pooling
   - Best performance
   - Acceptable speed overhead
   - Learns region importance

2. **Fast baseline:** Mean pooling
   - 2.3x faster training
   - 89% accuracy (vs 100%)
   - Good for rapid prototyping

3. **Avoid:** Max pooling
   - Incompatible with soft assignments
   - Poor gradient flow
   - Random-level performance

### Next Steps

- [ ] Test on real TCGA-BRCA data
- [ ] Ablate num_regions (8, 16, 32, 64)
- [ ] Ablate clustering methods (learnable, kmeans, grid)
- [ ] Compare hierarchical vs flat (no regions)

---

## Num Regions Ablation (Task 4.2)

**Status:** Pending

**Plan:**
- Test: 8, 16, 32, 64 regions
- Metrics: AUC, accuracy, speed, memory
- Hypothesis: 16-32 optimal (balance granularity vs overhead)

---

## Clustering Methods Ablation (Task 4.3)

**Status:** Pending

**Plan:**
- Learnable centers (gradient descent)
- K-means (sklearn, fixed)
- Grid (uniform, fixed)
- Hypothesis: Learnable > k-means > grid

---

## TCGA-BRCA Benchmark (Task 4.5)

**Status:** Blocked - no TCGA data loader

**Requirements:**
- TCGA-BRCA dataset with H5 features
- Patch coordinates
- Train/val/test splits
- Baseline comparison (TransnnMIL v1.0)

---

## References

- Script: `scripts/ablate_pooling_methods.py`
- Results: `experiments/results/pooling_ablation/pooling_ablation_results.json`
- Models: `src/models/hierarchical_pooling.py`
- Tests: `tests/models/test_hierarchical_pooling.py`
