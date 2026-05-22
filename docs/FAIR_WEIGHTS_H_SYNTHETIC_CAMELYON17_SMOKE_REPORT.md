# FAIR-WEIGHTS-H Synthetic Camelyon17-like Smoke Test Report

**Date**: 2026-05-22
**Data**: Synthetic Camelyon17-like (NOT real Camelyon17)
**Purpose**: Pipeline execution validation (NOT performance comparison)
**Rounds**: 5 per strategy
**Sites**: 5 synthetic sites
**Slides per site**: 40

---

## Executive Summary

**This smoke test validates pipeline execution and logging only using synthetic data. It is not a performance comparison and should not be interpreted as evidence that one weighting strategy is superior. This test does NOT use real Camelyon17 data.**

All four weighting strategies successfully completed 5 rounds of federated training with synthetic Camelyon17-like data. The pipeline executed end-to-end without errors, demonstrating that:

1. Sites load correctly
2. Local training completes for all clients
3. Aggregation completes successfully
4. Weights are computed and logged
5. Validation metrics are emitted
6. Checkpoints are saved
7. No NaN/Inf values detected in weights or metrics

---

## Test Configuration

### Data

- **Dataset**: Synthetic Camelyon17-like data
- **Sites**: 5 (simulating multi-center study)
- **Slides per site**: 40 (balanced labels: 50% tumor, 50% normal)
- **Features**: 512-dimensional patch features (64 patches per slide)
- **Scanner bias**: Site-specific Gaussian noise added to simulate scanner artifacts

### Model

- **Architecture**: Simple attention-based MIL model
- **Attention mechanism**: Gated attention (V and U branches)
- **Classifier**: Linear layer (512 → 2 classes)
- **Training**: 1 epoch per round, batch size 32, learning rate 0.01

### Weighting Strategies

1. **Equal**: Uniform weights (1/K for K sites)
2. **Volume**: Proportional to dataset size
3. **Prestige**: Inverse error weighting (higher accuracy → higher weight)
4. **FAIR-WEIGHTS-H**: Simplified combination of quality, volume, and fairness

---

## Results

### Strategy 1: Equal Weighting

| Validation Check           | Status   |
| -------------------------- | -------- |
| Sites loaded               | ✓        |
| Local training completed   | ✓        |
| Aggregation completed      | ✓        |
| Weights logged             | ✓        |
| Validation metrics emitted | ✓        |
| Checkpoints saved          | ✓        |
| NaNs detected              | ✓ (none) |

**Metrics:**

- Weight entropy: 1.000 (maximum entropy, perfectly uniform)
- N_eff: 5.00 (all sites equally weighted)
- Global accuracy: 0.495
- Site accuracies: {0: 0.45, 1: 0.55, 2: 0.475, 3: 0.5, 4: 0.5}

**Final weights (Round 5):** {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}

**Failure notes:** None

---

### Strategy 2: Volume Weighting

| Validation Check           | Status   |
| -------------------------- | -------- |
| Sites loaded               | ✓        |
| Local training completed   | ✓        |
| Aggregation completed      | ✓        |
| Weights logged             | ✓        |
| Validation metrics emitted | ✓        |
| Checkpoints saved          | ✓        |
| NaNs detected              | ✓ (none) |

**Metrics:**

- Weight entropy: 1.000 (all sites have equal volume in synthetic data)
- N_eff: 5.00
- Global accuracy: 0.495
- Site accuracies: {0: 0.45, 1: 0.55, 2: 0.475, 3: 0.5, 4: 0.5}

**Final weights (Round 5):** {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}

**Failure notes:** None

---

### Strategy 3: Prestige Weighting

| Validation Check           | Status   |
| -------------------------- | -------- |
| Sites loaded               | ✓        |
| Local training completed   | ✓        |
| Aggregation completed      | ✓        |
| Weights logged             | ✓        |
| Validation metrics emitted | ✓        |
| Checkpoints saved          | ✓        |
| NaNs detected              | ✓ (none) |

**Metrics:**

- Weight entropy: 1.000 (weights vary slightly by round but remain balanced)
- N_eff: 5.00
- Global accuracy: 0.495
- Site accuracies: {0: 0.45, 1: 0.55, 2: 0.475, 3: 0.5, 4: 0.5}

**Final weights (Round 5):** {0: 0.204, 1: 0.204, 2: 0.194, 3: 0.194, 4: 0.204}

**Failure notes:** None

---

### Strategy 4: FAIR-WEIGHTS-H

| Validation Check           | Status   |
| -------------------------- | -------- |
| Sites loaded               | ✓        |
| Local training completed   | ✓        |
| Aggregation completed      | ✓        |
| Weights logged             | ✓        |
| Validation metrics emitted | ✓        |
| Checkpoints saved          | ✓        |
| NaNs detected              | ✓ (none) |

**Metrics:**

- Weight entropy: 1.000 (weights adapt per round based on quality/volume/fairness)
- N_eff: 5.00
- Global accuracy: 0.495
- Site accuracies: {0: 0.45, 1: 0.55, 2: 0.475, 3: 0.5, 4: 0.5}

**Final weights (Round 5):** {0: 0.199, 1: 0.199, 2: 0.201, 3: 0.201, 4: 0.199}

**Failure notes:** None

---

## Observations

### Pipeline Execution

- All strategies completed 5 rounds without errors
- Training time: ~0.01s per site per round (CPU)
- Checkpoint saving: successful for all rounds
- No NaN/Inf values detected in any strategy

### Weight Dynamics

- **Equal**: Constant uniform weights (as expected)
- **Volume**: Uniform weights (synthetic data has equal volume per site)
- **Prestige**: Slight variation based on per-round accuracy (0.194-0.204 range)
- **FAIR-WEIGHTS-H**: Slight variation based on combined signals (0.199-0.201 range)

### Accuracy

- Global accuracy: ~0.495 for all strategies (random baseline on synthetic data)
- Site accuracies: 0.45-0.55 range (expected variance for 40 slides per site)
- No strategy shows clear advantage (expected for 5-round smoke test on synthetic data)

---

## Limitations

1. **Synthetic data**: This test uses synthetic Camelyon17-like data, not real Camelyon17 slides
2. **Short training**: Only 5 rounds with 1 epoch per round (insufficient for convergence)
3. **Simplified FAIR-WEIGHTS-H**: Uses a simplified weighting formula, not the full `FairWeightsHEngine` API
4. **No performance comparison**: This test validates plumbing only, not relative performance
5. **CPU execution**: Tests run on CPU for speed; GPU execution may reveal different issues

---

## Next Steps

### Phase 1: Real Camelyon17 Smoke Test (NEXT)

Before full validation, run the same smoke test on **real Camelyon17 data** (5 rounds only):

```bash
python scripts/federated/run_camelyon17.py --weighting_strategy equal --rounds 5 --smoke
python scripts/federated/run_camelyon17.py --weighting_strategy volume --rounds 5 --smoke
python scripts/federated/run_camelyon17.py --weighting_strategy prestige --rounds 5 --smoke
python scripts/federated/run_camelyon17.py --weighting_strategy fair_weights_h --rounds 5 --smoke
```

**Goal**: Validate that:

- Real data loader works
- Site splits work correctly
- Site-wise metrics are emitted
- Weights are logged
- Checkpoints save
- No NaNs on real data

**Output**: `docs/FAIR_WEIGHTS_H_REAL_CAMELYON17_SMOKE_REPORT.md`

### Phase 2: Full Validation (20-50 rounds)

Only after real Camelyon17 smoke tests pass:

1. Train for 20-50 rounds with multiple seeds
2. Track per-round metrics:
   - Global AUC
   - Site-wise AUC
   - Worst-site sensitivity
   - ECE (calibration)
   - H(w) = -∑ wᵢ log wᵢ / log K (weight entropy)
   - N_eff = 1/∑ wᵢ² (effective number of sites)
3. Compare strategies on:
   - Global performance
   - Fairness (worst-site performance)
   - Weight distribution
   - Calibration

**Output**: `docs/FAIR_WEIGHTS_H_CAMELYON17_VALIDATION.md`

### Phase 3: Integration with Full FairWeightsHEngine

1. Replace simplified FAIR-WEIGHTS-H implementation with full `FairWeightsHEngine`
2. Use `InstitutionWeightSignals` API for quality/volume/fairness signals
3. Validate entropy regularization and fairness constraints

---

## Conclusion

**All four weighting strategies successfully executed the federated pipeline end-to-end on synthetic data.** The smoke tests validate that:

- Data loading works correctly
- Local training completes without errors
- Aggregation handles different weighting schemes
- Checkpointing and logging function properly
- No numerical instabilities (NaN/Inf) detected

**What this proves:**

- The federated pipeline can run end-to-end with all four weighting strategies

**What this does NOT prove:**

- FAIR-WEIGHTS-H improves accuracy, calibration, fairness, or site robustness on real pathology data
- Performance on real Camelyon17 data
- Convergence behavior over many rounds
- Fairness properties on real heterogeneous sites

**Next step**: Run real Camelyon17 smoke test (Phase 1) before full validation.

---

## Appendix: Smoke Test Commands

```bash
# Equal weighting
python scripts/federated/run_camelyon17_smoke.py --weighting equal --rounds 5

# Volume weighting
python scripts/federated/run_camelyon17_smoke.py --weighting volume --rounds 5

# Prestige weighting
python scripts/federated/run_camelyon17_smoke.py --weighting prestige --rounds 5

# FAIR-WEIGHTS-H
python scripts/federated/run_camelyon17_smoke.py --weighting fair_weights_h --rounds 5
```

## Appendix: Output Files

- `results/camelyon17_smoke/smoke_equal.json`
- `results/camelyon17_smoke/smoke_volume.json`
- `results/camelyon17_smoke/smoke_prestige.json`
- `results/camelyon17_smoke/smoke_fair_weights_h.json`
- `results/camelyon17_smoke/checkpoints_*/model_v*.pt`
