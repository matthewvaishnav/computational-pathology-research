# PCam Federated Benchmark Plan

**Status:** 📋 Planned  
**Previous Step:** PCam federated smoke tests (5 rounds) — ✅ Complete  
**Next Step:** Full PCam federated benchmark (20-50 rounds)

## Objective

Run a comprehensive federated learning benchmark on real PCam pathology patches to compare institutional weighting strategies under controlled conditions with extended training.

## Motivation

The smoke tests (5 rounds) validated that:
- All strategies execute end-to-end without crashes
- Real pathology patches load and train correctly
- Weights are computed and logged properly
- No numerical instabilities (NaN/Inf)

However, 5 rounds is insufficient to:
- Observe convergence behavior
- Measure final performance differences
- Evaluate weight dynamics over time
- Assess calibration quality

The benchmark extends training to 20-50 rounds to capture these dynamics.

## Scope

### What This Benchmark IS
- ✅ Extended training on real PCam pathology patches
- ✅ Controlled comparison of 4 weighting strategies
- ✅ Measurement of convergence, performance, and weight dynamics
- ✅ Validation of FAIR-WEIGHTS-H behavior over extended training

### What This Benchmark IS NOT
- ❌ Real multi-center Camelyon17 validation
- ❌ True hospital-level domain shift evaluation
- ❌ Slide-level WSI aggregation
- ❌ Clinical validation

## Experimental Design

### Dataset
- **Source:** Real PCam pathology patches (Camelyon16-derived)
- **Size:** 5,000 training samples (for benchmark speed)
- **Sites:** 5 simulated federated sites (1,000 samples each)
- **Distribution:** Balanced positive rates across sites

### Strategies to Compare
1. **Equal** — Uniform weighting (baseline)
2. **Volume** — Dataset-size-based weighting
3. **Prestige** — Accuracy-based weighting
4. **FAIR-WEIGHTS-H** — Fairness-aware weighting (default settings)

### Training Configuration
- **Rounds:** 20-50 (to be determined based on convergence)
- **Local epochs per round:** 1
- **Batch size:** 32
- **Learning rate:** 0.01
- **Model:** Simple CNN (same as smoke tests)
- **Seeds:** 3 independent runs for statistical reliability

### Metrics to Track

#### Performance Metrics
- **Global AUC** — Overall model performance
- **Site-wise AUC** — Per-site performance (5 sites)
- **Worst-site AUC** — Minimum site performance (fairness proxy)
- **Global accuracy** — Overall classification accuracy
- **Site-wise accuracy** — Per-site accuracy

#### Calibration Metrics
- **ECE (Expected Calibration Error)** — Global calibration quality
- **Site-wise ECE** — Per-site calibration quality

#### Weight Dynamics
- **Weight entropy** — Distribution uniformity (H = -Σ w_i log w_i)
- **N_eff (Effective institution count)** — exp(H)
- **Weight trajectories** — How weights evolve over rounds
- **Weight variance** — Stability of weight assignments

#### Convergence Metrics
- **Training loss** — Per-round training loss
- **Validation loss** — Per-round validation loss
- **Rounds to convergence** — When performance plateaus

## Implementation Plan

### Phase 1: Script Enhancement
Extend `scripts/federated/run_pcam_federated_smoke.py` to support:
- Configurable number of rounds (--rounds 20-50)
- Multiple seeds (--seed 42, 43, 44)
- Extended metrics logging (AUC, ECE, weight dynamics)
- Validation set evaluation
- Convergence detection

### Phase 2: Execution
Run benchmark for each strategy × seed combination:
```bash
# Equal weighting
python scripts/federated/run_pcam_federated_smoke.py --weighting equal --rounds 30 --seed 42
python scripts/federated/run_pcam_federated_smoke.py --weighting equal --rounds 30 --seed 43
python scripts/federated/run_pcam_federated_smoke.py --weighting equal --rounds 30 --seed 44

# Volume weighting
python scripts/federated/run_pcam_federated_smoke.py --weighting volume --rounds 30 --seed 42
python scripts/federated/run_pcam_federated_smoke.py --weighting volume --rounds 30 --seed 43
python scripts/federated/run_pcam_federated_smoke.py --weighting volume --rounds 30 --seed 44

# Prestige weighting
python scripts/federated/run_pcam_federated_smoke.py --weighting prestige --rounds 30 --seed 42
python scripts/federated/run_pcam_federated_smoke.py --weighting prestige --rounds 30 --seed 43
python scripts/federated/run_pcam_federated_smoke.py --weighting prestige --rounds 30 --seed 44

# FAIR-WEIGHTS-H
python scripts/federated/run_pcam_federated_smoke.py --weighting fair_weights_h --rounds 30 --seed 42
python scripts/federated/run_pcam_federated_smoke.py --weighting fair_weights_h --rounds 30 --seed 43
python scripts/federated/run_pcam_federated_smoke.py --weighting fair_weights_h --rounds 30 --seed 44
```

Total: 12 runs (4 strategies × 3 seeds)

### Phase 3: Analysis
Generate comparison report with:
- Performance comparison tables (mean ± std across seeds)
- Weight dynamics plots (entropy, N_eff over rounds)
- Convergence curves (loss, accuracy, AUC over rounds)
- Calibration comparison (ECE across strategies)
- Statistical significance tests (paired t-tests)

### Phase 4: Documentation
Create `docs/validation/pcam-federated-benchmark-results.md` with:
- Executive summary
- Detailed results tables
- Visualizations
- Interpretation
- Limitations
- Next steps

## Expected Outcomes

### Hypothesis 1: Convergence
All strategies should converge to similar global AUC (~0.85-0.90 for PCam) since sites are balanced.

### Hypothesis 2: Weight Dynamics
- **Equal:** Constant uniform weights (entropy = log(5) ≈ 1.61)
- **Volume:** Constant uniform weights (sites have equal size)
- **Prestige:** Dynamic weights favoring better-performing sites
- **FAIR-WEIGHTS-H:** Balanced weights with slight adjustments for fairness

### Hypothesis 3: Calibration
FAIR-WEIGHTS-H may show better calibration if fairness-aware weighting reduces overconfidence.

### Hypothesis 4: Worst-Site Performance
FAIR-WEIGHTS-H should maintain competitive worst-site AUC compared to baselines.

## Success Criteria

The benchmark is successful if:
1. ✅ All 12 runs complete without crashes
2. ✅ Metrics are logged and saved correctly
3. ✅ Results are reproducible across seeds (low variance)
4. ✅ FAIR-WEIGHTS-H shows no performance degradation vs. baselines
5. ✅ Weight dynamics match theoretical expectations
6. ✅ Comprehensive report is generated

## Timeline Estimate

- **Script enhancement:** 2-4 hours
- **Execution:** 4-6 hours (12 runs × ~20-30 min each)
- **Analysis:** 2-3 hours
- **Documentation:** 1-2 hours
- **Total:** 1-2 days

## Risks and Mitigations

### Risk 1: Long Runtime
**Mitigation:** Use smaller dataset (5K samples) and efficient model. Consider GPU if available.

### Risk 2: High Variance Across Seeds
**Mitigation:** Use 3 seeds minimum. If variance is high, add more seeds.

### Risk 3: No Performance Differences
**Mitigation:** This is actually expected for balanced sites. The value is in validating weight dynamics and establishing baseline performance.

### Risk 4: Numerical Instabilities
**Mitigation:** Smoke tests already validated stability. Monitor for NaN/Inf during runs.

## Future Extensions

After this benchmark:
1. **Heterogeneous Sites:** Introduce class imbalance across sites
2. **New Mathematical Modes:** Test log_linear and mirror_descent
3. **Hyperparameter Sensitivity:** Vary beta, eta, temperature
4. **Real Camelyon17:** Move to true multi-center WSI validation

## References

- Smoke test report: `docs/FAIR_WEIGHTS_H_PCAM_FEDERATED_SMOKE_REPORT.md`
- Implementation: `src/features/federated/pathology_fl/weighting/fair_weights_h.py`
- Test script: `scripts/federated/run_pcam_federated_smoke.py`

## Validation Ladder Position

```
✅ 1. Synthetic Camelyon17-like smoke
✅ 2. PCam federated smoke (equal)
✅ 3. PCam federated smoke (all strategies)
📋 4. PCam federated benchmark ← YOU ARE HERE
⏭️ 5. Real Camelyon17 subset smoke
⏭️ 6. Real Camelyon17 full validation
```

This benchmark represents the transition from "does it run?" to "how does it perform?"
