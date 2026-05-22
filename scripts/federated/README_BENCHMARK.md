# PCam Federated Benchmark

## Overview

This benchmark extends the smoke tests (5 rounds) to comprehensive validation (30 rounds) to answer the key question:

> **Does FAIR-WEIGHTS-H produce meaningfully different or more stable behavior than equal, volume, and prestige weighting on real pathology patches?**

## Experimental Design

### Configuration
- **Strategies:** 4 (equal, volume, prestige, fair_weights_h)
- **Seeds:** 3 (42, 43, 44)
- **Rounds:** 30
- **Sites:** 5 simulated federated sites
- **Total runs:** 12 (4 strategies × 3 seeds)

### Dataset
- **Source:** Real PCam pathology patches (Camelyon16-derived)
- **Size:** 5,000 training samples
- **Distribution:** 1,000 samples per site (balanced)

### Metrics Tracked
- **Performance:** Global accuracy, site-wise accuracy
- **Weight dynamics:** Weight entropy, N_eff (effective institution count)
- **Convergence:** Training loss, validation accuracy over rounds
- **Stability:** Variance across seeds

## Execution

### Prerequisites
```bash
# Ensure PCam data is available
ls data/pcam_real/train/images.npy
ls data/pcam_real/train/labels.npy

# Ensure environment is activated
conda activate pathology  # or your environment name
```

### Run Benchmark

**Linux/Mac:**
```bash
bash scripts/federated/run_pcam_benchmark.sh
```

**Windows:**
```cmd
scripts\federated\run_pcam_benchmark.bat
```

**Manual execution (single run):**
```bash
python scripts/federated/run_pcam_federated_smoke.py \
    --weighting equal \
    --rounds 30 \
    --num-sites 5 \
    --seed 42 \
    --output-dir results/pcam_federated_benchmark/equal_seed42
```

### Expected Runtime
- **Per run:** 20-30 minutes
- **Total (12 runs):** 4-6 hours
- **Recommendation:** Run overnight or in background

### Monitoring Progress
```bash
# Watch logs in real-time
tail -f results/pcam_federated_benchmark/equal_seed42/smoke_equal.json

# Check completion status
ls -lh results/pcam_federated_benchmark/*/smoke_*.json
```

## Output Structure

```
results/pcam_federated_benchmark/
├── equal_seed42/
│   ├── smoke_equal.json          # Metrics and results
│   └── checkpoints_equal/        # Model checkpoints
│       ├── model_v1.pt
│       ├── model_v2.pt
│       └── ...
├── equal_seed43/
│   └── ...
├── equal_seed44/
│   └── ...
├── volume_seed42/
│   └── ...
├── prestige_seed42/
│   └── ...
├── fair_weights_h_seed42/
│   └── ...
└── ...
```

## Analysis

After all runs complete, generate the comparison report:

```bash
python scripts/federated/analyze_pcam_benchmark.py \
    --input-dir results/pcam_federated_benchmark \
    --output docs/validation/pcam-benchmark-report.md
```

The analysis script will:
1. Load all 12 result files
2. Compute mean ± std across seeds for each strategy
3. Generate convergence curves
4. Plot weight dynamics
5. Perform statistical significance tests
6. Create comprehensive markdown report

## Expected Outcomes

### Hypothesis 1: Convergence
All strategies should converge to similar global accuracy (~0.85-0.90) since sites are balanced.

### Hypothesis 2: Weight Dynamics
- **Equal:** Constant uniform weights (entropy ≈ 1.61)
- **Volume:** Constant uniform weights (equal site sizes)
- **Prestige:** Dynamic weights favoring better-performing sites
- **FAIR-WEIGHTS-H:** Balanced weights with fairness adjustments

### Hypothesis 3: Stability
FAIR-WEIGHTS-H should show low variance across seeds, indicating stable behavior.

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size in the script
# Edit scripts/federated/run_pcam_federated_smoke.py
# Change: batch_size = 32 → batch_size = 16
```

### Slow Execution
```bash
# Use GPU if available
export CUDA_VISIBLE_DEVICES=0

# Or reduce dataset size
# Edit script to use max_samples=2500 instead of 5000
```

### Failed Run
```bash
# Check logs
cat results/pcam_federated_benchmark/equal_seed42/smoke_equal.json

# Re-run specific combination
python scripts/federated/run_pcam_federated_smoke.py \
    --weighting equal \
    --rounds 30 \
    --seed 42 \
    --output-dir results/pcam_federated_benchmark/equal_seed42
```

## Next Steps

After benchmark completion:
1. **Generate report:** Run analysis script
2. **Review results:** Check `docs/validation/pcam-benchmark-report.md`
3. **Interpret findings:** Does FAIR-WEIGHTS-H show different behavior?
4. **Document limitations:** Note that sites are simulated, not real hospitals
5. **Plan next validation:** Move to real Camelyon17 multi-center data

## Important Notes

### What This Benchmark IS
- ✅ Extended training on real PCam pathology patches
- ✅ Controlled comparison of 4 weighting strategies
- ✅ Measurement of convergence and weight dynamics
- ✅ Validation of FAIR-WEIGHTS-H behavior over 30 rounds

### What This Benchmark IS NOT
- ❌ Real multi-center Camelyon17 validation
- ❌ True hospital-level domain shift evaluation
- ❌ Slide-level WSI aggregation
- ❌ Clinical validation

## References

- Smoke test report: `docs/FAIR_WEIGHTS_H_PCAM_FEDERATED_SMOKE_REPORT.md`
- Benchmark plan: `docs/validation/PCAM_FEDERATED_BENCHMARK_PLAN.md`
- Implementation: `src/features/federated/pathology_fl/weighting/fair_weights_h.py`
