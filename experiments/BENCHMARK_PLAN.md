# Competitor Benchmark Plan

## Objective
Run PathML, CLAM, and other frameworks on identical hardware (RTX 4070) with identical dataset (PCam) for fair comparison.

## Current Status
- ✅ Repository PCam model: recorded 0.9394 test ROC AUC on the official 32,768-patch test split
- ℹ️ Historical literature table: 0.9394 is numerically higher than all 10 collected external PCam AUC values, but those values are not a matched benchmark
- ⏳ Direct baseline PyTorch rerun: implemented, not yet executed under the matched GPU protocol
- ⏳ PathML: not yet benchmarked under the matched protocol
- ⏳ CLAM: not yet benchmarked under the matched protocol

## Requirements

### 1. PathML Setup
```bash
# Install PathML
pip install pathml

# Adapt PCam data loader
# PathML expects specific data format - need custom adapter

# Run benchmark
python experiments/benchmark_competitors.py --framework pathml
```

**Challenges**:
- PathML primarily designed for WSI, not patch-level
- May require data format conversion
- Windows compatibility unknown

### 2. CLAM Setup
```bash
# Clone CLAM repository
git clone https://github.com/mahmoodlab/CLAM.git external/CLAM

# Install dependencies
cd external/CLAM
pip install -r requirements.txt

# Convert PCam to CLAM format
# CLAM expects bag-of-patches format

# Run benchmark
python experiments/benchmark_competitors.py --framework clam
```

**Challenges**:
- CLAM designed for MIL on WSI
- Requires converting PCam patches to "bags"
- Primarily Linux-focused (Windows support unclear)

### 3. Baseline PyTorch
```bash
# Already implemented
python experiments/benchmark_competitors.py --framework baseline
```

**Status**: Ready to run (needs GPU access)

## Fair Comparison Criteria

All benchmarks must use:
1. **Same hardware**: RTX 4070 (8GB)
2. **Same dataset**: PCam (262K train, 32K val, 32K test)
3. **Same data splits**: Identical train/val/test sets
4. **Same metrics**: AUC, accuracy, training time
5. **Same hyperparameters** (where applicable):
   - Learning rate: 0.001
   - Optimizer: AdamW
   - Scheduler: Cosine annealing
   - Batch size: Largest that fits in memory

## Timeline

### Phase 1: Repository result + direct baseline
- [x] Repository PCam result recorded
- [ ] Direct baseline PyTorch benchmark under the matched GPU protocol

### Phase 2: PathML (Est. 1-2 days)
- [ ] Install PathML
- [ ] Create PCam data adapter
- [ ] Run training benchmark
- [ ] Collect metrics

### Phase 3: CLAM (Est. 2-3 days)
- [ ] Clone CLAM repository
- [ ] Convert PCam to CLAM format
- [ ] Adapt training script
- [ ] Run benchmark
- [ ] Collect metrics

### Phase 4: Documentation (Est. 1 day)
- [ ] Update PERFORMANCE_COMPARISON.md with real numbers
- [ ] Add methodology section
- [ ] Include reproducibility instructions
- [ ] Publish results

## Comparison target

The historical numerical lead is already documented. This plan exists to test
the **stronger experimental claim** under matched conditions.

No speed, cost, or accuracy advantage should be assumed in advance. The direct
benchmark should report whatever the controlled reruns show.

## Reproducibility

All benchmarks will be:
1. **Scripted**: Automated via `benchmark_competitors.py`
2. **Versioned**: Exact package versions recorded
3. **Documented**: Full methodology in docs
4. **Open**: Code and configs published

## Notes

- Historical external numbers may be retained as descriptive context, with their provenance limitations stated
- Direct matched benchmarks will determine whether experimental superiority is supported
- The repository result of 0.9394 AUC remains valid regardless of the later comparative outcome
- Statistical inference will be applied only to measurements generated under a defensible matched design

## Running Benchmarks

```bash
# Run all benchmarks (requires GPU)
python experiments/benchmark_competitors.py --framework all

# Run specific framework
python experiments/benchmark_competitors.py --framework pathml
python experiments/benchmark_competitors.py --framework clam
python experiments/benchmark_competitors.py --framework baseline

# Skip long-running benchmarks
python experiments/benchmark_competitors.py --skip-long
```

## Contact

For questions or to contribute benchmark results:
- Open an issue: https://github.com/matthewvaishnav/histocore/issues
- Email: [your email]
