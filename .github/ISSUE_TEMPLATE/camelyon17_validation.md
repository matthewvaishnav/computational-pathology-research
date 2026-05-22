---
name: Camelyon17 FAIR-WEIGHTS-H Validation
about: Empirical validation of FAIR-WEIGHTS-H on Camelyon17 dataset
title: "Validate FAIR-WEIGHTS-H on Camelyon17 dataset"
labels: ["validation", "fair-weights-h", "camelyon17", "federated-learning"]
assignees: ""
---

## Objective

Empirically validate FAIR-WEIGHTS-H weighting strategy against baseline strategies on the Camelyon17 federated learning benchmark.

## Prerequisites

- [x] Core federated integration tests passing (5/5)
- [x] FAIR-WEIGHTS-H property tests passing (27/27)
- [x] LocalTrainer API refactored and tested
- [ ] Camelyon17 dataset downloaded and preprocessed
- [ ] Smoke experiment configuration ready

## Validation Plan

### Phase 1: Smoke Experiment

**Goal:** Verify the federated training pipeline works end-to-end on Camelyon17

**Configuration:**

- **Rounds:** 5 (quick validation)
- **Sites:** All 5 Camelyon17 sites
- **Model:** Simple CNN or ResNet18
- **Weighting:** Equal weights only
- **Seeds:** Fixed seed for reproducibility

**Success Criteria:**

- Training completes without errors
- Global model converges (loss decreases)
- All sites participate successfully
- Metrics logged correctly

### Phase 2: Weighting Strategy Comparison

**Goal:** Compare FAIR-WEIGHTS-H against baseline strategies

**Strategies to Compare:**

1. **Equal** - Uniform weights (1/N per site)
2. **Volume** - Proportional to dataset size
3. **Prestige** - Inverse error weighting
4. **FAIR-WEIGHTS-H** - Our proposed algorithm

**Configuration:**

- **Rounds:** 20-50 (sufficient for convergence)
- **Sites:** All 5 Camelyon17 sites
- **Model:** Same architecture for all strategies
- **Seeds:** Same random seeds for fair comparison
- **Data splits:** Identical train/val/test splits

**Metrics to Track:**

**Global Performance:**

- Global AUC (primary metric)
- Global accuracy
- Global F1 score

**Site-wise Performance:**

- Per-site AUC
- Per-site accuracy
- Worst-site sensitivity (fairness metric)

**Calibration:**

- Expected Calibration Error (ECE)
- Reliability diagrams

**Weighting Analysis:**

- Weight entropy (diversity of weights)
- Effective number of sites (N_eff)
- Per-round weight evolution
- Weight stability across rounds

**Convergence:**

- Training loss curves
- Validation loss curves
- Rounds to convergence

## Expected Outcomes

**Hypothesis:** FAIR-WEIGHTS-H will:

1. Match or exceed global AUC of baseline strategies
2. Improve worst-site performance (better fairness)
3. Maintain stable, interpretable weights
4. Show higher N_eff than prestige weighting

## Deliverables

- [ ] Smoke experiment results (Phase 1)
- [ ] Full comparison results (Phase 2)
- [ ] Validation report with:
  - Methodology
  - Results tables and figures
  - Statistical significance tests
  - Discussion of findings
- [ ] Updated documentation with empirical results

## Implementation Notes

**Smoke Experiment Script:**

```bash
python scripts/federated/run_camelyon17_smoke.py \
  --rounds 5 \
  --weighting equal \
  --seed 42
```

**Full Comparison Script:**

```bash
python scripts/federated/run_camelyon17_comparison.py \
  --rounds 50 \
  --strategies equal,volume,prestige,fair-weights-h \
  --seeds 42,43,44 \
  --output results/camelyon17_validation/
```

## Validation Report Template

```markdown
# Camelyon17 FAIR-WEIGHTS-H Validation Report

## Executive Summary

[Brief summary of findings]

## Methodology

- Dataset: Camelyon17 (5 sites)
- Model: [architecture]
- Rounds: [N]
- Seeds: [list]

## Results

### Global Performance

| Strategy       | AUC | Accuracy | F1  |
| -------------- | --- | -------- | --- |
| Equal          | ... | ...      | ... |
| Volume         | ... | ...      | ... |
| Prestige       | ... | ...      | ... |
| FAIR-WEIGHTS-H | ... | ...      | ... |

### Fairness Metrics

| Strategy       | Worst-Site AUC | Worst-Site Sensitivity |
| -------------- | -------------- | ---------------------- |
| Equal          | ...            | ...                    |
| Volume         | ...            | ...                    |
| Prestige       | ...            | ...                    |
| FAIR-WEIGHTS-H | ...            | ...                    |

### Weighting Analysis

[Weight evolution plots, N_eff comparison, entropy analysis]

## Discussion

[Interpretation of results, comparison to hypothesis]

## Conclusion

[Summary and recommendations]

## Note

The broader test suite contains optional-dependency and legacy-test failures unrelated to the FAIR-WEIGHTS-H / core federated training path.
```

## Timeline

- **Week 1:** Smoke experiment + debugging
- **Week 2:** Full comparison runs
- **Week 3:** Analysis and report writing

## Related Issues

- #XXX - Clean up optional and legacy federated test failures
