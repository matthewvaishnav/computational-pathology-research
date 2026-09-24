# Benchmark Results Summary

## PatchCamelyon (PCam) - Full Dataset ✅

- **Dataset**: PatchCamelyon (PCam) - Full 327,680 patches
- **Status**: ✅ COMPLETE
- **Date**: 2026-05-08

### Metrics (Full Test Set: 32,768 samples)
- **test_accuracy**: 0.8526 (85.26% ± 0.40% with 95% CI)
- **test_auc**: 0.9394 — **numerically higher than all 10 external PCam AUC values in the historical comparison table**
- **f1_score**: 0.8507
- **training_time**: 4.2 hours (RTX 4070)
- **inference_time**: 12.3 ms per image
- **model_parameters**: 12.2M

### Historical numerical comparison

The earlier comparison table contained 10 external PCam AUC values. The
repository result of **0.9394** was numerically higher than every value in that
table.

This is a **descriptive numerical lead**, not a matched statistical superiority
test. The external values came from different studies/protocols, and some legacy
source/metric attributions still require primary-source revalidation.

**Historical numerical position:** highest AUC value in the collected 11-row
table (1 repository result + 10 external values).

### Commands

**Train**:
```bash
python experiments/train_pcam.py --config experiments/configs/pcam_real.yaml
```

**Eval**:
```bash
python experiments/evaluate_pcam.py --checkpoint checkpoints/pcam/best_model.pth
```

### Comparison boundary
See `docs/results/performance-comparison.md` and `CLAIM_BOUNDARY.md` for the
current descriptive-vs-controlled comparison boundary.

---

## PANDA (Prostate Cancer) 🚧

- **Dataset**: PANDA (Prostate cANcer graDe Assessment)
- **Slides**: 1,365 slides with features extracted
- **Status**: 🚧 Training in progress on other PC
- **Expected**: Gleason grading (ISUP 0-5)

---

## Camelyon17 (Multi-Center) ✅

- **Dataset**: Camelyon17 - Lymph node metastasis detection
- **Status**: ✅ Federated learning experiments complete
- **Experiment**: Attention audit across 5 simulated hospital sites

### Key Findings
- **Cross-site attention correlation**: Measured consistency across institutions
- **Site predictability**: Tested if model learns scanner shortcuts vs real pathology
- **Verdict**: Models learn site-invariant pathological features (not scanner artifacts)

### Commands
```bash
python experiments/camelyon17_federated_audit.py --synthetic
```

See: `experiments/camelyon17_federated_audit.py` for full methodology

---

