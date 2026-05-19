# Benchmark Suite

**Goal:** Rigorous empirical validation of DMI and TransnnMIL v2 architecture

## Quick Start

```bash
# Run all benchmarks (requires datasets)
python benchmarks/run_all.py

# Run specific dataset
python benchmarks/run_pcam.py
python benchmarks/run_panda.py

# Generate consolidated report
python benchmarks/generate_report.py
```

## Benchmark Structure

```
benchmarks/
├── pcam/                      # PatchCamelyon benchmarks
│   ├── baseline_cnn.py        # Simple CNN baseline
│   ├── standard_mil.py        # AttentionMIL baseline
│   ├── transnnmil_v1.py       # TransnnMIL v1
│   ├── transnnmil_v2_2branch.py  # v2 without topology
│   ├── transnnmil_v2_3branch.py  # v2 full (all branches)
│   ├── results.json           # Consolidated results
│   └── README.md              # PCam-specific notes
├── panda/                     # PANDA benchmarks
│   ├── [same structure]
│   └── results.json
├── camelyon17/                # Camelyon17 benchmarks
│   ├── [same structure]
│   └── results.json
├── federated/                 # Federated learning benchmarks
│   ├── centralized.py         # Upper bound (centralized training)
│   ├── local_only.py          # Lower bound (no collaboration)
│   ├── fedavg.py              # Standard FedAvg baseline
│   ├── dmi_expertise.py       # DMI with expertise weighting
│   ├── results.json
│   └── README.md
├── run_all.py                 # Run all benchmarks
├── generate_report.py         # Generate consolidated report
├── summary.md                 # Human-readable summary
└── README.md                  # This file
```

## Metrics Tracked

### Classification Metrics:
- **Accuracy** - Overall correctness
- **AUC** - Area under ROC curve
- **Sensitivity** - True positive rate
- **Specificity** - True negative rate
- **Precision** - Positive predictive value
- **F1 Score** - Harmonic mean of precision/recall
- **Calibration (ECE)** - Expected calibration error

### Performance Metrics:
- **Training time** - Total training duration
- **Inference time** - Per-sample prediction time
- **Throughput** - Samples per second
- **Memory usage** - Peak GPU/CPU memory
- **Model size** - Number of parameters

### Federated Learning Metrics:
- **Communication rounds** - Rounds to convergence
- **Communication cost** - Total data transferred
- **Per-institution performance** - Local accuracy/AUC
- **Rare class sensitivity** - Performance on rare subtypes

## Ablation Studies

### TransnnMIL v2 Ablations:
1. **1-branch** (TransMIL only)
2. **2-branch** (TransMIL + Hierarchical)
3. **3-branch** (TransMIL + Hierarchical + Topology)
4. **With/without adaptive pruning**
5. **Different GNN architectures** (GATv2, GraphSAGE, GIN)

### DMI Ablations:
1. **Centralized** (upper bound)
2. **Local-only** (lower bound)
3. **FedAvg** (uniform weighting)
4. **DMI** (expertise weighting)
5. **DMI + specialization matching**
6. **DMI + slide quality weighting**

## Reproducibility

### Requirements:
```bash
pip install -r requirements.txt
```

### Datasets:
- **PCam:** Download from [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection)
- **PANDA:** Download from [Kaggle](https://www.kaggle.com/c/prostate-cancer-grade-assessment)
- **Camelyon17:** Download from [CodaLab](https://camelyon17.grand-challenge.org/)

### Exact Commands:
```bash
# PCam full dataset
python benchmarks/pcam/baseline_cnn.py --data-root data/pcam --epochs 20
python benchmarks/pcam/transnnmil_v2_3branch.py --data-root data/pcam --epochs 20

# PANDA
python benchmarks/panda/transnnmil_v2_3branch.py \
  --data-root panda/features_resnet50_300patches \
  --splits-file panda/splits.json \
  --epochs 50

# Federated learning
python benchmarks/federated/dmi_expertise.py \
  --num-institutions 5 \
  --rounds 100 \
  --dataset pcam
```

## Current Status

| Dataset | Status | Baseline | TransnnMIL v1 | TransnnMIL v2 | Notes |
|---------|--------|----------|---------------|---------------|-------|
| PCam (synthetic) | ✅ Complete | - | - | 0.94 acc, 1.0 AUC | Synthetic data, 700 samples |
| PCam (real) | 📋 Planned | - | - | - | Full 327K dataset |
| PANDA | 🚧 In Progress | - | - | Training... | 1,365 slides |
| Camelyon17 | 📋 Planned | - | - | - | - |
| DMI vs FedAvg | 📋 Planned | - | - | - | Multi-institution simulation |

## Comparison to Published Baselines

### PCam (Literature):
- **ResNet-50:** 0.89 AUC
- **DenseNet-121:** 0.92 AUC
- **EfficientNet-B0:** 0.96 AUC
- **Our target:** Match or exceed 0.92 AUC

### PANDA (Challenge):
- **1st place:** 0.943 kappa
- **10th place:** 0.89 kappa
- **Our target:** Top 10 performance (>0.89 kappa)

## Caveats & Limitations

### Current Limitations:
- PCam results on synthetic subset, not comparable to published baselines
- PANDA training in progress, results pending
- No real multi-institution data yet (using simulated institutions)
- Federated learning experiments on single machine (simulated distribution)

### Future Work:
- Real multi-institution pilot with 2-3 hospitals
- Prospective clinical validation
- Comparison to commercial systems
- Rare subtype detection benchmarks

## Citation

If you use these benchmarks, please cite:

```bibtex
@software{vaishnav2026benchmarks,
  title={Computational Pathology Benchmarks},
  author={Vaishnav, Matthew},
  year={2026},
  url={https://github.com/matthewvaishnav/computational-pathology-research}
}
```

---

**Last Updated:** May 19, 2026  
**Status:** Benchmark suite under development
