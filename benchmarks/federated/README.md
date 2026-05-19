# Federated Learning Benchmarks

**Goal:** Prove that DMI with expertise weighting outperforms standard FedAvg

## Hypothesis

**Expertise-weighted aggregation improves performance over uniform averaging, especially for:**
1. Rare subtype detection
2. Noisy-label scenarios
3. Data heterogeneity across institutions

## Experimental Setup

### Simulated Institutions

We simulate 5 institutions with different characteristics:

```python
institutions = {
    "mayo_clinic": {
        "expertise_tier": "comprehensive_cancer_center",
        "expertise_weight": 8.5,
        "data_quality": "high",
        "label_noise": 0.02,  # 2% label noise
        "case_mix": "diverse",  # All subtypes
        "samples": 1000
    },
    "johns_hopkins": {
        "expertise_tier": "academic_medical_center",
        "expertise_weight": 6.2,
        "data_quality": "high",
        "label_noise": 0.03,
        "case_mix": "diverse",
        "samples": 800
    },
    "regional_hospital": {
        "expertise_tier": "regional",
        "expertise_weight": 3.0,
        "data_quality": "medium",
        "label_noise": 0.08,
        "case_mix": "common_cases",  # Mostly common subtypes
        "samples": 600
    },
    "community_hospital_1": {
        "expertise_tier": "community",
        "expertise_weight": 1.0,
        "data_quality": "medium",
        "label_noise": 0.12,
        "case_mix": "common_cases",
        "samples": 400
    },
    "community_hospital_2": {
        "expertise_tier": "community",
        "expertise_weight": 1.0,
        "data_quality": "low",
        "label_noise": 0.15,
        "case_mix": "common_cases",
        "samples": 300
    }
}
```

### Scenarios

#### Scenario 1: Balanced Data
- All institutions have similar case distributions
- Tests basic aggregation performance

#### Scenario 2: Rare Subtype Detection
- Only Mayo Clinic and Johns Hopkins have rare subtypes
- Community hospitals have mostly common cases
- Tests if expertise weighting helps rare class performance

#### Scenario 3: Noisy Labels
- Community hospitals have 15% label noise
- Academic centers have <5% label noise
- Tests robustness to noisy participants

#### Scenario 4: Data Heterogeneity
- Different scanners, staining protocols per institution
- Tests domain adaptation capabilities

## Baselines

### 1. Centralized Training (Upper Bound)
```python
# Train on all data pooled together
model = train_centralized(all_data)
```

### 2. Local-Only Training (Lower Bound)
```python
# Each institution trains independently
models = {inst: train_local(inst_data) for inst in institutions}
```

### 3. Standard FedAvg (Baseline)
```python
# Uniform averaging (standard federated learning)
global_model = average([model_1, model_2, model_3, model_4, model_5])
```

### 4. DMI with Expertise Weighting (Our Approach)
```python
# Weighted by institutional expertise
weights = [8.5, 6.2, 3.0, 1.0, 1.0]  # Normalized
global_model = weighted_average(models, weights)
```

### 5. DMI + Specialization Matching
```python
# Route cases to specialist institutions
if case.is_rare_subtype():
    route_to(["mayo_clinic", "johns_hopkins"])
else:
    route_to(all_institutions)
```

## Metrics

### Overall Performance:
- Accuracy, AUC, F1 score
- Per-class sensitivity/specificity
- Calibration (ECE)

### Rare Class Performance:
- Sensitivity on rare subtypes
- Precision on rare subtypes
- Confusion matrix

### Convergence:
- Communication rounds to convergence
- Training time
- Communication cost (MB transferred)

### Per-Institution:
- Local accuracy/AUC
- Improvement over local-only training
- Contribution to global model

## Expected Results

### Hypothesis 1: DMI beats FedAvg overall
```
Centralized:  0.95 AUC (upper bound)
DMI:          0.93 AUC ✅ Target
FedAvg:       0.90 AUC
Local-only:   0.85 AUC (lower bound)
```

### Hypothesis 2: DMI excels on rare subtypes
```
Rare subtype sensitivity:
DMI:          0.88 ✅ Target
FedAvg:       0.75
Local-only:   0.60
```

### Hypothesis 3: DMI robust to noisy labels
```
With 15% label noise at community hospitals:
DMI:          0.91 AUC ✅ (minimal degradation)
FedAvg:       0.85 AUC (significant degradation)
```

## Running Experiments

### Quick Test (Synthetic Data):
```bash
# Run all baselines on synthetic data (5 min)
python benchmarks/federated/run_all.py --dataset synthetic --rounds 10

# View results
python benchmarks/federated/generate_report.py
```

### Full Experiment (Real Data):
```bash
# Scenario 1: Balanced data
python benchmarks/federated/dmi_expertise.py \
  --dataset pcam \
  --num-institutions 5 \
  --rounds 100 \
  --scenario balanced

# Scenario 2: Rare subtype detection
python benchmarks/federated/dmi_expertise.py \
  --dataset panda \
  --num-institutions 5 \
  --rounds 100 \
  --scenario rare_subtypes

# Scenario 3: Noisy labels
python benchmarks/federated/dmi_expertise.py \
  --dataset pcam \
  --num-institutions 5 \
  --rounds 100 \
  --scenario noisy_labels \
  --noise-rate 0.15

# Compare all approaches
python benchmarks/federated/compare_all.py --dataset pcam
```

## Visualization

### Plots Generated:
1. **Convergence curves** - Accuracy vs communication rounds
2. **Per-institution performance** - Local vs global model
3. **Rare class sensitivity** - DMI vs FedAvg vs Local
4. **Confusion matrices** - Per approach
5. **Weight distribution** - Expertise weights over time

### Example Output:
```
benchmarks/federated/results/
├── convergence_curves.png
├── per_institution_performance.png
├── rare_class_sensitivity.png
├── confusion_matrices.png
├── weight_distribution.png
└── summary_table.csv
```

## Ablation Studies

### Expertise Weight Sensitivity:
```python
# Test different weight scales
weight_scales = [
    [1.0, 1.0, 1.0, 1.0, 1.0],  # Uniform (FedAvg)
    [2.0, 1.5, 1.2, 1.0, 1.0],  # Mild weighting
    [4.0, 3.0, 2.0, 1.0, 1.0],  # Moderate weighting
    [8.5, 6.2, 3.0, 1.0, 1.0],  # Strong weighting (our default)
]
```

### Aggregation Strategies:
- Uniform averaging (FedAvg)
- Expertise weighting (DMI)
- Expertise + slide quality weighting
- Expertise + attention-aware aggregation

## Current Status

| Experiment | Status | Results |
|------------|--------|---------|
| Synthetic data test | 📋 Planned | - |
| PCam balanced | 📋 Planned | - |
| PCam rare subtypes | 📋 Planned | - |
| PCam noisy labels | 📋 Planned | - |
| PANDA balanced | 📋 Planned | - |
| Real multi-institution | 📋 Future | Requires hospital partnerships |

## Limitations

### Current:
- Simulated institutions on single machine
- No real network latency or failures
- Synthetic data heterogeneity
- No real privacy constraints

### Future Work:
- Real multi-institution deployment
- Network simulation (latency, bandwidth)
- Byzantine attack scenarios
- Privacy budget tracking

## Citation

```bibtex
@article{vaishnav2026dmi,
  title={Expertise-Weighted Federated Learning for Medical AI},
  author={Vaishnav, Matthew},
  journal={In preparation},
  year={2026}
}
```

---

**Last Updated:** May 19, 2026  
**Status:** Experimental design complete, implementation in progress
