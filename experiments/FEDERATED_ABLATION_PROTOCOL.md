# Federated Ablation Study Protocol
**Goal:** Prove PathologyFL + DMI > PathologyFL > FedAvg on realistic multi-center data

**Status:** Ready to execute  
**Timeline:** 2-3 weeks  
**Hardware:** RTX 4070 (sufficient for federated simulation)

---

## Experimental Design

### Dataset: Camelyon17 (Multi-Center Lymph Node Metastasis)
- 5 hospital sites with natural distribution heterogeneity
- Whole slide images with metastasis labels
- Real-world multi-center characteristics

### Simulated Federation Setup

#### 5 Hospitals with Realistic Heterogeneity:

```python
hospitals = {
    "hospital_0": {  # Large academic center
        "samples": 200,
        "quality": {"sharpness": 0.9, "stain": 0.85, "artifacts": 0.1},
        "label_noise": 0.02,
        "hospital_type": "CANCER_CENTER",
        "specialties": ["breast", "lung"],
        "annual_cases": 10000,
        "diagnostic_accuracy": 0.95,
        "years_experience": 15
    },
    "hospital_1": {  # Medium teaching hospital
        "samples": 150,
        "quality": {"sharpness": 0.85, "stain": 0.80, "artifacts": 0.15},
        "label_noise": 0.05,
        "hospital_type": "TEACHING_HOSPITAL",
        "specialties": ["general"],
        "annual_cases": 5000,
        "diagnostic_accuracy": 0.90,
        "years_experience": 10
    },
    "hospital_2": {  # Community hospital (good quality)
        "samples": 100,
        "quality": {"sharpness": 0.80, "stain": 0.75, "artifacts": 0.20},
        "label_noise": 0.08,
        "hospital_type": "COMMUNITY_HOSPITAL",
        "specialties": ["general"],
        "annual_cases": 3000,
        "diagnostic_accuracy": 0.88,
        "years_experience": 8
    },
    "hospital_3": {  # Small community (lower quality)
        "samples": 50,
        "quality": {"sharpness": 0.70, "stain": 0.65, "artifacts": 0.30},
        "label_noise": 0.12,
        "hospital_type": "COMMUNITY_HOSPITAL",
        "specialties": ["general"],
        "annual_cases": 1500,
        "diagnostic_accuracy": 0.82,
        "years_experience": 5
    },
    "hospital_4": {  # Rural hospital (challenging)
        "samples": 30,
        "quality": {"sharpness": 0.65, "stain": 0.60, "artifacts": 0.35},
        "label_noise": 0.15,
        "hospital_type": "RURAL_HOSPITAL",
        "specialties": ["general"],
        "annual_cases": 800,
        "diagnostic_accuracy": 0.78,
        "years_experience": 3
    }
}
```

**Key heterogeneity factors:**
- **Volume imbalance**: 6.7x difference (200 vs 30 samples)
- **Quality variation**: Sharpness 0.65-0.90, artifacts 0.10-0.35
- **Label noise**: 2%-15% across sites
- **Expertise variation**: Cancer center vs rural hospital

---

## Methods to Compare

### 1. FedAvg (Baseline)
```python
# Standard federated averaging - uniform weights
global_model = average([model_0, model_1, model_2, model_3, model_4])
```

### 2. FedAvg + Quality Weighting
```python
# Only slide quality, no pathology hierarchy or expertise
quality_weights = [calculate_quality(site) for site in hospitals]
global_model = weighted_average(models, quality_weights)
```

### 3. PathologyFL (No DMI)
```python
# Full pathology-aware aggregation, uniform institutional weights
# - Cancer-type specific strategies
# - Slide quality weighting
# - Attention-aware aggregation
# - Hierarchical patch→slide→case→hospital→global
# BUT: All hospitals weighted equally (no expertise layer)
pathology_weights = calculate_pathology_weights(cancer_type, slide_quality)
global_model = weighted_average(models, pathology_weights)
```

### 4. PathologyFL + DMI (Full System)
```python
# Two-layer system:
# Layer 1: PathologyFL (domain knowledge)
# Layer 2: DMI (institutional expertise)
pathology_weights = calculate_pathology_weights(cancer_type, slide_quality)
expertise_weights = calculate_expertise_weights(hospital_metadata)
combined_weights = alpha * expertise_weights + beta * pathology_weights
global_model = weighted_average(models, combined_weights)
```

### 5. Oracle Weighting (Upper Bound)
```python
# Weights set proportional to true per-site validation accuracy
# Not deployable (requires knowing ground truth), but shows ceiling
oracle_weights = [site_val_accuracy for site in hospitals]
global_model = weighted_average(models, oracle_weights)
```

### 6. FedAvg + DMI-style Weighting (Ablation)
```python
# Isolate effect of expertise weighting without PathologyFL
# Uses DMI weights but standard FedAvg aggregation (no pathology hierarchy)
expertise_weights = calculate_expertise_weights(hospital_metadata)
global_model = weighted_average(models, expertise_weights)
```

---

## Metrics & Evaluation

### Primary Metrics:
1. **Global AUC** - Primary metric for medical AI
2. **Global Accuracy** - Overall correctness
3. **Calibration (ECE)** - Expected calibration error

### Fairness & Robustness:
4. **Per-site AUC** - Performance at each hospital
5. **Worst-site AUC** - Minimum across all sites
6. **AUC std dev** - Variance across sites

### Efficiency:
7. **Convergence rounds** - Rounds to reach 95% of final AUC
8. **Communication cost** - Total MB transferred
9. **Training time** - Wall-clock hours

### Statistical Rigor:
- **Bootstrap confidence intervals** (1000 samples, 95% CI)
- **DeLong test** for AUC comparisons
- **Multiple seeds** (3-5 random seeds per method)

---

## Experimental Protocol

### Training Configuration:
```yaml
model: AttentionMIL  # Or TransnnMIL v2
num_rounds: 100
local_epochs: 5
batch_size: 16
learning_rate: 1e-4
optimizer: Adam

# PathologyFL + DMI hyperparameters
alpha: 0.5  # Expertise weight
beta: 0.3   # Quality weight
cancer_type: "general"  # Camelyon17 is lymph node metastasis
```

### Ablation Studies:

#### A. DMI Factor Ablation (Turn on/off individually):
```python
ablations = [
    {"hospital_type": True, "specialization": False, "volume": False, "accuracy": False, "experience": False},
    {"hospital_type": True, "specialization": True, "volume": False, "accuracy": False, "experience": False},
    {"hospital_type": True, "specialization": True, "volume": True, "accuracy": False, "experience": False},
    {"hospital_type": True, "specialization": True, "volume": True, "accuracy": True, "experience": False},
    {"hospital_type": True, "specialization": True, "volume": True, "accuracy": True, "experience": True},  # Full
]
```

#### B. Hyperparameter Sensitivity:
```python
# Sweep alpha and beta
alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0]
beta_values = [0.0, 0.2, 0.3, 0.5, 0.7]
# Constraint: alpha + beta <= 1.0
```

#### C. Heterogeneity Stress Tests:
1. **Extreme volume imbalance**: 20x difference (400 vs 20 samples)
2. **High label noise**: 30% noise at 2 sites
3. **Severe quality degradation**: Artifacts 0.5+ at 2 sites
4. **New site joins mid-training**: Add hospital_5 at round 50

---

## Expected Results

### Hypothesis 1: PathologyFL beats FedAvg
```
Method                    Global AUC    Worst-site AUC    Convergence
FedAvg                    0.88          0.82              80 rounds
FedAvg + Quality          0.89          0.84              75 rounds
PathologyFL (no DMI)      0.91 ✅       0.86 ✅           65 rounds ✅
```

**Why:** Pathology-aware hierarchical aggregation + slide quality weighting handles heterogeneity better than uniform averaging.

### Hypothesis 2: DMI adds value on top of PathologyFL
```
Method                    Global AUC    Worst-site AUC    Convergence
PathologyFL (no DMI)      0.91          0.86              65 rounds
PathologyFL + DMI         0.93 ✅       0.88 ✅           60 rounds ✅
Oracle                    0.94          0.90              55 rounds
```

**Why:** Institutional expertise weighting (hospital type, specialization, volume, accuracy) further improves aggregation, especially for rare cases and noisy sites.

### Hypothesis 3: DMI factors contribute incrementally
```
DMI Factors Enabled                      Global AUC
Hospital type only                       0.915
+ Specialization                         0.920
+ Volume                                 0.925
+ Accuracy                               0.928
+ Experience (full)                      0.930 ✅
```

**Why:** Each expertise factor captures a different aspect of institutional capability.

---

## Visualization & Reporting

### Key Plots:

1. **Convergence curves** - AUC vs communication rounds
   - All 6 methods on same plot
   - Shaded regions for 95% CI

2. **Per-site performance** - Bar chart of AUC per hospital
   - Grouped by method
   - Shows fairness across sites

3. **Ablation heatmap** - DMI factors vs Global AUC
   - Shows incremental contribution

4. **Hyperparameter sensitivity** - 2D heatmap (alpha, beta) → AUC
   - Identifies stable regions

5. **Robustness analysis** - Performance under stress tests
   - Volume imbalance, label noise, quality degradation

### Statistical Tables:

```markdown
| Method              | Global AUC | 95% CI        | p-value vs FedAvg | Worst-site AUC | Convergence |
|---------------------|------------|---------------|-------------------|----------------|-------------|
| FedAvg              | 0.880      | [0.872-0.888] | -                 | 0.820          | 80 rounds   |
| FedAvg + Quality    | 0.890      | [0.883-0.897] | 0.023             | 0.840          | 75 rounds   |
| PathologyFL         | 0.910      | [0.904-0.916] | <0.001            | 0.860          | 65 rounds   |
| PathologyFL + DMI   | 0.930      | [0.925-0.935] | <0.001            | 0.880          | 60 rounds   |
| Oracle              | 0.940      | [0.935-0.945] | <0.001            | 0.900          | 55 rounds   |
```

---

## Implementation Plan

### Week 1: Setup & Baseline
- [ ] Prepare Camelyon17 federated splits (5 hospitals)
- [ ] Implement FedAvg baseline
- [ ] Implement FedAvg + Quality weighting
- [ ] Run initial experiments (3 seeds each)

### Week 2: PathologyFL & DMI
- [ ] Implement PathologyFL (no DMI)
- [ ] Implement PathologyFL + DMI (full system)
- [ ] Implement Oracle weighting
- [ ] Run main comparison (5 seeds each)

### Week 3: Ablations & Analysis
- [ ] DMI factor ablation (5 configurations)
- [ ] Hyperparameter sweep (alpha, beta)
- [ ] Stress tests (volume, noise, quality)
- [ ] Generate all plots and tables
- [ ] Write results report

---

## Success Criteria

### Minimum for "Validated":
- ✅ PathologyFL beats FedAvg by ≥2% AUC (p < 0.05)
- ✅ PathologyFL + DMI beats PathologyFL by ≥1% AUC (p < 0.05)
- ✅ Results hold across 3+ random seeds
- ✅ Improvement visible on worst-site AUC (fairness)

### Stretch for "Genius-Level":
- ✅ PathologyFL + DMI within 1-2% of Oracle (near-optimal)
- ✅ Consistent gains across multiple stress tests
- ✅ Clear ablation showing each DMI factor contributes
- ✅ Replicate trend on PANDA when training completes

---

## Compute Budget

**Hardware:** RTX 4070 (12GB VRAM)

**Estimated time per method:**
- Training: ~4 hours × 100 rounds × 5 hospitals = ~2000 GPU-hours
- With parallelization: ~8-12 wall-clock hours per method
- Total for 6 methods × 3 seeds: ~150-200 hours (~1 week continuous)

**Optimization:**
- Run multiple seeds in parallel (if multi-GPU available)
- Cache feature extraction (reuse across methods)
- Early stopping if convergence clear

---

## Deliverables

1. **Experimental report**: `results/federated_ablation/REPORT.md`
2. **All plots**: `results/federated_ablation/figures/`
3. **Statistical tables**: `results/federated_ablation/tables/`
4. **Reproducible configs**: `experiments/configs/federated_ablation/`
5. **Checkpoints**: `checkpoints/federated_ablation/`

---

## Next Steps After Validation

If results confirm hypotheses:

1. **Paper submission**: "PathologyFL + DMI: Hierarchical Expertise-Weighted Federated Learning for Computational Pathology"
   - Target: IEEE TMI, Nature Biomedical Engineering, or MICCAI

2. **Cross-dataset validation**: Replicate on PANDA (prostate cancer)

3. **Real multi-institution pilot**: 2-3 hospitals with IRB approval

4. **Theoretical framework**: Formalize when/why expertise weighting helps

---

**Status:** Ready to execute  
**Owner:** Matthew  
**Timeline:** 2-3 weeks  
**Priority:** CRITICAL - This is the missing piece for "genius-level" validation
