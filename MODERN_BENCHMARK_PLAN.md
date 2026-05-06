# Modern Benchmark Plan for HistoCore

## Why PCam (2016) Isn't Enough

**Current problem**: HistoCore's 93.94% AUC is on PatchCamelyon (2016):
- 8-year-old dataset
- 96×96 pixel patches (tiny)
- Single task (metastasis detection)
- Not representative of modern clinical challenges

**What reviewers/hospitals will say**:
- "This is outdated"
- "Show me results on current benchmarks"
- "How does it compare to foundation models?"

---

## Modern Benchmarks (2024-2025)

### 1. **Patho-Bench** (Mahmood Lab, Feb 2025)
**Status**: ⭐ **BEST CHOICE** - Most comprehensive

**What it is**:
- 95 tasks across 33 public datasets
- Canonical train-test splits (reproducible)
- Includes: linear probing, survival prediction, retrieval
- HuggingFace integration (easy download)
- Automated evaluation pipeline

**Tasks include**:
- Cancer detection (multiple organs)
- Biomarker prediction (ER, PR, HER2, KRAS, STK11, ALK)
- Survival analysis
- Mutation prediction

**Why it matters**:
- Industry standard (Mahmood Lab = Harvard/BWH)
- All major foundation models benchmarked here
- Direct comparison to UNI, Virchow, Prov-GigaPath, H-optimus-0

**How to use**:
```python
from patho_bench.SplitFactory import SplitFactory
from patho_bench.ExperimentFactory import ExperimentFactory

# Download task from HuggingFace
split, config = SplitFactory.from_hf(
    './splits', 
    'cptac_ccrcc', 
    'BAP1_mutation'
)

# Run HistoCore model
experiment = ExperimentFactory.linprobe(
    split=split,
    task_config=config,
    pooled_embeddings_dir='./histocore_features',
    model_name='histocore'
)
experiment.train()
experiment.test()
```

**Effort**: 2-3 weeks
**Impact**: ⭐⭐⭐⭐⭐ (Directly comparable to all major models)

---

### 2. **Nature Communications Clinical Benchmark** (April 2025)
**Status**: ⭐⭐⭐⭐ **CLINICAL VALIDATION**

**What it is**:
- 22 clinical tasks from 3 hospitals (MSKCC, Mount Sinai, SUH)
- Real clinical data (not curated research datasets)
- Detection tasks: breast, oral, bladder, kidney, colorectal, DCIS, IBD
- Biomarker tasks: ER, PR, HER2, HRD, BRAF, NRAS, KRAS, STK11, ALK
- ICI therapy response prediction

**Why it matters**:
- **Real clinical data** (not research datasets)
- Multi-institutional (generalization proof)
- Published in Nature Communications (high credibility)
- Automated external benchmarking available

**How to use**:
1. Fill out Microsoft Form
2. Upload Docker container with HistoCore
3. They run benchmark and return results
4. Optional: Add to public leaderboard

**Effort**: 1-2 weeks (mostly Docker packaging)
**Impact**: ⭐⭐⭐⭐⭐ (Clinical validation, Nature publication)

---

### 3. **TCGA Pan-Cancer** (26,565 cases)
**Status**: ⭐⭐⭐ **LARGE SCALE**

**What it is**:
- 26,565 digitized cases from TCGA
- 33 cancer types
- Survival prediction, subtype classification
- Publicly available

**Why it matters**:
- Largest public pathology dataset
- Multi-cancer generalization
- Standard benchmark for foundation models

**Effort**: 3-4 weeks
**Impact**: ⭐⭐⭐⭐ (Scale demonstration)

---

## Recommended Strategy

### Phase 1: Quick Win (1-2 weeks)
**Submit to Nature Communications Clinical Benchmark**

**Why first**:
- Automated (they run it for you)
- Clinical data (addresses "real-world" criticism)
- Nature publication (credibility)
- Public leaderboard (visibility)

**Steps**:
1. Package HistoCore in Docker
2. Write inference script (template provided)
3. Submit via Microsoft Form
4. Get results in ~1 week

**Expected results**:
- Compare directly to UNI, Virchow, Prov-GigaPath, H-optimus-0
- If HistoCore is competitive → instant credibility
- If HistoCore underperforms → identify gaps

---

### Phase 2: Comprehensive Benchmark (2-3 weeks)
**Run Patho-Bench locally**

**Why second**:
- 95 tasks (comprehensive)
- Reproducible (canonical splits)
- Direct comparison to all models
- Can iterate and improve

**Steps**:
1. Install Patho-Bench: `pip install -e .`
2. Extract HistoCore features with Trident
3. Run linear probing on all 95 tasks
4. Generate comparison report

**Expected results**:
- Detailed performance across 95 tasks
- Identify strengths/weaknesses
- Publishable benchmark paper

---

### Phase 3: Foundation Model Comparison (1 week)
**Compare HistoCore to foundation models**

**Benchmark against**:
- UNI (ViT-L, 307M params, 0.773 biomarker AUC)
- Virchow2 (ViT-H, 632M params, 0.765 biomarker AUC)
- Prov-GigaPath (ViT-G, 1.1B params, 0.780 biomarker AUC)
- H-optimus-0 (ViT-G, 1.1B params, 0.785 biomarker AUC)

**Key questions**:
- Does HistoCore's 8-12x training speedup sacrifice accuracy?
- How does federated learning affect performance?
- Is HistoCore competitive with billion-parameter models?

---

## Implementation Plan

### Week 1: Docker Packaging
```bash
# Create Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install HistoCore
COPY . /histocore
RUN pip install -e /histocore

# Inference script
COPY inference.py /inference.py
ENTRYPOINT ["python", "/inference.py"]
```

### Week 2: Nature Benchmark Submission
1. Test Docker locally
2. Submit via Microsoft Form
3. Wait for results

### Week 3-4: Patho-Bench
```bash
# Install
git clone https://github.com/mahmoodlab/Patho-Bench.git
cd Patho-Bench
pip install -r requirements.txt
pip install -e .

# Extract features
python extract_features.py \
  --model histocore \
  --data-dir ./data \
  --output-dir ./features

# Run benchmark
python run_benchmark.py \
  --model histocore \
  --features-dir ./features \
  --output-dir ./results
```

### Week 5: Analysis & Paper
1. Compare results to foundation models
2. Identify strengths/weaknesses
3. Write benchmark paper
4. Submit to MICCAI/CVPR

---

## Expected Outcomes

### Best Case (HistoCore is competitive)
- **Detection tasks**: 0.95-0.98 AUC (on par with foundation models)
- **Biomarker tasks**: 0.75-0.78 AUC (competitive with UNI/Prov-GigaPath)
- **Result**: "HistoCore achieves foundation model performance with 8-12x faster training"
- **Impact**: Instant credibility, paper acceptance, hospital interest

### Realistic Case (HistoCore is good but not best)
- **Detection tasks**: 0.93-0.96 AUC (slightly below top models)
- **Biomarker tasks**: 0.70-0.75 AUC (below UNI/H-optimus-0)
- **Result**: "HistoCore offers best speed/accuracy tradeoff"
- **Impact**: Niche positioning (fast training for resource-constrained settings)

### Worst Case (HistoCore underperforms)
- **Detection tasks**: <0.90 AUC
- **Biomarker tasks**: <0.65 AUC
- **Result**: Identify gaps (model size? training data? architecture?)
- **Impact**: Roadmap for improvement, still publishable as analysis

---

## Why This Solves the Adoption Problem

### Before (PCam 2016)
- ❌ "8-year-old benchmark"
- ❌ "Not clinically relevant"
- ❌ "Can't compare to foundation models"
- ❌ "No proof it generalizes"

### After (Modern Benchmarks)
- ✅ "Benchmarked on 2025 clinical data"
- ✅ "Validated on 22 clinical tasks from 3 hospitals"
- ✅ "Competitive with UNI/Virchow/Prov-GigaPath"
- ✅ "95 tasks across 33 datasets"
- ✅ "Published in Nature Communications benchmark"

---

## Cost/Benefit Analysis

### Time Investment
- Week 1-2: Nature benchmark (low effort, high impact)
- Week 3-4: Patho-Bench (medium effort, high impact)
- Week 5: Analysis (low effort, medium impact)
- **Total**: 5 weeks

### Compute Requirements
- Nature benchmark: Free (they run it)
- Patho-Bench: 1-2 GPUs for 2-3 weeks
- **Cost**: $500-1000 (cloud GPU) or free (if you have GPUs)

### Impact
- **Credibility**: 10x increase (modern benchmarks vs PCam)
- **Comparability**: Direct comparison to all major models
- **Publishability**: MICCAI/CVPR paper ready
- **Hospital interest**: "Show me your Nature benchmark results"

---

## Next Steps (This Week)

1. **Read Patho-Bench paper**: https://arxiv.org/pdf/2502.06750
2. **Clone Patho-Bench repo**: `git clone https://github.com/mahmoodlab/Patho-Bench.git`
3. **Test installation**: `pip install -e .`
4. **Download one task**: Test HistoCore on single task
5. **Package Docker**: Prepare for Nature benchmark submission

---

## The Bottom Line

**PCam (2016) was good for initial validation.**

**Modern benchmarks (2024-2025) are required for:**
- Academic publication (MICCAI/CVPR/Nature)
- Hospital adoption (clinical validation)
- Comparison to foundation models (UNI, Virchow, etc.)

**5 weeks of work = instant credibility.**

**Without this, HistoCore stays a "research project with outdated benchmarks."**

**With this, HistoCore becomes "validated on 2025 clinical data, competitive with billion-parameter foundation models."**
