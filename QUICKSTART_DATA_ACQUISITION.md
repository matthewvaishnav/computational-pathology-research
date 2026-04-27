# Quick Start: Data Acquisition for Multi-Disease Training

This guide will help you acquire the datasets needed to complete the Medical AI Revolution platform in **4-6 weeks**.

---

## 📋 Prerequisites

1. **Python Environment**
   ```bash
   pip install -r requirements.txt
   pip install openai requests tqdm pillow
   ```

2. **API Keys** (for vision-language)
   - OpenAI API key for GPT-4V caption generation
   - Set environment variable: `export OPENAI_API_KEY=your_key_here`

3. **Kaggle CLI** (optional, for PANDA dataset)
   ```bash
   pip install kaggle
   # Set up credentials: https://github.com/Kaggle/kaggle-api#api-credentials
   ```

---

## 🚀 Week 1: Download Public Datasets

### Step 1: Run Automated Download Script

```bash
# Download all automatically available datasets
python scripts/download_public_datasets.py --data-root data

# Or download specific datasets
python scripts/download_public_datasets.py \
  --data-root data \
  --datasets lc25000 nct_crc crc_val
```

This will automatically download:
- ✅ **LC25000** (25K lung cancer images)
- ✅ **NCT-CRC-HE-100K** (100K colon cancer patches)
- ✅ **CRC-VAL-HE-7K** (7K colon validation images)

**Time**: 2-4 hours (depending on internet speed)

### Step 2: Manual Downloads

Some datasets require manual registration:

#### HAM10000 (Melanoma)
1. Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
2. Download all files
3. Extract to: `data/multi_disease/melanoma/ham10000/`

**Alternative (Kaggle)**:
```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d data/multi_disease/melanoma/ham10000/
```

#### PANDA (Prostate Cancer)
1. Visit: https://www.kaggle.com/c/prostate-cancer-grade-assessment
2. Accept competition rules
3. Download via Kaggle CLI:
```bash
kaggle competitions download -c prostate-cancer-grade-assessment -p data/multi_disease/prostate/panda
```

#### SICAPv2 (Prostate Cancer)
1. Visit: https://data.mendeley.com/datasets/9xxm58dvs3/1
2. Download dataset
3. Extract to: `data/multi_disease/prostate/sicap/`

**Time**: 1-2 days (including download time)

### Step 3: Verify Downloads

```bash
python scripts/verify_datasets.py --data-root data
```

This will show:
- ✓ Which datasets are complete
- ⚠ Which datasets are partial
- ✗ Which datasets are missing
- 📊 Summary statistics

---

## 🔬 Week 2-3: Generate Vision-Language Pairs

### Option A: Use GPT-4V (Recommended)

Generate captions for your downloaded images:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_key_here

# Generate captions for all multi-disease images
python scripts/generate_captions_gpt4v.py \
  --image-dir data/multi_disease \
  --output-dir data/vision_language/generated \
  --max-images 10000 \
  --batch-delay 1.0
```

**Cost Estimate**:
- 10K images: ~$200-300
- 50K images: ~$1,000-1,500
- 100K images: ~$2,000-3,000

**Time**: 3-6 hours for 10K images (with rate limiting)

### Option B: Download Pre-captioned Datasets

#### OpenPath Dataset
```bash
git clone https://github.com/mahmoodlab/OpenPath.git
cd OpenPath
python download_dataset.py --output ../data/vision_language/openpath
```

#### PubMed Central (PMC)
```bash
# Download pathology subset from PMC Open Access
python scripts/download_pmc_pathology.py \
  --output-dir data/vision_language/pmc_pathology \
  --keywords "histopathology,pathology,microscopy" \
  --min-images 50000
```

**Note**: PMC download script needs to be created based on your specific needs.

### Verify Vision-Language Data

```bash
python scripts/verify_datasets.py --data-root data
```

Look for the "Vision-Language Datasets" section in the output.

---

## 🎯 Week 3-4: Start Multi-Disease Training

Once you have at least 3 complete datasets, you can start training:

```bash
# Train multi-disease model
python experiments/train_multi_disease.py \
  --config configs/multi_disease.yaml \
  --data-root data \
  --output-dir checkpoints/multi_disease \
  --epochs 50 \
  --batch-size 64
```

**Expected Results**:
- Training time: 2-3 days on RTX 4070
- Expected accuracy: 85-92% per disease
- Cross-disease transfer: 5-10% improvement over single-disease

---

## 🌐 Week 4-6: Vision-Language Training

Once you have 10K+ image-text pairs:

```bash
# Train vision-language model
python experiments/train_vision_language.py \
  --config configs/vision_language.yaml \
  --data-root data/vision_language \
  --output-dir checkpoints/vision_language \
  --epochs 100 \
  --batch-size 32
```

**Expected Results**:
- Training time: 3-5 days on RTX 4070
- Zero-shot capabilities: 60-75% accuracy on unseen diseases
- Natural language explanations: Qualitative improvement

---

## 📊 Dataset Summary

### Automatically Downloaded (Week 1)
| Dataset | Disease | Images | Size | Time |
|---------|---------|--------|------|------|
| LC25000 | Lung | 25K | ~2GB | 30min |
| NCT-CRC | Colon | 100K | ~8GB | 2hrs |
| CRC-VAL | Colon | 7K | ~500MB | 15min |

### Manual Download (Week 1)
| Dataset | Disease | Images | Size | Time |
|---------|---------|--------|------|------|
| HAM10000 | Melanoma | 10K | ~1GB | 1hr |
| PANDA | Prostate | 11K WSI | ~100GB | 1-2 days |
| SICAPv2 | Prostate | 19K | ~2GB | 1hr |

### Already Have
| Dataset | Disease | Images | Size | Status |
|---------|---------|--------|------|--------|
| PCam | Breast | 327K | ~7GB | ✅ Downloaded & Training |

### Vision-Language (Week 2-3)
| Source | Method | Pairs | Cost | Time |
|--------|--------|-------|------|------|
| GPT-4V | Generate | 10K | $200-300 | 3-6hrs |
| OpenPath | Download | 200K | Free | 1-2 days |
| PMC | Download | 500K | Free | 2-3 days |

---

## 💰 Budget Breakdown

### Minimal Budget (~$300)
- GPT-4V captions: $200-300 for 10K images
- Cloud storage: Free (use local)
- Compute: Free (use RTX 4070)
- **Total**: ~$300

### Recommended Budget (~$1,500)
- GPT-4V captions: $1,000-1,500 for 50K images
- Cloud storage: Free (use local)
- Compute: Free (use RTX 4070)
- **Total**: ~$1,500

### Full Scale (~$5,000)
- GPT-4V captions: $2,000-3,000 for 100K images
- Cloud compute: $1,000-2,000 (if needed)
- Commercial datasets: $0-5,000 (optional)
- **Total**: ~$5,000

---

## 🎯 Success Milestones

### Week 1 ✅
- [ ] Downloaded 3+ public datasets
- [ ] Verified dataset integrity
- [ ] Organized data directory structure

### Week 2-3 ✅
- [ ] Generated 10K+ image-text pairs
- [ ] Created vision-language manifest
- [ ] Verified caption quality

### Week 3-4 ✅
- [ ] Started multi-disease training
- [ ] Achieved 80%+ accuracy on 3+ diseases
- [ ] Validated cross-disease transfer

### Week 4-6 ✅
- [ ] Started vision-language training
- [ ] Demonstrated zero-shot capabilities
- [ ] Generated natural language explanations

---

## 🆘 Troubleshooting

### Download Issues

**Problem**: Slow download speeds
```bash
# Use aria2c for faster parallel downloads
sudo apt-get install aria2
aria2c -x 16 -s 16 <url>
```

**Problem**: Kaggle API not working
```bash
# Check credentials
cat ~/.kaggle/kaggle.json

# Re-authenticate
kaggle config set -n username -v <your_username>
kaggle config set -n key -v <your_key>
```

### GPT-4V Issues

**Problem**: Rate limit errors
```bash
# Increase batch delay
python scripts/generate_captions_gpt4v.py \
  --batch-delay 2.0  # Slower but more reliable
```

**Problem**: High costs
```bash
# Start with smaller batch
python scripts/generate_captions_gpt4v.py \
  --max-images 1000  # Test with 1K images first
```

### Training Issues

**Problem**: Out of memory
```bash
# Reduce batch size
python experiments/train_multi_disease.py \
  --batch-size 32  # Instead of 64
```

**Problem**: Slow training
```bash
# Enable mixed precision
python experiments/train_multi_disease.py \
  --mixed-precision \
  --num-workers 4
```

---

## 📞 Getting Help

### Community Resources
- GitHub Issues: https://github.com/matthewvaishnav/computational-pathology-research/issues
- Discussions: https://github.com/matthewvaishnav/computational-pathology-research/discussions

### Academic Collaborations
- Email pathology departments at universities
- Join r/computationalpathology
- Attend MICCAI, ISBI conferences

### Commercial Support
- Contact digital pathology vendors
- Reach out to PathAI, Paige.AI
- Join Digital Pathology Association

---

## ✅ Next Steps

1. **This Week**: Run automated download script
2. **Next Week**: Complete manual downloads
3. **Week 3**: Generate captions with GPT-4V
4. **Week 4**: Start multi-disease training
5. **Week 5-6**: Start vision-language training

**You can have a working multi-disease model with vision-language capabilities in 4-6 weeks!**

---

*Last Updated: April 27, 2026*
*Status: Ready for execution*
