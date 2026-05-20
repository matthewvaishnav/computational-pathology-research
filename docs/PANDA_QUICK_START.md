# PANDA Quick Start - Ready to Go!

## ✅ Environment Status
- **GPU**: RTX 3060 (12GB VRAM) - Ready
- **CUDA**: 12.1 - Working
- **PyTorch**: 2.5.1+cu121 - Working
- **PathologyFL**: Loaded and functional
- **All dependencies**: Installed

## 🎯 Current Status
- ⚠️ **Waiting for**: Kaggle competition rules acceptance
- ⚠️ **Dataset**: Not downloaded yet
- ✅ **Scripts**: All ready to go
- ✅ **Pipeline**: Fully prepared

---

## 🚀 Step-by-Step Instructions

### Step 1: Accept Competition Rules (DO THIS NOW)
**This is the ONLY manual step you need to do:**

1. Open browser: https://www.kaggle.com/c/prostate-cancer-grade-assessment
2. Click "Join Competition" or "Late Submission"
3. Check the box and accept the rules
4. Done! (takes 30 seconds)

### Step 2: Download Dataset
Once you've accepted the rules, run:

```bash
python scripts/download_panda.py --output_dir data/panda
```

**Download details:**
- Size: ~100GB
- Time: 1-3 hours (depends on internet speed)
- Files: 10,616 TIFF whole-slide images + CSV metadata

**What's downloading:**
- `train.csv` - Metadata (slide IDs, ISUP grades, Gleason scores)
- `train_images/` - 10,616 prostate biopsy WSIs
- `test.csv` - Test set metadata (optional)

### Step 3: Extract Features
After download completes, extract patch features:

```bash
# Using ResNet50 (fast, ~4-6 hours)
python scripts/extract_panda_features.py \
    --data_dir data/panda \
    --output_dir data/panda/features_resnet50 \
    --model resnet50 \
    --batch_size 64 \
    --num_workers 4
```

**What this does:**
- Extracts 224x224 patches from each slide
- Encodes patches with ResNet50
- Saves features to HDF5 files
- Time: ~4-6 hours for full dataset

### Step 4: Train Model
Once features are extracted, start training:

```bash
python experiments/train_panda.py \
    --data_dir data/panda \
    --features_dir data/panda/features_resnet50 \
    --index_path data/panda/slide_index.json \
    --model nnmil \
    --input_dim 2048 \
    --hidden_dim 256 \
    --ordinal \
    --epochs 40 \
    --batch_size 32 \
    --lr 5e-4 \
    --checkpoint_dir checkpoints/panda \
    --log_dir logs/panda
```

**Training details:**
- Time: ~2-4 hours for 40 epochs
- Metric: Quadratic Weighted Kappa
- Expected Kappa: 0.82-0.86 with ResNet50
- Checkpoints saved every 5 epochs

### Step 5: Check Progress
Monitor training status:

```bash
# Check dataset status
python check_panda_status.py

# Check GPU usage
nvidia-smi

# View training logs
type logs\panda\train.log
```

---

## 📊 Expected Timeline

| Step | Time | Status |
|------|------|--------|
| Accept rules | 30 seconds | ⏳ **DO THIS NOW** |
| Download dataset | 1-3 hours | ⏳ Waiting |
| Extract features | 4-6 hours | ⏳ Waiting |
| Train model | 2-4 hours | ⏳ Waiting |
| **Total** | **7-13 hours** | |

**Note**: Steps 2-4 run automatically once started. You can leave them running overnight.

---

## 🎓 What You're Building

**Task**: Automated Gleason grading for prostate cancer
- **Input**: Whole-slide images of prostate biopsies
- **Output**: ISUP grade (0-5)
  - Grade 0: Benign (no cancer)
  - Grade 1: Least aggressive
  - Grade 5: Most aggressive
- **Method**: Multiple Instance Learning (MIL)
- **Metric**: Quadratic Weighted Kappa

**Clinical Impact**: Automated grading can:
- Reduce pathologist workload
- Improve consistency
- Enable second opinions
- Scale to underserved areas

---

## 🔧 Troubleshooting

### Download fails with 401 Unauthorized
**Solution**: Accept competition rules (Step 1)

### Download is slow
**Options**:
1. Wait it out (1-3 hours is normal)
2. Download overnight
3. Use faster internet connection

### CUDA out of memory during feature extraction
**Solution**: Reduce batch size
```bash
python scripts/extract_panda_features.py --batch_size 32  # or 16
```

### CUDA out of memory during training
**Solution**: Reduce batch size
```bash
python experiments/train_panda.py --batch_size 16  # or 8
```

---

## 📈 Performance Expectations

### With ResNet50 Features
- **Kappa**: 0.82-0.86
- **Accuracy**: 72-76%
- **Training time**: 2-4 hours

### With Better Foundation Models (Optional)
If you want higher performance later:

| Model | Kappa | Accuracy | Feature Extraction Time |
|-------|-------|----------|------------------------|
| ResNet50 | 0.82-0.86 | 72-76% | 4-6 hours |
| Phikon | 0.85-0.88 | 75-78% | 6-8 hours |
| CONCH | 0.87-0.90 | 77-80% | 8-10 hours |
| UNI | 0.88-0.91 | 78-81% | 10-12 hours |

---

## 🎯 Quick Commands Reference

```bash
# Check environment
python check_gpu_env.py

# Check dataset status
python check_panda_status.py

# Download dataset
python scripts/download_panda.py --output_dir data/panda

# Extract features
python scripts/extract_panda_features.py \
    --data_dir data/panda \
    --output_dir data/panda/features_resnet50 \
    --model resnet50 \
    --batch_size 64

# Train model
python experiments/train_panda.py \
    --data_dir data/panda \
    --features_dir data/panda/features_resnet50 \
    --epochs 40 \
    --batch_size 32 \
    --ordinal

# Monitor GPU
nvidia-smi

# View logs
type logs\panda\train.log
```

---

## ✅ Ready to Start!

**Your system is fully prepared. The only thing left is:**

1. **Accept Kaggle competition rules** (30 seconds)
2. **Run the download command**
3. **Let it run!**

Everything else is automated and ready to go. Your RTX 3060 with 12GB VRAM is perfect for this task.

---

## 📚 Additional Resources

- **Full guide**: `PANDA_SETUP_GUIDE.md`
- **Competition page**: https://www.kaggle.com/c/prostate-cancer-grade-assessment
- **Paper**: Bulten et al. (2022), Lancet Oncology 23(2), 252-261

---

**Questions? Issues? Check the troubleshooting section or the full setup guide!**
