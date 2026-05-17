# PANDA Dataset Setup and Training Guide

## Overview
PANDA (Prostate cANcer graDe Assessment) is a Kaggle competition dataset for automated Gleason grading of prostate biopsies using whole-slide images (WSIs).

**Task**: Predict ISUP grade (0-5) from prostate biopsy WSIs
**Metric**: Quadratic Weighted Kappa
**Dataset Size**: ~100GB (10,616 slides)

---

## Step 1: Download Dataset from Kaggle

### Prerequisites
1. ✅ Kaggle API installed (`pip install kaggle`)
2. ✅ Kaggle credentials configured (`~/.kaggle/kaggle.json`)
3. ⚠️ **REQUIRED**: Accept competition rules

### Accept Competition Rules (REQUIRED)
**You must do this manually before downloading:**

1. Go to: https://www.kaggle.com/c/prostate-cancer-grade-assessment
2. Click "Join Competition" or "Late Submission"
3. Accept the competition rules

### Download Command
Once rules are accepted, run:

```bash
# Download everything (~100GB, takes 1-3 hours depending on connection)
python scripts/download_panda.py --output_dir data/panda

# Or download just CSVs first to test (< 1MB)
python scripts/download_panda.py --output_dir data/panda --no_images
```

**Downloaded files:**
- `train.csv` - Slide metadata (slide_id, isup_grade, gleason_score, data_provider)
- `train_images/` - 10,616 TIFF whole-slide images
- `test.csv` - Test set metadata (optional)

---

## Step 2: Extract Features from WSIs

PANDA uses Multiple Instance Learning (MIL) which requires:
1. Extract patches from WSIs
2. Encode patches with foundation model (CONCH, Phikon, or UNI)
3. Aggregate patch features for slide-level prediction

### Option A: Use Pre-trained Foundation Models

**CONCH (Recommended for prostate)**
```bash
python scripts/extract_panda_features.py \
    --data_dir data/panda \
    --output_dir data/panda/features_conch \
    --model conch \
    --batch_size 64 \
    --num_workers 4
```

**Phikon (Fast, good performance)**
```bash
python scripts/extract_panda_features.py \
    --data_dir data/panda \
    --output_dir data/panda/features_phikon \
    --model phikon \
    --batch_size 128 \
    --num_workers 4
```

**UNI (Highest quality, slower)**
```bash
python scripts/extract_panda_features.py \
    --data_dir data/panda \
    --output_dir data/panda/features_uni \
    --model uni \
    --batch_size 32 \
    --num_workers 4
```

**Feature extraction time**: ~6-12 hours for full dataset (depends on GPU)

**Output**: HDF5 files in `data/panda/features_<model>/`
- Each file: `{slide_id}.h5` containing:
  - `features`: [num_patches, feature_dim] patch embeddings
  - `coordinates`: [num_patches, 2] patch locations

---

## Step 3: Create Slide Index

The slide index organizes metadata and creates train/val/test splits:

```bash
python -c "
from pathlib import Path
from src.data.panda_dataset import PANDASlideIndex

index = PANDASlideIndex.from_csv(
    csv_path='data/panda/train.csv',
    image_dir='data/panda/train_images',
    split_ratios=(0.7, 0.15, 0.15),
    stratify=True,
    seed=42
)
index.save('data/panda/slide_index.json')
print(f'Created index with {len(index)} slides')
print(f'Grade distribution: {index.get_grade_distribution()}')
"
```

---

## Step 4: Train MIL Model

### Basic Training
```bash
python experiments/train_panda.py \
    --data_dir data/panda \
    --features_dir data/panda/features_phikon \
    --index_path data/panda/slide_index.json \
    --model nnmil \
    --input_dim 256 \
    --hidden_dim 256 \
    --epochs 40 \
    --batch_size 32 \
    --lr 5e-4 \
    --checkpoint_dir checkpoints/panda \
    --log_dir logs/panda
```

### Training with Ordinal Regression (Recommended)
Ordinal regression is better for ordered grades (0 < 1 < 2 < ... < 5):

```bash
python experiments/train_panda.py \
    --data_dir data/panda \
    --features_dir data/panda/features_phikon \
    --index_path data/panda/slide_index.json \
    --model nnmil \
    --input_dim 256 \
    --hidden_dim 256 \
    --ordinal \
    --epochs 40 \
    --batch_size 32 \
    --lr 5e-4 \
    --checkpoint_dir checkpoints/panda_ordinal \
    --log_dir logs/panda_ordinal
```

### Using Config Files
```bash
# Phikon config
python experiments/train_panda.py --config configs/panda_phikon.yaml

# CONCH config
python experiments/train_panda.py --config configs/panda_conch.yaml

# UNI config
python experiments/train_panda.py --config configs/panda_uni.yaml
```

**Training time**: ~2-4 hours for 40 epochs (depends on GPU and batch size)

---

## Step 5: Evaluate Model

```bash
python experiments/evaluate_panda.py \
    --checkpoint checkpoints/panda/best_model.pth \
    --data_dir data/panda \
    --features_dir data/panda/features_phikon \
    --index_path data/panda/slide_index.json \
    --output_dir results/panda
```

**Evaluation metrics:**
- Quadratic Weighted Kappa (primary metric)
- Accuracy
- Per-grade accuracy
- Confusion matrix

---

## Expected Performance

| Model | Feature Extractor | Kappa | Accuracy | Notes |
|-------|------------------|-------|----------|-------|
| nnMIL | Phikon | 0.85-0.88 | 75-78% | Fast, good baseline |
| nnMIL | CONCH | 0.87-0.90 | 77-80% | Best for prostate |
| nnMIL | UNI | 0.88-0.91 | 78-81% | Highest quality |
| CLAM | Phikon | 0.86-0.89 | 76-79% | Attention-based |
| TransMIL | UNI | 0.89-0.92 | 79-82% | State-of-the-art |

**Note**: Performance varies based on:
- Train/val split
- Hyperparameters
- Data augmentation
- Ordinal vs standard cross-entropy

---

## Troubleshooting

### Download Issues
- **401 Unauthorized**: Accept competition rules at https://www.kaggle.com/c/prostate-cancer-grade-assessment
- **403 Forbidden**: Check Kaggle API credentials in `~/.kaggle/kaggle.json`
- **Slow download**: Use `--no_images` first to test, then download images

### Feature Extraction Issues
- **CUDA out of memory**: Reduce `--batch_size` (try 32 or 16)
- **Missing slides**: Some slides may be corrupted, script will skip them
- **Slow extraction**: Use multiple workers (`--num_workers 4-8`)

### Training Issues
- **No feature files found**: Run feature extraction first (Step 2)
- **CUDA out of memory**: Reduce batch size or use gradient accumulation
- **Low kappa score**: Try ordinal regression, increase epochs, or use better features

---

## Quick Start (Minimal Setup)

If you want to test the pipeline quickly:

```bash
# 1. Download just CSVs (< 1MB)
python scripts/download_panda.py --output_dir data/panda --no_images

# 2. Create demo dataset with synthetic data
python scripts/create_panda_demo.py --output_dir data/panda_demo --num_slides 100

# 3. Train on demo data
python experiments/train_panda.py \
    --data_dir data/panda_demo \
    --features_dir data/panda_demo/features \
    --index_path data/panda_demo/slide_index.json \
    --epochs 10 \
    --batch_size 16
```

---

## PathologyFL Integration

PANDA training automatically uses PathologyFL for:
- Expertise-weighted aggregation (if using federated learning)
- Hospital-specific model adaptation
- Privacy-preserving multi-institutional training

To enable federated learning:
```bash
python experiments/train_panda_federated.py \
    --config configs/panda_federated.yaml \
    --num_hospitals 5 \
    --rounds 50
```

---

## GPU Requirements

- **Minimum**: 8GB VRAM (RTX 2070, RTX 3060)
- **Recommended**: 12GB+ VRAM (RTX 3060 Ti, RTX 3080, RTX 4070)
- **Optimal**: 24GB+ VRAM (RTX 3090, RTX 4090, A5000)

**Your system**: RTX 3060 (12GB) ✅ - Perfect for PANDA training!

---

## Next Steps

1. ✅ Accept Kaggle competition rules
2. ⏳ Download PANDA dataset (~100GB)
3. ⏳ Extract features with foundation model (~6-12 hours)
4. ⏳ Train MIL model (~2-4 hours)
5. ⏳ Evaluate and analyze results

**Current Status**: Waiting for competition rules acceptance to start download
