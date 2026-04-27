# Data Acquisition Scripts

This directory contains scripts for acquiring datasets needed for multi-disease and vision-language training.

---

## 📋 Available Scripts

### 1. `download_public_datasets.py`

**Purpose**: Automatically download publicly available medical imaging datasets.

**Usage**:
```bash
# Download all datasets
python scripts/download_public_datasets.py --data-root data

# Download specific datasets
python scripts/download_public_datasets.py \
  --data-root data \
  --datasets lc25000 nct_crc crc_val
```

**Datasets Downloaded**:
- ✅ **LC25000**: 25K lung cancer images (GitHub)
- ✅ **NCT-CRC-HE-100K**: 100K colon cancer patches (Zenodo)
- ✅ **CRC-VAL-HE-7K**: 7K colon validation images (Zenodo)
- ⚠ **HAM10000**: 10K melanoma images (requires manual download)
- ⚠ **PANDA**: 11K prostate WSI (requires Kaggle API)
- ⚠ **SICAPv2**: 19K prostate patches (requires manual download)

**Time**: 2-4 hours (automatic downloads only)

---

### 2. `generate_captions_gpt4v.py`

**Purpose**: Generate detailed medical captions for pathology images using GPT-4V.

**Prerequisites**:
```bash
pip install openai pillow
export OPENAI_API_KEY=your_key_here
```

**Usage**:
```bash
# Generate captions for all images in a directory
python scripts/generate_captions_gpt4v.py \
  --image-dir data/multi_disease/lung/lc25000 \
  --output-dir data/vision_language/lc25000_captions \
  --max-images 10000 \
  --batch-delay 1.0
```

**Parameters**:
- `--image-dir`: Directory containing pathology images
- `--output-dir`: Output directory for captions and processed images
- `--api-key`: OpenAI API key (or set OPENAI_API_KEY env var)
- `--model`: Model to use (default: gpt-4o)
- `--max-images`: Maximum number of images to process
- `--skip-existing`: Skip images that already have captions (default: True)
- `--batch-delay`: Delay between API calls in seconds (default: 1.0)

**Cost Estimates**:
- 1K images: ~$20-30
- 10K images: ~$200-300
- 50K images: ~$1,000-1,500
- 100K images: ~$2,000-3,000

**Time**: 3-6 hours for 10K images (with rate limiting)

**Output Structure**:
```
output_dir/
├── images/              # Processed images
├── captions/            # Generated captions (.txt files)
├── generation_results.json  # Statistics and costs
└── vision_language_manifest.json  # Training manifest
```

---

### 3. `verify_datasets.py`

**Purpose**: Verify downloaded datasets and generate statistics.

**Usage**:
```bash
# Verify all datasets
python scripts/verify_datasets.py --data-root data

# Save report to file
python scripts/verify_datasets.py \
  --data-root data \
  --output data/metadata/verification_report.json
```

**Output**:
```
=============================================================
Dataset Verification Report
=============================================================

📊 Multi-Disease Datasets:
-------------------------------------------------------------

BREAST:
  ✓ pcam: found
      Files: 6 / 6

LUNG:
  ✓ lc25000: found
      Images: 25,000 / 25,000

COLON:
  ✓ nct_crc: found
      Images: 100,000 / 100,000
  ✓ crc_val: found
      Images: 7,180 / 7,180

MELANOMA:
  ⚠ ham10000: not_found

PROSTATE:
  ⚠ panda: not_found
  ⚠ sicap: not_found


🔬 Vision-Language Datasets:
-------------------------------------------------------------
  ✓ lc25000_captions: 10,000 image-text pairs
  ✓ generated: 5,000 image-text pairs

  Total: 15,000 pairs


📈 Summary:
-------------------------------------------------------------
Total datasets: 7
  ✓ Complete: 4
  ⚠ Partial: 0
  ✗ Missing: 3


🎯 Training Readiness:
-------------------------------------------------------------
✓ Ready for multi-disease training
  4 complete datasets available

⚠ Limited vision-language training possible
  Only 15,000 pairs (recommend 100K+)
```

---

### 4. `download_pmc_pathology.py`

**Purpose**: Download pathology images from PubMed Central Open Access Subset.

**Prerequisites**:
```bash
pip install biopython requests
```

**Usage**:
```bash
python scripts/download_pmc_pathology.py \
  --output-dir data/vision_language/pmc_pathology \
  --keywords "histopathology,pathology,microscopy" \
  --max-articles 1000 \
  --min-images 50000 \
  --email your_email@example.com
```

**Parameters**:
- `--output-dir`: Output directory for images and captions
- `--keywords`: Comma-separated keywords for search
- `--max-articles`: Maximum number of articles to process
- `--min-images`: Minimum number of images to download
- `--email`: Your email for NCBI Entrez (required)

**Note**: This is a simplified implementation. Full PMC image extraction requires more sophisticated XML parsing.

**Time**: 1-2 days for 50K images

---

## 🚀 Quick Start Workflow

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
pip install openai biopython
```

### Step 2: Download Public Datasets
```bash
python scripts/download_public_datasets.py --data-root data
```

### Step 3: Verify Downloads
```bash
python scripts/verify_datasets.py --data-root data
```

### Step 4: Generate Captions
```bash
export OPENAI_API_KEY=your_key_here

python scripts/generate_captions_gpt4v.py \
  --image-dir data/multi_disease \
  --output-dir data/vision_language/generated \
  --max-images 10000
```

### Step 5: Verify Vision-Language Data
```bash
python scripts/verify_datasets.py --data-root data
```

### Step 6: Start Training
```bash
# Multi-disease training
python experiments/train_multi_disease.py \
  --config configs/multi_disease.yaml \
  --data-root data

# Vision-language training
python experiments/train_vision_language.py \
  --config configs/vision_language.yaml \
  --data-root data/vision_language
```

---

## 📊 Expected Dataset Structure

After running all scripts, your data directory should look like:

```
data/
├── pcam_real/                    # Already downloaded
│   ├── camelyonpatch_level_2_split_train_x.h5
│   ├── camelyonpatch_level_2_split_train_y.h5
│   ├── camelyonpatch_level_2_split_valid_x.h5
│   ├── camelyonpatch_level_2_split_valid_y.h5
│   ├── camelyonpatch_level_2_split_test_x.h5
│   └── camelyonpatch_level_2_split_test_y.h5
│
├── multi_disease/
│   ├── breast/
│   │   └── pcam/ -> ../../pcam_real/
│   ├── lung/
│   │   └── lc25000/
│   │       ├── lung_aca/
│   │       ├── lung_n/
│   │       └── lung_scc/
│   ├── colon/
│   │   ├── nct_crc/
│   │   │   ├── ADI/
│   │   │   ├── BACK/
│   │   │   ├── DEB/
│   │   │   └── ...
│   │   └── crc_val/
│   ├── melanoma/
│   │   └── ham10000/
│   │       ├── images/
│   │       └── metadata.csv
│   └── prostate/
│       ├── panda/
│       │   ├── train_images/
│       │   └── train.csv
│       └── sicap/
│
├── vision_language/
│   ├── generated/
│   │   ├── images/
│   │   ├── captions/
│   │   └── vision_language_manifest.json
│   ├── pmc_pathology/
│   │   ├── images/
│   │   ├── captions/
│   │   └── pmc_metadata.json
│   └── openpath/
│
└── metadata/
    ├── dataset_registry.json
    └── verification_report.json
```

---

## 💰 Cost Breakdown

### Free Resources
- Public datasets: $0
- Compute (RTX 4070): $0
- Storage (local): $0

### Paid Resources
- GPT-4V captions (10K images): $200-300
- GPT-4V captions (50K images): $1,000-1,500
- GPT-4V captions (100K images): $2,000-3,000

### Total to Get Started
- **Minimal**: ~$300 (10K captions)
- **Recommended**: ~$1,500 (50K captions)
- **Full Scale**: ~$3,000 (100K captions)

---

## 🆘 Troubleshooting

### Download Issues

**Problem**: `git clone` fails
```bash
# Use HTTPS instead of SSH
git clone https://github.com/tampapath/lung_colon_image_set.git
```

**Problem**: Zenodo download is slow
```bash
# Use aria2c for faster downloads
sudo apt-get install aria2
aria2c -x 16 -s 16 <zenodo_url>
```

### GPT-4V Issues

**Problem**: Rate limit errors
```bash
# Increase batch delay
python scripts/generate_captions_gpt4v.py --batch-delay 2.0
```

**Problem**: API key not found
```bash
# Set environment variable
export OPENAI_API_KEY=your_key_here

# Or pass directly
python scripts/generate_captions_gpt4v.py --api-key your_key_here
```

### Verification Issues

**Problem**: Dataset not found
```bash
# Check data-root path
python scripts/verify_datasets.py --data-root /absolute/path/to/data
```

**Problem**: Incomplete dataset
```bash
# Re-run download script
python scripts/download_public_datasets.py --data-root data
```

---

## 📞 Support

For issues or questions:
- GitHub Issues: https://github.com/matthewvaishnav/computational-pathology-research/issues
- Email: matthew.vaishnav@example.com

---

*Last Updated: April 27, 2026*
