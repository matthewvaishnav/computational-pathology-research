# Data Acquisition Guide for Medical AI Training

## Overview

This guide provides practical steps for acquiring the datasets and resources needed to complete the Medical AI Revolution platform.

---

## 🔬 Vision-Language Data for Zero-Shot Detection

### What You Need

**Image-Text Pairs**: 100K+ pathology images with corresponding text descriptions

### Public Datasets Available Now

#### 1. **PubMed Central Open Access Subset**
- **Source**: https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
- **Content**: Medical images with captions from research papers
- **Size**: ~3M images across all medical domains
- **Pathology subset**: ~500K images
- **License**: Various open licenses (check per article)
- **How to get**:
  ```bash
  # Install PMC OA tools
  pip install biopython requests
  
  # Download pathology subset
  python scripts/download_pmc_pathology.py \
    --output-dir data/pmc_pathology \
    --keywords "histopathology,pathology,microscopy" \
    --min-images 100000
  ```

#### 2. **ARCH (Archive of Cancer Histopathology)**
- **Source**: https://warwick.ac.uk/fac/cross_fac/tia/data/arch
- **Content**: Annotated histopathology images with diagnostic reports
- **Size**: 10K+ WSI with reports
- **License**: Academic use
- **How to get**: Request access via university email

#### 3. **PathologyOutlines.com Dataset**
- **Source**: https://www.pathologyoutlines.com/
- **Content**: Educational pathology images with detailed descriptions
- **Size**: 50K+ images with expert annotations
- **License**: Educational use (contact for research)
- **How to get**: Email info@pathologyoutlines.com with research proposal

#### 4. **TCGA (The Cancer Genome Atlas)**
- **Source**: https://portal.gdc.cancer.gov/
- **Content**: WSI with clinical reports and genomic data
- **Size**: 30K+ slides across 33 cancer types
- **License**: Open access
- **How to get**:
  ```bash
  # Install GDC Data Transfer Tool
  wget https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip
  unzip gdc-client_v1.6.1_Ubuntu_x64.zip
  
  # Download TCGA slides
  ./gdc-client download -m tcga_manifest.txt -d data/tcga
  ```

#### 5. **OpenPath Dataset**
- **Source**: https://github.com/mahmoodlab/OpenPath
- **Content**: Curated pathology images with captions
- **Size**: 200K+ image-text pairs
- **License**: CC BY 4.0
- **How to get**:
  ```bash
  git clone https://github.com/mahmoodlab/OpenPath.git
  cd OpenPath
  python download_dataset.py --output data/openpath
  ```

### Creating Your Own Vision-Language Dataset

#### Option 1: Mine Pathology Reports

```python
# scripts/create_vision_language_pairs.py
"""
Extract image-text pairs from pathology reports and slides.
"""

import os
from pathlib import Path
import pydicom
from PIL import Image

def extract_report_text(dicom_path):
    """Extract text from DICOM SR (Structured Report)."""
    ds = pydicom.dcmread(dicom_path)
    
    # Extract findings, diagnosis, description
    report_text = []
    
    if hasattr(ds, 'ContentSequence'):
        for item in ds.ContentSequence:
            if hasattr(item, 'TextValue'):
                report_text.append(item.TextValue)
    
    return ' '.join(report_text)

def create_image_text_pairs(slide_dir, report_dir, output_dir):
    """Create image-text pairs from slides and reports."""
    pairs = []
    
    for slide_path in Path(slide_dir).glob('*.svs'):
        slide_id = slide_path.stem
        report_path = Path(report_dir) / f"{slide_id}_report.txt"
        
        if report_path.exists():
            # Extract patches from slide
            patches = extract_representative_patches(slide_path, num_patches=10)
            
            # Read report text
            with open(report_path) as f:
                report_text = f.read()
            
            # Create pairs
            for i, patch in enumerate(patches):
                pair = {
                    'image': patch,
                    'text': report_text,
                    'slide_id': slide_id,
                    'patch_id': i
                }
                pairs.append(pair)
    
    return pairs
```

#### Option 2: Use GPT-4V for Caption Generation

```python
# scripts/generate_captions_gpt4v.py
"""
Generate captions for pathology images using GPT-4V.
"""

import openai
import base64
from PIL import Image
import io

def generate_caption(image_path, api_key):
    """Generate detailed caption for pathology image."""
    
    # Load and encode image
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # Call GPT-4V
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Describe this pathology image in detail. Include:
                        1. Tissue type and staining
                        2. Cellular features and morphology
                        3. Pathological findings
                        4. Diagnostic impression
                        Be specific and use medical terminology."""
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_data}"
                    }
                ]
            }
        ],
        max_tokens=500
    )
    
    return response.choices[0].message.content

# Usage
for image_path in Path('data/slides').glob('*.jpg'):
    caption = generate_caption(image_path, api_key=YOUR_API_KEY)
    
    # Save caption
    caption_path = image_path.with_suffix('.txt')
    with open(caption_path, 'w') as f:
        f.write(caption)
```

#### Option 3: Collaborate with Pathology Labs

**Template Email for Pathology Labs:**

```
Subject: Research Collaboration - Medical AI for Pathology

Dear [Lab Director],

I'm developing an open-source medical AI platform for pathology diagnosis 
and would like to collaborate on creating a vision-language dataset.

What we need:
- De-identified pathology slides (WSI)
- Corresponding diagnostic reports (anonymized)
- Approximately 10,000-50,000 cases

What we offer:
- Co-authorship on publications
- Access to trained models for your lab
- Technical support for deployment
- IRB protocol assistance

Our platform: https://github.com/matthewvaishnav/computational-pathology-research

Would you be interested in discussing this further?

Best regards,
[Your Name]
```

---

## 🏥 Multi-Disease Datasets

### Publicly Available Datasets

#### 1. **Breast Cancer**
- **CAMELYON16/17**: https://camelyon17.grand-challenge.org/
  - 1,000 WSI with lymph node metastases
  - Download: Register and download from challenge site
  
- **BRACS**: https://www.bracs.icar.cnr.it/
  - 4,539 breast cancer images, 7 classes
  - Download: Request access via website

#### 2. **Lung Cancer**
- **LC25000**: https://github.com/tampapath/lung_colon_image_set
  - 25,000 lung histopathology images
  - 5 classes (normal, adenocarcinoma, squamous cell carcinoma)
  - Download: Direct from GitHub

- **NLST**: https://cdas.cancer.gov/nlst/
  - National Lung Screening Trial data
  - Request access through NCI

#### 3. **Prostate Cancer**
- **PANDA**: https://www.kaggle.com/c/prostate-cancer-grade-assessment
  - 10,616 WSI with Gleason grading
  - Download: Kaggle competition data

- **SICAPv2**: https://data.mendeley.com/datasets/9xxm58dvs3/1
  - 18,783 patches, 4 Gleason grades
  - Download: Direct from Mendeley Data

#### 4. **Colon Cancer**
- **CRC-VAL-HE-7K**: https://zenodo.org/record/1214456
  - 7,180 colorectal cancer images
  - 9 tissue classes
  - Download: Direct from Zenodo

- **NCT-CRC-HE-100K**: https://zenodo.org/record/1214456
  - 100,000 colorectal cancer patches
  - Download: Direct from Zenodo

#### 5. **Melanoma**
- **ISIC Archive**: https://www.isic-archive.com/
  - 50,000+ dermoscopy images
  - Download: API or bulk download

- **HAM10000**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
  - 10,015 dermatoscopic images
  - 7 diagnostic categories

### Download Script for All Datasets

```bash
# scripts/download_all_datasets.sh

#!/bin/bash

# Create data directory
mkdir -p data/{breast,lung,prostate,colon,melanoma}

# Breast Cancer
echo "Downloading breast cancer datasets..."
# CAMELYON (requires registration)
echo "Please register at https://camelyon17.grand-challenge.org/ and download manually"

# Lung Cancer
echo "Downloading lung cancer datasets..."
git clone https://github.com/tampapath/lung_colon_image_set.git data/lung/lc25000

# Prostate Cancer
echo "Downloading prostate cancer datasets..."
kaggle competitions download -c prostate-cancer-grade-assessment -p data/prostate/panda

# Colon Cancer
echo "Downloading colon cancer datasets..."
wget https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip -P data/colon/
unzip data/colon/NCT-CRC-HE-100K.zip -d data/colon/nct_crc

# Melanoma
echo "Downloading melanoma datasets..."
wget https://dataverse.harvard.edu/api/access/datafile/3358126 -O data/melanoma/ham10000.zip
unzip data/melanoma/ham10000.zip -d data/melanoma/ham10000

echo "Dataset download complete!"
echo "Note: Some datasets require manual registration and download"
```

---

## 📊 Dataset Organization

### Recommended Structure

```
data/
├── vision_language/
│   ├── pmc_pathology/
│   │   ├── images/
│   │   └── captions/
│   ├── openpath/
│   └── custom/
├── multi_disease/
│   ├── breast/
│   │   ├── camelyon16/
│   │   └── bracs/
│   ├── lung/
│   │   └── lc25000/
│   ├── prostate/
│   │   └── panda/
│   ├── colon/
│   │   └── nct_crc/
│   └── melanoma/
│       └── ham10000/
└── metadata/
    ├── dataset_registry.json
    └── statistics.json
```

### Dataset Registry

```python
# scripts/create_dataset_registry.py
"""
Create a registry of all available datasets.
"""

import json
from pathlib import Path

def create_registry():
    registry = {
        "vision_language": {
            "pmc_pathology": {
                "path": "data/vision_language/pmc_pathology",
                "num_pairs": 500000,
                "source": "PubMed Central",
                "license": "Various (check per article)"
            },
            "openpath": {
                "path": "data/vision_language/openpath",
                "num_pairs": 200000,
                "source": "OpenPath",
                "license": "CC BY 4.0"
            }
        },
        "multi_disease": {
            "breast": {
                "camelyon16": {
                    "path": "data/multi_disease/breast/camelyon16",
                    "num_slides": 400,
                    "classes": ["normal", "tumor"],
                    "license": "Academic use"
                }
            },
            "lung": {
                "lc25000": {
                    "path": "data/multi_disease/lung/lc25000",
                    "num_images": 25000,
                    "classes": ["normal", "adenocarcinoma", "squamous"],
                    "license": "CC0 1.0"
                }
            }
            # ... add all datasets
        }
    }
    
    with open('data/metadata/dataset_registry.json', 'w') as f:
        json.dump(registry, f, indent=2)

if __name__ == "__main__":
    create_registry()
```

---

## 🚀 Quick Start: Minimal Viable Dataset

If you want to start training immediately with publicly available data:

### 1. Download Core Datasets (1 week)

```bash
# Breast cancer (already have PCam)
# ✓ Already downloaded

# Lung cancer
git clone https://github.com/tampapath/lung_colon_image_set.git data/lung/

# Prostate cancer
kaggle competitions download -c prostate-cancer-grade-assessment

# Colon cancer
wget https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip

# Melanoma
wget https://dataverse.harvard.edu/api/access/datafile/3358126
```

### 2. Create Vision-Language Pairs (2 weeks)

```bash
# Use GPT-4V to generate captions
python scripts/generate_captions_gpt4v.py \
  --image-dir data/multi_disease \
  --output-dir data/vision_language/generated \
  --api-key YOUR_OPENAI_API_KEY
```

### 3. Start Training (immediate)

```bash
# Train multi-disease model
python experiments/train_multi_disease.py \
  --config configs/multi_disease.yaml \
  --data-registry data/metadata/dataset_registry.json

# Train vision-language model
python experiments/train_vision_language.py \
  --config configs/vision_language.yaml \
  --image-dir data/vision_language/generated/images \
  --caption-dir data/vision_language/generated/captions
```

---

## 💰 Budget Considerations

### Free Options
- **Public datasets**: $0
- **GPT-4V captions**: ~$500 for 100K images
- **Compute (your RTX 4070)**: $0 (already have)

### Paid Options
- **Commercial datasets**: $5K-50K per dataset
- **Cloud compute (if needed)**: $1K-5K/month
- **Data annotation services**: $0.10-1.00 per image

### Recommended Approach
1. Start with free public datasets (0-4 weeks)
2. Generate captions with GPT-4V ($500)
3. Train on your local GPU (free)
4. Scale to cloud if needed ($1K-5K)

**Total to get started**: ~$500

---

## 📞 Getting Help

### Academic Collaborations
- Email pathology departments at universities
- Join pathology AI research groups
- Attend MICCAI, ISBI conferences

### Industry Partnerships
- Contact digital pathology vendors (Leica, Hamamatsu)
- Reach out to PathAI, Paige.AI for data sharing
- Join Digital Pathology Association

### Online Communities
- r/computationalpathology
- PathologyOutlines forums
- Grand Challenge forums

---

## ✅ Next Steps

1. **This week**: Download public datasets (LC25000, NCT-CRC, HAM10000)
2. **Next week**: Generate captions with GPT-4V
3. **Week 3**: Start multi-disease training
4. **Week 4**: Start vision-language training
5. **Month 2**: Evaluate and iterate

You can have a working multi-disease model with vision-language capabilities in **4-6 weeks** using publicly available data and your existing hardware.
