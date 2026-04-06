# Dataset Acquisition Guide

This guide provides instructions for acquiring and preparing publicly available datasets for computational pathology research. All datasets referenced are publicly accessible and require appropriate access permissions.

## Public Datasets

### 1. TCGA (The Cancer Genome Atlas)

**Description**: Comprehensive multi-omics cancer dataset including whole-slide images, genomic data, and clinical information across multiple cancer types.

**Access URL**: https://portal.gdc.cancer.gov/

**Dataset Version**: GDC Data Portal (latest release)

**Access Requirements**:
- Create a free account at the GDC Data Portal
- Controlled-access data requires dbGaP authorization
- Open-access data (including diagnostic slides) available without additional authorization

**Recommended Cancer Types for Multimodal Analysis**:
- TCGA-BRCA (Breast Invasive Carcinoma)
- TCGA-LUAD (Lung Adenocarcinoma)
- TCGA-KIRC (Kidney Renal Clear Cell Carcinoma)

**Data Components**:
- **WSI**: Diagnostic slides in SVS format (open access)
- **Genomic**: RNA-seq, DNA methylation, copy number variation (controlled access)
- **Clinical**: Patient demographics, treatment history, survival data (open access)

**Download Instructions**:
1. Navigate to https://portal.gdc.cancer.gov/
2. Use the Repository page to filter by data type (Slide Image, Gene Expression, Clinical)
3. Add files to cart and download using GDC Data Transfer Tool
4. For bulk downloads, use the GDC API or manifest files

### 2. CAMELYON (Cancer Metastases in Lymph Nodes)

**Description**: Large-scale dataset of whole-slide images for breast cancer metastasis detection in sentinel lymph nodes.

**Access URL**: https://camelyon17.grand-challenge.org/

**Dataset Version**: CAMELYON16 and CAMELYON17

**Access Requirements**:
- Register for a free account on Grand Challenge
- Accept the dataset terms of use
- Download access is immediate after registration

**Data Components**:
- **WSI**: H&E stained lymph node sections in TIFF format
- **Annotations**: Pixel-level metastasis annotations (CAMELYON16)
- **Clinical**: Patient-level labels for metastasis presence

**Download Instructions**:
1. Register at https://camelyon17.grand-challenge.org/
2. Navigate to the Data page
3. Download training and test sets (note: large files, ~1TB total)
4. Download annotation XML files for ground truth

**Note**: CAMELYON provides only WSI data. For multimodal experiments, this can be combined with TCGA genomic/clinical data or used for WSI-only experiments.

## Dataset Versions and Reproducibility

To ensure reproducibility, document the specific versions and access dates:

- **TCGA**: Specify GDC Data Release version (e.g., "Release 38.0, accessed March 2024")
- **CAMELYON**: Specify dataset version (CAMELYON16 or CAMELYON17)
- **Preprocessing**: Document all preprocessing steps and software versions

## Ethical Considerations

- All datasets contain de-identified patient data
- Use only for research purposes as specified in dataset terms
- Do not attempt to re-identify patients
- Cite original dataset publications in any resulting work
- Follow institutional IRB guidelines for human subjects research

## Dataset Citations

**TCGA**:
```
The Cancer Genome Atlas Research Network. (2013). 
The Cancer Genome Atlas Pan-Cancer analysis project. 
Nature Genetics, 45(10), 1113-1120.
```

**CAMELYON**:
```
Bandi, P., et al. (2019). 
From Detection of Individual Metastases to Classification of Lymph Node Status at the Patient Level: 
The CAMELYON17 Challenge. 
IEEE Transactions on Medical Imaging, 38(2), 550-560.
```


## Preprocessing Instructions

### WSI Preprocessing

**Objective**: Extract patch-level features from whole-slide images for efficient training.

**Steps**:

1. **Tissue Segmentation**:
   - Convert WSI to lower resolution (e.g., 32x downsampling)
   - Apply Otsu thresholding to separate tissue from background
   - Filter out small artifacts and background regions

2. **Patch Extraction**:
   - Extract patches at desired magnification (e.g., 20x or 40x)
   - Typical patch size: 224x224 or 256x256 pixels
   - Use stride to control overlap (e.g., stride=224 for non-overlapping)
   - Skip patches with >50% background

3. **Feature Extraction**:
   - Use pretrained CNN (e.g., ResNet50, ImageNet pretrained) to extract features
   - Extract features from penultimate layer (e.g., 2048-dim for ResNet50)
   - Store features in HDF5 format for efficient loading
   - Typical output: [num_patches, feature_dim] per slide

4. **Stain Normalization** (Optional):
   - Apply traditional methods (Macenko, Reinhard) or use the transformer-based approach
   - Normalize to a reference slide to reduce color variation
   - Apply before or after feature extraction depending on approach

**Tools**:
- OpenSlide for WSI reading
- PyTorch/TensorFlow for feature extraction
- scikit-image for tissue segmentation

**Example Command** (pseudocode):
```bash
python scripts/preprocess_wsi.py \
    --input_dir data/raw/slides/ \
    --output_dir data/processed/wsi_features/ \
    --patch_size 224 \
    --magnification 20x \
    --feature_extractor resnet50
```

### Genomic Data Preprocessing

**Objective**: Normalize and filter genomic features for integration with imaging data.

**Steps**:

1. **RNA-seq Processing**:
   - Use FPKM or TPM normalized counts from TCGA
   - Log-transform: log2(value + 1)
   - Filter low-variance genes (e.g., keep top 5000 by variance)
   - Z-score normalization across samples

2. **Copy Number Variation**:
   - Use gene-level CNV estimates from GISTIC2
   - Filter for high-confidence alterations
   - Encode as continuous values or discrete categories

3. **Mutation Data**:
   - Extract binary mutation indicators for key genes
   - Focus on driver mutations or frequently mutated genes
   - Encode as binary vector [num_genes]

**Output Format**:
- CSV file with rows=patients, columns=genes
- Typical dimensions: [num_patients, 5000-20000 features]

**Example Command** (pseudocode):
```bash
python scripts/preprocess_genomics.py \
    --input_dir data/raw/tcga_genomics/ \
    --output_file data/processed/genomic_features.csv \
    --num_genes 5000 \
    --normalization zscore
```

### Clinical Text Preprocessing

**Objective**: Tokenize and encode clinical notes for text encoder input.

**Steps**:

1. **Text Extraction**:
   - Extract relevant clinical fields (diagnosis, pathology reports, treatment notes)
   - Concatenate fields with special separators

2. **Tokenization**:
   - Use domain-specific tokenizer (e.g., BioBERT, ClinicalBERT) or general tokenizer
   - Truncate or pad to fixed length (e.g., 512 tokens)
   - Add special tokens ([CLS], [SEP])

3. **Encoding**:
   - Convert tokens to integer IDs using tokenizer vocabulary
   - Store as integer arrays for efficient loading

**Output Format**:
- JSON file with patient_id -> token_ids mapping
- Or CSV with patient_id, tokenized_text columns

**Example Command** (pseudocode):
```bash
python scripts/preprocess_clinical_text.py \
    --input_file data/raw/clinical_data.csv \
    --output_file data/processed/clinical_text.json \
    --tokenizer bert-base-uncased \
    --max_length 512
```

### Expected Directory Structure After Preprocessing

```
data/
├── README.md                          # This file
├── raw/                               # Raw downloaded data (not in git)
│   ├── slides/                        # Raw WSI files (.svs, .tiff)
│   ├── tcga_genomics/                 # Raw TCGA genomic data
│   └── clinical_data.csv              # Raw clinical information
├── processed/                         # Preprocessed data (not in git)
│   ├── wsi_features/                  # Extracted WSI features
│   │   ├── patient_001_slide_01.h5   # HDF5 files with patch features
│   │   ├── patient_001_slide_02.h5
│   │   └── ...
│   ├── genomic_features.csv           # Normalized genomic data
│   ├── clinical_text.json             # Tokenized clinical text
│   └── metadata.json                  # Dataset splits and patient info
└── .gitkeep                           # Placeholder for git

```

**Storage Requirements**:
- Raw WSI: ~100GB - 1TB depending on dataset size
- Processed features: ~10-50GB (much smaller than raw images)
- Genomic/clinical: ~100MB - 1GB


## Data Format Specifications

### HDF5 Format for WSI Features

**File Structure**:
```
patient_001_slide_01.h5
├── features          # Dataset [num_patches, feature_dim], dtype=float32
├── coordinates       # Dataset [num_patches, 2], dtype=int32 (x, y positions)
└── metadata          # Group
    ├── slide_id      # Attribute: string
    ├── patient_id    # Attribute: string
    ├── magnification # Attribute: string (e.g., "20x")
    ├── patch_size    # Attribute: int
    └── feature_extractor  # Attribute: string (e.g., "resnet50")
```

**Example Loading Code**:
```python
import h5py
import numpy as np

# Load WSI features
with h5py.File('data/processed/wsi_features/patient_001_slide_01.h5', 'r') as f:
    features = f['features'][:]  # Shape: [num_patches, 2048]
    coordinates = f['coordinates'][:]  # Shape: [num_patches, 2]
    slide_id = f['metadata'].attrs['slide_id']
    patient_id = f['metadata'].attrs['patient_id']

print(f"Loaded {features.shape[0]} patches for patient {patient_id}")
print(f"Feature dimension: {features.shape[1]}")
```

### JSON Format for Metadata

**File**: `data/processed/metadata.json`

**Structure**:
```json
{
  "dataset_info": {
    "name": "TCGA-BRCA",
    "version": "GDC Release 38.0",
    "preprocessing_date": "2024-03-15",
    "num_patients": 1000
  },
  "splits": {
    "train": ["patient_001", "patient_002", ...],
    "val": ["patient_800", "patient_801", ...],
    "test": ["patient_900", "patient_901", ...]
  },
  "patients": {
    "patient_001": {
      "slides": [
        {
          "slide_id": "slide_01",
          "file_path": "wsi_features/patient_001_slide_01.h5",
          "timestamp": 0,
          "stain_protocol": "H&E_lab_A"
        }
      ],
      "genomic_available": true,
      "clinical_text_available": true,
      "label": 1,
      "survival_months": 48.5
    }
  }
}
```

**Example Loading Code**:
```python
import json

# Load metadata
with open('data/processed/metadata.json', 'r') as f:
    metadata = json.load(f)

# Get training patient IDs
train_patients = metadata['splits']['train']

# Get patient information
patient_info = metadata['patients']['patient_001']
print(f"Patient has {len(patient_info['slides'])} slides")
print(f"Genomic data available: {patient_info['genomic_available']}")
```

### CSV Format for Genomic Data

**File**: `data/processed/genomic_features.csv`

**Structure**:
```csv
patient_id,gene_0001,gene_0002,gene_0003,...,gene_5000
patient_001,0.523,-1.234,0.891,...,2.103
patient_002,-0.234,0.456,-0.123,...,1.234
...
```

- First column: `patient_id` (string)
- Remaining columns: Gene expression values (float, z-score normalized)
- Column names: `gene_XXXX` where XXXX is gene index or gene symbol

**Example Loading Code**:
```python
import pandas as pd

# Load genomic features
genomic_df = pd.read_csv('data/processed/genomic_features.csv', index_col='patient_id')

# Get features for a specific patient
patient_genomic = genomic_df.loc['patient_001'].values  # Shape: [5000]

print(f"Genomic feature dimension: {len(patient_genomic)}")
print(f"Feature range: [{patient_genomic.min():.2f}, {patient_genomic.max():.2f}]")
```

### JSON Format for Clinical Text

**File**: `data/processed/clinical_text.json`

**Structure**:
```json
{
  "patient_001": {
    "text": "Patient diagnosed with invasive ductal carcinoma...",
    "token_ids": [101, 5776, 11441, 2007, 13204, ...],
    "attention_mask": [1, 1, 1, 1, 1, ...],
    "length": 256
  },
  "patient_002": {
    "text": "Pathology report shows...",
    "token_ids": [101, 3507, 2227, 3189, ...],
    "attention_mask": [1, 1, 1, 1, ...],
    "length": 128
  }
}
```

**Example Loading Code**:
```python
import json
import torch

# Load clinical text
with open('data/processed/clinical_text.json', 'r') as f:
    clinical_data = json.load(f)

# Get tokenized text for a patient
patient_text = clinical_data['patient_001']
token_ids = torch.tensor(patient_text['token_ids'])  # Shape: [seq_len]
attention_mask = torch.tensor(patient_text['attention_mask'])  # Shape: [seq_len]

print(f"Text length: {patient_text['length']} tokens")
print(f"Original text: {patient_text['text'][:100]}...")
```

## Complete Data Loading Example

**Multimodal Data Loading**:
```python
import h5py
import json
import pandas as pd
import torch

def load_patient_data(patient_id, data_dir='data/processed'):
    """Load all modalities for a single patient."""
    
    # Load metadata
    with open(f'{data_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    patient_info = metadata['patients'][patient_id]
    
    # Load WSI features (first slide)
    wsi_path = f"{data_dir}/{patient_info['slides'][0]['file_path']}"
    with h5py.File(wsi_path, 'r') as f:
        wsi_features = torch.tensor(f['features'][:])  # [num_patches, feature_dim]
    
    # Load genomic features
    genomic_df = pd.read_csv(f'{data_dir}/genomic_features.csv', index_col='patient_id')
    genomic_features = torch.tensor(genomic_df.loc[patient_id].values)  # [num_genes]
    
    # Load clinical text
    with open(f'{data_dir}/clinical_text.json', 'r') as f:
        clinical_data = json.load(f)
    clinical_tokens = torch.tensor(clinical_data[patient_id]['token_ids'])  # [seq_len]
    
    # Get label
    label = patient_info['label']
    
    return {
        'wsi_features': wsi_features,
        'genomic': genomic_features,
        'clinical_text': clinical_tokens,
        'label': label,
        'patient_id': patient_id
    }

# Example usage
data = load_patient_data('patient_001')
print(f"WSI features: {data['wsi_features'].shape}")
print(f"Genomic features: {data['genomic'].shape}")
print(f"Clinical text: {data['clinical_text'].shape}")
print(f"Label: {data['label']}")
```

## Notes

- All file paths in metadata.json are relative to `data/processed/`
- Missing modalities should be handled by returning `None` or empty tensors
- Ensure consistent patient IDs across all data files
- Use the same random seed for train/val/test splits to ensure reproducibility
- Store large files (WSI features) in HDF5 for memory-efficient loading
- Consider using data augmentation during training (not during preprocessing)

