# Data Preparation Scripts

This directory contains scripts for preparing and validating datasets for training.

## Scripts

### 1. prepare_dataset.py

Prepares raw data for training by organizing files and creating train/val/test splits.

**Features:**
- Automatic train/val/test splitting with stratification
- Metadata organization
- File copying and organization
- Dataset info generation
- Validation checks

**Usage:**
```bash
# Basic usage
python scripts/data/prepare_dataset.py \
    --raw-dir data/raw \
    --output-dir data/processed

# Custom split ratios
python scripts/data/prepare_dataset.py \
    --raw-dir data/raw \
    --output-dir data/processed \
    --split-ratio 0.8 0.1 0.1

# With custom random seed
python scripts/data/prepare_dataset.py \
    --raw-dir data/raw \
    --output-dir data/processed \
    --random-seed 123
```

**Input Structure:**
```
data/raw/
├── metadata.csv          # Required: Contains sample information
├── slide1.svs           # WSI files
├── slide2.svs
└── ...
```

**Output Structure:**
```
data/processed/
├── dataset_info.json    # Dataset statistics
├── train/
│   ├── metadata.csv
│   └── ...
├── val/
│   ├── metadata.csv
│   └── ...
└── test/
    ├── metadata.csv
    └── ...
```

**Metadata CSV Format:**
The metadata CSV should contain at least:
- `sample_id`: Unique identifier for each sample
- `wsi_path`: Path to WSI file (relative to raw_dir)
- `label`: Class label (for stratified splitting)
- Additional clinical features as needed

Example:
```csv
sample_id,wsi_path,label,age,stage
patient_001,slide1.svs,0,65,2
patient_002,slide2.svs,1,58,3
```

### 2. validate_data.py

Validates dataset integrity and quality.

**Features:**
- Directory structure validation
- Metadata validation
- Label distribution analysis
- Feature statistics
- Missing value detection
- Outlier detection
- File existence checks
- Class imbalance detection

**Usage:**
```bash
# Validate entire dataset
python scripts/data/validate_data.py --data-dir data/processed

# Validate specific split
python scripts/data/validate_data.py --data-dir data/processed --split train

# Quiet mode (only show issues)
python scripts/data/validate_data.py --data-dir data/processed --quiet
```

**Validation Checks:**
1. **Directory Structure**
   - Checks for train/val/test directories
   - Verifies metadata files exist

2. **Metadata Validation**
   - Checks for required columns
   - Detects duplicate sample IDs
   - Reports missing values

3. **Label Distribution**
   - Analyzes class distribution
   - Detects class imbalance
   - Reports imbalance ratios

4. **Feature Statistics**
   - Identifies constant features
   - Detects outliers using IQR method
   - Reports feature statistics

5. **File Validation**
   - Checks if referenced files exist
   - Reports missing files

**Exit Codes:**
- `0`: All validation checks passed
- `1`: Critical issues found or validation failed

## Workflow

### Complete Data Preparation Workflow

```bash
# 1. Prepare dataset
python scripts/data/prepare_dataset.py \
    --raw-dir data/raw \
    --output-dir data/processed \
    --split-ratio 0.7 0.15 0.15

# 2. Validate prepared dataset
python scripts/data/validate_data.py --data-dir data/processed

# 3. If validation passes, proceed to training
python experiments/train.py --data-dir data/processed
```

### Makefile Integration

These scripts are integrated into the Makefile:

```bash
# Prepare dataset
make prepare-data RAW_DIR=data/raw OUTPUT_DIR=data/processed

# Validate dataset
make validate-data DATA_DIR=data/processed

# Check data status
make data-check
```

## Best Practices

### 1. Data Organization

- Keep raw data separate from processed data
- Use consistent naming conventions
- Include comprehensive metadata
- Document data sources and preprocessing steps

### 2. Split Ratios

- **Standard**: 70/15/15 (train/val/test)
- **Large datasets**: 80/10/10
- **Small datasets**: 60/20/20
- Always use stratified splitting for classification

### 3. Validation

- Always validate data after preparation
- Check for class imbalance
- Monitor missing values
- Verify file integrity

### 4. Reproducibility

- Use fixed random seeds
- Document split ratios
- Save dataset info
- Version control metadata

## Troubleshooting

### Issue: "No metadata CSV found"

**Solution:** Ensure your raw directory contains a CSV file with sample information. The file should be named `metadata.csv` or contain "metadata" in the filename.

### Issue: "Severe class imbalance detected"

**Solution:** Consider:
- Using weighted loss functions
- Oversampling minority classes
- Undersampling majority classes
- Using stratified sampling

### Issue: "Missing files"

**Solution:** 
- Check that `wsi_path` in metadata is correct
- Verify files exist in raw directory
- Check file permissions
- Ensure paths are relative to raw_dir

### Issue: "Duplicate sample IDs"

**Solution:**
- Review metadata for duplicate entries
- Ensure each sample has a unique ID
- Remove or merge duplicates

## Advanced Usage

### Custom Preprocessing

You can extend these scripts for custom preprocessing:

```python
from scripts.data.prepare_dataset import DatasetPreparer

class CustomPreparer(DatasetPreparer):
    def preprocess_sample(self, sample):
        # Add custom preprocessing
        return processed_sample

preparer = CustomPreparer(...)
preparer.prepare()
```

### Integration with CI/CD

Add data validation to your CI pipeline:

```yaml
# .github/workflows/ci.yml
- name: Validate Data
  run: |
    python scripts/data/validate_data.py \
      --data-dir data/processed \
      --quiet
```

## Requirements

These scripts require:
- pandas
- numpy
- scikit-learn
- tqdm

Install with:
```bash
pip install pandas numpy scikit-learn tqdm
```

Or use the project requirements:
```bash
pip install -r requirements.txt
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the script help: `python scripts/data/prepare_dataset.py --help`
3. Check the main project README
4. Open an issue on GitHub
