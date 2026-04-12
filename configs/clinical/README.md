# Clinical Workflow Configuration

This directory contains configuration files for the clinical workflow integration system.

## Configuration Files

### Disease Taxonomies

Disease taxonomy configurations define the classification schemes for different clinical specialties:

- **`cancer_grading.yaml`**: Cancer grading taxonomy (normal, benign, grade 1-4)
- **`cardiac_pathology.yaml`**: Cardiac pathology taxonomy (myocardium, ischemia, infarction, etc.)
- **`tissue_classification.yaml`**: General tissue type classification (epithelial, connective, muscle, etc.)

Each taxonomy file includes:
- Disease definitions with hierarchical relationships
- Clinical decision thresholds per disease state
- Time horizon thresholds for risk prediction

### Patient Metadata Schema

- **`patient_metadata_schema.yaml`**: Schema for structured patient clinical metadata

Defines required and optional fields for patient data including:
- Demographics (age, sex, ethnicity)
- Medical history (comorbidities, previous diagnoses)
- Lifestyle factors (smoking, alcohol, exercise)
- Family history
- Current medications

## Usage

### Loading a Disease Taxonomy

```python
from src.clinical.taxonomy import DiseaseTaxonomy

# Load from YAML file
taxonomy = DiseaseTaxonomy(config_file="configs/clinical/cancer_grading.yaml")

# Get taxonomy information
num_classes = taxonomy.get_num_classes()
disease_ids = list(taxonomy.diseases.keys())
```

### Loading Clinical Thresholds

```python
from src.clinical.thresholds import ClinicalThresholdSystem

# Initialize threshold system
threshold_system = ClinicalThresholdSystem()

# Load thresholds from taxonomy config
threshold_system.load_from_file("configs/clinical/cancer_grading.yaml")

# Evaluate risk scores
flagged = threshold_system.evaluate_risk_scores(risk_scores)
```

### Creating Patient Metadata

```python
from src.clinical.patient_context import ClinicalMetadata

# Create patient metadata following schema
metadata = ClinicalMetadata(
    age=65,
    sex="M",
    smoking_status="former",
    alcohol_consumption="moderate",
    medications=["aspirin", "metformin"],
    exercise_frequency="light",
    family_history={"cancer": True, "heart_disease": False}
)
```

## Creating Custom Taxonomies

To create a custom disease taxonomy:

1. Copy an existing taxonomy file as a template
2. Define your disease states with unique IDs
3. Set parent-child relationships for hierarchical taxonomies
4. Configure clinical decision thresholds for each disease state
5. Set time horizon thresholds for risk prediction

### Example Taxonomy Structure

```yaml
name: "my_taxonomy"
version: "1.0"
description: "Custom disease taxonomy"

diseases:
  - id: "disease_1"
    name: "Disease Name"
    description: "Disease description"
    parent: null  # or parent disease ID
    
  - id: "disease_2"
    name: "Another Disease"
    description: "Another description"
    parent: "disease_1"  # child of disease_1

thresholds:
  disease_1:
    risk_threshold: 0.5
    confidence_threshold: 0.85
    anomaly_threshold: 0.6
    
  disease_2:
    risk_threshold: 0.6
    confidence_threshold: 0.90
    anomaly_threshold: 0.7

time_horizon_thresholds:
  "1_year": 0.6
  "5_year": 0.5
  "10_year": 0.4
```

## Threshold Configuration

### Risk Thresholds
- **Purpose**: Flag cases with elevated disease risk
- **Range**: 0.0 - 1.0
- **Typical values**: 0.5 - 0.8 depending on disease severity

### Confidence Thresholds
- **Purpose**: Ensure prediction reliability
- **Range**: 0.0 - 1.0
- **Typical values**: 0.80 - 0.95 (higher for critical diagnoses)

### Anomaly Thresholds
- **Purpose**: Detect unusual patterns requiring review
- **Range**: 0.0 - 1.0
- **Typical values**: 0.5 - 0.9

### Time Horizon Thresholds
- **Purpose**: Flag elevated risk at specific time horizons
- **Horizons**: 1-year, 5-year, 10-year
- **Typical values**: Decreasing with longer horizons (0.6, 0.5, 0.4)

## Validation

All configuration files are validated on load:
- Disease IDs must be unique
- Parent references must exist
- Thresholds must be in valid ranges (0.0 - 1.0)
- Required fields must be present

## Examples

See the `examples/` directory for complete usage examples:
- `clinical_inference.py`: End-to-end clinical inference
- `longitudinal_analysis.py`: Patient tracking over time

## Support

For questions or issues with configuration files, please refer to:
- Clinical workflow integration documentation
- Disease taxonomy API documentation
- Example scripts in `examples/` directory
