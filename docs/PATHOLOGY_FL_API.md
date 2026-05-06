# PathologyFL API Reference

## Core Classes

### PathologyFederatedAggregator

Main aggregation engine for hierarchical medical expertise weighting.

```python
from src.federated.pathology_fl import PathologyFederatedAggregator

aggregator = PathologyFederatedAggregator(alpha=0.5, beta=0.3)
```

**Parameters:**
- `alpha` (float): Expertise weighting factor (0.0-1.0)
- `beta` (float): Quality weighting factor (0.0-1.0)

**Methods:**

#### `calculate_expertise_weight(metadata, cancer_type)`
Calculate hospital expertise weight for specific cancer type.

**Returns:** float - Expertise weight (typically 0.5-10.0)

#### `aggregate_updates(client_updates, hospital_metadata, slide_quality, cancer_type)`
Perform hierarchical aggregation of model updates.

**Returns:** Dict[str, torch.Tensor] - Aggregated model parameters

### PathologyFLCoordinator

Federated learning coordinator with medical expertise integration.

```python
from src.federated.pathology_fl_coordinator import PathologyFLCoordinator

coordinator = PathologyFLCoordinator("config.yaml")
```

**Methods:**

#### `register_hospital(hospital_id, metadata)`
Register hospital with medical expertise metadata.

#### `federated_round(client_updates, slide_qualities, cancer_type)`
Execute one round of PathologyFL with expertise weighting.

### PathologyFLClient

Hospital-side client with slide quality assessment.

```python
from src.federated.pathology_fl_client import PathologyFLClient

client = PathologyFLClient("hospital_id", "config.yaml")
```

**Methods:**

#### `train_local_model(train_loader, epochs, cancer_type)`
Train local model and return updates with quality metrics.

**Returns:** Tuple[Dict, Dict] - Model updates and quality metrics

## Data Classes

### HospitalMetadata
```python
@dataclass
class HospitalMetadata:
    hospital_id: str
    hospital_type: HospitalType
    annual_cases: int
    cancer_specialties: List[CancerType]
    diagnostic_accuracy: float
    years_experience: int
```

### SlideQuality
```python
@dataclass
class SlideQuality:
    image_sharpness: float      # 0.0-1.0
    stain_consistency: float    # 0.0-1.0
    label_confidence: float     # 0.0-1.0
    artifact_level: float       # 0.0-1.0 (lower is better)
```

## Enums

### HospitalType
- `CANCER_CENTER`: Specialized cancer treatment centers
- `TEACHING_HOSPITAL`: Academic medical centers
- `COMMUNITY_HOSPITAL`: General community hospitals
- `RURAL_HOSPITAL`: Rural and critical access hospitals

### CancerType
- `BREAST`: Breast cancer specialization
- `LUNG`: Lung cancer specialization
- `PROSTATE`: Prostate cancer specialization
- `COLORECTAL`: Colorectal cancer specialization
- `GENERAL`: General pathology

## Configuration

### YAML Configuration Format
```yaml
coordinator:
  expertise_weight: 0.5
  quality_weight: 0.3
  num_rounds: 10

hospitals:
  cancer_centers:
    base_weight: 2.0
    specialty_bonus: 1.5
```

## Examples

### Basic Usage
```python
# Initialize coordinator
coordinator = PathologyFLCoordinator("config.yaml")

# Register hospital
coordinator.register_hospital("mayo_clinic", {
    "hospital_type": "cancer_center",
    "annual_cases": 15000,
    "cancer_specialties": ["breast", "lung"],
    "diagnostic_accuracy": 0.96,
    "years_experience": 20
})

# Execute federated round
await coordinator.federated_round(
    client_updates, slide_qualities, "breast"
)
```

### Advanced Configuration
```python
# Custom aggregator settings
aggregator = PathologyFederatedAggregator(
    alpha=0.6,  # Higher expertise weighting
    beta=0.2    # Lower quality weighting
)

# Cancer-specific aggregation
result = aggregator.pathology_type_specific_aggregation(
    client_updates, CancerType.BREAST, hospital_metadata
)
```