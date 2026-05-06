# PathologyFL Deployment Guide

## Quick Deployment

### 1. Install HistoCore
```bash
pip install histocore
```

### 2. Start Coordinator
```bash
python -m src.federated.pathology_fl_coordinator \
    --config configs/pathology_fl_config.yaml
```

### 3. Register Hospitals
```python
coordinator.register_hospital("mayo_clinic", {
    "hospital_type": "cancer_center",
    "annual_cases": 15000,
    "cancer_specialties": ["breast", "lung", "prostate"],
    "diagnostic_accuracy": 0.96,
    "years_experience": 20
})
```

### 4. Start Clients
```bash
python -m src.federated.pathology_fl_client \
    --hospital-id mayo_clinic \
    --config client_config.yaml
```

## Hospital Types

- **Cancer Centers**: 2.0x base weight, specialty bonuses
- **Teaching Hospitals**: 1.5x base weight, research focus
- **Community Hospitals**: 1.0x base weight, general care
- **Rural Hospitals**: 0.8x base weight, limited resources

## Configuration

Edit `configs/pathology_fl_config.yaml`:
- Adjust expertise and quality weights
- Set quality thresholds
- Configure cancer-type specific parameters

## Monitoring

PathologyFL provides detailed logging:
- Expertise weight calculations
- Quality assessments
- Aggregation results
- Performance metrics