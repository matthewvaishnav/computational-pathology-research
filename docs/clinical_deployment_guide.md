# Clinical Deployment Guide

This guide provides instructions for deploying the computational pathology system in clinical environments.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Clinical Workflow Setup](#clinical-workflow-setup)
5. [Integration with Clinical Systems](#integration-with-clinical-systems)
6. [Security and Compliance](#security-and-compliance)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware Requirements

**Minimum Configuration:**
- CPU: 8 cores (Intel Xeon or AMD EPYC)
- RAM: 32 GB
- GPU: NVIDIA RTX 4070 or equivalent (12GB VRAM)
- Storage: 500 GB SSD

**Recommended Configuration:**
- CPU: 16+ cores
- RAM: 64 GB
- GPU: NVIDIA A100 or H100 (40GB+ VRAM)
- Storage: 2 TB NVMe SSD

### Software Requirements

- Operating System: Ubuntu 20.04 LTS or later
- Python: 3.9+
- CUDA: 11.8+ (for GPU acceleration)
- Docker: 20.10+ (for containerized deployment)
- PostgreSQL: 13+ (for audit logging and patient data)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-org/computational-pathology-research.git
cd computational-pathology-research
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
pip install -r requirements-clinical.txt
```

### 3. Install System Dependencies

```bash
# Install OpenSlide for WSI processing
sudo apt-get install openslide-tools python3-openslide

# Install DICOM toolkit
sudo apt-get install dcmtk
```

## Configuration

### 1. Disease Taxonomy Configuration

Choose or create a disease taxonomy for your clinical specialty:

```bash
# Use existing taxonomy
cp configs/clinical/cancer_grading.yaml configs/clinical/active_taxonomy.yaml

# Or create custom taxonomy
nano configs/clinical/custom_taxonomy.yaml
```

See `configs/clinical/README.md` for taxonomy configuration details.

### 2. Clinical Threshold Configuration

Configure clinical decision thresholds in your taxonomy file:

```yaml
thresholds:
  disease_state_id:
    risk_threshold: 0.6        # Flag cases above this risk
    confidence_threshold: 0.85  # Minimum confidence for reporting
    anomaly_threshold: 0.7      # Flag unusual patterns
```

### 3. Model Configuration

Configure model paths and versions:

```bash
# Edit model configuration
nano configs/clinical/model_config.yaml
```

```yaml
models:
  classifier:
    path: "models/classifier_v1.0.pth"
    version: "1.0.0"
    embedding_dim: 1024
    
  risk_analyzer:
    path: "models/risk_analyzer_v1.0.pth"
    version: "1.0.0"
    time_horizons: ["1_year", "5_year", "10_year"]
```

## Clinical Workflow Setup

### 1. Initialize Clinical System

```python
from src.clinical.workflow import ClinicalWorkflowSystem

# Initialize system
system = ClinicalWorkflowSystem(
    taxonomy_config="configs/clinical/active_taxonomy.yaml",
    model_config="configs/clinical/model_config.yaml",
    enable_audit_logging=True,
    enable_privacy_controls=True
)
```

### 2. Configure User Roles

Set up role-based access control (RBAC):

```python
from src.clinical.privacy import PrivacyManager

privacy_manager = PrivacyManager()

# Define roles
privacy_manager.add_role("pathologist", permissions=[
    "view_patient_data",
    "view_predictions",
    "generate_reports",
    "amend_reports"
])

privacy_manager.add_role("technician", permissions=[
    "upload_images",
    "view_predictions"
])

privacy_manager.add_role("administrator", permissions=[
    "all"
])
```

### 3. Set Up Audit Logging

Configure audit logging for regulatory compliance:

```python
from src.clinical.audit import AuditLogger

audit_logger = AuditLogger(
    storage_path="audit_logs",
    retention_years=7,  # FDA requirement
    enable_encryption=True
)
```

## Integration with Clinical Systems

### DICOM Integration

Configure DICOM adapter for PACS integration:

```python
from src.clinical.dicom_adapter import DICOMAdapter

dicom_adapter = DICOMAdapter(
    pacs_host="pacs.hospital.org",
    pacs_port=11112,
    ae_title="PATHOLOGY_AI"
)

# Read WSI from DICOM
wsi_data, metadata = dicom_adapter.read_wsi("path/to/dicom/file.dcm")

# Write results to DICOM SR
dicom_adapter.write_structured_report(
    prediction_result,
    metadata,
    output_path="results/report.dcm"
)
```

### HL7 FHIR Integration

Configure FHIR adapter for EHR integration:

```python
from src.clinical.fhir_adapter import FHIRAdapter

fhir_adapter = FHIRAdapter(
    fhir_server_url="https://fhir.hospital.org",
    auth_method="oauth2",
    client_id="pathology_ai_client",
    client_secret="your_secret"
)

# Read patient metadata
patient_metadata = fhir_adapter.read_patient_metadata("Patient/12345")

# Write diagnostic report
fhir_adapter.write_diagnostic_report(
    patient_id="Patient/12345",
    prediction_result=result,
    imaging_study_id="ImagingStudy/67890"
)
```

## Security and Compliance

### 1. Enable Encryption

```python
from src.clinical.privacy import PrivacyManager

privacy_manager = PrivacyManager(
    encryption_key_path="/secure/keys/encryption.key",
    enable_at_rest_encryption=True,
    enable_in_transit_encryption=True
)
```

### 2. Configure Session Timeout

```python
privacy_manager.set_session_timeout(minutes=15)
```

### 3. Enable Audit Trail Signatures

```python
audit_logger.enable_cryptographic_signatures(
    signing_key_path="/secure/keys/signing.key"
)
```

### 4. HIPAA Compliance Checklist

- ✓ Patient data encrypted at rest (AES-256)
- ✓ Patient data encrypted in transit (TLS 1.3+)
- ✓ Patient identifiers anonymized in logs
- ✓ Role-based access controls enforced
- ✓ Audit logging enabled (7-year retention)
- ✓ Session timeout configured
- ✓ Data deletion support (right to be forgotten)

## Monitoring and Maintenance

### 1. Performance Monitoring

```python
from src.clinical.validation import ModelValidator, PerformanceMonitor

# Initialize performance monitor
monitor = PerformanceMonitor(
    validator=model_validator,
    check_interval_hours=24,
    alert_on_degradation=True
)

# Start monitoring
monitor.start_monitoring()
```

### 2. Model Validation

Run periodic validation to detect performance degradation:

```bash
# Run validation script
python scripts/validate_clinical_model.py \
    --model-path models/classifier_v1.0.pth \
    --validation-data data/validation/ \
    --output-report validation_report.json
```

### 3. Backup and Recovery

```bash
# Backup audit logs
python scripts/backup_audit_logs.py \
    --source audit_logs/ \
    --destination /backup/audit_logs/ \
    --verify-integrity

# Backup patient timelines
python scripts/backup_patient_data.py \
    --source patient_timelines/ \
    --destination /backup/patient_timelines/ \
    --encrypt
```

## Troubleshooting

### Common Issues

#### Issue: Inference too slow (>5 seconds)

**Solution:**
1. Check GPU utilization: `nvidia-smi`
2. Enable batch processing
3. Optimize patch processing parameters
4. Consider upgrading GPU hardware

#### Issue: High false positive rate

**Solution:**
1. Review clinical decision thresholds
2. Recalibrate uncertainty quantifier
3. Retrain model with more diverse data
4. Enable OOD detection

#### Issue: DICOM integration failures

**Solution:**
1. Verify PACS connectivity: `ping pacs.hospital.org`
2. Check DICOM transfer syntax support
3. Validate DICOM file integrity
4. Review PACS logs for errors

#### Issue: FHIR authentication failures

**Solution:**
1. Verify OAuth2 credentials
2. Check token expiration
3. Ensure SMART on FHIR scope permissions
4. Review FHIR server logs

### Logs and Diagnostics

```bash
# View system logs
tail -f logs/clinical_system.log

# View audit logs
python scripts/view_audit_logs.py --date 2026-04-12

# Run diagnostics
python scripts/run_diagnostics.py --full
```

### Support

For technical support:
- Email: support@your-org.com
- Documentation: https://docs.your-org.com
- Issue tracker: https://github.com/your-org/computational-pathology-research/issues

## Regulatory Compliance

### FDA Submission Support

The system includes regulatory compliance infrastructure:

```python
from src.clinical.regulatory import RegulatoryComplianceManager

compliance_manager = RegulatoryComplianceManager()

# Initialize device compliance
dmr = compliance_manager.initialize_device_compliance(
    device_name="Pathology AI System",
    device_version="1.0.0",
    manufacturer="Your Organization",
    intended_use="Computer-aided diagnosis for histopathology",
    indications_for_use="Classification of tissue samples",
    regulatory_standards=[
        RegulatoryStandard.FDA_510K,
        RegulatoryStandard.ISO_14971
    ]
)

# Generate submission package
package_path = compliance_manager.generate_regulatory_submission_package(
    device_name="Pathology AI System",
    device_version="1.0.0",
    submission_type="510k",
    output_path="regulatory_submission"
)
```

See `docs/regulatory_compliance.md` for detailed regulatory documentation.

## Next Steps

1. Complete system configuration
2. Run validation tests
3. Perform user acceptance testing (UAT)
4. Train clinical staff
5. Deploy to production environment
6. Monitor performance and collect feedback
7. Iterate and improve

For detailed examples, see:
- `examples/clinical_inference.py`
- `examples/longitudinal_analysis.py`
- `examples/regulatory_compliance_example.py`
