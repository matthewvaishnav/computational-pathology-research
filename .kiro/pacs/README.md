# PACS Integration System - Configuration Guide

## Overview

The PACS Integration System provides comprehensive integration with hospital Picture Archiving and Communication Systems (PACS) for clinical deployment of HistoCore AI pathology analysis.

## Quick Start

### 1. Installation

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Configuration

Choose a configuration profile based on your environment:

- **Development**: `config.development.yaml` - Local testing with mock PACS
- **Staging**: `config.staging.yaml` - Pre-production testing
- **Production**: `config.production.yaml` - Hospital deployment

### 3. Basic Usage

```python
from src.clinical.pacs import PACSService

# Initialize service with configuration
service = PACSService(
    config_path=".kiro/pacs/config.production.yaml",
    profile="production"
)

# Start the service
service.start()

# Check health
health = service.health_check()
print(f"Service status: {health['overall_status']}")

# Get statistics
stats = service.get_statistics()

# Shutdown when done
service.shutdown()
```

### 4. Context Manager Usage

```python
from src.clinical.pacs import PACSService

with PACSService(config_path=".kiro/pacs/config.production.yaml") as service:
    # Service automatically starts
    health = service.health_check()
    # Service automatically shuts down on exit
```

## Configuration Profiles

### Development Profile

For local development and testing:
- Uses localhost PACS server
- TLS disabled
- Reduced performance limits
- Verbose logging
- Mock notifications

### Production Profile

For hospital deployment:
- Real PACS endpoints with failover
- TLS 1.3 encryption required
- HIPAA-compliant audit logging
- Full performance capabilities
- Real notification channels

## Security Configuration

### TLS Certificates

Production deployments require TLS certificates:

1. **Server Certificate**: PACS server certificate for validation
2. **Client Certificate**: HistoCore client certificate for mutual authentication
3. **CA Bundle**: Certificate Authority bundle for chain validation

Place certificates in `/etc/ssl/certs/` and private keys in `/etc/ssl/private/`:

```yaml
security:
  tls_enabled: true
  tls_version: "1.3"
  certificate_path: "/etc/ssl/certs/pacs.crt"
  client_cert_path: "/etc/ssl/private/histocore.crt"
  client_key_path: "/etc/ssl/private/histocore.key"
  ca_bundle_path: "/etc/ssl/certs/ca-bundle.crt"
  verify_certificates: true
  mutual_authentication: true
```

### Environment Variables

Sensitive credentials should be stored in environment variables:

```bash
export HISTOCORE_SMTP_PASSWORD="your_smtp_password"
export TWILIO_ACCOUNT_SID="your_twilio_sid"
export TWILIO_AUTH_TOKEN="your_twilio_token"
```

## PACS Vendor Configuration

The system supports multiple PACS vendors with vendor-specific optimizations:

### GE Healthcare PACS

```yaml
vendor: "GE"
```

### Philips IntelliSpace PACS

```yaml
vendor: "Philips"
```

### Siemens syngo PACS

```yaml
vendor: "Siemens"
```

### Agfa Enterprise Imaging

```yaml
vendor: "Agfa"
```

## Workflow Configuration

### Automated Polling

Configure automatic polling for new WSI studies:

```yaml
workflow:
  poll_interval: "5m"  # Poll every 5 minutes
  auto_start_polling: true
  priority_processing_enabled: true
  max_processing_time_minutes: 60
```

### Manual Processing

For manual control:

```python
# Query for studies
studies, result = service.pacs_adapter.query_studies(
    patient_id="12345",
    modality="SM"
)

# Process specific studies
results = service.workflow_orchestrator.process_new_studies(studies)
```

## Error Handling

### Dead Letter Queue

Failed operations are queued for retry:

```yaml
error_handling:
  dead_letter_queue_path: "/var/histocore/dlq"
  auto_retry_enabled: true
  retry_interval_minutes: 15
```

View failed operations:

```python
dlq_stats = service.dead_letter_queue.get_statistics()
print(f"Failed operations: {dlq_stats['queue_size']}")
```

### Failover

Automatic failover to backup PACS:

```yaml
failover:
  enabled: true
  health_check_interval_seconds: 60
  auto_failover_enabled: true
```

## Audit Logging

### HIPAA Compliance

Audit logs are HIPAA-compliant with:
- Tamper-evident storage (cryptographic signatures)
- 7-year retention (configurable 1-10 years)
- PHI access tracking
- Encrypted storage

```yaml
audit:
  enabled: true
  retention_days: 2555  # 7 years
  enable_encryption: true
  tamper_evident_enabled: true
```

### Searching Audit Logs

```python
# Search audit logs
results = service.audit_logger.search_logs(
    start_date="2026-01-01",
    end_date="2026-01-31",
    event_type="dicom_query"
)
```

## Notifications

### Email Notifications

```yaml
notifications:
  email:
    enabled: true
    smtp_host: "smtp.hospital.org"
    smtp_port: 587
    smtp_use_tls: true
    from_address: "histocore@hospital.org"
    pathologist_addresses:
      - "pathology-team@hospital.org"
```

### SMS Notifications

```yaml
notifications:
  sms:
    enabled: true
    provider: "twilio"
    admin_numbers:
      - "+1234567890"
```

### HL7 Messages

```yaml
notifications:
  hl7:
    enabled: true
    host: "hl7.hospital.org"
    port: 2575
    facility: "HISTOCORE"
```

## Monitoring

### Health Checks

```python
health = service.health_check()
print(f"Overall status: {health['overall_status']}")
print(f"PACS endpoints: {health['components']['pacs_endpoints']['status']}")
print(f"Workflow: {health['components']['workflow_orchestrator']['status']}")
```

### Statistics

```python
stats = service.get_statistics()
print(f"Studies processed: {stats['workflow']['studies_processed']}")
print(f"Studies failed: {stats['workflow']['studies_failed']}")
print(f"Active processing: {stats['workflow']['active_processing']}")
```

## Troubleshooting

### Connection Issues

1. **Check PACS endpoint connectivity**:
   ```bash
   telnet pacs.hospital.org 11112
   ```

2. **Verify TLS certificates**:
   ```bash
   openssl s_client -connect pacs.hospital.org:11112 -cert /etc/ssl/private/histocore.crt -key /etc/ssl/private/histocore.key
   ```

3. **Check firewall rules**:
   - Ensure port 11112 (DICOM) is open
   - Verify VPN/network access

### Performance Issues

1. **Increase concurrent processing**:
   ```yaml
   performance:
     max_concurrent_studies: 100
     connection_pool_size: 20
   ```

2. **Check disk space**:
   ```bash
   df -h /var/histocore/pacs_cache
   ```

3. **Monitor transfer rates**:
   ```python
   stats = service.get_statistics()
   print(stats['workflow']['statistics'])
   ```

### Audit Log Issues

1. **Check log directory permissions**:
   ```bash
   ls -la /var/log/histocore/pacs_audit
   ```

2. **Verify encryption key**:
   ```bash
   ls -la /etc/histocore/audit_encryption.key
   ```

3. **Check retention settings**:
   ```python
   audit_stats = service.audit_logger.get_statistics()
   print(f"Total logs: {audit_stats['total_logs']}")
   ```

## Support

For issues or questions:
- Check logs: `/var/log/histocore/pacs_service.log`
- Review audit logs: `/var/log/histocore/pacs_audit/`
- Contact: pacs-admin@hospital.org
