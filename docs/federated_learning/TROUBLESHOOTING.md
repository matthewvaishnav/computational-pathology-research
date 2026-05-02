# Federated Learning Troubleshooting Guide

## Connection Issues

### TLS Certificate Verification Failed

**Symptom:**
```
ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

**Causes:**
- Certificate hostname mismatch
- Expired certificate
- Self-signed cert not trusted

**Solutions:**

1. **Regenerate certificates with correct hostname:**
```bash
python -m src.federated.communication.tls_utils generate \
    --output-dir ./certs \
    --coordinator-host <actual-ip-or-hostname>
```

2. **Verify certificate details:**
```bash
openssl x509 -in certs/coordinator-cert.pem -text -noout | grep "Subject:"
```

3. **Check certificate expiry:**
```bash
openssl x509 -in certs/coordinator-cert.pem -noout -dates
```

4. **For development only - disable verification (NOT for production):**
```yaml
# client.yaml
coordinator:
  verify_ssl: false  # INSECURE - dev only
```

### Client Cannot Connect to Coordinator

**Symptom:**
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Causes:**
- Coordinator not running
- Firewall blocking ports
- Wrong coordinator URL

**Solutions:**

1. **Verify coordinator is running:**
```bash
curl https://coordinator.example.com:8080/health
```

2. **Check firewall rules:**
```bash
# Linux
sudo ufw status
sudo ufw allow 8080/tcp

# Windows
netsh advfirewall firewall add rule name="FL Coordinator" dir=in action=allow protocol=TCP localport=8080
```

3. **Verify coordinator URL in client config:**
```yaml
# client.yaml
coordinator:
  url: "https://<correct-ip>:8080"  # Check IP/hostname
```

4. **Test network connectivity:**
```bash
telnet coordinator.example.com 8080
# Or
nc -zv coordinator.example.com 8080
```

### Mutual Authentication Failed

**Symptom:**
```
ssl.SSLError: [SSL] PEM lib (_ssl.c:4067)
```

**Causes:**
- Missing client certificate
- Client cert not signed by coordinator CA

**Solutions:**

1. **Generate client certificate:**
```bash
python -m src.federated.communication.tls_utils generate_client_cert \
    --output-dir ./certs \
    --client-id hospital-001 \
    --ca-cert ./certs/coordinator-cert.pem \
    --ca-key ./certs/coordinator-key.pem
```

2. **Verify client cert in config:**
```yaml
# client.yaml
coordinator:
  client_cert: "./certs/client-cert.pem"
  client_key: "./certs/client-key.pem"
```

## Training Issues

### Privacy Budget Exhausted

**Symptom:**
```
PrivacyBudgetExhausted: Cumulative epsilon (1.05) exceeds target (1.0)
```

**Causes:**
- Too many training rounds
- Noise multiplier too low
- Target epsilon too strict

**Solutions:**

1. **Increase target epsilon (less privacy, more utility):**
```yaml
# coordinator.yaml
privacy:
  target_epsilon: 2.0  # Increase from 1.0
```

2. **Increase noise multiplier (more privacy, slower convergence):**
```yaml
privacy:
  noise_multiplier: 1.5  # Increase from 1.1
```

3. **Reduce number of rounds:**
```yaml
training:
  num_rounds: 50  # Reduce from 100
```

4. **Reset privacy budget for new training phase:**
```python
privacy_engine.reset_privacy_budget()
```

### Byzantine Attack Detected

**Symptom:**
```
ByzantineAttackDetected: Client hospital-003 flagged in 3 consecutive rounds
```

**Causes:**
- Malicious client
- Client hardware failure
- Data quality issues

**Solutions:**

1. **Investigate flagged client:**
```bash
# Check client logs
tail -f logs/client/hospital-003.log

# Check client metrics
python -m src.federated.monitoring.client_diagnostics \
    --client-id hospital-003
```

2. **Adjust Byzantine threshold:**
```yaml
# coordinator.yaml
byzantine:
  threshold_std: 4.0  # Increase from 3.0 (less sensitive)
```

3. **Temporarily exclude client:**
```bash
python -m src.federated.coordinator.exclude_client \
    --client-id hospital-003 \
    --reason "Under investigation"
```

4. **Switch Byzantine detection algorithm:**
```yaml
byzantine:
  algorithm: "trimmed_mean"  # Try different algorithm
```

### Training Not Converging

**Symptom:**
- Loss not decreasing after 20+ rounds
- Accuracy stuck at random baseline

**Causes:**
- Learning rate too high/low
- Data heterogeneity across clients
- Privacy noise too high

**Solutions:**

1. **Adjust learning rate:**
```yaml
# client.yaml
training:
  learning_rate: 0.0001  # Reduce from 0.001
```

2. **Use FedProx for heterogeneous data:**
```yaml
# coordinator.yaml
aggregation:
  algorithm: "fedprox"
  fedprox_mu: 0.01
```

3. **Reduce privacy noise:**
```yaml
privacy:
  noise_multiplier: 0.8  # Reduce from 1.1
```

4. **Increase local epochs:**
```yaml
# client.yaml
training:
  local_epochs: 10  # Increase from 5
```

5. **Check data quality:**
```bash
python -m src.federated.client.data_diagnostics \
    --config configs/federated/client.yaml
```

### Client Timeout

**Symptom:**
```
ClientDropoutError: Client hospital-002 exceeded timeout (600s)
```

**Causes:**
- Slow network
- Insufficient client resources
- Large model/dataset

**Solutions:**

1. **Increase timeout:**
```yaml
# coordinator.yaml
training:
  client_timeout_seconds: 1800  # 30 minutes
```

2. **Enable gradient compression:**
```yaml
# client.yaml
compression:
  enable: true
  mode: "quantization"
  quantization_bits: 8
```

3. **Reduce batch size:**
```yaml
# client.yaml
training:
  batch_size: 16  # Reduce from 32
```

4. **Use async training:**
```yaml
# coordinator.yaml
training:
  sync_mode: "semi_synchronous"
  min_client_percentage: 0.8
```

## Resource Issues

### CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. **Reduce batch size:**
```yaml
# client.yaml
training:
  batch_size: 8  # Reduce from 32
  gradient_accumulation_steps: 4  # Compensate
```

2. **Enable mixed precision:**
```yaml
training:
  mixed_precision: true  # FP16
```

3. **Set GPU memory limit:**
```yaml
resources:
  max_gpu_memory_gb: 6  # Reduce from 8
```

4. **Clear GPU cache:**
```python
import torch
torch.cuda.empty_cache()
```

5. **Use CPU training (slower):**
```yaml
training:
  device: "cpu"
```

### Disk Space Exhausted

**Symptom:**
```
OSError: [Errno 28] No space left on device
```

**Solutions:**

1. **Clean old checkpoints:**
```bash
python -m src.federated.model_registry.cleanup \
    --keep-last 5
```

2. **Reduce checkpoint frequency:**
```yaml
# coordinator.yaml
model_registry:
  save_interval: 5  # Save every 5 rounds instead of 1
```

3. **Clean PACS cache:**
```bash
python -m src.federated.client.pacs_cache_cleanup \
    --max-age-days 7
```

4. **Increase disk limit:**
```yaml
# client.yaml
resources:
  max_disk_space_gb: 200  # Increase from 100
```

### High CPU Usage

**Symptom:**
- System unresponsive during training
- Other services impacted

**Solutions:**

1. **Limit CPU cores:**
```yaml
# client.yaml
resources:
  max_cpu_cores: 2  # Reduce from 4
```

2. **Reduce data loader workers:**
```yaml
training:
  num_workers: 2  # Reduce from 4
```

3. **Schedule training during off-hours:**
```yaml
resources:
  enable_scheduling: true
  training_windows:
    - start: "22:00"
      end: "06:00"
      days: ["monday", "tuesday", "wednesday", "thursday", "friday"]
```

## PACS Integration Issues

### PACS Connection Timeout

**Symptom:**
```
PACSConnectionError: Connection to pacs.hospital.local:11112 timed out
```

**Solutions:**

1. **Verify PACS connectivity:**
```bash
python -m src.clinical.pacs.pacs_service test \
    --host pacs.hospital.local \
    --port 11112 \
    --aet HISTOCORE
```

2. **Check PACS firewall:**
```bash
telnet pacs.hospital.local 11112
```

3. **Increase connection timeout:**
```yaml
# client.yaml
pacs:
  connection_timeout: 60  # Increase from 30
```

4. **Verify AE titles:**
```yaml
pacs:
  aet: "HISTOCORE"  # Your AE title
  calling_aet: "FL_CLIENT"  # PACS expects this
```

### No WSI Studies Found

**Symptom:**
```
PACSDataError: No studies found matching criteria
```

**Solutions:**

1. **Expand date range:**
```yaml
# client.yaml
pacs:
  date_range_days: 90  # Increase from 30
```

2. **Verify modality filter:**
```yaml
pacs:
  modality: "SM"  # Slide Microscopy
```

3. **Test PACS query manually:**
```bash
python -m src.clinical.pacs.query_studies \
    --host pacs.hospital.local \
    --port 11112 \
    --modality SM \
    --date-range 90
```

4. **Check PACS logs for errors:**
```bash
# Check PACS server logs for rejected queries
```

### DICOM Retrieval Failed

**Symptom:**
```
PACSRetrievalError: C-MOVE operation failed for study 1.2.3.4.5
```

**Solutions:**

1. **Verify storage permissions:**
```bash
# Check write permissions
ls -la data/pacs_cache/
```

2. **Increase max studies:**
```yaml
pacs:
  max_studies: 2000  # Increase from 1000
```

3. **Retry failed retrievals:**
```bash
python -m src.federated.client.retry_pacs_retrieval \
    --study-id 1.2.3.4.5
```

## Monitoring Issues

### Prometheus Metrics Not Appearing

**Symptom:**
- Prometheus dashboard empty
- No metrics at `/metrics` endpoint

**Solutions:**

1. **Verify Prometheus enabled:**
```yaml
# coordinator.yaml
monitoring:
  enable_prometheus: true
  prometheus_port: 9090
```

2. **Check metrics endpoint:**
```bash
curl http://localhost:9090/metrics
```

3. **Verify Prometheus scrape config:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'federated_coordinator'
    static_configs:
      - targets: ['localhost:9090']
```

4. **Restart monitoring:**
```bash
python -m src.federated.monitoring.restart
```

### TensorBoard Not Showing Logs

**Symptom:**
- TensorBoard dashboard empty
- No training curves

**Solutions:**

1. **Verify TensorBoard enabled:**
```yaml
monitoring:
  enable_tensorboard: true
  tensorboard_dir: "./runs"
```

2. **Check log directory:**
```bash
ls -la runs/
```

3. **Start TensorBoard manually:**
```bash
tensorboard --logdir=./runs --port=6006
```

4. **Verify logging calls:**
```python
monitor.log_round_metrics(round_id, metrics)
```

## Performance Issues

### Slow Training Rounds

**Symptom:**
- Each round takes >10 minutes
- Expected: 2-3 minutes per round

**Solutions:**

1. **Enable gradient compression:**
```yaml
compression:
  enable: true
  mode: "sparsification"
  sparsification_ratio: 0.1
```

2. **Use async training:**
```yaml
training:
  sync_mode: "semi_synchronous"
  min_client_percentage: 0.8
```

3. **Profile bottlenecks:**
```bash
python -m src.federated.profiler \
    --config configs/federated/coordinator.yaml \
    --rounds 5
```

4. **Reduce local epochs:**
```yaml
training:
  local_epochs: 3  # Reduce from 5
```

### High Network Bandwidth Usage

**Symptom:**
- Network saturated during training
- Other services impacted

**Solutions:**

1. **Enable aggressive compression:**
```yaml
compression:
  mode: "quantization"
  quantization_bits: 4  # Reduce from 8
```

2. **Reduce update frequency:**
```yaml
training:
  local_epochs: 10  # More local training, fewer updates
```

3. **Use sparsification:**
```yaml
compression:
  mode: "sparsification"
  sparsification_ratio: 0.01  # Top 1% only
```

## Debugging Tools

### Enable Debug Logging

```yaml
# coordinator.yaml / client.yaml
logging:
  level: "DEBUG"
```

### Collect Diagnostics

```bash
# Coordinator diagnostics
python -m src.federated.coordinator.diagnostics \
    --output diagnostics_coordinator.json

# Client diagnostics
python -m src.federated.client.diagnostics \
    --output diagnostics_client.json
```

### Simulate Training Locally

```bash
# Test with 3 virtual clients
python -m src.federated.production.simulate \
    --num-clients 3 \
    --num-rounds 5 \
    --dataset synthetic
```

### Validate Configuration

```bash
# Validate coordinator config
python -m src.federated.production.config validate \
    --config configs/federated/coordinator.yaml \
    --type coordinator

# Validate client config
python -m src.federated.production.config validate \
    --config configs/federated/client.yaml \
    --type client
```

### Check System Requirements

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check disk space
df -h

# Check memory
free -h

# Check network
ping coordinator.example.com
```

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `PrivacyBudgetExhausted` | Privacy budget exceeded | Increase epsilon or reduce rounds |
| `ByzantineAttackDetected` | Malicious/faulty client | Investigate client, adjust threshold |
| `ClientDropoutError` | Client timeout | Increase timeout, enable compression |
| `AggregationError` | Incompatible updates | Verify all clients use same model |
| `CheckpointNotFoundError` | Missing checkpoint | Check checkpoint directory |
| `PACSConnectionError` | PACS unreachable | Verify network, firewall, AE titles |
| `CUDAOutOfMemoryError` | GPU memory exhausted | Reduce batch size, enable FP16 |
| `ConfigValidationError` | Invalid config | Run config validation tool |

## Getting Help

### Check Logs

```bash
# Coordinator logs
tail -f logs/coordinator/coordinator.log

# Client logs
tail -f logs/client/client.log

# Audit logs
tail -f logs/audit/audit.log
```

### Report Issues

When reporting issues, include:

1. **Error message** (full stack trace)
2. **Configuration files** (redact secrets)
3. **System info** (OS, Python version, GPU)
4. **Logs** (last 100 lines)
5. **Steps to reproduce**

```bash
# Generate issue report
python -m src.federated.support.generate_report \
    --output issue_report.zip
```

### Community Support

- GitHub Issues: https://github.com/yourusername/computational-pathology-research/issues
- Documentation: https://histocore.readthedocs.io
- Email: support@histocore.org

## Next Steps

- [Installation Guide](INSTALLATION.md) - Setup instructions
- [Configuration Guide](CONFIGURATION.md) - Config reference
- [API Reference](API_REFERENCE.md) - API documentation

