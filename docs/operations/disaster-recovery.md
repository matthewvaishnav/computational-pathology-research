# HistoCore Disaster Recovery Plan

## Overview

Comprehensive disaster recovery procedures for HistoCore streaming system.

## Recovery Time Objectives (RTO)

- **Critical Services**: 1 hour
- **Processing Pipeline**: 2 hours
- **Full System**: 4 hours

## Recovery Point Objectives (RPO)

- **Configuration**: 0 (version controlled)
- **Models**: 0 (immutable artifacts)
- **Metrics**: 15 minutes
- **Processing State**: 5 minutes

## Backup Strategy

### Automated Backups

```bash
# Daily full backup
0 2 * * * /app/scripts/histocore-backup.py backup --compress

# Hourly incremental (configs only)
0 * * * * /app/scripts/histocore-backup.py backup -c configs

# Weekly model backup
0 3 * * 0 /app/scripts/histocore-backup.py backup -c models
```

### Backup Components

1. **Prometheus Data** (15-day retention)
2. **Grafana Dashboards** (version controlled + backup)
3. **ML Models** (versioned artifacts)
4. **Configuration Files** (git + backup)
5. **Processing State** (Redis snapshots)

### Backup Locations

- **Primary**: Local disk (`/var/backups/histocore`)
- **Secondary**: S3/Azure Blob (encrypted)
- **Tertiary**: Off-site cold storage

## Disaster Scenarios

### Scenario 1: Single Node Failure

**Detection**: Health checks fail, node unreachable

**Recovery**:
```bash
# 1. Verify node status
kubectl get nodes

# 2. Drain node
kubectl drain <node-name> --ignore-daemonsets

# 3. Pods auto-reschedule to healthy nodes
kubectl get pods -n histocore -o wide

# 4. Replace/repair node
# 5. Uncordon node
kubectl uncordon <node-name>
```

**RTO**: 15 minutes  
**RPO**: 0 (stateless pods)

### Scenario 2: GPU Node Failure

**Detection**: GPU metrics missing, OOM events spike

**Recovery**:
```bash
# 1. Check GPU node status
kubectl describe node <gpu-node>

# 2. Drain GPU node
kubectl drain <gpu-node> --ignore-daemonsets --delete-emptydir-data

# 3. Verify GPU pods rescheduled
kubectl get pods -n histocore -l node-type=gpu

# 4. If no spare GPU capacity, scale up
kubectl scale deployment histocore-streaming --replicas=<N>
```

**RTO**: 30 minutes  
**RPO**: Current slide processing lost

### Scenario 3: Database/Redis Failure

**Detection**: Cache miss rate 100%, connection errors

**Recovery**:
```bash
# 1. Check Redis status
kubectl get pods -n histocore -l app=redis

# 2. If pod failed, delete to trigger restart
kubectl delete pod <redis-pod> -n histocore

# 3. If persistent volume corrupted, restore from snapshot
kubectl apply -f k8s/redis-restore.yaml

# 4. Verify cache operational
redis-cli -h <redis-host> ping
```

**RTO**: 10 minutes  
**RPO**: 5 minutes (last snapshot)

### Scenario 4: Complete Cluster Failure

**Detection**: All services down, cluster unreachable

**Recovery**:
```bash
# 1. Provision new cluster
terraform apply -var="environment=disaster-recovery"

# 2. Restore from backup
./scripts/histocore-backup.py restore /backups/latest.tar.gz

# 3. Deploy application
kubectl apply -f k8s/

# 4. Verify services
kubectl get pods -n histocore
./scripts/histocore-admin.py health status

# 5. Update DNS/load balancer
# 6. Verify processing
./scripts/histocore-admin.py processing submit test-slide.svs --wait
```

**RTO**: 4 hours  
**RPO**: 15 minutes (metrics), 0 (configs/models)

### Scenario 5: Data Corruption

**Detection**: Processing errors, invalid results

**Recovery**:
```bash
# 1. Stop processing
kubectl scale deployment histocore-streaming --replicas=0

# 2. Identify corrupted data
./scripts/histocore-admin.py logs tail -n 1000 -l ERROR

# 3. Restore from backup
./scripts/histocore-backup.py restore /backups/pre-corruption.tar.gz -c models

# 4. Verify model integrity
python -c "import torch; torch.load('models/histocore.pth')"

# 5. Resume processing
kubectl scale deployment histocore-streaming --replicas=2
```

**RTO**: 1 hour  
**RPO**: Last known good backup

### Scenario 6: Security Breach

**Detection**: Unauthorized access, suspicious activity

**Recovery**:
```bash
# 1. IMMEDIATE: Isolate system
kubectl delete ingress histocore-ingress -n histocore

# 2. Audit logs
kubectl logs -n histocore -l app=histocore --since=24h > audit.log

# 3. Rotate credentials
kubectl delete secret histocore-secrets -n histocore
kubectl create secret generic histocore-secrets --from-env-file=.env.new

# 4. Rebuild from clean images
kubectl set image deployment/histocore-streaming histocore=histocore:clean

# 5. Restore from pre-breach backup
./scripts/histocore-backup.py restore /backups/pre-breach.tar.gz

# 6. Security scan
trivy image histocore:latest

# 7. Restore access with new credentials
kubectl apply -f k8s/ingress.yaml
```

**RTO**: 2 hours  
**RPO**: Pre-breach state

## Recovery Procedures

### Full System Restore

```bash
#!/bin/bash
# Full disaster recovery script

set -e

BACKUP_FILE=$1
ENVIRONMENT=${2:-"production"}

echo "Starting disaster recovery..."

# 1. Provision infrastructure
cd terraform
terraform init
terraform apply -var="environment=$ENVIRONMENT" -auto-approve

# 2. Configure kubectl
aws eks update-kubeconfig --name histocore-$ENVIRONMENT

# 3. Install NVIDIA plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# 4. Restore from backup
cd ../
./scripts/histocore-backup.py restore $BACKUP_FILE

# 5. Deploy application
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/streaming.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/monitoring.yaml

# 6. Wait for ready
kubectl wait --for=condition=available --timeout=600s deployment/histocore-streaming -n histocore

# 7. Verify health
./scripts/histocore-admin.py health status

# 8. Run smoke test
./scripts/histocore-admin.py processing submit test-data/sample.svs --wait

echo "Disaster recovery complete!"
```

### Partial Component Restore

```bash
# Restore specific component
./scripts/histocore-backup.py restore backup.tar.gz -c prometheus

# Restart affected services
kubectl rollout restart deployment/prometheus -n histocore
```

## Testing

### Monthly DR Drill

```bash
# 1. Schedule maintenance window
# 2. Take snapshot of current state
# 3. Simulate failure
kubectl delete namespace histocore

# 4. Execute recovery
./scripts/disaster-recovery.sh /backups/latest.tar.gz dr-test

# 5. Verify functionality
./scripts/histocore-admin.py health status
./scripts/histocore-admin.py processing submit test.svs --wait

# 6. Document results
# 7. Cleanup DR environment
terraform destroy -var="environment=dr-test"
```

### Quarterly Full DR Test

- Complete cluster rebuild
- Cross-region failover
- Team coordination exercise
- Documentation update

## Monitoring

### Backup Health

```promql
# Backup age
time() - histocore_last_backup_timestamp > 86400

# Backup size trend
rate(histocore_backup_size_bytes[7d])

# Backup failures
rate(histocore_backup_failures_total[1h]) > 0
```

### Recovery Readiness

- Backup integrity checks (weekly)
- Restore test (monthly)
- DR drill (quarterly)
- Documentation review (quarterly)

## Contacts

### Escalation Path

1. **On-Call Engineer** (immediate)
2. **Team Lead** (15 min)
3. **Engineering Manager** (30 min)
4. **CTO** (1 hour)

### External Contacts

- **Cloud Provider Support**: [support link]
- **Vendor Support**: [support link]
- **Security Team**: [contact]

## Post-Incident

### Required Actions

1. **Incident Report** (24 hours)
2. **Root Cause Analysis** (1 week)
3. **Remediation Plan** (2 weeks)
4. **DR Plan Update** (as needed)

### Metrics to Track

- Actual RTO vs target
- Actual RPO vs target
- Data loss (if any)
- Cost of downtime
- Lessons learned

## Appendix

### Backup Verification

```bash
# Verify backup integrity
./scripts/histocore-backup.py info /backups/latest.tar.gz

# Test restore (dry run)
./scripts/histocore-backup.py restore /backups/latest.tar.gz --dry-run
```

### Emergency Contacts

```yaml
oncall:
  primary: +1-XXX-XXX-XXXX
  secondary: +1-XXX-XXX-XXXX
  
escalation:
  manager: manager@example.com
  director: director@example.com
  
vendors:
  aws: aws-support-case-id
  nvidia: nvidia-support-id
```