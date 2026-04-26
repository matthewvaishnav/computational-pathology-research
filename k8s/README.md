# HistoCore Kubernetes Deployment

Production-ready Kubernetes manifests for real-time WSI streaming with GPU support.

## Quick Start

```bash
# Deploy everything
./deploy.sh

# Port forward for local access
kubectl port-forward -n histocore svc/histocore-streaming 8000:8000
```

## Prerequisites

- Kubernetes 1.20+
- GPU nodes with NVIDIA drivers
- NVIDIA Device Plugin
- Ingress controller (nginx)
- Storage class (fast-ssd recommended)

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Ingress       │    │   Streaming     │    │     Redis       │
│   (nginx)       │───▶│   (2+ pods)     │───▶│   (cache)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Prometheus    │    │    Grafana      │
                       │  (monitoring)   │───▶│  (dashboards)   │
                       └─────────────────┘    └─────────────────┘
```

## Components

### Core Services
- **Streaming**: Main application (2+ replicas)
- **Redis**: Cache and session storage
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

### GPU Support
- Node selector: `accelerator=nvidia-tesla-k80`
- Resource requests: `nvidia.com/gpu: 1`
- CUDA 12.1 runtime

### Storage
- **Data**: 100Gi (ReadWriteMany)
- **Logs**: 20Gi (ReadWriteMany)  
- **Models**: 50Gi (ReadWriteMany)
- **Cache**: 5Gi (EmptyDir)

### Autoscaling
- Min replicas: 2
- Max replicas: 10
- CPU target: 70%
- Memory target: 80%

## Configuration

### Environment Variables
Edit `configmap.yaml`:
```yaml
HISTOCORE_MAX_MEMORY_GB: "8"
HISTOCORE_BATCH_SIZE: "32"
CUDA_VISIBLE_DEVICES: "0"
```

### Secrets
Edit `secret.yaml` (base64 encoded):
```bash
echo -n "your-secret" | base64
```

### GPU Nodes
Label GPU nodes:
```bash
kubectl label nodes <node-name> accelerator=nvidia-tesla-k80
```

## Deployment

### Manual Steps
```bash
# 1. Create namespace
kubectl apply -f namespace.yaml

# 2. Configuration
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml

# 3. Storage and cache
kubectl apply -f redis.yaml

# 4. Main application
kubectl apply -f streaming.yaml

# 5. Monitoring
kubectl apply -f monitoring.yaml
kubectl apply -f prometheus-config.yaml
kubectl apply -f grafana-config.yaml

# 6. External access
kubectl apply -f ingress.yaml

# 7. Autoscaling
kubectl apply -f hpa.yaml
```

### Automated
```bash
./deploy.sh
```

## Monitoring

### Grafana Dashboards
- GPU utilization per pod
- Memory usage trends
- Processing throughput
- Error rates and latency
- Cache hit rates

### Prometheus Metrics
- `histocore_processing_time_seconds`
- `histocore_memory_usage_bytes`
- `histocore_gpu_utilization_percent`
- `histocore_requests_total`
- `histocore_errors_total`

### Alerts (TODO)
- High memory usage (>90%)
- GPU utilization low (<50%)
- Processing queue backlog
- Pod restart frequency

## Scaling

### Manual Scaling
```bash
kubectl scale deployment histocore-streaming --replicas=5 -n histocore
```

### Auto Scaling
HPA configured for:
- CPU: 70% target
- Memory: 80% target
- Custom: Queue length

### Cluster Scaling
Add GPU nodes:
```bash
# GKE example
gcloud container node-pools create gpu-pool \
  --accelerator type=nvidia-tesla-k80,count=1 \
  --zone us-central1-a \
  --cluster histocore-cluster
```

## Security

### RBAC (TODO)
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: histocore-role
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list"]
```

### Network Policies (TODO)
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: histocore-netpol
spec:
  podSelector:
    matchLabels:
      app: histocore-streaming
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
```

### Pod Security Standards
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  allowPrivilegeEscalation: false
```

## Troubleshooting

### Pod Issues
```bash
# Check pod status
kubectl get pods -n histocore

# Check logs
kubectl logs -f deployment/histocore-streaming -n histocore

# Describe pod
kubectl describe pod <pod-name> -n histocore
```

### GPU Issues
```bash
# Check GPU nodes
kubectl get nodes -l accelerator=nvidia-tesla-k80

# Check device plugin
kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system

# Test GPU in pod
kubectl exec -it <pod-name> -n histocore -- nvidia-smi
```

### Storage Issues
```bash
# Check PVCs
kubectl get pvc -n histocore

# Check storage class
kubectl get storageclass

# Check volume mounts
kubectl describe pod <pod-name> -n histocore
```

### Network Issues
```bash
# Check services
kubectl get svc -n histocore

# Check ingress
kubectl get ingress -n histocore

# Test internal connectivity
kubectl exec -it <pod-name> -n histocore -- curl http://histocore-redis:6379
```

## Backup and Recovery

### Data Backup
```bash
# Backup Redis
kubectl exec histocore-redis-xxx -n histocore -- redis-cli BGSAVE

# Backup persistent volumes
kubectl get pv
# Use volume snapshots or backup tools
```

### Disaster Recovery
```bash
# Restore from backup
kubectl apply -f backup-restore.yaml

# Scale up from zero
kubectl scale deployment histocore-streaming --replicas=2 -n histocore
```

## Performance Tuning

### Resource Limits
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "1000m"
    nvidia.com/gpu: 1
  limits:
    memory: "8Gi"
    cpu: "2000m"
    nvidia.com/gpu: 1
```

### Node Affinity
```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: node-type
          operator: In
          values:
          - gpu-optimized
```

### Pod Disruption Budget
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: histocore-pdb
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: histocore-streaming
```