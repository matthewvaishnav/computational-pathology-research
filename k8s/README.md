# Kubernetes Deployment for HistoCore

Production-ready k8s configs for HistoCore deployment.

## Quick Start

```bash
# Create namespace
kubectl create namespace histocore

# Apply configs
kubectl apply -f k8s/ -n histocore

# Check status
kubectl get pods -n histocore
kubectl get svc -n histocore
```

## Components

### ConfigMap (`configmap.yaml`)
- Application settings (log level, paths, workers)
- GPU settings (CUDA devices, memory allocation)
- Inference settings (batch size, timeout)
- Rate limiting config
- Monitoring settings

### Deployment (`deployment.yaml`)
- 3 replicas for HA
- GPU support (1 GPU per pod)
- Health checks (liveness + readiness)
- Resource limits (4Gi RAM, 2 CPU, 1 GPU)
- Volume mounts for models + cache
- ConfigMap integration

### Service (`service.yaml`)
- LoadBalancer for external access
- ClusterIP for metrics (Prometheus)
- Port 80 → 8000

### HPA (`hpa.yaml`)
- Auto-scaling: 2-10 replicas
- CPU target: 70%
- Memory target: 80%
- Custom metric: active requests
- Scale-up: fast (100% every 30s)
- Scale-down: slow (50% every 60s, 5min stabilization)

### PVC (`pvc.yaml`)
- Models: 50Gi (ReadOnlyMany, fast-ssd)
- Data: 500Gi (ReadWriteMany, standard)

### Ingress (`ingress.yaml`)
- NGINX ingress controller
- TLS/SSL termination
- Rate limiting (100 RPS, 50 connections)
- CORS support
- Large file uploads (1024MB)
- Timeouts (300s)

### NetworkPolicy (`networkpolicy.yaml`)
- Ingress: Allow from ingress controller, Prometheus, same namespace
- Egress: Allow DNS, HTTPS, HTTP, same namespace
- Zero-trust network security

## Prerequisites

### GPU Support
```bash
# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml
```

### Metrics Server
```bash
# Install metrics server for HPA
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### Prometheus (optional)
```bash
# Install Prometheus for custom metrics
helm install prometheus prometheus-community/kube-prometheus-stack
```

### Configuration

### ConfigMap
Edit `configmap.yaml` for application settings:
```yaml
data:
  LOG_LEVEL: "INFO"
  BATCH_SIZE: "32"
  RATE_LIMIT_PER_MINUTE: "60"
```

### Environment Variables
ConfigMap automatically loaded via `envFrom` in deployment.

### Resource Limits
Edit `deployment.yaml`:
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
    nvidia.com/gpu: "1"
  limits:
    memory: "8Gi"
    cpu: "4"
    nvidia.com/gpu: "1"
```

### Scaling
Edit `hpa.yaml`:
```yaml
minReplicas: 2
maxReplicas: 10
```

## Monitoring

### Health Checks
```bash
# Liveness probe
curl http://<service-ip>/health/live

# Readiness probe
curl http://<service-ip>/health/ready

# Detailed health
curl http://<service-ip>/health/detailed
```

### Metrics
```bash
# Prometheus metrics
curl http://<service-ip>/metrics
```

### Logs
```bash
# View logs
kubectl logs -f deployment/histocore -n histocore

# View logs from specific pod
kubectl logs -f <pod-name> -n histocore
```

## Troubleshooting

### Pod not starting
```bash
# Check events
kubectl describe pod <pod-name> -n histocore

# Check logs
kubectl logs <pod-name> -n histocore
```

### GPU not available
```bash
# Check GPU nodes
kubectl get nodes -l accelerator=nvidia-gpu

# Check device plugin
kubectl get pods -n kube-system | grep nvidia
```

### OOM errors
```bash
# Increase memory limits in deployment.yaml
resources:
  limits:
    memory: "16Gi"
```

### Slow scaling
```bash
# Check HPA status
kubectl get hpa -n histocore

# Check metrics
kubectl top pods -n histocore
```

## Production Checklist

- [ ] GPU nodes available
- [ ] Metrics server installed
- [ ] Prometheus installed (for custom metrics)
- [ ] NGINX ingress controller installed
- [ ] cert-manager installed (for TLS)
- [ ] PVCs created and bound
- [ ] Models uploaded to PVC
- [ ] ConfigMap values configured
- [ ] Ingress hostname configured
- [ ] TLS certificates issued
- [ ] Resource limits tuned
- [ ] HPA thresholds configured
- [ ] NetworkPolicy rules reviewed
- [ ] Health checks passing
- [ ] Monitoring dashboards setup
- [ ] Alerts configured
- [ ] Backup strategy in place
- [ ] Disaster recovery tested

## Security

### Ingress TLS
```yaml
# Edit ingress.yaml
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: histocore-tls
```

### NetworkPolicy
Network policies enforce zero-trust security:
- Ingress: Only from ingress controller, Prometheus, same namespace
- Egress: Only DNS, HTTPS, HTTP, same namespace

Edit `networkpolicy.yaml` to customize rules.

### RBAC
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: histocore
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: histocore
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
```

## Updates

### Rolling Update
```bash
# Update image
kubectl set image deployment/histocore histocore=histocore:v2 -n histocore

# Check rollout status
kubectl rollout status deployment/histocore -n histocore

# Rollback if needed
kubectl rollout undo deployment/histocore -n histocore
```

### Blue-Green Deployment
```bash
# Deploy new version
kubectl apply -f k8s/deployment-v2.yaml -n histocore

# Switch traffic
kubectl patch service histocore -p '{"spec":{"selector":{"version":"v2"}}}' -n histocore

# Remove old version
kubectl delete deployment histocore-v1 -n histocore
```
