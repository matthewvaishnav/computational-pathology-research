# Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the Computational Pathology API.

## Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Docker image built and pushed to registry
- Persistent storage provisioner
- (Optional) NGINX Ingress Controller
- (Optional) cert-manager for TLS
- (Optional) Prometheus Operator for monitoring
- (Optional) NVIDIA GPU Operator for GPU nodes

## Quick Start

### 1. Create Namespace
```bash
kubectl apply -f namespace.yaml
```

### 2. Create ConfigMap and Secrets
```bash
# Edit secret.yaml with actual values first!
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
```

### 3. Create Persistent Volume Claims
```bash
kubectl apply -f pvc.yaml
```

### 4. Deploy Application
```bash
# CPU-only deployment
kubectl apply -f deployment.yaml

# Or GPU deployment
kubectl apply -f gpu-deployment.yaml
```

### 5. Create Service
```bash
kubectl apply -f service.yaml
```

### 6. (Optional) Create Ingress
```bash
# Edit ingress.yaml with your domain first!
kubectl apply -f ingress.yaml
```

### 7. (Optional) Enable Autoscaling
```bash
kubectl apply -f hpa.yaml
```

## File Descriptions

### Core Resources

#### `namespace.yaml`
Creates the `pathology` namespace for all resources.

#### `deployment.yaml`
Main CPU-based deployment with:
- 3 replicas
- Rolling update strategy
- Health checks
- Resource limits
- Anti-affinity rules

#### `gpu-deployment.yaml`
GPU-accelerated deployment with:
- NVIDIA GPU support
- 2 replicas (GPUs are expensive)
- Node selector for GPU nodes
- GPU tolerations

#### `service.yaml`
Two services:
- `pathology-api`: LoadBalancer for external access
- `pathology-api-internal`: ClusterIP for internal access

#### `ingress.yaml`
NGINX Ingress with:
- TLS/SSL termination
- Rate limiting
- Proxy timeouts
- cert-manager integration

### Configuration

#### `configmap.yaml`
Application configuration:
- Server settings
- Model configuration
- Logging configuration
- Monitoring settings

#### `secret.yaml`
Sensitive data (⚠️ **DO NOT commit real secrets!**):
- API keys
- Database credentials
- Cloud provider credentials

### Storage

#### `pvc.yaml`
Persistent Volume Claims:
- `pathology-models-pvc`: 10Gi for model files (ReadOnlyMany)
- `pathology-data-pvc`: 100Gi for data (ReadWriteMany)

### Scaling

#### `hpa.yaml`
Horizontal Pod Autoscaler:
- Min replicas: 3
- Max replicas: 10
- CPU target: 70%
- Memory target: 80%
- Smart scale-up/down policies

### Monitoring

#### `servicemonitor.yaml`
Prometheus ServiceMonitor for metrics collection.

## Deployment Scenarios

### Scenario 1: Development/Testing

```bash
# Single replica, minimal resources
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f pvc.yaml

# Edit deployment.yaml: set replicas: 1
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### Scenario 2: Production (CPU)

```bash
# Full deployment with autoscaling
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml
kubectl apply -f servicemonitor.yaml
```

### Scenario 3: Production (GPU)

```bash
# GPU-accelerated deployment
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f pvc.yaml
kubectl apply -f gpu-deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
```

### Scenario 4: Hybrid (CPU + GPU)

```bash
# Deploy both CPU and GPU versions
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f gpu-deployment.yaml

# Create separate services
kubectl apply -f service.yaml

# Route traffic based on requirements
# (requires custom ingress configuration)
```

## Configuration

### Update Docker Image

Edit `deployment.yaml` or `gpu-deployment.yaml`:
```yaml
spec:
  template:
    spec:
      containers:
      - name: api
        image: your-registry/pathology-api:v1.0.0  # Update this
```

### Update Resource Limits

Edit resource requests/limits in deployment files:
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
```

### Update Replicas

Edit `deployment.yaml`:
```yaml
spec:
  replicas: 5  # Change this
```

Or use kubectl:
```bash
kubectl scale deployment pathology-api -n pathology --replicas=5
```

### Update Environment Variables

Edit `configmap.yaml` or add to deployment:
```yaml
env:
- name: LOG_LEVEL
  value: "debug"
- name: MAX_BATCH_SIZE
  value: "64"
```

## Secrets Management

### Option 1: Kubernetes Secrets (Basic)

```bash
# Create secret from literal values
kubectl create secret generic pathology-secrets \
  --from-literal=api-key=your-key \
  --from-literal=db-password=your-password \
  -n pathology
```

### Option 2: Sealed Secrets

```bash
# Install sealed-secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/controller.yaml

# Seal your secret
kubeseal --format=yaml < secret.yaml > sealed-secret.yaml
kubectl apply -f sealed-secret.yaml
```

### Option 3: External Secrets Operator

```bash
# Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets -n external-secrets-system --create-namespace

# Configure AWS Secrets Manager backend
# See: https://external-secrets.io/latest/provider/aws-secrets-manager/
```

### Option 4: HashiCorp Vault

```bash
# Install Vault
helm repo add hashicorp https://helm.releases.hashicorp.com
helm install vault hashicorp/vault

# Configure Vault integration
# See: https://www.vaultproject.io/docs/platform/k8s
```

## Storage

### Prepare Model Files

```bash
# Copy model files to PVC
kubectl run -it --rm copy-models --image=busybox -n pathology -- sh

# Inside the pod:
# Mount the PVC and copy files
# Or use kubectl cp:
kubectl cp models/best_model.pth pathology/copy-models:/models/
```

### Using Cloud Storage

For AWS S3, GCS, or Azure Blob:

1. Update deployment to use init container:
```yaml
initContainers:
- name: download-models
  image: amazon/aws-cli
  command:
  - sh
  - -c
  - aws s3 cp s3://your-bucket/models/ /models/ --recursive
  volumeMounts:
  - name: models
    mountPath: /models
```

2. Add AWS credentials to secrets

## Networking

### Access the API

**Via LoadBalancer**:
```bash
# Get external IP
kubectl get svc pathology-api -n pathology

# Access API
curl http://<EXTERNAL-IP>/health
```

**Via Ingress**:
```bash
# Access via domain
curl https://pathology-api.example.com/health
```

**Via Port Forward** (development):
```bash
kubectl port-forward svc/pathology-api 8000:8000 -n pathology
curl http://localhost:8000/health
```

### Internal Access

From within the cluster:
```bash
# Use internal service
curl http://pathology-api-internal.pathology.svc.cluster.local:8000/health
```

## Monitoring

### View Logs

```bash
# All pods
kubectl logs -f -l app=pathology-api -n pathology

# Specific pod
kubectl logs -f pathology-api-<pod-id> -n pathology

# Previous container (if crashed)
kubectl logs --previous pathology-api-<pod-id> -n pathology
```

### Check Pod Status

```bash
# List pods
kubectl get pods -n pathology

# Describe pod
kubectl describe pod pathology-api-<pod-id> -n pathology

# Get events
kubectl get events -n pathology --sort-by='.lastTimestamp'
```

### Check Resource Usage

```bash
# Pod resource usage
kubectl top pods -n pathology

# Node resource usage
kubectl top nodes
```

### Prometheus Metrics

If ServiceMonitor is deployed:
```bash
# Access Prometheus
kubectl port-forward -n monitoring svc/prometheus-k8s 9090:9090

# View metrics at http://localhost:9090
```

## Scaling

### Manual Scaling

```bash
# Scale deployment
kubectl scale deployment pathology-api -n pathology --replicas=5

# Check status
kubectl get deployment pathology-api -n pathology
```

### Autoscaling

```bash
# Check HPA status
kubectl get hpa -n pathology

# Describe HPA
kubectl describe hpa pathology-api-hpa -n pathology

# Watch autoscaling
kubectl get hpa -n pathology --watch
```

## Updates and Rollouts

### Update Image

```bash
# Update image
kubectl set image deployment/pathology-api api=pathology-api:v2.0.0 -n pathology

# Check rollout status
kubectl rollout status deployment/pathology-api -n pathology

# View rollout history
kubectl rollout history deployment/pathology-api -n pathology
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/pathology-api -n pathology

# Rollback to specific revision
kubectl rollout undo deployment/pathology-api --to-revision=2 -n pathology
```

### Zero-Downtime Updates

The deployment uses `RollingUpdate` strategy with:
- `maxSurge: 1` - One extra pod during update
- `maxUnavailable: 0` - No downtime

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n pathology

# Common issues:
# 1. Image pull error - check image name and registry access
# 2. Resource limits - check node capacity
# 3. Volume mount error - check PVC status
# 4. Config error - check configmap and secrets
```

### Service Not Accessible

```bash
# Check service
kubectl get svc pathology-api -n pathology
kubectl describe svc pathology-api -n pathology

# Check endpoints
kubectl get endpoints pathology-api -n pathology

# Test from within cluster
kubectl run -it --rm debug --image=busybox -n pathology -- sh
wget -O- http://pathology-api:8000/health
```

### High Memory Usage

```bash
# Check memory usage
kubectl top pods -n pathology

# Solutions:
# 1. Increase memory limits
# 2. Reduce batch size (update configmap)
# 3. Enable memory profiling
# 4. Check for memory leaks
```

### Slow Response Times

```bash
# Check pod logs for slow requests
kubectl logs -f <pod-name> -n pathology

# Solutions:
# 1. Scale up replicas
# 2. Enable GPU deployment
# 3. Optimize model inference
# 4. Add caching layer
```

## Cleanup

### Remove All Resources

```bash
# Delete all resources in namespace
kubectl delete namespace pathology

# Or delete individually
kubectl delete -f .
```

### Remove Specific Resources

```bash
kubectl delete deployment pathology-api -n pathology
kubectl delete svc pathology-api -n pathology
kubectl delete ingress pathology-api-ingress -n pathology
```

## Security Best Practices

1. **Use RBAC**: Create service accounts with minimal permissions
2. **Network Policies**: Restrict pod-to-pod communication
3. **Pod Security Policies**: Enforce security standards
4. **Secrets Management**: Use external secret managers
5. **Image Scanning**: Scan images for vulnerabilities
6. **TLS**: Always use TLS for external access
7. **Resource Limits**: Set appropriate limits to prevent resource exhaustion
8. **Read-Only Filesystem**: Mount volumes as read-only when possible

## Production Checklist

- [ ] Update all placeholder values (domain, secrets, etc.)
- [ ] Configure proper resource limits
- [ ] Set up persistent storage
- [ ] Configure TLS/SSL certificates
- [ ] Set up monitoring and alerting
- [ ] Configure log aggregation
- [ ] Set up backup strategy
- [ ] Test disaster recovery
- [ ] Configure autoscaling
- [ ] Set up CI/CD pipeline
- [ ] Document runbooks
- [ ] Train operations team

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/)
- [cert-manager](https://cert-manager.io/)
- [Prometheus Operator](https://prometheus-operator.dev/)
