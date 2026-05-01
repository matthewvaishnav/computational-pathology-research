# HistoCore Helm Chart

Production-grade Helm chart for deploying HistoCore computational pathology framework on Kubernetes.

## Features

- **High Availability**: Multi-replica deployment with pod anti-affinity
- **Auto-scaling**: Horizontal Pod Autoscaler (HPA) based on CPU/memory
- **GPU Support**: NVIDIA GPU scheduling and resource management
- **Security**: Network policies, pod security contexts, secrets management
- **Monitoring**: Prometheus metrics, health checks, liveness/readiness probes
- **Storage**: Persistent volume claims for data storage
- **Ingress**: TLS-enabled ingress with cert-manager integration

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- NVIDIA GPU Operator (for GPU support)
- Ingress controller (nginx recommended)
- cert-manager (for TLS certificates)
- Storage provisioner (for persistent volumes)

## Installation

### Quick Start

```bash
# Add Helm repository (if published)
helm repo add histocore https://charts.histocore.io
helm repo update

# Install with default values
helm install histocore histocore/histocore

# Install with custom values
helm install histocore histocore/histocore -f custom-values.yaml
```

### Local Installation

```bash
# From source
cd k8s/helm
helm install histocore ./histocore

# With custom values
helm install histocore ./histocore -f custom-values.yaml

# Dry run to preview
helm install histocore ./histocore --dry-run --debug
```

## Configuration

### Basic Configuration

```yaml
# values.yaml
replicaCount: 2

image:
  repository: histocore/histocore
  tag: "1.0.0"

resources:
  limits:
    cpu: 4000m
    memory: 16Gi
    nvidia.com/gpu: 1
  requests:
    cpu: 2000m
    memory: 8Gi
    nvidia.com/gpu: 1

ingress:
  enabled: true
  hosts:
    - host: histocore.example.com
      paths:
        - path: /
          pathType: Prefix
```

### GPU Configuration

```yaml
# Enable GPU scheduling
nodeSelector:
  gpu: "true"

tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule

resources:
  limits:
    nvidia.com/gpu: 1
  requests:
    nvidia.com/gpu: 1
```

### Auto-scaling Configuration

```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

### Storage Configuration

```yaml
persistence:
  enabled: true
  storageClass: "fast-ssd"
  accessMode: ReadWriteOnce
  size: 100Gi
  mountPath: /data
```

### Security Configuration

```yaml
# Pod security context
podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

securityContext:
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false

# Network policy
networkPolicy:
  enabled: true
  policyTypes:
    - Ingress
    - Egress
```

### PACS Integration

```yaml
config:
  pacs:
    enabled: true
    host: "pacs.hospital.org"
    port: 11112
    aet: "HISTOCORE"
    calledAet: "PACS"
```

### Secrets Management

**Option 1: Helm values (development only)**

```yaml
secrets:
  database:
    host: "postgres.example.com"
    port: "5432"
    name: "histocore"
    user: "histocore_user"
    password: "changeme"
  apiKeys:
    admin: "admin-key-here"
    service: "service-key-here"
```

**Option 2: External Secrets (production)**

```bash
# Create secrets manually
kubectl create secret generic histocore-secrets \
  --from-literal=DATABASE_HOST=postgres.example.com \
  --from-literal=DATABASE_PASSWORD=secure-password

# Or use external-secrets operator
kubectl apply -f external-secret.yaml
```

## Upgrading

```bash
# Upgrade to new version
helm upgrade histocore histocore/histocore --version 1.1.0

# Upgrade with new values
helm upgrade histocore histocore/histocore -f new-values.yaml

# Rollback if needed
helm rollback histocore
```

## Uninstallation

```bash
# Uninstall release
helm uninstall histocore

# Delete PVCs (if needed)
kubectl delete pvc histocore-pvc
```

## Values Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `2` |
| `image.repository` | Image repository | `histocore/histocore` |
| `image.tag` | Image tag | `""` (uses appVersion) |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `service.type` | Service type | `ClusterIP` |
| `service.port` | Service port | `8000` |
| `ingress.enabled` | Enable ingress | `true` |
| `ingress.className` | Ingress class | `nginx` |
| `ingress.hosts` | Ingress hosts | `[histocore.example.com]` |
| `resources.limits.cpu` | CPU limit | `4000m` |
| `resources.limits.memory` | Memory limit | `16Gi` |
| `resources.limits.nvidia.com/gpu` | GPU limit | `1` |
| `autoscaling.enabled` | Enable HPA | `true` |
| `autoscaling.minReplicas` | Min replicas | `2` |
| `autoscaling.maxReplicas` | Max replicas | `10` |
| `persistence.enabled` | Enable persistence | `true` |
| `persistence.size` | PVC size | `100Gi` |
| `networkPolicy.enabled` | Enable network policy | `true` |
| `podDisruptionBudget.enabled` | Enable PDB | `true` |

See [values.yaml](values.yaml) for complete configuration options.

## Examples

### Development Environment

```yaml
# dev-values.yaml
replicaCount: 1

resources:
  limits:
    cpu: 2000m
    memory: 8Gi
  requests:
    cpu: 1000m
    memory: 4Gi

autoscaling:
  enabled: false

persistence:
  enabled: false

ingress:
  enabled: false

config:
  monitoring:
    logLevel: "DEBUG"
```

```bash
helm install histocore ./histocore -f dev-values.yaml
```

### Production Environment

```yaml
# prod-values.yaml
replicaCount: 3

image:
  tag: "1.0.0"

resources:
  limits:
    cpu: 4000m
    memory: 16Gi
    nvidia.com/gpu: 1
  requests:
    cpu: 2000m
    memory: 8Gi
    nvidia.com/gpu: 1

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20

persistence:
  enabled: true
  storageClass: "fast-ssd"
  size: 500Gi

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: histocore.prod.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: histocore-tls
      hosts:
        - histocore.prod.example.com

networkPolicy:
  enabled: true

podDisruptionBudget:
  enabled: true
  minAvailable: 2

config:
  security:
    enableAuth: true
  monitoring:
    prometheusEnabled: true
    logLevel: "INFO"
```

```bash
helm install histocore ./histocore -f prod-values.yaml
```

### High-Availability Setup

```yaml
# ha-values.yaml
replicaCount: 5

autoscaling:
  enabled: true
  minReplicas: 5
  maxReplicas: 50
  targetCPUUtilizationPercentage: 60
  targetMemoryUtilizationPercentage: 70

affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
            - key: app.kubernetes.io/name
              operator: In
              values:
                - histocore
        topologyKey: kubernetes.io/hostname

podDisruptionBudget:
  enabled: true
  minAvailable: 3

priorityClassName: "high-priority"
```

## Monitoring

### Prometheus Integration

The chart includes Prometheus annotations for automatic scraping:

```yaml
podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"
```

### Health Checks

- **Liveness Probe**: `/health/live` - Checks if pod is alive
- **Readiness Probe**: `/health/ready` - Checks if pod is ready to serve traffic

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods -l app.kubernetes.io/name=histocore
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

### Check Resources

```bash
kubectl top pods -l app.kubernetes.io/name=histocore
kubectl get hpa histocore
```

### Check Ingress

```bash
kubectl get ingress histocore
kubectl describe ingress histocore
```

### Common Issues

**GPU Not Available**

```bash
# Check GPU operator
kubectl get pods -n gpu-operator-resources

# Check node labels
kubectl get nodes --show-labels | grep gpu
```

**PVC Not Binding**

```bash
# Check PVC status
kubectl get pvc histocore-pvc

# Check storage class
kubectl get storageclass
```

**Ingress Not Working**

```bash
# Check ingress controller
kubectl get pods -n ingress-nginx

# Check certificate
kubectl get certificate histocore-tls
```

## Support

- **Documentation**: https://github.com/matthewvaishnav/histocore/tree/main/docs
- **Issues**: https://github.com/matthewvaishnav/histocore/issues
- **Discussions**: https://github.com/matthewvaishnav/histocore/discussions

## License

MIT License - see [LICENSE](../../../LICENSE) for details.
