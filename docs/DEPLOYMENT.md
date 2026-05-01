# Deployment Guide

This guide covers deploying HistoCore to Kubernetes using the automated CD pipeline.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [CD Pipeline Configuration](#cd-pipeline-configuration)
- [Deployment Process](#deployment-process)
- [Blue-Green Deployment](#blue-green-deployment)
- [Rollback Procedures](#rollback-procedures)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Overview

HistoCore uses a multi-stage deployment pipeline:

```
Code Push → CI Tests → Build Image → Dev → Staging → Production
                                      ↓       ↓          ↓
                                   Auto    Auto      Manual
```

**Environments**:
- **Development**: Automatic deployment on main branch push
- **Staging**: Automatic deployment after dev succeeds
- **Production**: Manual deployment with blue-green strategy

## Prerequisites

### Required Tools
- kubectl v1.28+
- Helm v3.13+
- Docker (for local testing)
- GitHub CLI (optional, for manual deployments)

### Kubernetes Clusters
You need three Kubernetes clusters (or namespaces):
- Development cluster
- Staging cluster
- Production cluster

### GitHub Secrets
Configure these secrets in repository settings:

```bash
# Development cluster
KUBECONFIG_DEV: <base64-encoded kubeconfig>

# Staging cluster
KUBECONFIG_STAGING: <base64-encoded kubeconfig>

# Production cluster
KUBECONFIG_PROD: <base64-encoded kubeconfig>
```

## Environment Setup

### 1. Prepare Kubeconfig Files

```bash
# For each cluster, create a service account with deployment permissions
kubectl create serviceaccount histocore-deployer -n kube-system

# Create cluster role binding
kubectl create clusterrolebinding histocore-deployer \
  --clusterrole=cluster-admin \
  --serviceaccount=kube-system:histocore-deployer

# Get service account token
TOKEN=$(kubectl get secret $(kubectl get serviceaccount histocore-deployer -n kube-system -o jsonpath='{.secrets[0].name}') -n kube-system -o jsonpath='{.data.token}' | base64 -d)

# Create kubeconfig
cat > kubeconfig-dev.yaml <<EOF
apiVersion: v1
kind: Config
clusters:
- cluster:
    certificate-authority-data: <CA_DATA>
    server: <CLUSTER_URL>
  name: dev-cluster
contexts:
- context:
    cluster: dev-cluster
    user: histocore-deployer
  name: dev
current-context: dev
users:
- name: histocore-deployer
  user:
    token: $TOKEN
EOF

# Encode for GitHub secret
cat kubeconfig-dev.yaml | base64 -w 0 > kubeconfig-dev.b64
```

### 2. Add Secrets to GitHub

1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add each kubeconfig:
   - Name: `KUBECONFIG_DEV`
   - Value: Contents of `kubeconfig-dev.b64`
4. Repeat for staging and production

### 3. Configure Helm Values

Edit environment-specific values files:

**Development** (`k8s/helm/histocore/values-dev.yaml`):
```yaml
replicaCount: 1
resources:
  limits:
    cpu: 2000m
    memory: 8Gi
autoscaling:
  enabled: false
ingress:
  hosts:
    - host: dev.histocore.example.com
```

**Staging** (`k8s/helm/histocore/values.yaml`):
```yaml
replicaCount: 2
resources:
  limits:
    cpu: 4000m
    memory: 16Gi
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 5
ingress:
  hosts:
    - host: staging.histocore.example.com
```

**Production** (`k8s/helm/histocore/values-prod.yaml`):
```yaml
replicaCount: 3
resources:
  limits:
    cpu: 8000m
    memory: 32Gi
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
ingress:
  hosts:
    - host: histocore.example.com
```

## CD Pipeline Configuration

### Automatic Deployment

The CD pipeline automatically deploys to dev and staging when code is pushed to main:

```yaml
# Triggers automatic deployment
git push origin main
```

**Pipeline Flow**:
1. Run tests (can be skipped with manual trigger)
2. Build Docker image and push to GitHub Container Registry
3. Deploy to development environment
4. Run smoke tests
5. Deploy to staging environment (if dev succeeds)
6. Run integration tests

### Manual Deployment

Deploy to production manually:

```bash
# Using GitHub CLI
gh workflow run cd.yml -f environment=prod -f skip_tests=false

# Or via GitHub Actions UI:
# 1. Go to Actions → CD
# 2. Click "Run workflow"
# 3. Select environment: prod
# 4. Click "Run workflow"
```

## Deployment Process

### Development Deployment

Automatic on main branch push:

```bash
# Push to main
git push origin main

# Monitor deployment
gh run watch

# Check deployment status
kubectl get pods -n dev -l app.kubernetes.io/name=histocore
kubectl logs -n dev -l app.kubernetes.io/name=histocore
```

### Staging Deployment

Automatic after dev deployment succeeds:

```bash
# Monitor deployment
gh run watch

# Check deployment status
kubectl get pods -n staging -l app.kubernetes.io/name=histocore

# Test staging endpoint
curl https://staging.histocore.example.com/health
```

### Production Deployment

Manual deployment with blue-green strategy:

```bash
# Trigger production deployment
gh workflow run cd.yml -f environment=prod

# Monitor deployment
gh run watch

# Verify green deployment
kubectl get pods -n prod -l version=green

# Check service routing
kubectl get service histocore -n prod -o yaml | grep version
```

## Blue-Green Deployment

Production uses blue-green deployment for zero-downtime updates:

### How It Works

1. **Deploy Green**: New version deployed alongside blue (current)
2. **Verify Green**: Health checks and smoke tests
3. **Switch Traffic**: Service selector updated to green
4. **Monitor**: 5-minute monitoring period
5. **Remove Blue**: Old version removed if successful
6. **Rollback**: Automatic rollback to blue on failure

### Manual Blue-Green Operations

```bash
# Check current version
kubectl get service histocore -n prod -o jsonpath='{.spec.selector.version}'

# Switch to green manually
kubectl patch service histocore -n prod -p '{"spec":{"selector":{"version":"green"}}}'

# Switch back to blue (rollback)
kubectl patch service histocore -n prod -p '{"spec":{"selector":{"version":"blue"}}}'

# View both deployments
kubectl get deployments -n prod -l app.kubernetes.io/name=histocore
```

## Rollback Procedures

### Automatic Rollback

The CD pipeline automatically rolls back on failure:
- Failed health checks
- High error rate during monitoring
- Deployment timeout

### Manual Rollback

#### Using Helm

```bash
# List releases
helm list -n prod

# Rollback to previous version
helm rollback histocore-green -n prod

# Rollback to specific revision
helm rollback histocore-green 3 -n prod
```

#### Using kubectl

```bash
# Rollback deployment
kubectl rollout undo deployment/histocore-green -n prod

# Rollback to specific revision
kubectl rollout undo deployment/histocore-green -n prod --to-revision=2

# Check rollout history
kubectl rollout history deployment/histocore-green -n prod
```

#### Blue-Green Rollback

```bash
# Switch traffic back to blue
kubectl patch service histocore -n prod -p '{"spec":{"selector":{"version":"blue"}}}'

# Remove failed green deployment
kubectl delete deployment histocore-green -n prod
```

## Monitoring

### Deployment Status

```bash
# Watch deployment progress
kubectl rollout status deployment/histocore-green -n prod

# View pod status
kubectl get pods -n prod -l app.kubernetes.io/name=histocore

# View events
kubectl get events -n prod --sort-by='.lastTimestamp'
```

### Application Health

```bash
# Health check
curl https://histocore.example.com/health

# Metrics
curl https://histocore.example.com/metrics

# Readiness probe
kubectl get pods -n prod -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}'
```

### Logs

```bash
# View logs
kubectl logs -n prod -l app.kubernetes.io/name=histocore

# Follow logs
kubectl logs -n prod -l app.kubernetes.io/name=histocore -f

# View logs from specific version
kubectl logs -n prod -l version=green

# View logs from previous deployment
kubectl logs -n prod -l app.kubernetes.io/name=histocore --previous
```

### Prometheus Metrics

```bash
# Port forward to Prometheus
kubectl port-forward -n monitoring svc/prometheus 9090:9090

# Query metrics
curl http://localhost:9090/api/v1/query?query=histocore_inference_requests_total
```

## Troubleshooting

### Deployment Fails

**Symptom**: Deployment stuck in pending or failing state

**Diagnosis**:
```bash
# Check pod status
kubectl describe pod <pod-name> -n prod

# Check events
kubectl get events -n prod --field-selector involvedObject.name=<pod-name>

# Check logs
kubectl logs <pod-name> -n prod
```

**Common Causes**:
- Insufficient resources (CPU/memory/GPU)
- Image pull errors
- Configuration errors
- Health check failures

**Solutions**:
```bash
# Check resource availability
kubectl top nodes
kubectl describe node <node-name>

# Verify image exists
docker pull ghcr.io/your-org/histocore:main-<sha>

# Validate Helm chart
helm lint ./k8s/helm/histocore

# Check configuration
kubectl get configmap histocore-config -n prod -o yaml
```

### Health Checks Failing

**Symptom**: Pods restarting or not becoming ready

**Diagnosis**:
```bash
# Check readiness probe
kubectl get pods -n prod -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")]}'

# Check liveness probe
kubectl describe pod <pod-name> -n prod | grep -A 10 "Liveness"

# Test health endpoint
kubectl exec <pod-name> -n prod -- curl localhost:8000/health
```

**Solutions**:
- Increase `initialDelaySeconds` in probe configuration
- Check application startup time
- Verify health endpoint implementation
- Review application logs for errors

### Traffic Not Routing

**Symptom**: Service not accessible or routing to wrong version

**Diagnosis**:
```bash
# Check service selector
kubectl get service histocore -n prod -o yaml | grep -A 5 selector

# Check endpoints
kubectl get endpoints histocore -n prod

# Check ingress
kubectl get ingress -n prod
kubectl describe ingress histocore -n prod
```

**Solutions**:
```bash
# Verify service selector matches pod labels
kubectl get pods -n prod --show-labels

# Update service selector
kubectl patch service histocore -n prod -p '{"spec":{"selector":{"version":"green"}}}'

# Check ingress controller
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx
```

### High Error Rate

**Symptom**: Increased errors after deployment

**Diagnosis**:
```bash
# Check error logs
kubectl logs -n prod -l version=green | grep ERROR

# Check metrics
curl https://histocore.example.com/metrics | grep error

# Check Prometheus alerts
kubectl get prometheusrules -n monitoring
```

**Solutions**:
- Rollback to previous version
- Review application logs
- Check configuration changes
- Verify database connectivity
- Check external service dependencies

### Image Pull Errors

**Symptom**: Pods stuck in `ImagePullBackOff`

**Diagnosis**:
```bash
# Check pod events
kubectl describe pod <pod-name> -n prod | grep -A 10 Events

# Check image pull secrets
kubectl get secrets -n prod
```

**Solutions**:
```bash
# Verify image exists
docker pull ghcr.io/your-org/histocore:main-<sha>

# Create image pull secret (if needed)
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=<username> \
  --docker-password=<token> \
  -n prod

# Update deployment to use secret
kubectl patch deployment histocore-green -n prod -p '{"spec":{"template":{"spec":{"imagePullSecrets":[{"name":"ghcr-secret"}]}}}}'
```

## Best Practices

1. **Always test in dev/staging first** - Never deploy directly to production
2. **Monitor deployments** - Watch logs and metrics during rollout
3. **Use semantic versioning** - Tag releases with meaningful versions
4. **Document changes** - Update changelog for each deployment
5. **Backup before deployment** - Backup databases and configurations
6. **Plan rollback strategy** - Know how to rollback before deploying
7. **Communicate deployments** - Notify team of production deployments
8. **Schedule maintenance windows** - Deploy during low-traffic periods
9. **Test rollback procedures** - Regularly test rollback in staging
10. **Keep deployment history** - Maintain records of all deployments

## Security Considerations

1. **Rotate secrets regularly** - Update kubeconfig tokens periodically
2. **Use RBAC** - Limit deployment permissions to necessary resources
3. **Scan images** - Trivy scans run automatically in pipeline
4. **Network policies** - Restrict pod-to-pod communication
5. **TLS everywhere** - Use HTTPS for all external endpoints
6. **Audit logs** - Enable Kubernetes audit logging
7. **Secret management** - Use external secret managers (Vault, AWS Secrets Manager)
8. **Image signing** - Sign Docker images for verification

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Blue-Green Deployment Pattern](https://martinfowler.com/bliki/BlueGreenDeployment.html)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
