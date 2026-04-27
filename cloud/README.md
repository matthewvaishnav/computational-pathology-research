# HistoCore Cloud Deployment Guide

This directory contains infrastructure-as-code and deployment scripts for deploying HistoCore to major cloud providers.

## Supported Cloud Providers

- **AWS** - Amazon Web Services with EKS
- **Azure** - Microsoft Azure with AKS  
- **GCP** - Google Cloud Platform with GKE

## Prerequisites

### Common Requirements
- Docker installed and configured
- kubectl installed
- Terraform >= 1.0 installed

### AWS Requirements
- AWS CLI installed and configured
- AWS credentials with appropriate permissions
- Terraform AWS provider

### Azure Requirements
- Azure CLI installed and configured
- Azure subscription with appropriate permissions
- Terraform Azure provider

### GCP Requirements
- Google Cloud SDK installed and configured
- GCP project with billing enabled
- Terraform Google provider

## Quick Start

### AWS Deployment

```bash
# Navigate to AWS directory
cd cloud/aws

# Configure AWS credentials
aws configure

# Deploy infrastructure
./deploy.sh production us-west-2
```

### Azure Deployment

```bash
# Navigate to Azure directory
cd cloud/azure

# Login to Azure
az login

# Deploy infrastructure
./deploy.sh production "East US"
```

### GCP Deployment

```bash
# Navigate to GCP directory
cd cloud/gcp

# Authenticate with GCP
gcloud auth login
gcloud auth application-default login

# Deploy infrastructure
./deploy.sh my-project-id production us-central1
```

## Architecture Overview

### AWS Architecture
- **EKS Cluster** with CPU and GPU node groups
- **ElastiCache Redis** for caching
- **S3 Bucket** for data storage
- **Application Load Balancer** for ingress
- **VPC** with public/private subnets
- **IAM roles** for secure access

### Azure Architecture
- **AKS Cluster** with system, CPU, and GPU node pools
- **Azure Cache for Redis** for caching
- **Azure Storage Account** for data storage
- **Azure Load Balancer** for ingress
- **Virtual Network** with subnets
- **Key Vault** for secrets management

### GCP Architecture
- **GKE Cluster** with system, CPU, and GPU node pools
- **Cloud Memorystore Redis** for caching
- **Cloud Storage** for data storage
- **Cloud Load Balancer** for ingress
- **VPC Network** with subnets
- **Cloud KMS** for encryption
- **Cloud SQL PostgreSQL** for metadata

## GPU Support

All cloud deployments include GPU support for ML inference:

- **AWS**: Uses `p3.2xlarge` instances with NVIDIA V100 GPUs
- **Azure**: Uses `Standard_NC6s_v3` instances with NVIDIA V100 GPUs  
- **GCP**: Uses `n1-standard-4` instances with NVIDIA T4 GPUs

GPU nodes are automatically tainted to ensure only GPU workloads are scheduled on them.

## Scaling Configuration

### Horizontal Pod Autoscaling (HPA)
All deployments include HPA configuration:
- CPU utilization: 70% target
- Memory utilization: 80% target
- Min replicas: 2
- Max replicas: 10

### Cluster Autoscaling
Node pools are configured with autoscaling:
- **CPU nodes**: 1-10 nodes
- **GPU nodes**: 1-5 nodes (cost optimization)

## Security Features

### Network Security
- Private subnets for worker nodes
- Security groups/NSGs with minimal required access
- TLS encryption for all communications

### Data Security
- Encryption at rest for all storage
- Encryption in transit with TLS 1.3
- Secrets management via cloud-native services

### Access Control
- RBAC enabled on Kubernetes clusters
- Cloud IAM integration
- Service accounts with minimal permissions

## Monitoring and Observability

### Metrics Collection
- Prometheus for metrics collection
- Grafana for visualization
- Cloud-native monitoring integration

### Logging
- Centralized logging to cloud providers
- Structured JSON logging
- Log retention policies

### Health Checks
- Kubernetes liveness/readiness probes
- Load balancer health checks
- Application-level health endpoints

## Cost Optimization

### Resource Management
- Spot/preemptible instances where appropriate
- Automatic scaling based on demand
- Resource quotas and limits

### Storage Optimization
- Lifecycle policies for data retention
- Compression for archived data
- Tiered storage classes

## Disaster Recovery

### Backup Strategy
- Automated database backups
- Cross-region replication for critical data
- Point-in-time recovery capabilities

### High Availability
- Multi-AZ/region deployments
- Load balancer health checks
- Automatic failover mechanisms

## Deployment Customization

### Environment Variables
Each deployment script accepts environment-specific configuration:

```bash
# AWS
./deploy.sh <environment> <region>

# Azure  
./deploy.sh <environment> <region>

# GCP
./deploy.sh <project_id> <environment> <region>
```

### Terraform Variables
Customize deployments by modifying `variables.tf` files:
- Instance/VM sizes
- Node counts
- Storage sizes
- Network configurations

### Kubernetes Manifests
Modify the shared `k8s/` manifests for application-specific changes:
- Resource requests/limits
- Environment variables
- Service configurations

## Troubleshooting

### Common Issues

1. **Insufficient Permissions**
   - Ensure cloud credentials have required permissions
   - Check IAM roles and policies

2. **Resource Quotas**
   - Verify cloud provider quotas for GPUs
   - Request quota increases if needed

3. **Network Connectivity**
   - Check security group/NSG rules
   - Verify subnet configurations

4. **GPU Driver Issues**
   - Ensure NVIDIA device plugin is installed
   - Check node taints and tolerations

### Debug Commands

```bash
# Check cluster status
kubectl get nodes
kubectl get pods -n histocore

# Check logs
kubectl logs -n histocore deployment/histocore-streaming

# Check GPU availability
kubectl describe nodes | grep nvidia.com/gpu

# Port forward for local access
kubectl port-forward -n histocore svc/histocore-streaming 8000:8000
```

## Support

For deployment issues:
1. Check cloud provider documentation
2. Review Terraform logs
3. Examine Kubernetes events
4. Contact support with deployment logs

## Next Steps

After successful deployment:
1. Configure DNS for production domains
2. Set up SSL certificates
3. Configure monitoring alerts
4. Implement backup procedures
5. Set up CI/CD pipelines