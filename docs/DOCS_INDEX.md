# HistoCore Documentation Index

Complete documentation for HistoCore Real-Time WSI Streaming.

## Quick Start

- **[Installation Guide](INSTALLATION.md)** - Get started with HistoCore
- **[Quick Start Tutorial](QUICK_START.md)** - Process your first slide in 5 minutes
- **[FAQ](FAQ.md)** - Frequently asked questions

## API Documentation

- **[API Reference](api/README.md)** - Complete REST API documentation
- **[OpenAPI Specification](api/openapi.yaml)** - Machine-readable API spec
- **[WebSocket Streaming](api/README.md#websocket-streaming)** - Real-time updates
- **[Code Examples](api/README.md#code-examples)** - Python, JavaScript, cURL

## Deployment

- **[Deployment Guide](deployment/DEPLOYMENT_GUIDE.md)** - Complete deployment guide
  - [Docker Deployment](deployment/DEPLOYMENT_GUIDE.md#docker-deployment)
  - [Kubernetes Deployment](deployment/DEPLOYMENT_GUIDE.md#kubernetes-deployment)
  - [Cloud Deployment](deployment/DEPLOYMENT_GUIDE.md#cloud-deployment) (AWS, Azure, GCP)
- **[Configuration Guide](deployment/CONFIGURATION_GUIDE.md)** - Configuration reference
  - [Core Settings](deployment/CONFIGURATION_GUIDE.md#core-settings)
  - [GPU Configuration](deployment/CONFIGURATION_GUIDE.md#gpu-configuration)
  - [PACS Integration](deployment/CONFIGURATION_GUIDE.md#pacs-integration)
  - [Performance Tuning](deployment/CONFIGURATION_GUIDE.md#performance-tuning)

## Operations

- **[Monitoring Guide](../monitoring/README.md)** - Monitoring and observability
  - [Prometheus Metrics](../monitoring/README.md#prometheus-metrics)
  - [Grafana Dashboards](../monitoring/README.md#grafana-dashboards)
  - [Alerting](../monitoring/README.md#alerting)
  - [Distributed Tracing](../monitoring/README.md#tracing)
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions
  - [Quick Diagnostics](TROUBLESHOOTING.md#quick-diagnostics)
  - [GPU Issues](TROUBLESHOOTING.md#gpu-issues)
  - [Performance Issues](TROUBLESHOOTING.md#performance-issues)
  - [PACS Integration Issues](TROUBLESHOOTING.md#pacs-integration-issues)

## Architecture

- **[System Architecture](ARCHITECTURE.md)** - High-level system design
- **[Architecture Diagrams](ARCHITECTURE_DIAGRAMS.md)** - Visual system documentation
- **[Design Document](../.kiro/specs/real-time-wsi-streaming/design.md)** - Detailed technical design
- **[Requirements Document](../.kiro/specs/real-time-wsi-streaming/requirements.md)** - System requirements

## Features

### Real-Time WSI Streaming
- **[Real-Time Streaming System](REALTIME_STREAMING.md)** - Complete streaming system documentation
- **[Streaming Overview](features/STREAMING.md)** - Real-time processing architecture
- **[Progressive Confidence](features/CONFIDENCE.md)** - Confidence building and early stopping
- **[Attention Visualization](features/ATTENTION.md)** - Attention heatmap generation

### Model Management
- **[Model Hot-Swapping](REALTIME_STREAMING.md#model-hot-swapping)** - Zero-downtime model updates
- **[A/B Testing](REALTIME_STREAMING.md#ab-testing)** - Model comparison and validation
- **[Model Versioning](REALTIME_STREAMING.md#model-hot-swapping)** - Version control and rollback

### Performance & Testing
- **[Stress Testing](REALTIME_STREAMING.md#stress-testing)** - Concurrent load validation
- **[Performance Regression](REALTIME_STREAMING.md#performance-regression-testing)** - Automated baseline tracking
- **[Benchmarks](REALTIME_STREAMING.md#performance-benchmarks)** - Performance metrics

### PACS Integration
- **[PACS Integration Guide](features/PACS_INTEGRATION.md)** - Hospital PACS integration
- **[DICOM Support](features/DICOM.md)** - DICOM networking and WSI support
- **[Clinical Workflow](features/CLINICAL_WORKFLOW.md)** - Clinical reporting and workflow

### Model Interpretability
- **[Grad-CAM Visualization](features/GRADCAM.md)** - CNN feature visualization
- **[Attention Heatmaps](features/ATTENTION_HEATMAPS.md)** - MIL attention visualization
- **[Failure Analysis](features/FAILURE_ANALYSIS.md)** - Model debugging and analysis

## Development

- **[Development Guide](development/DEVELOPMENT_GUIDE.md)** - Development setup
- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute
- **[Testing Guide](development/TESTING_GUIDE.md)** - Testing infrastructure
- **[Code Style Guide](development/CODE_STYLE.md)** - Coding standards

## Benchmarks and Results

- **[PCam Real Results](PCAM_REAL_RESULTS.md)** - Full dataset benchmark results
- **[Performance Benchmark](../PERFORMANCE_BENCHMARK_COMPLETE.md)** - Comprehensive performance analysis
- **[Superiority Report](../results/comprehensive_benchmark/HISTOCORE_SUPERIORITY_REPORT.md)** - Competitive analysis

## Reference

- **[CLI Reference](reference/CLI_REFERENCE.md)** - Command-line interface
- **[Configuration Reference](deployment/CONFIGURATION_GUIDE.md)** - All configuration options
- **[Metrics Reference](reference/METRICS_REFERENCE.md)** - Prometheus metrics
- **[Error Codes](reference/ERROR_CODES.md)** - Error code reference

## Use Cases

- **[Hospital Deployment](use-cases/HOSPITAL_DEPLOYMENT.md)** - Clinical deployment guide
- **[Research Workflows](use-cases/RESEARCH_WORKFLOWS.md)** - Research use cases
- **[Batch Processing](use-cases/BATCH_PROCESSING.md)** - High-throughput processing
- **[Live Demos](use-cases/LIVE_DEMOS.md)** - Hospital demonstration setup

## Security and Compliance

- **[Security Guide](security/SECURITY_GUIDE.md)** - Security best practices
- **[HIPAA Compliance](security/HIPAA_COMPLIANCE.md)** - HIPAA compliance guide
- **[FDA Pathway](security/FDA_PATHWAY.md)** - FDA 510(k) preparation
- **[Audit Logging](security/AUDIT_LOGGING.md)** - Audit trail documentation

## Cloud Providers

- **[AWS Deployment](../cloud/aws/README.md)** - Amazon Web Services
- **[Azure Deployment](../cloud/azure/README.md)** - Microsoft Azure
- **[GCP Deployment](../cloud/gcp/README.md)** - Google Cloud Platform

## Docker and Kubernetes

- **[Docker Guide](../docker/README.md)** - Docker deployment
- **[Kubernetes Guide](../k8s/README.md)** - Kubernetes deployment
- **[Helm Charts](../k8s/helm/README.md)** - Helm deployment

## Additional Resources

- **[Changelog](../CHANGELOG.md)** - Version history
- **[Roadmap](../ROADMAP.md)** - Future plans
- **[License](../LICENSE)** - MIT License
- **[Citation](../CITATION.cff)** - How to cite HistoCore

## Getting Help

- **[FAQ](FAQ.md)** - Frequently asked questions
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues
- **[GitHub Issues](https://github.com/histocore/histocore/issues)** - Bug reports and feature requests
- **[Email Support](mailto:support@histocore.ai)** - Direct support
- **[Slack Community](https://histocore.slack.com)** - Community chat

## Documentation by Role

### For Developers
1. [Installation Guide](INSTALLATION.md)
2. [Development Guide](development/DEVELOPMENT_GUIDE.md)
3. [API Reference](api/README.md)
4. [Testing Guide](development/TESTING_GUIDE.md)
5. [Contributing Guide](../CONTRIBUTING.md)

### For DevOps Engineers
1. [Deployment Guide](deployment/DEPLOYMENT_GUIDE.md)
2. [Configuration Guide](deployment/CONFIGURATION_GUIDE.md)
3. [Monitoring Guide](../monitoring/README.md)
4. [Troubleshooting Guide](TROUBLESHOOTING.md)
5. [Security Guide](security/SECURITY_GUIDE.md)

### For Clinical Users
1. [Quick Start Tutorial](QUICK_START.md)
2. [Clinical Workflow](features/CLINICAL_WORKFLOW.md)
3. [PACS Integration](features/PACS_INTEGRATION.md)
4. [FAQ](FAQ.md)
5. [Troubleshooting](TROUBLESHOOTING.md)

### For Researchers
1. [Quick Start Tutorial](QUICK_START.md)
2. [Research Workflows](use-cases/RESEARCH_WORKFLOWS.md)
3. [Model Interpretability](features/GRADCAM.md)
4. [Benchmarks](PCAM_REAL_RESULTS.md)
5. [API Reference](api/README.md)

### For System Administrators
1. [Deployment Guide](deployment/DEPLOYMENT_GUIDE.md)
2. [Configuration Guide](deployment/CONFIGURATION_GUIDE.md)
3. [Security Guide](security/SECURITY_GUIDE.md)
4. [Monitoring Guide](../monitoring/README.md)
5. [Backup and Recovery](deployment/DEPLOYMENT_GUIDE.md#backup-and-recovery)

## Documentation Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| API Reference | ✅ Complete | 2026-04-26 |
| Deployment Guide | ✅ Complete | 2026-04-26 |
| Configuration Guide | ✅ Complete | 2026-04-26 |
| Troubleshooting Guide | ✅ Complete | 2026-04-26 |
| FAQ | ✅ Complete | 2026-04-26 |
| Monitoring Guide | ✅ Complete | 2026-04-08 |
| Docker Guide | ✅ Complete | 2026-04-08 |
| Kubernetes Guide | ✅ Complete | 2026-04-08 |
| Cloud Deployment | ✅ Complete | 2026-04-08 |
| Installation Guide | 🚧 In Progress | - |
| Quick Start Tutorial | 🚧 In Progress | - |
| Development Guide | 📝 Planned | - |
| Testing Guide | 📝 Planned | - |
| Security Guide | 📝 Planned | - |

## Contributing to Documentation

Documentation improvements are welcome! To contribute:

1. **Fork the repository**
2. **Edit documentation** in `docs/` directory
3. **Follow style guide**:
   - Use Markdown format
   - Include code examples
   - Add diagrams where helpful
   - Keep language clear and concise
4. **Submit pull request**

See [Contributing Guide](../CONTRIBUTING.md) for details.

## Documentation Feedback

Have feedback on the documentation?

- **GitHub Issues**: [Report documentation issues](https://github.com/histocore/histocore/issues/new?labels=documentation)
- **Email**: docs@histocore.ai
- **Slack**: #documentation channel

## License

Documentation is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Code examples in documentation are licensed under [MIT License](../LICENSE).
