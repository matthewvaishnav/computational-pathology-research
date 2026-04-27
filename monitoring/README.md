# HistoCore Monitoring Stack

Complete monitoring, alerting, and observability solution for HistoCore streaming.

## Components

- **Prometheus** - Metrics collection and alerting
- **Grafana** - Visualization and dashboards  
- **Alertmanager** - Alert routing and notifications
- **Jaeger** - Distributed tracing
- **Node Exporter** - System metrics
- **cAdvisor** - Container metrics

## Quick Start

```bash
# Start monitoring stack
cd monitoring
docker-compose up -d

# Check status
docker-compose ps
```

## Access URLs

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093
- **Jaeger**: http://localhost:16686

## Dashboards

### HistoCore Overview
- Slide processing rate
- Processing time percentiles
- Memory and GPU usage
- Error rates

### Performance Metrics
- Detailed processing duration
- Attention computation time
- Model quality metrics
- PACS response times
- Cache hit rates

### GPU Metrics
- GPU memory usage and utilization
- Processing throughput
- Patch processing rates
- Out of memory events

## Alerts

### Performance Alerts
- High processing time (>30s warning, >60s critical)
- Low throughput (<100 patches/sec warning, <50 critical)
- Slow attention computation (>0.5s warning, >1.0s critical)

### Resource Alerts
- High memory usage (>80% warning, >90% critical)
- High GPU memory usage (>85% warning, >95% critical)
- High CPU usage (>70% warning, >90% critical)

### Quality Alerts
- Low model confidence (<0.7 warning, <0.5 critical)
- High early stopping rate (>80%)
- High error rates (>0.1/sec warning, >0.5/sec critical)

### Business Alerts
- Processing backlog (no slides processed in 10min)
- High failure rate (>10% warning, >25% critical)
- SLA violation (>30s end-to-end processing)

## Configuration

### Prometheus
- Scrape interval: 15s
- Retention: 15 days
- Alert evaluation: 15s

### Grafana
- Auto-provisioned dashboards
- Prometheus datasource
- Admin password: admin (change in production)

### Alertmanager
- Email notifications
- Slack integration
- PagerDuty integration
- Alert grouping and routing

## Health Checks

HistoCore provides comprehensive health check endpoints:

### Endpoints
- `/health` - Basic health status
- `/health/live` - Kubernetes liveness probe
- `/health/ready` - Kubernetes readiness probe  
- `/health/detailed` - Full component status
- `/status` - HTML status page

### Components Monitored
- System resources (CPU, memory, disk)
- GPU availability and memory
- Metrics collection system
- Application dependencies

## Metrics Reference

### Processing Metrics
```
histocore_slides_processed_total - Total slides processed
histocore_patches_processed_total - Total patches processed
histocore_processing_duration_seconds - Processing time histogram
histocore_throughput_patches_per_second - Current throughput
```

### Memory Metrics
```
histocore_memory_usage_bytes - System memory usage
histocore_gpu_memory_usage_bytes - GPU memory usage
```

### Quality Metrics
```
histocore_confidence_score - Model confidence histogram
histocore_early_stopping_rate - Early stopping rate
histocore_attention_computation_seconds - Attention time
```

### Error Metrics
```
histocore_errors_total - Error counter by type/component
histocore_oom_events_total - Out of memory events
```

### Network Metrics
```
histocore_pacs_requests_total - PACS request counter
histocore_pacs_response_time_seconds - PACS response time
```

### Cache Metrics
```
histocore_cache_hits_total - Cache hits
histocore_cache_misses_total - Cache misses
```

## Tracing

### OpenTelemetry Integration
- Automatic instrumentation for requests/asyncio
- Manual span creation for custom operations
- Jaeger export for visualization

### Traced Operations
- WSI loading and processing
- GPU batch processing
- Attention computation
- PACS operations
- Cache operations

### Usage
```python
from src.streaming.tracing import traced_operation, trace_span

@traced_operation('wsi.process', 'wsi_processor')
def process_slide(slide_path):
    # Automatically traced
    pass

# Manual tracing
with trace_span('custom_operation', {'param': value}):
    # Custom operation
    pass
```

## Logging

### Structured Logging
- JSON format for machine parsing
- Correlation IDs for request tracing
- Component-based organization
- Performance metadata

### Log Levels
- DEBUG: Detailed debugging info
- INFO: General operational messages
- WARNING: Potential issues
- ERROR: Error conditions
- CRITICAL: System failures

### Usage
```python
from src.streaming.logging_config import get_streaming_logger

logger = get_streaming_logger('component_name')
logger.log_slide_processing_start(slide_id, slide_path)
```

## Production Deployment

### Security
- Change default passwords
- Enable TLS for external access
- Configure authentication
- Restrict network access

### Scaling
- Use external Prometheus for HA
- Configure Grafana clustering
- Set up alert routing
- Monitor resource usage

### Backup
- Export Grafana dashboards
- Backup Prometheus data
- Save alert configurations
- Document runbooks

## Troubleshooting

### Common Issues

1. **Metrics not appearing**
   - Check Prometheus targets
   - Verify network connectivity
   - Check firewall rules

2. **Alerts not firing**
   - Verify alert rules syntax
   - Check Alertmanager config
   - Test notification channels

3. **High resource usage**
   - Adjust scrape intervals
   - Reduce retention periods
   - Optimize queries

### Debug Commands
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Test alert rules
docker exec histocore-prometheus promtool check rules /etc/prometheus/alerts/*.yml

# Check Grafana logs
docker logs histocore-grafana

# Validate Alertmanager config
docker exec histocore-alertmanager amtool config check
```