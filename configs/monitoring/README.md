# Federated Learning Monitoring System

This directory contains configuration files for the federated learning monitoring stack, which provides real-time metrics, alerting, and visualization for FL training.

## Components

### 1. Prometheus (Metrics Collection)
- **Purpose**: Collects and stores time-series metrics from FL coordinator and clients
- **Port**: 9090
- **Configuration**: `prometheus.yml`
- **Alert Rules**: `alert_rules.yml`

### 2. Grafana (Visualization)
- **Purpose**: Provides dashboards for visualizing FL metrics
- **Port**: 3000
- **Default Credentials**: admin / federated_learning_2026
- **Dashboard**: `grafana_dashboard.json`
- **Datasource**: `grafana_datasource.yml`

### 3. Alertmanager (Alert Routing)
- **Purpose**: Routes alerts to appropriate channels (email, Slack, PagerDuty)
- **Port**: 9093
- **Configuration**: `alertmanager.yml`

### 4. Node Exporter (System Metrics)
- **Purpose**: Exports system-level metrics (CPU, memory, disk)
- **Port**: 9100

### 5. TensorBoard (Training Visualization)
- **Purpose**: Visualizes training metrics, model parameters, and convergence
- **Port**: 6006 (when running standalone)
- **Log Directory**: `./fl_tensorboard`

## Quick Start

### 1. Start Monitoring Stack

```bash
cd configs/monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

### 2. Access Dashboards

- **Grafana**: http://localhost:3000
  - Login: admin / federated_learning_2026
  - Navigate to "Federated Learning Monitoring" dashboard

- **Prometheus**: http://localhost:9090
  - Query metrics directly
  - View alert status

- **Alertmanager**: http://localhost:9093
  - View active alerts
  - Silence alerts

- **TensorBoard**: 
  ```bash
  tensorboard --logdir=./fl_tensorboard --port=6006
  ```
  Then visit http://localhost:6006

### 3. Configure Alerts

Edit `alertmanager.yml` to configure alert routing:

```yaml
receivers:
  - name: 'critical-alerts'
    email_configs:
      - to: 'your-email@example.com'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK'
        channel: '#alerts'
```

## Metrics Exported

### Training Metrics
- `fl_current_round`: Current training round number
- `fl_model_version`: Current global model version
- `fl_model_loss`: Global model loss
- `fl_model_accuracy`: Global model accuracy
- `fl_gradient_norm`: L2 norm of aggregated gradients

### Performance Metrics
- `fl_round_duration_seconds`: Time taken per training round
- `fl_aggregation_time_seconds`: Time taken for gradient aggregation
- `fl_client_participation_total`: Number of clients participating

### Security Metrics
- `fl_byzantine_detections_total`: Total Byzantine updates detected
- `fl_client_dropout_total`: Total client dropouts

### Privacy Metrics
- `fl_privacy_epsilon`: Current privacy budget (epsilon)

## Alert Rules

### Critical Alerts
- **PrivacyBudgetExhausted**: Privacy budget (epsilon) >= 1.0
- **HighByzantineDetectionRate**: Byzantine detection rate > 20%
- **CoordinatorDown**: FL Coordinator is down

### Warning Alerts
- **HighClientDropoutRate**: Client dropout rate > 50%
- **LowClientParticipation**: < 3 clients participating
- **TrainingStalled**: Loss hasn't changed in 10 minutes
- **ModelPerformanceDegraded**: Loss increased significantly
- **SlowRoundDuration**: Round taking > 10 minutes
- **HighGradientNorm**: Gradient norm > 10.0 (potential instability)

## Grafana Dashboard Panels

1. **Training Progress**: Current round and model version
2. **Model Performance**: Loss and accuracy over time
3. **Client Participation**: Number of active clients
4. **Round Duration**: Time per round and aggregation time
5. **Byzantine Detections**: Detection rate over time
6. **Client Dropouts**: Dropout rate over time
7. **Privacy Budget**: Current epsilon value with alert threshold
8. **Gradient Norm**: Gradient L2 norm over time
9. **System Health**: Overall system status

## Integration with FL Coordinator

### Python Code Example

```python
from src.federated.coordinator.monitoring import (
    PrometheusMetricsExporter,
    TensorBoardLogger,
    ConvergenceDetector,
    AlertGenerator,
    MonitoringDashboard
)

# Initialize monitoring components
prometheus = PrometheusMetricsExporter(metrics_file="./fl_metrics/prometheus.txt")
tensorboard = TensorBoardLogger(log_dir="./fl_tensorboard")
convergence = ConvergenceDetector(patience=5, min_delta=0.001, metric_name="loss")
alerts = AlertGenerator(alert_file="./fl_alerts/alerts.jsonl")

# Create dashboard
dashboard = MonitoringDashboard(prometheus, tensorboard, convergence, alerts)

# Start training round
dashboard.start_round(round_id=1, num_clients=10, model_version=0)

# Start aggregation
dashboard.start_aggregation(round_id=1)
# ... perform aggregation ...
dashboard.end_aggregation(round_id=1)

# End round with metrics
metrics = {"loss": 0.5, "accuracy": 0.95}
dashboard.end_round(round_id=1, metrics=metrics, num_byzantine=2)

# Log additional metrics
dashboard.log_gradient_norm(round_id=1, gradient_norm=2.5)
dashboard.log_privacy_budget(round_id=1, epsilon=0.8, client_id="hospital_a")

# Check convergence
if dashboard.convergence.is_converged():
    print("Training has converged!")

# Get summary
summary = dashboard.get_summary()
print(f"Converged: {summary['converged']}")
print(f"Total alerts: {summary['total_alerts']}")
```

## Troubleshooting

### Prometheus Not Scraping Metrics

1. Check metrics file exists:
   ```bash
   ls -la ./fl_metrics/prometheus.txt
   ```

2. Verify Prometheus configuration:
   ```bash
   docker exec fl_prometheus promtool check config /etc/prometheus/prometheus.yml
   ```

3. Check Prometheus targets:
   - Visit http://localhost:9090/targets
   - Ensure all targets are "UP"

### Grafana Dashboard Not Showing Data

1. Verify Prometheus datasource:
   - Grafana → Configuration → Data Sources
   - Test connection to Prometheus

2. Check metric names in queries:
   - Ensure metric names match those exported by FL coordinator

3. Verify time range:
   - Adjust time range in Grafana to match training period

### Alerts Not Firing

1. Check alert rules:
   ```bash
   docker exec fl_prometheus promtool check rules /etc/prometheus/alert_rules.yml
   ```

2. Verify Alertmanager configuration:
   ```bash
   docker exec fl_alertmanager amtool check-config /etc/alertmanager/alertmanager.yml
   ```

3. Check alert status in Prometheus:
   - Visit http://localhost:9090/alerts

### TensorBoard Not Loading

1. Verify log directory exists:
   ```bash
   ls -la ./fl_tensorboard
   ```

2. Check TensorBoard logs:
   ```bash
   tensorboard --logdir=./fl_tensorboard --port=6006 --debug
   ```

## Production Deployment

### Security Considerations

1. **Change default passwords**:
   - Grafana admin password
   - Alertmanager credentials

2. **Enable TLS**:
   - Configure TLS for Prometheus, Grafana, Alertmanager
   - Use certificates from trusted CA

3. **Restrict access**:
   - Use firewall rules to limit access to monitoring ports
   - Enable authentication for all services

4. **Secure alert channels**:
   - Use encrypted channels (HTTPS, TLS SMTP)
   - Rotate API keys and webhooks regularly

### Scaling

1. **Prometheus**:
   - Increase retention period: `--storage.tsdb.retention.time=30d`
   - Enable remote write for long-term storage (e.g., Thanos, Cortex)

2. **Grafana**:
   - Use external database (PostgreSQL, MySQL) instead of SQLite
   - Enable caching for better performance

3. **Alertmanager**:
   - Deploy in HA mode with multiple replicas
   - Use external storage for silences and notifications

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)

## Support

For issues or questions:
1. Check logs: `docker-compose -f docker-compose.monitoring.yml logs`
2. Review Prometheus targets: http://localhost:9090/targets
3. Check Grafana datasource: http://localhost:3000/datasources
4. Review alert rules: http://localhost:9090/alerts
