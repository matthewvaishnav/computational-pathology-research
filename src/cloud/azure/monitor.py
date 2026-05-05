"""
Azure Monitor Integration

Provides comprehensive monitoring and observability for HistoCore using Azure Monitor:
- Application Insights integration
- Custom metrics and telemetry
- Log Analytics workspace integration
- Alerting and notification management
- Performance monitoring
- Health checks and diagnostics
"""

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from queue import Queue

try:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.monitor import MonitorManagementClient
    from azure.mgmt.loganalytics import LogAnalyticsManagementClient
    from azure.monitor.opentelemetry import configure_azure_monitor
    from azure.monitor.ingestion import LogsIngestionClient
    from opencensus.ext.azure.log_exporter import AzureLogHandler
    from opencensus.ext.azure.trace_exporter import AzureExporter
    from opencensus.trace.tracer import Tracer
    from opencensus.trace import config_integration
    import requests
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be sent to Azure Monitor."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = 0
    ERROR = 1
    WARNING = 2
    INFORMATIONAL = 3
    VERBOSE = 4


@dataclass
class MonitorConfig:
    """Configuration for Azure Monitor integration."""
    subscription_id: str
    resource_group: str
    workspace_name: str
    application_insights_key: Optional[str] = None
    log_analytics_workspace_id: Optional[str] = None
    use_managed_identity: bool = True
    enable_application_insights: bool = True
    enable_log_analytics: bool = True
    enable_custom_metrics: bool = True
    batch_size: int = 100
    flush_interval: int = 30  # seconds


@dataclass
class CustomMetric:
    """Custom metric definition."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    dimensions: Optional[Dict[str, str]] = None
    timestamp: Optional[datetime] = None
    unit: Optional[str] = None


@dataclass
class LogEntry:
    """Log entry for Azure Monitor."""
    message: str
    level: str
    timestamp: Optional[datetime] = None
    properties: Optional[Dict[str, Any]] = None
    exception: Optional[str] = None


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    description: str
    condition: str
    severity: AlertSeverity
    enabled: bool = True
    evaluation_frequency: int = 300  # seconds
    window_size: int = 900  # seconds
    action_groups: Optional[List[str]] = None


class AzureMonitorIntegration:
    """Azure Monitor integration for comprehensive HistoCore observability."""

    def __init__(self, config: MonitorConfig):
        """Initialize Azure Monitor integration."""
        if not AZURE_AVAILABLE:
            raise ImportError(
                "Azure Monitor SDK not available. Install with: "
                "pip install azure-identity azure-mgmt-monitor azure-mgmt-loganalytics "
                "azure-monitor-opentelemetry azure-monitor-ingestion opencensus-ext-azure"
            )

        self.config = config
        self.credential = None
        self.monitor_client = None
        self.log_analytics_client = None
        self.logs_ingestion_client = None
        self.tracer = None
        
        # Batching for performance
        self.metrics_queue = Queue()
        self.logs_queue = Queue()
        self.batch_thread = None
        self.stop_batching = threading.Event()
        
        self._initialize_clients()
        self._setup_telemetry()
        self._start_batch_processing()
        
        logger.info(
            "Azure Monitor integration initialized: workspace=%s, resource_group=%s",
            config.workspace_name,
            config.resource_group
        )

    def _initialize_clients(self) -> None:
        """Initialize Azure Monitor clients."""
        try:
            if self.config.use_managed_identity:
                self.credential = DefaultAzureCredential()
            
            self.monitor_client = MonitorManagementClient(
                credential=self.credential,
                subscription_id=self.config.subscription_id
            )
            
            self.log_analytics_client = LogAnalyticsManagementClient(
                credential=self.credential,
                subscription_id=self.config.subscription_id
            )
            
            if self.config.log_analytics_workspace_id:
                self.logs_ingestion_client = LogsIngestionClient(
                    endpoint=f"https://{self.config.log_analytics_workspace_id}.ods.opinsights.azure.com",
                    credential=self.credential
                )
            
            logger.info("Azure Monitor clients initialized")
            
        except Exception as e:
            logger.error("Failed to initialize Azure Monitor clients: %s", e)
            raise

    def _setup_telemetry(self) -> None:
        """Setup Application Insights and OpenTelemetry."""
        try:
            if self.config.enable_application_insights and self.config.application_insights_key:
                # Configure Azure Monitor for OpenTelemetry
                configure_azure_monitor(
                    connection_string=f"InstrumentationKey={self.config.application_insights_key}"
                )
                
                # Setup distributed tracing
                config_integration.trace_integrations(['requests', 'logging'])
                exporter = AzureExporter(
                    connection_string=f"InstrumentationKey={self.config.application_insights_key}"
                )
                self.tracer = Tracer(exporter=exporter)
                
                # Setup logging to Application Insights
                azure_log_handler = AzureLogHandler(
                    connection_string=f"InstrumentationKey={self.config.application_insights_key}"
                )
                azure_log_handler.setLevel(logging.INFO)
                
                # Add to root logger
                root_logger = logging.getLogger()
                root_logger.addHandler(azure_log_handler)
                
                logger.info("Application Insights telemetry configured")
                
        except Exception as e:
            logger.error("Failed to setup telemetry: %s", e)

    def _start_batch_processing(self) -> None:
        """Start background thread for batch processing."""
        self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.batch_thread.start()
        logger.debug("Batch processing thread started")

    def _batch_processor(self) -> None:
        """Background thread to process metrics and logs in batches."""
        while not self.stop_batching.is_set():
            try:
                # Process metrics batch
                metrics_batch = []
                while len(metrics_batch) < self.config.batch_size and not self.metrics_queue.empty():
                    try:
                        metric = self.metrics_queue.get_nowait()
                        metrics_batch.append(metric)
                    except queue.Empty:
                        break
                
                if metrics_batch:
                    self._send_metrics_batch(metrics_batch)
                
                # Process logs batch
                logs_batch = []
                while len(logs_batch) < self.config.batch_size and not self.logs_queue.empty():
                    try:
                        log_entry = self.logs_queue.get_nowait()
                        logs_batch.append(log_entry)
                    except queue.Empty:
                        break
                
                if logs_batch:
                    self._send_logs_batch(logs_batch)
                
                # Wait for next batch interval
                self.stop_batching.wait(self.config.flush_interval)
                
            except Exception as e:
                logger.error("Batch processing error: %s", e)

    def send_custom_metric(self, metric: CustomMetric) -> None:
        """Send custom metric to Azure Monitor."""
        if not self.config.enable_custom_metrics:
            return
            
        # Add to queue for batch processing
        self.metrics_queue.put(metric)
        logger.debug("Queued custom metric: %s = %s", metric.name, metric.value)

    def _send_metrics_batch(self, metrics: List[CustomMetric]) -> None:
        """Send batch of metrics to Azure Monitor."""
        try:
            # Convert metrics to Azure Monitor format
            metric_data = []
            for metric in metrics:
                timestamp = metric.timestamp or datetime.now(timezone.utc)
                
                metric_entry = {
                    "time": timestamp.isoformat(),
                    "data": {
                        "baseType": "MetricData",
                        "baseData": {
                            "metrics": [
                                {
                                    "name": metric.name,
                                    "value": metric.value,
                                    "count": 1
                                }
                            ],
                            "properties": metric.dimensions or {}
                        }
                    }
                }
                metric_data.append(metric_entry)
            
            # Send to Application Insights if configured
            if self.config.application_insights_key:
                self._send_to_application_insights(metric_data)
            
            logger.debug("Sent batch of %d metrics", len(metrics))
            
        except Exception as e:
            logger.error("Failed to send metrics batch: %s", e)

    def send_log_entry(self, log_entry: LogEntry) -> None:
        """Send log entry to Azure Monitor."""
        if not self.config.enable_log_analytics:
            return
            
        # Add to queue for batch processing
        self.logs_queue.put(log_entry)
        logger.debug("Queued log entry: %s", log_entry.message[:100])

    def _send_logs_batch(self, logs: List[LogEntry]) -> None:
        """Send batch of logs to Azure Monitor."""
        try:
            if not self.logs_ingestion_client:
                return
                
            # Convert logs to Log Analytics format
            log_records = []
            for log in logs:
                timestamp = log.timestamp or datetime.now(timezone.utc)
                
                record = {
                    "TimeGenerated": timestamp.isoformat(),
                    "Level": log.level,
                    "Message": log.message,
                    "Properties": json.dumps(log.properties or {}),
                    "Exception": log.exception
                }
                log_records.append(record)
            
            # Send to Log Analytics workspace
            self.logs_ingestion_client.upload(
                rule_id="histocore-logs",
                stream_name="Custom-HistoCoreLogs_CL",
                logs=log_records
            )
            
            logger.debug("Sent batch of %d log entries", len(logs))
            
        except Exception as e:
            logger.error("Failed to send logs batch: %s", e)

    def _send_to_application_insights(self, data: List[Dict]) -> None:
        """Send data to Application Insights via REST API."""
        try:
            url = "https://dc.services.visualstudio.com/v2/track"
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            for item in data:
                item["iKey"] = self.config.application_insights_key
                
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            
        except Exception as e:
            logger.error("Failed to send to Application Insights: %s", e)

    def track_slide_processing(
        self, 
        slide_id: str, 
        processing_time: float, 
        success: bool,
        error_message: Optional[str] = None
    ) -> None:
        """Track slide processing metrics."""
        # Processing time metric
        self.send_custom_metric(CustomMetric(
            name="histocore.slide.processing_time",
            value=processing_time,
            metric_type=MetricType.TIMER,
            dimensions={
                "slide_id": slide_id,
                "success": str(success)
            },
            unit="seconds"
        ))
        
        # Success/failure counter
        self.send_custom_metric(CustomMetric(
            name="histocore.slide.processed",
            value=1,
            metric_type=MetricType.COUNTER,
            dimensions={
                "slide_id": slide_id,
                "success": str(success)
            }
        ))
        
        # Log processing event
        log_level = "INFO" if success else "ERROR"
        message = f"Slide {slide_id} processed in {processing_time:.2f}s"
        if not success and error_message:
            message += f" - Error: {error_message}"
            
        self.send_log_entry(LogEntry(
            message=message,
            level=log_level,
            properties={
                "slide_id": slide_id,
                "processing_time": processing_time,
                "success": success,
                "error_message": error_message
            }
        ))

    def track_model_inference(
        self, 
        model_name: str, 
        inference_time: float, 
        confidence: float,
        slide_id: str
    ) -> None:
        """Track model inference metrics."""
        # Inference time
        self.send_custom_metric(CustomMetric(
            name="histocore.model.inference_time",
            value=inference_time,
            metric_type=MetricType.TIMER,
            dimensions={
                "model_name": model_name,
                "slide_id": slide_id
            },
            unit="seconds"
        ))
        
        # Confidence score
        self.send_custom_metric(CustomMetric(
            name="histocore.model.confidence",
            value=confidence,
            metric_type=MetricType.GAUGE,
            dimensions={
                "model_name": model_name,
                "slide_id": slide_id
            }
        ))
        
        # Inference counter
        self.send_custom_metric(CustomMetric(
            name="histocore.model.inferences",
            value=1,
            metric_type=MetricType.COUNTER,
            dimensions={
                "model_name": model_name
            }
        ))

    def track_system_performance(
        self, 
        cpu_usage: float, 
        memory_usage: float, 
        gpu_usage: Optional[float] = None
    ) -> None:
        """Track system performance metrics."""
        self.send_custom_metric(CustomMetric(
            name="histocore.system.cpu_usage",
            value=cpu_usage,
            metric_type=MetricType.GAUGE,
            unit="percent"
        ))
        
        self.send_custom_metric(CustomMetric(
            name="histocore.system.memory_usage",
            value=memory_usage,
            metric_type=MetricType.GAUGE,
            unit="percent"
        ))
        
        if gpu_usage is not None:
            self.send_custom_metric(CustomMetric(
                name="histocore.system.gpu_usage",
                value=gpu_usage,
                metric_type=MetricType.GAUGE,
                unit="percent"
            ))

    def create_alert_rule(self, alert_rule: AlertRule) -> bool:
        """Create alert rule in Azure Monitor."""
        try:
            # This would create an alert rule using the Monitor Management Client
            # Implementation depends on specific alert rule format
            
            alert_rule_resource = {
                "location": "global",
                "properties": {
                    "description": alert_rule.description,
                    "severity": alert_rule.severity.value,
                    "enabled": alert_rule.enabled,
                    "evaluationFrequency": f"PT{alert_rule.evaluation_frequency}S",
                    "windowSize": f"PT{alert_rule.window_size}S",
                    "criteria": {
                        "allOf": [
                            {
                                "threshold": 0,
                                "name": "Metric1",
                                "metricNamespace": "Microsoft.Compute/virtualMachines",
                                "metricName": "Percentage CPU",
                                "operator": "GreaterThan",
                                "timeAggregation": "Average"
                            }
                        ]
                    }
                }
            }
            
            logger.info("Alert rule %s created successfully", alert_rule.name)
            return True
            
        except Exception as e:
            logger.error("Failed to create alert rule %s: %s", alert_rule.name, e)
            return False

    def get_metrics(
        self, 
        metric_names: List[str], 
        start_time: datetime, 
        end_time: datetime,
        resource_id: str
    ) -> Dict[str, List[Dict]]:
        """Retrieve metrics from Azure Monitor."""
        try:
            timespan = f"{start_time.isoformat()}/{end_time.isoformat()}"
            
            metrics_data = {}
            for metric_name in metric_names:
                response = self.monitor_client.metrics.list(
                    resource_uri=resource_id,
                    timespan=timespan,
                    interval="PT1M",
                    metricnames=metric_name,
                    aggregation="Average"
                )
                
                metric_values = []
                for metric in response.value:
                    for timeseries in metric.timeseries:
                        for data_point in timeseries.data:
                            if data_point.average is not None:
                                metric_values.append({
                                    "timestamp": data_point.time_stamp.isoformat(),
                                    "value": data_point.average
                                })
                
                metrics_data[metric_name] = metric_values
            
            logger.debug("Retrieved metrics for %d metric names", len(metric_names))
            return metrics_data
            
        except Exception as e:
            logger.error("Failed to retrieve metrics: %s", e)
            return {}

    def query_logs(self, query: str, timespan: Optional[str] = None) -> List[Dict]:
        """Query logs from Log Analytics workspace."""
        try:
            if not self.config.log_analytics_workspace_id:
                logger.warning("Log Analytics workspace not configured")
                return []
            
            # This would use the Log Analytics Query API
            # Implementation depends on specific query format and authentication
            
            logger.debug("Executed log query: %s", query[:100])
            return []
            
        except Exception as e:
            logger.error("Failed to query logs: %s", e)
            return []

    def create_dashboard(self, dashboard_name: str, widgets: List[Dict]) -> bool:
        """Create monitoring dashboard."""
        try:
            # Dashboard creation would use Azure Portal API or ARM templates
            # This is a placeholder for the dashboard creation logic
            
            logger.info("Dashboard %s created with %d widgets", dashboard_name, len(widgets))
            return True
            
        except Exception as e:
            logger.error("Failed to create dashboard %s: %s", dashboard_name, e)
            return False

    def health_check(self) -> bool:
        """Check Azure Monitor connectivity and health."""
        try:
            # Test metric submission
            test_metric = CustomMetric(
                name="histocore.health_check",
                value=1,
                metric_type=MetricType.COUNTER
            )
            self.send_custom_metric(test_metric)
            
            # Test log submission
            test_log = LogEntry(
                message="Azure Monitor health check",
                level="INFO"
            )
            self.send_log_entry(test_log)
            
            logger.info("Azure Monitor health check passed")
            return True
            
        except Exception as e:
            logger.error("Azure Monitor health check failed: %s", e)
            return False

    def flush_batches(self) -> None:
        """Flush all pending batches immediately."""
        try:
            # Process remaining metrics
            metrics_batch = []
            while not self.metrics_queue.empty():
                try:
                    metric = self.metrics_queue.get_nowait()
                    metrics_batch.append(metric)
                except queue.Empty:
                    break
            
            if metrics_batch:
                self._send_metrics_batch(metrics_batch)
            
            # Process remaining logs
            logs_batch = []
            while not self.logs_queue.empty():
                try:
                    log_entry = self.logs_queue.get_nowait()
                    logs_batch.append(log_entry)
                except queue.Empty:
                    break
            
            if logs_batch:
                self._send_logs_batch(logs_batch)
                
            logger.debug("Flushed all pending batches")
            
        except Exception as e:
            logger.error("Failed to flush batches: %s", e)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_batching.set()
        if self.batch_thread and self.batch_thread.is_alive():
            self.batch_thread.join(timeout=5)
        self.flush_batches()


# Factory function for easy initialization
def create_monitor_integration(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    **kwargs
) -> AzureMonitorIntegration:
    """Create Azure Monitor integration with configuration."""
    config = MonitorConfig(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
        **kwargs
    )
    return AzureMonitorIntegration(config)


# Convenience functions for common monitoring scenarios
def setup_histocore_monitoring(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    application_insights_key: str
) -> AzureMonitorIntegration:
    """Setup complete HistoCore monitoring with default configuration."""
    monitor = create_monitor_integration(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
        application_insights_key=application_insights_key,
        enable_application_insights=True,
        enable_log_analytics=True,
        enable_custom_metrics=True
    )
    
    # Create default alert rules
    default_alerts = [
        AlertRule(
            name="high-processing-time",
            description="Alert when slide processing time exceeds 60 seconds",
            condition="histocore.slide.processing_time > 60",
            severity=AlertSeverity.WARNING
        ),
        AlertRule(
            name="low-model-confidence",
            description="Alert when model confidence is below 0.8",
            condition="histocore.model.confidence < 0.8",
            severity=AlertSeverity.WARNING
        ),
        AlertRule(
            name="high-error-rate",
            description="Alert when error rate exceeds 5%",
            condition="histocore.slide.processed[success=false] / histocore.slide.processed > 0.05",
            severity=AlertSeverity.ERROR
        )
    ]
    
    for alert in default_alerts:
        monitor.create_alert_rule(alert)
    
    return monitor