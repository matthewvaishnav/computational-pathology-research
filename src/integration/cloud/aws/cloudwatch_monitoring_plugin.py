"""
AWS CloudWatch Monitoring Plugin

Provides comprehensive monitoring and observability for medical AI systems
using CloudWatch metrics, logs, alarms, and dashboards.
"""

import asyncio
import json
import logging
import statistics
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import aioboto3
import boto3
import botocore
from botocore.exceptions import ClientError, NoCredentialsError

from ..plugin_interface import MonitoringPlugin, PluginCapability
from ..plugin_manager import PluginMetadata


class MetricUnit(Enum):
    """CloudWatch metric units"""

    SECONDS = "Seconds"
    MICROSECONDS = "Microseconds"
    MILLISECONDS = "Milliseconds"
    BYTES = "Bytes"
    KILOBYTES = "Kilobytes"
    MEGABYTES = "Megabytes"
    GIGABYTES = "Gigabytes"
    TERABYTES = "Terabytes"
    BITS = "Bits"
    KILOBITS = "Kilobits"
    MEGABITS = "Megabits"
    GIGABITS = "Gigabits"
    TERABITS = "Terabits"
    PERCENT = "Percent"
    COUNT = "Count"
    COUNT_PER_SECOND = "Count/Second"
    NONE = "None"


class AlarmState(Enum):
    """CloudWatch alarm states"""

    OK = "OK"
    ALARM = "ALARM"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ComparisonOperator(Enum):
    """CloudWatch alarm comparison operators"""

    GREATER_THAN_THRESHOLD = "GreaterThanThreshold"
    GREATER_THAN_OR_EQUAL_TO_THRESHOLD = "GreaterThanOrEqualToThreshold"
    LESS_THAN_THRESHOLD = "LessThanThreshold"
    LESS_THAN_OR_EQUAL_TO_THRESHOLD = "LessThanOrEqualToThreshold"


class Statistic(Enum):
    """CloudWatch statistics"""

    SAMPLE_COUNT = "SampleCount"
    AVERAGE = "Average"
    SUM = "Sum"
    MINIMUM = "Minimum"
    MAXIMUM = "Maximum"


@dataclass
class CloudWatchMetric:
    """CloudWatch metric data point"""

    metric_name: str
    namespace: str
    value: float
    unit: MetricUnit
    timestamp: datetime
    dimensions: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["unit"] = self.unit.value
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class CloudWatchAlarm:
    """CloudWatch alarm configuration"""

    alarm_name: str
    alarm_description: str
    metric_name: str
    namespace: str
    statistic: Statistic
    period: int
    evaluation_periods: int
    threshold: float
    comparison_operator: ComparisonOperator
    alarm_actions: List[str]
    ok_actions: List[str]
    dimensions: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["statistic"] = self.statistic.value
        data["comparison_operator"] = self.comparison_operator.value
        return data


@dataclass
class LogEvent:
    """CloudWatch log event"""

    timestamp: datetime
    message: str
    log_group: str
    log_stream: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class Dashboard:
    """CloudWatch dashboard"""

    dashboard_name: str
    dashboard_body: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class AWSCloudWatchMonitoringPlugin(MonitoringPlugin):
    """
    AWS CloudWatch monitoring integration plugin

    Provides comprehensive monitoring including:
    - Custom metrics collection and publishing
    - Log aggregation and analysis
    - Alarm management and notifications
    - Dashboard creation and management
    - Performance monitoring
    - Health checks and alerting
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize AWS CloudWatch monitoring plugin"""
        super().__init__(config)

        # AWS configuration
        self.aws_region = config.get("aws_region", "us-east-1")
        self.aws_access_key_id = config.get("aws_access_key_id")
        self.aws_secret_access_key = config.get("aws_secret_access_key")
        self.aws_session_token = config.get("aws_session_token")

        # CloudWatch settings
        self.namespace = config.get("namespace", "MedicalAI")
        self.log_group_prefix = config.get("log_group_prefix", "/medical-ai/")
        self.retention_days = config.get("retention_days", 30)

        # Metric settings
        self.metric_buffer_size = config.get("metric_buffer_size", 20)
        self.metric_flush_interval = config.get("metric_flush_interval", 60)  # seconds

        # Alarm settings
        self.alarm_prefix = config.get("alarm_prefix", "MedicalAI-")
        self.sns_topic_arn = config.get("sns_topic_arn")

        # Connection settings
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)

        # AWS clients
        self.cloudwatch_client = None
        self.logs_client = None
        self.session = None

        # Metric buffering
        self.metric_buffer = []
        self.last_flush_time = datetime.now()

        # Registered alarms and dashboards
        self.alarms = {}
        self.dashboards = {}

        # Log streams
        self.log_streams = {}

        self.logger = logging.getLogger(__name__)

    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        return PluginMetadata(
            name="aws-cloudwatch-monitoring",
            version="1.0.0",
            description="AWS CloudWatch Monitoring and Observability",
            vendor="Amazon Web Services",
            capabilities=[
                PluginCapability.METRICS_COLLECTION,
                PluginCapability.LOG_AGGREGATION,
                PluginCapability.ALERTING,
                PluginCapability.DASHBOARDS,
                PluginCapability.PERFORMANCE_MONITORING,
                PluginCapability.HEALTH_CHECKS,
            ],
            supported_formats=["JSON", "CloudWatch Logs", "Custom Metrics"],
            configuration_schema={
                "aws_region": {"type": "string", "required": True},
                "namespace": {"type": "string", "default": "MedicalAI"},
                "log_group_prefix": {"type": "string", "default": "/medical-ai/"},
                "sns_topic_arn": {"type": "string", "required": False},
                "aws_access_key_id": {"type": "string", "required": False, "sensitive": True},
                "aws_secret_access_key": {"type": "string", "required": False, "sensitive": True},
            },
        )

    async def initialize(self) -> bool:
        """Initialize AWS CloudWatch connection"""
        try:
            # Create aioboto3 session
            session_kwargs = {"region_name": self.aws_region}

            if self.aws_access_key_id and self.aws_secret_access_key:
                session_kwargs.update(
                    {
                        "aws_access_key_id": self.aws_access_key_id,
                        "aws_secret_access_key": self.aws_secret_access_key,
                    }
                )

                if self.aws_session_token:
                    session_kwargs["aws_session_token"] = self.aws_session_token

            # Create aioboto3 session for async operations
            self.session = aioboto3.Session(**session_kwargs)

            # Create sync boto3 clients for some operations
            boto_session = boto3.Session(**session_kwargs)
            self.cloudwatch_client = boto_session.client("cloudwatch")
            self.logs_client = boto_session.client("logs")

            # Test connection
            if not await self._test_connection():
                self.logger.error("CloudWatch connection test failed")
                return False

            # Setup log groups
            await self._setup_log_groups()

            # Start metric flushing task
            asyncio.create_task(self._metric_flush_task())

            self.logger.info("AWS CloudWatch monitoring plugin initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"AWS CloudWatch initialization failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup resources"""
        # Flush remaining metrics
        await self._flush_metrics()

    async def _test_connection(self) -> bool:
        """Test CloudWatch connection"""
        try:
            async with self.session.client("cloudwatch") as cloudwatch:
                # Test by listing metrics
                response = await cloudwatch.list_metrics(Namespace=self.namespace, MaxRecords=1)
                return "Metrics" in response

        except Exception as e:
            self.logger.error(f"CloudWatch connection test failed: {e}")
            return False

    async def _setup_log_groups(self):
        """Setup CloudWatch log groups"""
        try:
            # Create main log groups
            log_groups = [
                f"{self.log_group_prefix}application",
                f"{self.log_group_prefix}inference",
                f"{self.log_group_prefix}training",
                f"{self.log_group_prefix}errors",
            ]

            for log_group in log_groups:
                try:
                    self.logs_client.create_log_group(
                        logGroupName=log_group, retentionInDays=self.retention_days
                    )
                    self.logger.info(f"Created log group: {log_group}")
                except ClientError as e:
                    if e.response["Error"]["Code"] != "ResourceAlreadyExistsException":
                        raise

        except Exception as e:
            self.logger.error(f"Error setting up log groups: {e}")

    async def put_metric(
        self,
        metric_name: str,
        value: float,
        unit: MetricUnit,
        dimensions: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Put custom metric to CloudWatch"""
        try:
            timestamp = timestamp or datetime.now()

            metric = CloudWatchMetric(
                metric_name=metric_name,
                namespace=self.namespace,
                value=value,
                unit=unit,
                timestamp=timestamp,
                dimensions=dimensions or {},
            )

            # Add to buffer
            self.metric_buffer.append(metric)

            # Flush if buffer is full
            if len(self.metric_buffer) >= self.metric_buffer_size:
                await self._flush_metrics()

        except Exception as e:
            self.logger.error(f"Error putting metric {metric_name}: {e}")

    async def put_metrics_batch(self, metrics: List[CloudWatchMetric]):
        """Put multiple metrics in batch"""
        try:
            self.metric_buffer.extend(metrics)

            # Flush if buffer is full
            if len(self.metric_buffer) >= self.metric_buffer_size:
                await self._flush_metrics()

        except Exception as e:
            self.logger.error(f"Error putting metrics batch: {e}")

    async def _flush_metrics(self):
        """Flush metrics buffer to CloudWatch"""
        try:
            if not self.metric_buffer:
                return

            # Prepare metric data
            metric_data = []

            for metric in self.metric_buffer:
                metric_datum = {
                    "MetricName": metric.metric_name,
                    "Value": metric.value,
                    "Unit": metric.unit.value,
                    "Timestamp": metric.timestamp,
                }

                if metric.dimensions:
                    metric_datum["Dimensions"] = [
                        {"Name": k, "Value": v} for k, v in metric.dimensions.items()
                    ]

                metric_data.append(metric_datum)

            # Send metrics in batches of 20 (CloudWatch limit)
            async with self.session.client("cloudwatch") as cloudwatch:
                for i in range(0, len(metric_data), 20):
                    batch = metric_data[i : i + 20]

                    await cloudwatch.put_metric_data(Namespace=self.namespace, MetricData=batch)

            # Clear buffer
            self.metric_buffer.clear()
            self.last_flush_time = datetime.now()

            self.logger.debug(f"Flushed {len(metric_data)} metrics to CloudWatch")

        except Exception as e:
            self.logger.error(f"Error flushing metrics: {e}")

    async def _metric_flush_task(self):
        """Background task to flush metrics periodically"""
        try:
            while True:
                await asyncio.sleep(self.metric_flush_interval)

                # Check if it's time to flush
                if (
                    datetime.now() - self.last_flush_time
                ).total_seconds() >= self.metric_flush_interval:
                    await self._flush_metrics()

        except asyncio.CancelledError:
            # Flush remaining metrics on cancellation
            await self._flush_metrics()
        except Exception as e:
            self.logger.error(f"Error in metric flush task: {e}")

    async def put_log_event(
        self, log_group: str, log_stream: str, message: str, timestamp: Optional[datetime] = None
    ):
        """Put log event to CloudWatch Logs"""
        try:
            timestamp = timestamp or datetime.now()

            # Ensure log stream exists
            await self._ensure_log_stream(log_group, log_stream)

            # Get sequence token
            sequence_token = self.log_streams.get(f"{log_group}/{log_stream}")

            async with self.session.client("logs") as logs:
                put_params = {
                    "logGroupName": log_group,
                    "logStreamName": log_stream,
                    "logEvents": [
                        {"timestamp": int(timestamp.timestamp() * 1000), "message": message}
                    ],
                }

                if sequence_token:
                    put_params["sequenceToken"] = sequence_token

                response = await logs.put_log_events(**put_params)

                # Update sequence token
                self.log_streams[f"{log_group}/{log_stream}"] = response.get("nextSequenceToken")

        except Exception as e:
            self.logger.error(f"Error putting log event: {e}")

    async def _ensure_log_stream(self, log_group: str, log_stream: str):
        """Ensure log stream exists"""
        try:
            stream_key = f"{log_group}/{log_stream}"

            if stream_key not in self.log_streams:
                try:
                    self.logs_client.create_log_stream(
                        logGroupName=log_group, logStreamName=log_stream
                    )
                    self.log_streams[stream_key] = None
                except ClientError as e:
                    if e.response["Error"]["Code"] != "ResourceAlreadyExistsException":
                        raise
                    self.log_streams[stream_key] = None

        except Exception as e:
            self.logger.error(f"Error ensuring log stream: {e}")

    async def create_alarm(self, alarm: CloudWatchAlarm) -> bool:
        """Create CloudWatch alarm"""
        try:
            alarm_actions = alarm.alarm_actions.copy()
            ok_actions = alarm.ok_actions.copy()

            # Add SNS topic if configured
            if self.sns_topic_arn:
                if self.sns_topic_arn not in alarm_actions:
                    alarm_actions.append(self.sns_topic_arn)
                if self.sns_topic_arn not in ok_actions:
                    ok_actions.append(self.sns_topic_arn)

            alarm_params = {
                "AlarmName": f"{self.alarm_prefix}{alarm.alarm_name}",
                "AlarmDescription": alarm.alarm_description,
                "MetricName": alarm.metric_name,
                "Namespace": alarm.namespace,
                "Statistic": alarm.statistic.value,
                "Period": alarm.period,
                "EvaluationPeriods": alarm.evaluation_periods,
                "Threshold": alarm.threshold,
                "ComparisonOperator": alarm.comparison_operator.value,
                "AlarmActions": alarm_actions,
                "OKActions": ok_actions,
            }

            if alarm.dimensions:
                alarm_params["Dimensions"] = [
                    {"Name": k, "Value": v} for k, v in alarm.dimensions.items()
                ]

            async with self.session.client("cloudwatch") as cloudwatch:
                await cloudwatch.put_metric_alarm(**alarm_params)

                self.alarms[alarm.alarm_name] = alarm

                self.logger.info(f"Created alarm: {alarm.alarm_name}")
                return True

        except Exception as e:
            self.logger.error(f"Error creating alarm {alarm.alarm_name}: {e}")
            return False

    async def get_alarm_state(self, alarm_name: str) -> Optional[AlarmState]:
        """Get alarm state"""
        try:
            full_alarm_name = f"{self.alarm_prefix}{alarm_name}"

            async with self.session.client("cloudwatch") as cloudwatch:
                response = await cloudwatch.describe_alarms(AlarmNames=[full_alarm_name])

                if response["MetricAlarms"]:
                    state = response["MetricAlarms"][0]["StateValue"]
                    return AlarmState(state)

                return None

        except Exception as e:
            self.logger.error(f"Error getting alarm state: {e}")
            return None

    async def get_metric_statistics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        period: int,
        statistics: List[Statistic],
        dimensions: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get metric statistics"""
        try:
            params = {
                "Namespace": self.namespace,
                "MetricName": metric_name,
                "StartTime": start_time,
                "EndTime": end_time,
                "Period": period,
                "Statistics": [stat.value for stat in statistics],
            }

            if dimensions:
                params["Dimensions"] = [{"Name": k, "Value": v} for k, v in dimensions.items()]

            async with self.session.client("cloudwatch") as cloudwatch:
                response = await cloudwatch.get_metric_statistics(**params)

                return response.get("Datapoints", [])

        except Exception as e:
            self.logger.error(f"Error getting metric statistics: {e}")
            return []

    async def create_dashboard(self, dashboard_name: str, widgets: List[Dict[str, Any]]) -> bool:
        """Create CloudWatch dashboard"""
        try:
            dashboard_body = {"widgets": widgets}

            async with self.session.client("cloudwatch") as cloudwatch:
                await cloudwatch.put_dashboard(
                    DashboardName=dashboard_name, DashboardBody=json.dumps(dashboard_body)
                )

                self.dashboards[dashboard_name] = Dashboard(
                    dashboard_name=dashboard_name, dashboard_body=dashboard_body
                )

                self.logger.info(f"Created dashboard: {dashboard_name}")
                return True

        except Exception as e:
            self.logger.error(f"Error creating dashboard {dashboard_name}: {e}")
            return False

    async def query_logs(
        self, log_group: str, query: str, start_time: datetime, end_time: datetime, limit: int = 100
    ) -> List[LogEvent]:
        """Query CloudWatch Logs"""
        try:
            async with self.session.client("logs") as logs:
                # Start query
                response = await logs.start_query(
                    logGroupName=log_group,
                    startTime=int(start_time.timestamp()),
                    endTime=int(end_time.timestamp()),
                    queryString=query,
                    limit=limit,
                )

                query_id = response["queryId"]

                # Wait for query to complete
                while True:
                    result = await logs.get_query_results(queryId=query_id)

                    if result["status"] == "Complete":
                        break
                    elif result["status"] == "Failed":
                        raise Exception("Query failed")

                    await asyncio.sleep(1)

                # Parse results
                log_events = []
                for result_row in result["results"]:
                    event_data = {field["field"]: field["value"] for field in result_row}

                    log_event = LogEvent(
                        timestamp=datetime.fromtimestamp(
                            int(event_data.get("@timestamp", 0)) / 1000
                        ),
                        message=event_data.get("@message", ""),
                        log_group=log_group,
                        log_stream=event_data.get("@logStream", ""),
                    )
                    log_events.append(log_event)

                return log_events

        except Exception as e:
            self.logger.error(f"Error querying logs: {e}")
            return []

    # Convenience methods for common medical AI metrics
    async def record_inference_time(self, model_name: str, inference_time_ms: float):
        """Record model inference time"""
        await self.put_metric(
            metric_name="InferenceTime",
            value=inference_time_ms,
            unit=MetricUnit.MILLISECONDS,
            dimensions={"ModelName": model_name},
        )

    async def record_accuracy_metric(self, model_name: str, accuracy: float):
        """Record model accuracy"""
        await self.put_metric(
            metric_name="ModelAccuracy",
            value=accuracy,
            unit=MetricUnit.PERCENT,
            dimensions={"ModelName": model_name},
        )

    async def record_error_count(self, error_type: str, count: int = 1):
        """Record error count"""
        await self.put_metric(
            metric_name="ErrorCount",
            value=count,
            unit=MetricUnit.COUNT,
            dimensions={"ErrorType": error_type},
        )

    async def record_processing_volume(self, data_type: str, volume: int):
        """Record data processing volume"""
        await self.put_metric(
            metric_name="ProcessingVolume",
            value=volume,
            unit=MetricUnit.COUNT,
            dimensions={"DataType": data_type},
        )

    async def validate_connection(self) -> Dict[str, Any]:
        """Validate CloudWatch connection"""
        try:
            # Test CloudWatch access
            cloudwatch_accessible = await self._test_connection()

            # Test logs access
            logs_accessible = False
            try:
                async with self.session.client("logs") as logs:
                    await logs.describe_log_groups(limit=1)
                logs_accessible = True
            except Exception as e:
                logger.warning(f"CloudWatch health check failed: {type(e).__name__}")
                pass

            # Test metric publishing
            metrics_writable = False
            try:
                await self.put_metric("TestMetric", 1.0, MetricUnit.COUNT)
                await self._flush_metrics()
                metrics_writable = True
            except:
                pass

            # Test SNS topic if configured
            sns_accessible = True
            if self.sns_topic_arn:
                try:
                    sns_client = boto3.client("sns", region_name=self.aws_region)
                    sns_client.get_topic_attributes(TopicArn=self.sns_topic_arn)
                except Exception as e:
                    logger.warning(f"SNS health check failed: {type(e).__name__}")
                    sns_accessible = False

            return {
                "connected": cloudwatch_accessible and logs_accessible,
                "cloudwatch_accessible": cloudwatch_accessible,
                "logs_accessible": logs_accessible,
                "metrics_writable": metrics_writable,
                "sns_accessible": sns_accessible,
                "namespace": self.namespace,
                "alarms_configured": len(self.alarms),
                "dashboards_configured": len(self.dashboards),
                "aws_region": self.aws_region,
                "last_check": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"connected": False, "error": str(e), "last_check": datetime.now().isoformat()}


# Plugin factory function
def create_plugin(config: Dict[str, Any]) -> AWSCloudWatchMonitoringPlugin:
    """Create AWS CloudWatch monitoring plugin instance"""
    return AWSCloudWatchMonitoringPlugin(config)
