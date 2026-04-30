"""
Production monitoring and alerting system for federated learning.

Provides comprehensive monitoring of federated learning systems with
real-time metrics collection, alerting, and dashboard integration.
"""

import json
import logging
import smtplib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil
import requests

from src.utils.safe_threading import GracefulThread

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Alert delivery channels."""

    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class Alert:
    """Alert message."""

    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "federated_learning"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class SystemMetrics:
    """System performance metrics."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    network_bytes_sent: Optional[int] = None
    network_bytes_recv: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": self.memory_used_gb,
            "memory_total_gb": self.memory_total_gb,
            "disk_percent": self.disk_percent,
            "disk_used_gb": self.disk_used_gb,
            "disk_total_gb": self.disk_total_gb,
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_percent": self.gpu_memory_percent,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
        }


@dataclass
class FederatedLearningMetrics:
    """Federated learning specific metrics."""

    timestamp: datetime
    round_id: int
    active_clients: int
    total_clients: int
    round_duration_seconds: float
    aggregation_time_seconds: float
    model_accuracy: Optional[float] = None
    model_loss: Optional[float] = None
    convergence_rate: Optional[float] = None
    communication_overhead_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "round_id": self.round_id,
            "active_clients": self.active_clients,
            "total_clients": self.total_clients,
            "round_duration_seconds": self.round_duration_seconds,
            "aggregation_time_seconds": self.aggregation_time_seconds,
            "model_accuracy": self.model_accuracy,
            "model_loss": self.model_loss,
            "convergence_rate": self.convergence_rate,
            "communication_overhead_mb": self.communication_overhead_mb,
        }


class SlackAlerter:
    """Slack webhook alerting."""

    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        """
        Initialize Slack alerter.

        Args:
            webhook_url: Slack webhook URL
            channel: Slack channel name
        """
        self.webhook_url = webhook_url
        self.channel = channel

    def send_alert(self, alert: Alert) -> bool:
        """
        Send alert to Slack.

        Args:
            alert: Alert to send

        Returns:
            True if successful
        """
        try:
            # Color coding by severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",  # Green
                AlertSeverity.WARNING: "#ff9500",  # Orange
                AlertSeverity.ERROR: "#ff0000",  # Red
                AlertSeverity.CRITICAL: "#8B0000",  # Dark red
            }

            # Create Slack message
            payload = {
                "channel": self.channel,
                "username": "FL Monitor",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#cccccc"),
                        "title": f"[{alert.severity.upper()}] {alert.title}",
                        "text": alert.message,
                        "fields": [
                            {"title": "Source", "value": alert.source, "short": True},
                            {
                                "title": "Timestamp",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True,
                            },
                        ],
                        "footer": "Federated Learning Monitor",
                        "ts": int(alert.timestamp.timestamp()),
                    }
                ],
            }

            # Add metadata fields
            if alert.metadata:
                for key, value in alert.metadata.items():
                    payload["attachments"][0]["fields"].append(
                        {"title": key.replace("_", " ").title(), "value": str(value), "short": True}
                    )

            # Send to Slack
            response = requests.post(self.webhook_url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.info(f"Slack alert sent: {alert.title}")
                return True
            else:
                logger.error(f"Slack alert failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Slack alert error: {e}")
            return False


class EmailAlerter:
    """Email SMTP alerting."""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: List[str],
    ):
        """
        Initialize email alerter.

        Args:
            smtp_server: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: From email address
            to_emails: List of recipient email addresses
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails

    def send_alert(self, alert: Alert) -> bool:
        """
        Send alert via email.

        Args:
            alert: Alert to send

        Returns:
            True if successful
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)
            msg["Subject"] = f"[{alert.severity.upper()}] FL Alert: {alert.title}"

            # Create HTML body
            html_body = f"""
            <html>
            <body>
                <h2 style="color: {'red' if alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL] else 'orange' if alert.severity == AlertSeverity.WARNING else 'green'};">
                    [{alert.severity.upper()}] {alert.title}
                </h2>
                <p><strong>Message:</strong> {alert.message}</p>
                <p><strong>Source:</strong> {alert.source}</p>
                <p><strong>Timestamp:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                {f'<h3>Additional Information:</h3><ul>' + ''.join([f'<li><strong>{k.replace("_", " ").title()}:</strong> {v}</li>' for k, v in alert.metadata.items()]) + '</ul>' if alert.metadata else ''}
                
                <hr>
                <p><em>This alert was generated by the Federated Learning Monitoring System.</em></p>
            </body>
            </html>
            """

            msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Email alert sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Email alert error: {e}")
            return False


class WebhookAlerter:
    """Custom webhook alerting."""

    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        """
        Initialize webhook alerter.

        Args:
            webhook_url: Webhook URL
            headers: Optional HTTP headers
        """
        self.webhook_url = webhook_url
        self.headers = headers or {}

    def send_alert(self, alert: Alert) -> bool:
        """
        Send alert to webhook.

        Args:
            alert: Alert to send

        Returns:
            True if successful
        """
        try:
            payload = alert.to_dict()

            response = requests.post(
                self.webhook_url, json=payload, headers=self.headers, timeout=10
            )

            if response.status_code in [200, 201, 202]:
                logger.info(f"Webhook alert sent: {alert.title}")
                return True
            else:
                logger.error(f"Webhook alert failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Webhook alert error: {e}")
            return False


class SystemMonitor:
    """System resource monitoring."""

    def __init__(self):
        """Initialize system monitor."""
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import pynvml

            pynvml.nvmlInit()
            return True
        except ImportError:
            logger.warning("GPU monitoring unavailable (pynvml not installed)")
            return False
        except Exception:
            logger.warning("GPU monitoring unavailable (no NVIDIA GPU)")
            return False

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Network (optional)
        try:
            network = psutil.net_io_counters()
            network_sent = network.bytes_sent
            network_recv = network.bytes_recv
        except Exception:
            network_sent = None
            network_recv = None

        # GPU metrics (if available)
        gpu_util = None
        gpu_memory = None

        if self.gpu_available:
            try:
                import pynvml

                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory = (memory_info.used / memory_info.total) * 100
            except Exception as e:
                logger.debug(f"GPU metrics error: {e}")

        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_percent=disk.percent,
            disk_used_gb=disk.used / (1024**3),
            disk_total_gb=disk.total / (1024**3),
            gpu_utilization=gpu_util,
            gpu_memory_percent=gpu_memory,
            network_bytes_sent=network_sent,
            network_bytes_recv=network_recv,
        )


class FederatedLearningMonitor:
    """
    Comprehensive monitoring system for federated learning.

    Features:
    - Real-time system metrics collection
    - FL-specific metrics tracking
    - Multi-channel alerting (Slack, email, webhook)
    - Configurable thresholds
    - Metric persistence and querying
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize FL monitor.

        Args:
            config_path: Path to monitoring configuration file
        """
        self.config = self._load_config(config_path)
        self.system_monitor = SystemMonitor()
        self.alerters = self._setup_alerters()

        # Metrics storage
        self.system_metrics_history: List[SystemMetrics] = []
        self.fl_metrics_history: List[FederatedLearningMetrics] = []

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[GracefulThread] = None

        logger.info("Federated Learning Monitor initialized")

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load monitoring configuration."""
        default_config = {
            "monitoring_interval_seconds": 30,
            "metrics_retention_hours": 24,
            "thresholds": {
                "cpu_percent": 80,
                "memory_percent": 85,
                "disk_percent": 90,
                "gpu_utilization": 95,
                "round_duration_minutes": 30,
                "client_dropout_rate": 0.3,
            },
            "alerting": {"enabled": True, "channels": ["log"], "rate_limit_minutes": 5},
        }

        if config_path and config_path.exists():
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return default_config

    def _setup_alerters(self) -> Dict[AlertChannel, Any]:
        """Setup alert channels."""
        alerters = {}

        # Always include log alerter
        alerters[AlertChannel.LOG] = logger

        # Setup other alerters based on config
        alerting_config = self.config.get("alerting", {})

        # Slack
        if "slack" in alerting_config:
            slack_config = alerting_config["slack"]
            if slack_config.get("webhook_url"):
                alerters[AlertChannel.SLACK] = SlackAlerter(
                    webhook_url=slack_config["webhook_url"],
                    channel=slack_config.get("channel", "#alerts"),
                )

        # Email
        if "email" in alerting_config:
            email_config = alerting_config["email"]
            if all(
                k in email_config
                for k in ["smtp_server", "username", "password", "from_email", "to_emails"]
            ):
                alerters[AlertChannel.EMAIL] = EmailAlerter(
                    smtp_server=email_config["smtp_server"],
                    smtp_port=email_config.get("smtp_port", 587),
                    username=email_config["username"],
                    password=email_config["password"],
                    from_email=email_config["from_email"],
                    to_emails=email_config["to_emails"],
                )

        # Webhook
        if "webhook" in alerting_config:
            webhook_config = alerting_config["webhook"]
            if webhook_config.get("url"):
                alerters[AlertChannel.WEBHOOK] = WebhookAlerter(
                    webhook_url=webhook_config["url"], headers=webhook_config.get("headers", {})
                )

        return alerters

    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        
        def cleanup_callback():
            """Cleanup callback for graceful shutdown."""
            logger.info("Production monitoring cleanup completed")
        
        self.monitoring_thread = GracefulThread(
            target=self._monitoring_loop,
            name="production_monitor",
            daemon=False,
            cleanup_callback=cleanup_callback
        )
        self.monitoring_thread.start()

        logger.info("Started continuous monitoring")

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            if not self.monitoring_thread.stop(timeout=5.0):
                logger.warning("Production monitor thread did not stop within timeout")
            self.monitoring_thread = None

        logger.info("Stopped continuous monitoring")

    def _monitoring_loop(self, thread: GracefulThread):
        """Main monitoring loop.
        
        Args:
            thread: GracefulThread instance for shutdown coordination
        """
        interval = self.config.get("monitoring_interval_seconds", 30)

        while not thread.should_stop():
            try:
                # Collect system metrics
                system_metrics = self.system_monitor.get_system_metrics()
                self.system_metrics_history.append(system_metrics)

                # Check thresholds and generate alerts
                self._check_system_thresholds(system_metrics)

                # Clean old metrics
                self._cleanup_old_metrics()

                # Sleep for interval - exit immediately if stop requested
                if thread.wait_or_stop(interval):
                    break

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                if thread.wait_or_stop(interval):
                    break

    def record_fl_metrics(self, metrics: FederatedLearningMetrics):
        """
        Record federated learning metrics.

        Args:
            metrics: FL metrics to record
        """
        self.fl_metrics_history.append(metrics)

        # Check FL-specific thresholds
        self._check_fl_thresholds(metrics)

        logger.debug(f"Recorded FL metrics for round {metrics.round_id}")

    def _check_system_thresholds(self, metrics: SystemMetrics):
        """Check system metrics against thresholds."""
        thresholds = self.config.get("thresholds", {})

        # CPU threshold
        if metrics.cpu_percent > thresholds.get("cpu_percent", 80):
            self.send_alert(
                Alert(
                    severity=AlertSeverity.WARNING,
                    title="High CPU Usage",
                    message=f"CPU usage is {metrics.cpu_percent:.1f}%",
                    metadata={"cpu_percent": metrics.cpu_percent},
                )
            )

        # Memory threshold
        if metrics.memory_percent > thresholds.get("memory_percent", 85):
            self.send_alert(
                Alert(
                    severity=AlertSeverity.WARNING,
                    title="High Memory Usage",
                    message=f"Memory usage is {metrics.memory_percent:.1f}%",
                    metadata={"memory_percent": metrics.memory_percent},
                )
            )

        # Disk threshold
        if metrics.disk_percent > thresholds.get("disk_percent", 90):
            self.send_alert(
                Alert(
                    severity=AlertSeverity.ERROR,
                    title="High Disk Usage",
                    message=f"Disk usage is {metrics.disk_percent:.1f}%",
                    metadata={"disk_percent": metrics.disk_percent},
                )
            )

        # GPU threshold (if available)
        if metrics.gpu_utilization and metrics.gpu_utilization > thresholds.get(
            "gpu_utilization", 95
        ):
            self.send_alert(
                Alert(
                    severity=AlertSeverity.WARNING,
                    title="High GPU Utilization",
                    message=f"GPU utilization is {metrics.gpu_utilization:.1f}%",
                    metadata={"gpu_utilization": metrics.gpu_utilization},
                )
            )

    def _check_fl_thresholds(self, metrics: FederatedLearningMetrics):
        """Check FL metrics against thresholds."""
        thresholds = self.config.get("thresholds", {})

        # Round duration threshold
        round_duration_minutes = metrics.round_duration_seconds / 60
        if round_duration_minutes > thresholds.get("round_duration_minutes", 30):
            self.send_alert(
                Alert(
                    severity=AlertSeverity.WARNING,
                    title="Long Training Round",
                    message=f"Round {metrics.round_id} took {round_duration_minutes:.1f} minutes",
                    metadata={
                        "round_id": metrics.round_id,
                        "duration_minutes": round_duration_minutes,
                    },
                )
            )

        # Client participation threshold
        if metrics.total_clients > 0:
            participation_rate = metrics.active_clients / metrics.total_clients
            dropout_rate = 1 - participation_rate

            if dropout_rate > thresholds.get("client_dropout_rate", 0.3):
                self.send_alert(
                    Alert(
                        severity=AlertSeverity.ERROR,
                        title="High Client Dropout",
                        message=f"Only {metrics.active_clients}/{metrics.total_clients} clients participated in round {metrics.round_id}",
                        metadata={
                            "round_id": metrics.round_id,
                            "active_clients": metrics.active_clients,
                            "total_clients": metrics.total_clients,
                            "dropout_rate": dropout_rate,
                        },
                    )
                )

    def send_alert(self, alert: Alert):
        """
        Send alert through configured channels.

        Args:
            alert: Alert to send
        """
        if not self.config.get("alerting", {}).get("enabled", True):
            return

        # Rate limiting (simple implementation)
        # TODO: Implement proper rate limiting per alert type

        channels = self.config.get("alerting", {}).get("channels", ["log"])

        for channel_name in channels:
            try:
                channel = AlertChannel(channel_name)
                alerter = self.alerters.get(channel)

                if channel == AlertChannel.LOG:
                    # Log alert
                    log_level = {
                        AlertSeverity.INFO: logging.INFO,
                        AlertSeverity.WARNING: logging.WARNING,
                        AlertSeverity.ERROR: logging.ERROR,
                        AlertSeverity.CRITICAL: logging.CRITICAL,
                    }.get(alert.severity, logging.INFO)

                    logger.log(
                        log_level, f"[{alert.severity.upper()}] {alert.title}: {alert.message}"
                    )

                elif alerter:
                    # Send through external alerter
                    alerter.send_alert(alert)

            except Exception as e:
                logger.error(f"Failed to send alert via {channel_name}: {e}")

    def _cleanup_old_metrics(self):
        """Remove old metrics based on retention policy."""
        retention_hours = self.config.get("metrics_retention_hours", 24)
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)

        # Clean system metrics
        self.system_metrics_history = [
            m for m in self.system_metrics_history if m.timestamp > cutoff_time
        ]

        # Clean FL metrics
        self.fl_metrics_history = [m for m in self.fl_metrics_history if m.timestamp > cutoff_time]

    def get_recent_metrics(self, hours: int = 1) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get recent metrics.

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary with system and FL metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_system = [
            m.to_dict() for m in self.system_metrics_history if m.timestamp > cutoff_time
        ]

        recent_fl = [m.to_dict() for m in self.fl_metrics_history if m.timestamp > cutoff_time]

        return {"system_metrics": recent_system, "fl_metrics": recent_fl}

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall system health status.

        Returns:
            Health status summary
        """
        if not self.system_metrics_history:
            return {"status": "unknown", "message": "No metrics available"}

        latest_metrics = self.system_metrics_history[-1]
        thresholds = self.config.get("thresholds", {})

        # Check critical thresholds
        issues = []

        if latest_metrics.cpu_percent > thresholds.get("cpu_percent", 80):
            issues.append(f"High CPU: {latest_metrics.cpu_percent:.1f}%")

        if latest_metrics.memory_percent > thresholds.get("memory_percent", 85):
            issues.append(f"High Memory: {latest_metrics.memory_percent:.1f}%")

        if latest_metrics.disk_percent > thresholds.get("disk_percent", 90):
            issues.append(f"High Disk: {latest_metrics.disk_percent:.1f}%")

        # Determine overall status
        if not issues:
            status = "healthy"
            message = "All systems operating normally"
        elif len(issues) == 1:
            status = "warning"
            message = f"Issue detected: {issues[0]}"
        else:
            status = "critical"
            message = f"Multiple issues: {', '.join(issues)}"

        return {
            "status": status,
            "message": message,
            "timestamp": latest_metrics.timestamp.isoformat(),
            "issues": issues,
            "metrics": latest_metrics.to_dict(),
        }


# Example usage and configuration
def create_production_monitor(config_path: Optional[Path] = None) -> FederatedLearningMonitor:
    """
    Create production-ready FL monitor.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured FL monitor
    """
    monitor = FederatedLearningMonitor(config_path)
    monitor.start_monitoring()

    return monitor


if __name__ == "__main__":
    # Demo: Create and run monitor
    monitor = create_production_monitor()

    try:
        # Simulate FL metrics
        fl_metrics = FederatedLearningMetrics(
            timestamp=datetime.now(),
            round_id=1,
            active_clients=8,
            total_clients=10,
            round_duration_seconds=120,
            aggregation_time_seconds=5,
            model_accuracy=0.85,
            model_loss=0.15,
        )

        monitor.record_fl_metrics(fl_metrics)

        # Check health
        health = monitor.get_health_status()
        print(f"System Health: {health}")

        # Keep running
        time.sleep(60)

    except KeyboardInterrupt:
        monitor.stop_monitoring()
