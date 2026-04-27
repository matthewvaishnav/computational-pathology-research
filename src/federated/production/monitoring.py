"""Production monitoring and observability."""

import logging
import time
import psutil
import structlog
from datetime import datetime
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Info
import sentry_sdk

from .config import get_config
from .database import get_db_manager

config = get_config()
db_manager = get_db_manager()

# Prometheus metrics
SYSTEM_CPU_USAGE = Gauge('fl_system_cpu_usage_percent', 'System CPU usage percentage')
SYSTEM_MEMORY_USAGE = Gauge('fl_system_memory_usage_percent', 'System memory usage percentage')
SYSTEM_DISK_USAGE = Gauge('fl_system_disk_usage_percent', 'System disk usage percentage')

FL_CLIENTS_TOTAL = Gauge('fl_clients_total', 'Total number of registered clients')
FL_CLIENTS_ACTIVE = Gauge('fl_clients_active', 'Number of active clients')
FL_ROUNDS_TOTAL = Counter('fl_rounds_total', 'Total number of training rounds', ['status'])
FL_ROUND_DURATION = Histogram('fl_round_duration_seconds', 'Training round duration')

MODEL_SIZE_BYTES = Gauge('fl_model_size_bytes', 'Model size in bytes')
MODEL_PARAMETERS = Gauge('fl_model_parameters_total', 'Total number of model parameters')

PRIVACY_BUDGET_USED = Gauge('fl_privacy_budget_used', 'Privacy budget used per client', ['client_id'])
BYZANTINE_CLIENTS_DETECTED = Counter('fl_byzantine_clients_detected_total', 'Byzantine clients detected')

DATABASE_CONNECTIONS = Gauge('fl_database_connections_active', 'Active database connections')
DATABASE_QUERY_DURATION = Histogram('fl_database_query_duration_seconds', 'Database query duration', ['operation'])

API_REQUESTS_TOTAL = Counter('fl_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
API_REQUEST_DURATION = Histogram('fl_api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])

# System info
SYSTEM_INFO = Info('fl_system_info', 'System information')


def setup_logging():
    """Setup structured logging."""
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if config.monitoring.log_format == "json" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, config.monitoring.log_level),
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.monitoring.log_file_path) if config.monitoring.log_file_path else logging.NullHandler()
        ]
    )


class MetricsManager:
    """Manages system and application metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self._setup_system_info()
    
    def _setup_system_info(self):
        """Setup system information metrics."""
        import platform
        
        SYSTEM_INFO.info({
            'version': '1.0.0',
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'hostname': platform.node(),
        })
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU_USAGE.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            SYSTEM_DISK_USAGE.set(disk_percent)
            
            # Record to database
            db_manager.record_metric("cpu_usage_percent", cpu_percent, "system")
            db_manager.record_metric("memory_usage_percent", memory.percent, "system")
            db_manager.record_metric("disk_usage_percent", disk_percent, "system")
            
        except Exception as e:
            structlog.get_logger().error(f"Failed to update system metrics: {e}")
    
    def update_fl_metrics(self):
        """Update federated learning metrics."""
        try:
            with db_manager.get_session() as session:
                # Client metrics
                total_clients = session.query(db_manager.Client).count()
                active_clients = session.query(db_manager.Client).filter(
                    db_manager.Client.status == "active"
                ).count()
                
                FL_CLIENTS_TOTAL.set(total_clients)
                FL_CLIENTS_ACTIVE.set(active_clients)
                
                # Training round metrics
                completed_rounds = session.query(db_manager.TrainingRound).filter(
                    db_manager.TrainingRound.status == "completed"
                ).count()
                
                # Privacy budget metrics
                clients = session.query(db_manager.Client).all()
                for client in clients:
                    PRIVACY_BUDGET_USED.labels(client_id=client.client_id).set(
                        client.privacy_budget_epsilon
                    )
                
                # Database connection metrics
                DATABASE_CONNECTIONS.set(session.bind.pool.checkedout())
                
        except Exception as e:
            structlog.get_logger().error(f"Failed to update FL metrics: {e}")
    
    def record_training_round_start(self, round_id: int, participants: int):
        """Record training round start."""
        FL_ROUNDS_TOTAL.labels(status="started").inc()
        
        db_manager.record_metric(
            "training_round_participants", 
            participants, 
            "coordinator",
            labels={"round_id": round_id}
        )
    
    def record_training_round_complete(self, round_id: int, duration: float, participants: int):
        """Record training round completion."""
        FL_ROUNDS_TOTAL.labels(status="completed").inc()
        FL_ROUND_DURATION.observe(duration)
        
        db_manager.record_metric("training_round_duration_seconds", duration, "coordinator")
        db_manager.record_metric("training_round_participants", participants, "coordinator")
    
    def record_model_metrics(self, model_size_bytes: int, num_parameters: int):
        """Record model metrics."""
        MODEL_SIZE_BYTES.set(model_size_bytes)
        MODEL_PARAMETERS.set(num_parameters)
        
        db_manager.record_metric("model_size_bytes", model_size_bytes, "coordinator")
        db_manager.record_metric("model_parameters", num_parameters, "coordinator")
    
    def record_byzantine_detection(self, round_id: int, num_detected: int):
        """Record Byzantine client detection."""
        BYZANTINE_CLIENTS_DETECTED.inc(num_detected)
        
        db_manager.record_metric(
            "byzantine_clients_detected",
            num_detected,
            "coordinator",
            labels={"round_id": round_id}
        )
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics."""
        API_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=status_code).inc()
        API_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_database_query(self, operation: str, duration: float):
        """Record database query metrics."""
        DATABASE_QUERY_DURATION.labels(operation=operation).observe(duration)
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self.start_time


class HealthChecker:
    """Health check manager."""
    
    def __init__(self):
        self.checks = {}
    
    def register_check(self, name: str, check_func):
        """Register a health check."""
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                results[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "details": result if isinstance(result, dict) else {}
                }
                if not result:
                    overall_healthy = False
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e)
                }
                overall_healthy = False
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": results
        }


class AlertManager:
    """Alert management for critical events."""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
    
    def send_alert(self, level: str, title: str, message: str, tags: Dict[str, str] = None):
        """Send alert."""
        alert_data = {
            "level": level,
            "title": title,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "tags": tags or {}
        }
        
        # Log alert
        self.logger.warning("Alert triggered", **alert_data)
        
        # Send to Sentry if configured
        if config.monitoring.sentry_dsn:
            with sentry_sdk.push_scope() as scope:
                scope.set_level(level)
                scope.set_tag("alert", True)
                for key, value in (tags or {}).items():
                    scope.set_tag(key, value)
                sentry_sdk.capture_message(f"{title}: {message}")
        
        # Additional alerting channels
        self._send_slack_alert(level, title, message, tags)
        self._send_email_alert(level, title, message, tags)
        self._send_webhook_alert(level, title, message, tags)
    
    def _send_slack_alert(self, level: str, title: str, message: str, tags: Optional[Dict] = None):
        """Send alert to Slack webhook."""
        try:
            slack_webhook = getattr(config.monitoring, 'slack_webhook_url', None)
            if not slack_webhook:
                return
                
            import requests
            
            # Color coding for different alert levels
            color_map = {
                'error': '#FF0000',    # Red
                'warning': '#FFA500',  # Orange
                'info': '#0000FF',     # Blue
                'debug': '#808080'     # Gray
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(level, '#808080'),
                    "title": f"🚨 {title}",
                    "text": message,
                    "fields": [
                        {"title": "Level", "value": level.upper(), "short": True},
                        {"title": "Timestamp", "value": datetime.now().isoformat(), "short": True}
                    ],
                    "footer": "Medical AI Monitoring",
                    "ts": int(time.time())
                }]
            }
            
            # Add tag fields
            if tags:
                for key, value in tags.items():
                    payload["attachments"][0]["fields"].append({
                        "title": key.replace('_', ' ').title(),
                        "value": str(value),
                        "short": True
                    })
            
            response = requests.post(slack_webhook, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_email_alert(self, level: str, title: str, message: str, tags: Optional[Dict] = None):
        """Send alert via email."""
        try:
            email_config = getattr(config.monitoring, 'email', None)
            if not email_config or not email_config.get('enabled', False):
                return
                
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = email_config['from_address']
            msg['To'] = ', '.join(email_config['to_addresses'])
            msg['Subject'] = f"[{level.upper()}] {title}"
            
            # Email body
            body = f"""
Medical AI System Alert

Level: {level.upper()}
Title: {title}
Message: {message}
Timestamp: {datetime.now().isoformat()}

"""
            
            if tags:
                body += "Additional Information:\n"
                for key, value in tags.items():
                    body += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_host'], email_config['smtp_port'])
            if email_config.get('use_tls', True):
                server.starttls()
            if email_config.get('username') and email_config.get('password'):
                server.login(email_config['username'], email_config['password'])
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, level: str, title: str, message: str, tags: Optional[Dict] = None):
        """Send alert to custom webhook."""
        try:
            webhook_url = getattr(config.monitoring, 'webhook_url', None)
            if not webhook_url:
                return
                
            import requests
            
            payload = {
                "level": level,
                "title": title,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "tags": tags or {},
                "source": "medical_ai_monitoring"
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def alert_high_cpu_usage(self, cpu_percent: float):
        """Alert for high CPU usage."""
        if cpu_percent > 90:
            self.send_alert(
                "error",
                "High CPU Usage",
                f"CPU usage is {cpu_percent:.1f}%",
                {"metric": "cpu_usage", "value": str(cpu_percent)}
            )
    
    def alert_high_memory_usage(self, memory_percent: float):
        """Alert for high memory usage."""
        if memory_percent > 90:
            self.send_alert(
                "error",
                "High Memory Usage",
                f"Memory usage is {memory_percent:.1f}%",
                {"metric": "memory_usage", "value": str(memory_percent)}
            )
    
    def alert_training_round_failed(self, round_id: int, error: str):
        """Alert for training round failure."""
        self.send_alert(
            "error",
            "Training Round Failed",
            f"Training round {round_id} failed: {error}",
            {"round_id": str(round_id), "error": error}
        )
    
    def alert_byzantine_clients_detected(self, round_id: int, num_detected: int):
        """Alert for Byzantine client detection."""
        if num_detected > 0:
            self.send_alert(
                "warning",
                "Byzantine Clients Detected",
                f"Detected {num_detected} Byzantine clients in round {round_id}",
                {"round_id": str(round_id), "byzantine_count": str(num_detected)}
            )
    
    def alert_privacy_budget_exhausted(self, client_id: str):
        """Alert for privacy budget exhaustion."""
        self.send_alert(
            "warning",
            "Privacy Budget Exhausted",
            f"Client {client_id} has exhausted their privacy budget",
            {"client_id": client_id}
        )


# Global instances
metrics_manager = MetricsManager()
health_checker = HealthChecker()
alert_manager = AlertManager()


def get_metrics_manager() -> MetricsManager:
    """Get the global metrics manager instance."""
    return metrics_manager


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    return health_checker


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    return alert_manager


# Default health checks
def database_health_check():
    """Check database connectivity."""
    try:
        with db_manager.get_session() as session:
            session.execute("SELECT 1")
        return True
    except Exception:
        return False


def redis_health_check():
    """Check Redis connectivity."""
    try:
        # TODO: Add Redis client and check
        return True
    except Exception:
        return False


# Register default health checks
health_checker.register_check("database", database_health_check)
health_checker.register_check("redis", redis_health_check)