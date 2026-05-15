"""
Production Optimization Module for Medical AI Revolution
Handles performance optimization, scaling, monitoring, and operational excellence.
"""

import json
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric measurement."""

    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    site_id: str
    component: str  # api, model, database, etc.


@dataclass
class SystemAlert:
    """System alert definition."""

    alert_id: str
    severity: str  # critical, warning, info
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class CapacityPlan:
    """Capacity planning recommendation."""

    component: str
    current_utilization: float
    projected_utilization: float
    recommendation: str
    timeline_days: int
    estimated_cost: float


class PerformanceMonitor:
    """Monitors system performance and identifies bottlenecks."""

    def __init__(self, db_path: str = "data/performance_metrics.db"):
        self.db_path = db_path
        self.monitoring_active = False
        self.monitor_thread = None
        self._init_database()

    def _init_database(self):
        """Initialize performance metrics database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    metric_name TEXT,
                    value REAL,
                    unit TEXT,
                    site_id TEXT,
                    component TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_alerts (
                    alert_id TEXT PRIMARY KEY,
                    severity TEXT,
                    component TEXT,
                    message TEXT,
                    timestamp TEXT,
                    resolved BOOLEAN,
                    resolution_time TEXT
                )
            """)

    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval_seconds,), daemon=True
        )
        self.monitor_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        timestamp = datetime.now()
        site_id = "local"  # Could be configured per deployment

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self._record_metric(
            PerformanceMetric(
                timestamp=timestamp,
                metric_name="cpu_utilization",
                value=cpu_percent,
                unit="percent",
                site_id=site_id,
                component="system",
            )
        )

        # Memory metrics
        memory = psutil.virtual_memory()
        self._record_metric(
            PerformanceMetric(
                timestamp=timestamp,
                metric_name="memory_utilization",
                value=memory.percent,
                unit="percent",
                site_id=site_id,
                component="system",
            )
        )

        # Disk metrics
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100
        self._record_metric(
            PerformanceMetric(
                timestamp=timestamp,
                metric_name="disk_utilization",
                value=disk_percent,
                unit="percent",
                site_id=site_id,
                component="system",
            )
        )

        # Network metrics
        network = psutil.net_io_counters()
        self._record_metric(
            PerformanceMetric(
                timestamp=timestamp,
                metric_name="network_bytes_sent",
                value=network.bytes_sent,
                unit="bytes",
                site_id=site_id,
                component="network",
            )
        )

        # Check for alerts
        self._check_performance_alerts(
            timestamp, site_id, {"cpu": cpu_percent, "memory": memory.percent, "disk": disk_percent}
        )

    def _record_metric(self, metric: PerformanceMetric):
        """Record performance metric to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO performance_metrics 
                (timestamp, metric_name, value, unit, site_id, component)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    metric.timestamp.isoformat(),
                    metric.metric_name,
                    metric.value,
                    metric.unit,
                    metric.site_id,
                    metric.component,
                ),
            )

    def _check_performance_alerts(
        self, timestamp: datetime, site_id: str, metrics: Dict[str, float]
    ):
        """Check for performance alerts."""
        alerts = []

        if metrics["cpu"] > 90:
            alerts.append(
                SystemAlert(
                    alert_id=f"cpu_high_{int(timestamp.timestamp())}",
                    severity="critical",
                    component="system",
                    message=f"CPU utilization critical: {metrics['cpu']:.1f}%",
                    timestamp=timestamp,
                )
            )

        if metrics["memory"] > 85:
            alerts.append(
                SystemAlert(
                    alert_id=f"memory_high_{int(timestamp.timestamp())}",
                    severity="warning",
                    component="system",
                    message=f"Memory utilization high: {metrics['memory']:.1f}%",
                    timestamp=timestamp,
                )
            )

        if metrics["disk"] > 80:
            alerts.append(
                SystemAlert(
                    alert_id=f"disk_high_{int(timestamp.timestamp())}",
                    severity="warning",
                    component="system",
                    message=f"Disk utilization high: {metrics['disk']:.1f}%",
                    timestamp=timestamp,
                )
            )

        for alert in alerts:
            self._record_alert(alert)

    def _record_alert(self, alert: SystemAlert):
        """Record system alert to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO system_alerts
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    alert.alert_id,
                    alert.severity,
                    alert.component,
                    alert.message,
                    alert.timestamp.isoformat(),
                    alert.resolved,
                    alert.resolution_time.isoformat() if alert.resolution_time else None,
                ),
            )

        logger.warning(f"Alert: {alert.message}")

    def identify_bottlenecks(self, hours_back: int = 24) -> Dict[str, Any]:
        """Identify system bottlenecks from recent metrics."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT metric_name, component, AVG(value) as avg_value, 
                       MAX(value) as max_value, COUNT(*) as count
                FROM performance_metrics 
                WHERE timestamp >= ?
                GROUP BY metric_name, component
            """
            cursor = conn.execute(query, [cutoff_time.isoformat()])
            metrics = cursor.fetchall()

        bottlenecks = []
        recommendations = []

        for metric_name, component, avg_value, max_value, count in metrics:
            if metric_name == "cpu_utilization":
                if avg_value > 70:
                    bottlenecks.append(
                        {
                            "component": component,
                            "metric": metric_name,
                            "severity": "high" if avg_value > 85 else "medium",
                            "avg_value": avg_value,
                            "max_value": max_value,
                        }
                    )
                    recommendations.append("Consider CPU scaling or optimization")

            elif metric_name == "memory_utilization":
                if avg_value > 75:
                    bottlenecks.append(
                        {
                            "component": component,
                            "metric": metric_name,
                            "severity": "high" if avg_value > 90 else "medium",
                            "avg_value": avg_value,
                            "max_value": max_value,
                        }
                    )
                    recommendations.append("Consider memory scaling or optimization")

        return {
            "analysis_period_hours": hours_back,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "total_metrics": len(metrics),
        }


class AutoScaler:
    """Handles automatic scaling of system resources."""

    def __init__(self, performance_monitor: PerformanceMonitor):
        self.monitor = performance_monitor
        self.scaling_policies = {}
        self.scaling_active = False

    def add_scaling_policy(
        self,
        component: str,
        metric_name: str,
        scale_up_threshold: float,
        scale_down_threshold: float,
        scale_up_action: Callable,
        scale_down_action: Callable,
    ):
        """Add auto-scaling policy."""
        self.scaling_policies[component] = {
            "metric_name": metric_name,
            "scale_up_threshold": scale_up_threshold,
            "scale_down_threshold": scale_down_threshold,
            "scale_up_action": scale_up_action,
            "scale_down_action": scale_down_action,
            "last_action": None,
            "cooldown_minutes": 10,
        }

    def start_auto_scaling(self):
        """Start auto-scaling monitoring."""
        self.scaling_active = True
        threading.Thread(target=self._scaling_loop, daemon=True).start()
        logger.info("Auto-scaling started")

    def stop_auto_scaling(self):
        """Stop auto-scaling."""
        self.scaling_active = False
        logger.info("Auto-scaling stopped")

    def _scaling_loop(self):
        """Auto-scaling monitoring loop."""
        while self.scaling_active:
            try:
                self._check_scaling_conditions()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")

    def _check_scaling_conditions(self):
        """Check if scaling actions are needed."""
        current_time = datetime.now()

        for component, policy in self.scaling_policies.items():
            # Get recent metric value
            recent_value = self._get_recent_metric_value(
                policy["metric_name"], component, minutes_back=5
            )

            if recent_value is None:
                continue

            # Check cooldown period
            if (
                policy["last_action"]
                and (current_time - policy["last_action"]).total_seconds()
                < policy["cooldown_minutes"] * 60
            ):
                continue

            # Check scaling conditions
            if recent_value > policy["scale_up_threshold"]:
                logger.info(f"Scaling up {component}: {policy['metric_name']} = {recent_value}")
                policy["scale_up_action"]()
                policy["last_action"] = current_time

            elif recent_value < policy["scale_down_threshold"]:
                logger.info(f"Scaling down {component}: {policy['metric_name']} = {recent_value}")
                policy["scale_down_action"]()
                policy["last_action"] = current_time

    def _get_recent_metric_value(
        self, metric_name: str, component: str, minutes_back: int = 5
    ) -> Optional[float]:
        """Get recent average metric value."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes_back)

        with sqlite3.connect(self.monitor.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT AVG(value) FROM performance_metrics
                WHERE metric_name = ? AND component = ? AND timestamp >= ?
            """,
                [metric_name, component, cutoff_time.isoformat()],
            )

            result = cursor.fetchone()
            return result[0] if result and result[0] is not None else None


class CapacityPlanner:
    """Plans future capacity requirements."""

    def __init__(self, performance_monitor: PerformanceMonitor):
        self.monitor = performance_monitor

    def analyze_capacity_trends(self, days_back: int = 30) -> Dict[str, Any]:
        """Analyze capacity trends and project future needs."""
        cutoff_time = datetime.now() - timedelta(days=days_back)

        with sqlite3.connect(self.monitor.db_path) as conn:
            query = """
                SELECT DATE(timestamp) as date, metric_name, component, AVG(value) as avg_value
                FROM performance_metrics
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp), metric_name, component
                ORDER BY date
            """
            cursor = conn.execute(query, [cutoff_time.isoformat()])
            data = cursor.fetchall()

        # Group by metric and component
        trends = {}
        for date, metric_name, component, avg_value in data:
            key = f"{component}_{metric_name}"
            if key not in trends:
                trends[key] = []
            trends[key].append((date, avg_value))

        # Calculate trends and projections
        projections = {}
        for key, values in trends.items():
            if len(values) < 7:  # Need at least a week of data
                continue

            dates = [datetime.strptime(v[0], "%Y-%m-%d") for v in values]
            vals = [v[1] for v in values]

            # Simple linear regression for trend
            x = np.array([(d - dates[0]).days for d in dates])
            y = np.array(vals)

            if len(x) > 1:
                slope, intercept = np.polyfit(x, y, 1)

                # Project 30 days ahead
                future_days = 30
                projected_value = slope * (x[-1] + future_days) + intercept

                component, metric = key.split("_", 1)
                projections[key] = {
                    "component": component,
                    "metric": metric,
                    "current_avg": np.mean(vals[-7:]),  # Last week average
                    "projected_value": max(0, projected_value),
                    "trend_slope": slope,
                    "days_projected": future_days,
                }

        return {
            "analysis_period_days": days_back,
            "projections": projections,
            "recommendations": self._generate_capacity_recommendations(projections),
        }

    def _generate_capacity_recommendations(self, projections: Dict[str, Any]) -> List[CapacityPlan]:
        """Generate capacity planning recommendations."""
        recommendations = []

        for key, proj in projections.items():
            component = proj["component"]
            metric = proj["metric"]
            current = proj["current_avg"]
            projected = proj["projected_value"]

            if metric == "cpu_utilization":
                if projected > 80:
                    recommendations.append(
                        CapacityPlan(
                            component=component,
                            current_utilization=current,
                            projected_utilization=projected,
                            recommendation="Scale up CPU resources",
                            timeline_days=30,
                            estimated_cost=2000.0,
                        )
                    )

            elif metric == "memory_utilization":
                if projected > 85:
                    recommendations.append(
                        CapacityPlan(
                            component=component,
                            current_utilization=current,
                            projected_utilization=projected,
                            recommendation="Scale up memory resources",
                            timeline_days=30,
                            estimated_cost=1500.0,
                        )
                    )

            elif metric == "disk_utilization":
                if projected > 75:
                    recommendations.append(
                        CapacityPlan(
                            component=component,
                            current_utilization=current,
                            projected_utilization=projected,
                            recommendation="Expand storage capacity",
                            timeline_days=14,
                            estimated_cost=1000.0,
                        )
                    )

        return recommendations


class OperationalExcellence:
    """Manages operational excellence practices."""

    def __init__(self):
        self.incident_procedures = {}
        self.backup_schedules = {}
        self.security_policies = {}

    def setup_monitoring_alerting(self) -> Dict[str, Any]:
        """Setup comprehensive monitoring and alerting."""
        monitoring_config = {
            "metrics_to_monitor": [
                {
                    "name": "API Response Time",
                    "threshold": 5.0,
                    "unit": "seconds",
                    "severity": "warning",
                },
                {
                    "name": "Model Inference Time",
                    "threshold": 30.0,
                    "unit": "seconds",
                    "severity": "critical",
                },
                {
                    "name": "System Uptime",
                    "threshold": 99.5,
                    "unit": "percent",
                    "severity": "critical",
                },
                {"name": "Error Rate", "threshold": 1.0, "unit": "percent", "severity": "warning"},
            ],
            "alert_channels": [
                {"type": "email", "recipients": ["ops@hospital.com", "support@medai.com"]},
                {"type": "slack", "webhook": "https://hooks.slack.com/services/..."},
                {"type": "pagerduty", "service_key": "your-pagerduty-key"},
            ],
            "dashboards": [
                {"name": "System Overview", "metrics": ["cpu", "memory", "disk", "network"]},
                {
                    "name": "Application Performance",
                    "metrics": ["response_time", "throughput", "error_rate"],
                },
                {
                    "name": "Medical AI Metrics",
                    "metrics": ["inference_time", "accuracy", "queue_length"],
                },
            ],
        }

        return monitoring_config

    def create_incident_response_procedures(self) -> Dict[str, Any]:
        """Create incident response procedures."""
        procedures = {
            "severity_levels": {
                "P1_Critical": {
                    "description": "System down, patient safety impact",
                    "response_time": "15 minutes",
                    "escalation": "Immediate to on-call engineer and medical director",
                },
                "P2_High": {
                    "description": "Major functionality impaired",
                    "response_time": "1 hour",
                    "escalation": "On-call engineer within 30 minutes",
                },
                "P3_Medium": {
                    "description": "Minor functionality issues",
                    "response_time": "4 hours",
                    "escalation": "Next business day",
                },
            },
            "response_procedures": {
                "detection": [
                    "Automated monitoring alerts",
                    "User reports",
                    "Scheduled health checks",
                ],
                "initial_response": [
                    "Acknowledge incident within SLA",
                    "Assess severity and impact",
                    "Notify stakeholders",
                    "Begin investigation",
                ],
                "investigation": [
                    "Gather logs and metrics",
                    "Identify root cause",
                    "Implement temporary workaround if possible",
                    "Document findings",
                ],
                "resolution": [
                    "Implement permanent fix",
                    "Test solution thoroughly",
                    "Deploy to production",
                    "Verify resolution",
                ],
                "post_incident": [
                    "Conduct post-mortem review",
                    "Document lessons learned",
                    "Update procedures if needed",
                    "Implement preventive measures",
                ],
            },
            "communication_templates": {
                "initial_notification": "Incident detected: {title}. Severity: {severity}. Investigation in progress.",
                "status_update": "Incident update: {title}. Status: {status}. ETA: {eta}.",
                "resolution": "Incident resolved: {title}. Root cause: {cause}. Duration: {duration}.",
            },
        }

        return procedures

    def setup_backup_recovery(self) -> Dict[str, Any]:
        """Setup backup and recovery procedures."""
        backup_config = {
            "backup_schedules": {
                "database": {
                    "frequency": "daily",
                    "time": "02:00",
                    "retention_days": 30,
                    "location": "encrypted_cloud_storage",
                },
                "model_artifacts": {
                    "frequency": "weekly",
                    "time": "sunday_03:00",
                    "retention_weeks": 12,
                    "location": "model_registry",
                },
                "configuration": {
                    "frequency": "daily",
                    "time": "01:00",
                    "retention_days": 90,
                    "location": "version_control",
                },
                "logs": {
                    "frequency": "hourly",
                    "retention_days": 7,
                    "location": "log_aggregation_service",
                },
            },
            "recovery_procedures": {
                "database_recovery": [
                    "Identify backup to restore from",
                    "Stop application services",
                    "Restore database from backup",
                    "Verify data integrity",
                    "Restart services",
                    "Validate system functionality",
                ],
                "application_recovery": [
                    "Deploy previous known good version",
                    "Restore configuration from backup",
                    "Verify all integrations",
                    "Run smoke tests",
                    "Monitor for issues",
                ],
                "disaster_recovery": [
                    "Activate disaster recovery site",
                    "Restore all systems from backups",
                    "Update DNS and routing",
                    "Notify all stakeholders",
                    "Monitor system stability",
                ],
            },
            "rto_rpo_targets": {
                "database": {"rto_hours": 4, "rpo_hours": 1},
                "application": {"rto_hours": 2, "rpo_hours": 0.5},
                "full_system": {"rto_hours": 8, "rpo_hours": 4},
            },
        }

        return backup_config

    def implement_security_hardening(self) -> Dict[str, Any]:
        """Implement security hardening measures."""
        security_config = {
            "access_controls": {
                "authentication": [
                    "Multi-factor authentication required",
                    "Strong password policies",
                    "Regular password rotation",
                    "Account lockout policies",
                ],
                "authorization": [
                    "Role-based access control (RBAC)",
                    "Principle of least privilege",
                    "Regular access reviews",
                    "Automated deprovisioning",
                ],
            },
            "network_security": {
                "firewalls": [
                    "Web application firewall (WAF)",
                    "Network segmentation",
                    "Intrusion detection system (IDS)",
                    "DDoS protection",
                ],
                "encryption": [
                    "TLS 1.3 for data in transit",
                    "AES-256 for data at rest",
                    "Key rotation policies",
                    "Certificate management",
                ],
            },
            "monitoring": {
                "security_events": [
                    "Failed login attempts",
                    "Privilege escalations",
                    "Data access patterns",
                    "System configuration changes",
                ],
                "compliance": [
                    "HIPAA audit logging",
                    "SOX compliance monitoring",
                    "Regular vulnerability scans",
                    "Penetration testing",
                ],
            },
            "incident_response": {
                "security_incidents": [
                    "Immediate containment",
                    "Forensic investigation",
                    "Regulatory notification",
                    "Recovery and lessons learned",
                ]
            },
        }

        return security_config


# Demo usage and testing
def demo_production_optimization():
    """Demonstrate production optimization capabilities."""
    # Initialize components
    monitor = PerformanceMonitor()
    scaler = AutoScaler(monitor)
    planner = CapacityPlanner(monitor)
    ops = OperationalExcellence()

    # Start monitoring
    monitor.start_monitoring(interval_seconds=10)

    # Add scaling policies
    def scale_up_cpu():
        logger.info("Scaling up CPU resources")

    def scale_down_cpu():
        logger.info("Scaling down CPU resources")

    scaler.add_scaling_policy(
        component="system",
        metric_name="cpu_utilization",
        scale_up_threshold=80.0,
        scale_down_threshold=30.0,
        scale_up_action=scale_up_cpu,
        scale_down_action=scale_down_cpu,
    )

    # Start auto-scaling
    scaler.start_auto_scaling()

    # Let it run for a bit
    time.sleep(30)

    # Analyze bottlenecks
    bottlenecks = monitor.identify_bottlenecks(hours_back=1)
    print(f"Identified {len(bottlenecks['bottlenecks'])} bottlenecks")

    # Capacity planning
    capacity_analysis = planner.analyze_capacity_trends(days_back=1)
    print(f"Generated {len(capacity_analysis['recommendations'])} capacity recommendations")

    # Setup operational procedures
    monitoring_config = ops.setup_monitoring_alerting()
    incident_procedures = ops.create_incident_response_procedures()
    backup_config = ops.setup_backup_recovery()
    security_config = ops.implement_security_hardening()

    print("Production optimization setup complete")

    # Stop monitoring
    monitor.stop_monitoring()
    scaler.stop_auto_scaling()


if __name__ == "__main__":
    demo_production_optimization()
