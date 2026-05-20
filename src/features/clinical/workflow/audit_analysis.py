"""Audit log analysis and utility functions."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from .audit_compliance import ComplianceAuditLogger
from .audit_logger import AuditContextManager, AuditLogger
from .audit_models import AuditEventType, AuditSeverity
from .audit_storage import FileAuditStorage


class AuditLogAnalyzer:
    """Analyzer for audit log patterns and compliance monitoring."""

    def __init__(self, audit_logger: AuditLogger):
        """Initialize with audit logger."""
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)

    def detect_anomalous_patterns(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Detect anomalous patterns in audit logs."""
        start_time = datetime.now() - timedelta(days=lookback_days)
        records = self.audit_logger.get_audit_records(start_time=start_time)

        anomalies = {
            "unusual_access_patterns": [],
            "failed_operations_spike": [],
            "off_hours_activity": [],
            "bulk_data_access": [],
            "analysis_timestamp": datetime.now().isoformat(),
        }

        # Analyze access patterns by user
        user_activity = {}
        for record in records:
            user_id = record.event.user_id or "system"
            if user_id not in user_activity:
                user_activity[user_id] = {
                    "total_events": 0,
                    "failed_events": 0,
                    "event_types": {},
                    "hourly_distribution": [0] * 24,
                }

            user_activity[user_id]["total_events"] += 1

            if record.event.severity in [AuditSeverity.WARNING, AuditSeverity.ERROR]:
                user_activity[user_id]["failed_events"] += 1

            event_type = record.event.event_type.value
            user_activity[user_id]["event_types"][event_type] = (
                user_activity[user_id]["event_types"].get(event_type, 0) + 1
            )

            # Track hourly distribution
            hour = record.event.timestamp.hour
            user_activity[user_id]["hourly_distribution"][hour] += 1

        # Detect anomalies
        for user_id, activity in user_activity.items():
            # High failure rate
            if activity["total_events"] > 10:
                failure_rate = activity["failed_events"] / activity["total_events"]
                if failure_rate > 0.3:  # More than 30% failures
                    anomalies["failed_operations_spike"].append(
                        {
                            "user_id": user_id,
                            "failure_rate": failure_rate,
                            "total_events": activity["total_events"],
                            "failed_events": activity["failed_events"],
                        }
                    )

            # Off-hours activity (10 PM to 6 AM)
            off_hours_activity = sum(activity["hourly_distribution"][22:24]) + sum(
                activity["hourly_distribution"][0:6]
            )
            total_activity = sum(activity["hourly_distribution"])

            if total_activity > 20 and off_hours_activity / total_activity > 0.4:
                anomalies["off_hours_activity"].append(
                    {
                        "user_id": user_id,
                        "off_hours_percentage": off_hours_activity / total_activity,
                        "total_events": total_activity,
                        "off_hours_events": off_hours_activity,
                    }
                )

            # Bulk data access patterns
            prediction_events = activity["event_types"].get("prediction_operation", 0)
            if prediction_events > 100:  # More than 100 predictions
                anomalies["bulk_data_access"].append(
                    {
                        "user_id": user_id,
                        "prediction_count": prediction_events,
                        "total_events": activity["total_events"],
                    }
                )

        return anomalies

    def generate_usage_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate usage report for audit logs."""
        records = self.audit_logger.get_audit_records(start_date, end_date)

        report = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_days": (end_date - start_date).days,
            },
            "summary_statistics": {
                "total_events": len(records),
                "unique_users": len(set(r.event.user_id for r in records if r.event.user_id)),
                "event_types": {},
                "severity_distribution": {},
                "daily_activity": {},
            },
            "top_users": {},
            "system_health": {
                "error_rate": 0,
                "average_processing_time": 0,
                "peak_activity_hour": 0,
            },
        }

        # Calculate statistics
        processing_times = []
        hourly_activity = [0] * 24
        daily_activity = {}

        for record in records:
            # Event type distribution
            event_type = record.event.event_type.value
            report["summary_statistics"]["event_types"][event_type] = (
                report["summary_statistics"]["event_types"].get(event_type, 0) + 1
            )

            # Severity distribution
            severity = record.event.severity.value
            report["summary_statistics"]["severity_distribution"][severity] = (
                report["summary_statistics"]["severity_distribution"].get(severity, 0) + 1
            )

            # Daily activity
            date_str = record.event.timestamp.date().isoformat()
            daily_activity[date_str] = daily_activity.get(date_str, 0) + 1

            # Hourly activity
            hour = record.event.timestamp.hour
            hourly_activity[hour] += 1

            # Processing times for prediction operations
            if (
                record.event.event_type == AuditEventType.PREDICTION_OPERATION
                and "processing_time_ms" in record.event.details
            ):
                processing_times.append(record.event.details["processing_time_ms"])

        # Calculate derived metrics
        error_count = report["summary_statistics"]["severity_distribution"].get(
            "error", 0
        ) + report["summary_statistics"]["severity_distribution"].get("critical", 0)

        report["summary_statistics"]["daily_activity"] = daily_activity
        report["system_health"]["error_rate"] = error_count / len(records) if records else 0
        report["system_health"]["average_processing_time"] = (
            sum(processing_times) / len(processing_times) if processing_times else 0
        )
        report["system_health"]["peak_activity_hour"] = hourly_activity.index(max(hourly_activity))

        return report


# Utility functions for audit logging integration


def audit_operation(
    audit_logger: AuditLogger,
    operation_type: str,
    user_id: Optional[str] = None,
    session_token: Optional[str] = None,
    model_version: Optional[str] = None,
):
    """Decorator for automatic audit logging of operations."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            with AuditContextManager(
                audit_logger, operation_type, user_id, session_token, model_version
            ) as audit_ctx:
                result = func(*args, **kwargs)

                # Extract input/output data for logging
                input_data = {"args": str(args)[:1000], "kwargs": str(kwargs)[:1000]}
                output_data = {"result": str(result)[:1000]}

                audit_ctx.log_success(input_data, output_data)
                return result

        return wrapper

    return decorator


def create_default_audit_logger(storage_dir: Optional[Path] = None) -> ComplianceAuditLogger:
    """Create default audit logger with file storage."""
    if storage_dir is None:
        storage_dir = Path.home() / ".clinical_audit_logs"

    primary_storage = FileAuditStorage(storage_dir / "primary")
    backup_storage = FileAuditStorage(storage_dir / "backup")

    return ComplianceAuditLogger(
        storage=primary_storage,
        backup_storage=backup_storage,
        retention_days=2555,  # 7 years for FDA compliance
    )
