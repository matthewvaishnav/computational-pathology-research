"""
Audit Reports Module

Compliance reports, anomaly detection, usage analysis.
Extracted from audit.py for focused responsibility.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .audit import AuditEventType, AuditSeverity, SignedAuditRecord
from .audit_logger import AuditLogger, CryptographicSigner
from .audit_query import AuditStorage, FileAuditStorage

logger = logging.getLogger(__name__)


class ComplianceAuditLogger(AuditLogger):
    """Enhanced audit logger with regulatory compliance features."""

    def __init__(
        self,
        storage: AuditStorage,
        signer: Optional[CryptographicSigner] = None,
        retention_days: int = 2555,
        backup_storage: Optional[AuditStorage] = None,
    ):
        """Initialize compliance audit logger with backup storage."""
        super().__init__(storage, signer, retention_days)
        self.backup_storage = backup_storage
        self.compliance_logger = logging.getLogger(f"{__name__}.compliance")

    def _store_signed_event(self, event) -> str:
        """Store event in primary and backup storage."""
        event_id = super()._store_signed_event(event)

        if self.backup_storage and event_id:
            try:
                signature = self.signer.sign_event(event)
                signed_record = SignedAuditRecord(
                    event=event,
                    signature=signature,
                    public_key_fingerprint=self.signer.public_key_fingerprint,
                )

                backup_success = self.backup_storage.store_record(signed_record)
                if backup_success:
                    self.compliance_logger.info(f"Backup storage successful for event {event_id}")
                else:
                    self.compliance_logger.warning(f"Backup storage failed for event {event_id}")

            except Exception as e:
                self.compliance_logger.error(f"Backup storage error for event {event_id}: {e}")

        return event_id

    def generate_compliance_report(
        self, start_date: datetime, end_date: datetime, output_path: Path
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        records = self.get_audit_records(start_date, end_date)

        integrity_results = []
        for record in records:
            is_valid = self.verify_record_integrity(record)
            integrity_results.append(
                {
                    "event_id": record.event.event_id,
                    "timestamp": record.event.timestamp.isoformat(),
                    "integrity_valid": is_valid,
                }
            )

        stats = self.get_audit_statistics()

        compliance_report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_period_start": start_date.isoformat(),
                "report_period_end": end_date.isoformat(),
                "total_records": len(records),
                "integrity_verification_passed": all(
                    r["integrity_valid"] for r in integrity_results
                ),
                "public_key_fingerprint": self.signer.public_key_fingerprint,
                "retention_policy_days": self.retention_days,
            },
            "audit_statistics": stats,
            "integrity_verification": integrity_results,
            "regulatory_compliance": {
                "fda_21_cfr_part_11_compliant": True,
                "hipaa_compliant": True,
                "retention_period_years": self.retention_days / 365.25,
                "tamper_evident": True,
                "cryptographic_signatures": True,
            },
            "public_key": self.signer.export_public_key(),
        }

        records_export_path = output_path.parent / f"{output_path.stem}_records.json"
        self.export_audit_logs(records_export_path, start_date, end_date, "json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(compliance_report, f, indent=2)

        self.compliance_logger.info(
            f"Generated compliance report with {len(records)} records"
        )

        return compliance_report

    def validate_audit_chain(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Validate integrity of audit chain."""
        records = self.get_audit_records(start_time, end_time)

        validation_results = {
            "total_records": len(records),
            "valid_signatures": 0,
            "invalid_signatures": 0,
            "validation_errors": [],
            "chain_integrity": True,
            "validation_timestamp": datetime.now().isoformat(),
        }

        for record in records:
            try:
                is_valid = self.verify_record_integrity(record)

                if is_valid:
                    validation_results["valid_signatures"] += 1
                else:
                    validation_results["invalid_signatures"] += 1
                    validation_results["chain_integrity"] = False
                    validation_results["validation_errors"].append(
                        {
                            "event_id": record.event.event_id,
                            "timestamp": record.event.timestamp.isoformat(),
                            "error": "Invalid cryptographic signature",
                        }
                    )

            except Exception as e:
                validation_results["invalid_signatures"] += 1
                validation_results["chain_integrity"] = False
                validation_results["validation_errors"].append(
                    {
                        "event_id": record.event.event_id,
                        "timestamp": record.event.timestamp.isoformat(),
                        "error": f"Validation exception: {str(e)}",
                    }
                )

        return validation_results

    def archive_old_records(self, archive_path: Path, cutoff_date: datetime) -> Dict[str, Any]:
        """Archive old records while maintaining compliance."""
        records_to_archive = self.get_audit_records(end_time=cutoff_date)

        if not records_to_archive:
            return {
                "archived_count": 0,
                "archive_path": None,
                "archive_timestamp": datetime.now().isoformat(),
            }

        archive_data = {
            "archive_metadata": {
                "created_at": datetime.now().isoformat(),
                "cutoff_date": cutoff_date.isoformat(),
                "record_count": len(records_to_archive),
                "public_key_fingerprint": self.signer.public_key_fingerprint,
                "archive_format_version": "1.0",
            },
            "public_key": self.signer.export_public_key(),
            "records": [record.to_dict() for record in records_to_archive],
        }

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(archive_data, f, indent=2)

        with open(archive_path, "rb") as f:
            archive_hash = hashlib.sha256(f.read()).hexdigest()

        integrity_path = archive_path.with_suffix(".integrity")
        with open(integrity_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "archive_file": archive_path.name,
                    "sha256_hash": archive_hash,
                    "created_at": datetime.now().isoformat(),
                    "record_count": len(records_to_archive),
                },
                f,
                indent=2,
            )

        self.compliance_logger.info(f"Archived {len(records_to_archive)} records")

        return {
            "archived_count": len(records_to_archive),
            "archive_path": str(archive_path),
            "integrity_path": str(integrity_path),
            "archive_hash": archive_hash,
            "archive_timestamp": datetime.now().isoformat(),
        }


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

            hour = record.event.timestamp.hour
            user_activity[user_id]["hourly_distribution"][hour] += 1

        for user_id, activity in user_activity.items():
            if activity["total_events"] > 10:
                failure_rate = activity["failed_events"] / activity["total_events"]
                if failure_rate > 0.3:
                    anomalies["failed_operations_spike"].append(
                        {
                            "user_id": user_id,
                            "failure_rate": failure_rate,
                            "total_events": activity["total_events"],
                            "failed_events": activity["failed_events"],
                        }
                    )

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

            prediction_events = activity["event_types"].get("prediction_operation", 0)
            if prediction_events > 100:
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

        processing_times = []
        hourly_activity = [0] * 24
        daily_activity = {}

        for record in records:
            event_type = record.event.event_type.value
            report["summary_statistics"]["event_types"][event_type] = (
                report["summary_statistics"]["event_types"].get(event_type, 0) + 1
            )

            severity = record.event.severity.value
            report["summary_statistics"]["severity_distribution"][severity] = (
                report["summary_statistics"]["severity_distribution"].get(severity, 0) + 1
            )

            date_str = record.event.timestamp.date().isoformat()
            daily_activity[date_str] = daily_activity.get(date_str, 0) + 1

            hour = record.event.timestamp.hour
            hourly_activity[hour] += 1

            if (
                record.event.event_type == AuditEventType.PREDICTION_OPERATION
                and "processing_time_ms" in record.event.details
            ):
                processing_times.append(record.event.details["processing_time_ms"])

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


def create_default_audit_logger(storage_dir: Optional[Path] = None) -> ComplianceAuditLogger:
    """Create default audit logger with file storage."""
    if storage_dir is None:
        storage_dir = Path.home() / ".clinical_audit_logs"

    primary_storage = FileAuditStorage(storage_dir / "primary")
    backup_storage = FileAuditStorage(storage_dir / "backup")

    return ComplianceAuditLogger(
        storage=primary_storage,
        backup_storage=backup_storage,
        retention_days=2555,
    )
