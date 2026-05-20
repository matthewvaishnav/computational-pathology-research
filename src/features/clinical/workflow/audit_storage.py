"""Audit log storage implementations."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from .audit_models import AuditEventType, SignedAuditRecord


class AuditStorage(ABC):
    """Abstract base class for audit log storage."""

    @abstractmethod
    def store_record(self, record: SignedAuditRecord) -> bool:
        """Store signed audit record."""

    @abstractmethod
    def retrieve_records(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SignedAuditRecord]:
        """Retrieve audit records with filtering."""

    @abstractmethod
    def export_records(
        self,
        output_path: Path,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "json",
    ) -> bool:
        """Export audit records for regulatory submissions."""

    @abstractmethod
    def get_record_count(self) -> int:
        """Get total number of stored records."""

    @abstractmethod
    def cleanup_old_records(self, retention_days: int) -> int:
        """Clean up records older than retention period."""


class FileAuditStorage(AuditStorage):
    """File-based audit log storage implementation."""

    def __init__(self, storage_directory: Path):
        """Initialize with storage directory."""
        self.storage_directory = Path(storage_directory)
        self.storage_directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def _get_daily_log_file(self, date: datetime) -> Path:
        """Get log file path for specific date."""
        date_str = date.strftime("%Y-%m-%d")
        return self.storage_directory / f"audit_{date_str}.jsonl"

    def store_record(self, record: SignedAuditRecord) -> bool:
        """Store signed audit record in daily log file."""
        try:
            log_file = self._get_daily_log_file(record.created_at)

            with open(log_file, "a", encoding="utf-8") as f:
                json.dump(record.to_dict(), f)
                f.write("\n")

            return True
        except Exception as e:
            self.logger.error(f"Failed to store audit record: {e}")
            return False

    def retrieve_records(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SignedAuditRecord]:
        """Retrieve audit records with filtering."""
        records = []

        # Determine date range for file scanning
        if start_time is None:
            start_time = datetime.now() - timedelta(days=30)  # Default to last 30 days
        if end_time is None:
            end_time = datetime.now()

        current_date = start_time.date()
        end_date = end_time.date()

        while current_date <= end_date:
            log_file = self._get_daily_log_file(datetime.combine(current_date, datetime.min.time()))

            if log_file.exists():
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                record_data = json.loads(line)
                                record = SignedAuditRecord.from_dict(record_data)

                                # Apply filters
                                if start_time and record.event.timestamp < start_time:
                                    continue
                                if end_time and record.event.timestamp > end_time:
                                    continue
                                if event_type and record.event.event_type != event_type:
                                    continue
                                if user_id and record.event.user_id != user_id:
                                    continue

                                records.append(record)

                                if limit and len(records) >= limit:
                                    return records

                except Exception as e:
                    self.logger.error(f"Failed to read audit log {log_file}: {e}")

            current_date += timedelta(days=1)

        return records

    def export_records(
        self,
        output_path: Path,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "json",
    ) -> bool:
        """Export audit records for regulatory submissions."""
        try:
            records = self.retrieve_records(start_time, end_time)

            if format.lower() == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    export_data = {
                        "export_timestamp": datetime.now().isoformat(),
                        "start_time": start_time.isoformat() if start_time else None,
                        "end_time": end_time.isoformat() if end_time else None,
                        "record_count": len(records),
                        "records": [record.to_dict() for record in records],
                    }
                    json.dump(export_data, f, indent=2)

            elif format.lower() == "csv":
                import csv

                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    if records:
                        fieldnames = [
                            "event_id",
                            "event_type",
                            "timestamp",
                            "user_id",
                            "severity",
                            "description",
                            "model_version",
                            "ip_address",
                            "signature",
                        ]
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()

                        for record in records:
                            row = {
                                "event_id": record.event.event_id,
                                "event_type": record.event.event_type.value,
                                "timestamp": record.event.timestamp.isoformat(),
                                "user_id": record.event.user_id or "",
                                "severity": record.event.severity.value,
                                "description": record.event.description,
                                "model_version": record.event.model_version or "",
                                "ip_address": record.event.ip_address or "",
                                "signature": record.signature,
                            }
                            writer.writerow(row)

            return True

        except Exception as e:
            self.logger.error(f"Failed to export audit records: {e}")
            return False

    def get_record_count(self) -> int:
        """Get total number of stored records."""
        count = 0

        for log_file in self.storage_directory.glob("audit_*.jsonl"):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    count += sum(1 for line in f if line.strip())
            except Exception as e:
                self.logger.error(f"Failed to count records in {log_file}: {e}")

        return count

    def cleanup_old_records(self, retention_days: int) -> int:
        """Clean up records older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        deleted_count = 0

        for log_file in self.storage_directory.glob("audit_*.jsonl"):
            try:
                # Extract date from filename
                date_str = log_file.stem.replace("audit_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")

                if file_date < cutoff_date:
                    # Count records before deletion
                    with open(log_file, "r", encoding="utf-8") as f:
                        file_record_count = sum(1 for line in f if line.strip())

                    log_file.unlink()
                    deleted_count += file_record_count
                    self.logger.info(
                        f"Deleted old audit log {log_file} with {file_record_count} records"
                    )

            except Exception as e:
                self.logger.error(f"Failed to process audit log {log_file}: {e}")

        return deleted_count
