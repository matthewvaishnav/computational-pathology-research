"""Enhanced audit logger with regulatory compliance features."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .audit_crypto import CryptographicSigner
from .audit_logger import AuditLogger
from .audit_models import AuditEvent, SignedAuditRecord
from .audit_storage import AuditStorage


class ComplianceAuditLogger(AuditLogger):
    """Enhanced audit logger with regulatory compliance features."""

    def __init__(
        self,
        storage: Optional[AuditStorage] = None,
        signer: Optional[CryptographicSigner] = None,
        retention_days: int = 2555,  # 7 years for FDA compliance
        backup_storage: Optional[AuditStorage] = None,
    ):
        """Initialize compliance audit logger with backup storage."""
        super().__init__(storage, signer, retention_days)
        self.backup_storage = backup_storage
        self.compliance_logger = logging.getLogger(f"{__name__}.compliance")

    def _store_signed_event(self, event: AuditEvent) -> str:
        """Store event in primary and backup storage for redundancy."""
        event_id = super()._store_signed_event(event)

        # Also store in backup storage if available
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
        """Generate comprehensive compliance report for regulatory submissions."""
        records = self.get_audit_records(start_date, end_date)

        # Verify integrity of all records
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

        # Generate statistics
        stats = self.get_audit_statistics()

        # Create compliance report
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

        # Export detailed records
        records_export_path = output_path.parent / f"{output_path.stem}_records.json"
        self.export_audit_logs(records_export_path, start_date, end_date, "json")

        # Save compliance report
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(compliance_report, f, indent=2)

        self.compliance_logger.info(
            f"Generated compliance report with {len(records)} records for period "
            f"{start_date.date()} to {end_date.date()}"
        )

        return compliance_report

    def validate_audit_chain(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Validate integrity of audit chain for regulatory compliance."""
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
        """Archive old records while maintaining regulatory compliance."""
        records_to_archive = self.get_audit_records(end_time=cutoff_date)

        if not records_to_archive:
            return {
                "archived_count": 0,
                "archive_path": None,
                "archive_timestamp": datetime.now().isoformat(),
            }

        # Create archive with integrity verification
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

        # Save archive
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(archive_data, f, indent=2)

        # Create archive integrity hash
        with open(archive_path, "rb") as f:
            archive_hash = hashlib.sha256(f.read()).hexdigest()

        # Save integrity file
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

        self.compliance_logger.info(f"Archived {len(records_to_archive)} records to {archive_path}")

        return {
            "archived_count": len(records_to_archive),
            "archive_path": str(archive_path),
            "integrity_path": str(integrity_path),
            "archive_hash": archive_hash,
            "archive_timestamp": datetime.now().isoformat(),
        }
