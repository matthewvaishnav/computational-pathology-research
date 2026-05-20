"""HIPAA-compliant DICOM audit logging for PACS operations."""

import hashlib
import hmac
import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class AuditParticipant:
    user_id: str
    user_name: str
    user_role: str
    network_access_point: Optional[str] = None
    is_requestor: bool = True


@dataclass
class AuditStudyObject:
    study_instance_uid: str
    patient_id: str
    patient_name: str
    accession_number: Optional[str] = None
    sop_class_uid: Optional[str] = None


@dataclass
class AuditMessage:
    message_id: str
    event_id: str
    event_action: str
    event_outcome: int
    event_datetime: datetime
    event_type: str
    participants: List[AuditParticipant]
    study_objects: List[AuditStudyObject]
    description: str
    source_system: str = "HistoCore PACS Integration"
    phi_accessed: bool = False
    phi_fields: Optional[List[str]] = None
    additional_data: Optional[Dict[str, Any]] = None

    # DICOM Part 15 Annex A event codes
    EVENT_CODES: Dict[str, str] = field(
        default_factory=lambda: {
            "DICOM_QUERY": "110112",
            "DICOM_RETRIEVE": "110107",
            "DICOM_STORE": "110106",
            "PHI_ACCESS": "110103",
            "SECURITY_EVENT": "110114",
            "SYSTEM_EVENT": "110100",
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "event_id": self.event_id,
            "event_action": self.event_action,
            "event_outcome": self.event_outcome,
            "event_datetime": self.event_datetime.isoformat(),
            "event_type": self.event_type,
            "source_system": self.source_system,
            "description": self.description,
            "phi_accessed": self.phi_accessed,
            "phi_fields": self.phi_fields,
            "additional_data": self.additional_data,
            "participants": [asdict(p) for p in self.participants],
            "study_objects": [asdict(s) for s in self.study_objects],
        }

    def to_hipaa_format(self) -> Dict[str, Any]:
        # HIPAA requires explicit participant identity, action, outcome, and
        # patient identifier — all in a single traceable record per 45 CFR §164.312(b)
        return {
            "event_type": self.event_type,
            "event_datetime": self.event_datetime.isoformat(),
            "outcome": self.event_outcome,
            "action": self.event_action,
            "source_system": self.source_system,
            "phi_accessed": self.phi_accessed,
            "phi_fields": self.phi_fields or [],
            "participants": [
                {
                    "user_id": p.user_id,
                    "user_name": p.user_name,
                    "user_role": p.user_role,
                    "is_requestor": p.is_requestor,
                    "network_access_point": p.network_access_point,
                }
                for p in self.participants
            ],
            "study_objects": [
                {
                    "study_instance_uid": s.study_instance_uid,
                    "patient_id": s.patient_id,
                    "patient_name": s.patient_name,
                    "accession_number": s.accession_number,
                }
                for s in self.study_objects
            ],
        }


class TamperEvidentStorage:
    def __init__(self, storage_path: Path, signing_key: Optional[bytes] = None):
        self.storage_path = storage_path
        # A session-only key means integrity cannot be verified across restarts;
        # callers in production MUST supply a persisted key from a secrets vault.
        if signing_key is None:
            logger.warning(
                "No signing_key supplied — generating ephemeral session key. "
                "Log integrity cannot be verified after restart."
            )
        self._key = signing_key or os.urandom(32)
        storage_path.mkdir(parents=True, exist_ok=True)

    def write_entry(self, entry: Dict[str, Any]) -> str:
        entry_json = json.dumps(entry, sort_keys=True).encode()
        signature = self._compute_signature(entry_json)
        message_id = entry.get("message_id", uuid.uuid4().hex)
        dt = datetime.fromisoformat(entry["event_datetime"])
        day_dir = self.storage_path / dt.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        file_path = day_dir / f"{message_id}.json"
        payload = {
            "data": entry,
            "signature": signature,
            "timestamp": datetime.utcnow().isoformat(),
        }
        file_path.write_text(json.dumps(payload), encoding="utf-8")
        return str(file_path.relative_to(self.storage_path))

    def verify_entry(self, entry_path: Path) -> bool:
        try:
            raw = entry_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            stored_sig = payload["signature"]
            entry_json = json.dumps(payload["data"], sort_keys=True).encode()
            expected_sig = self._compute_signature(entry_json)
            # Constant-time comparison prevents timing attacks against the HMAC
            return hmac.compare_digest(stored_sig, expected_sig)
        except Exception:
            return False

    def read_entry(self, entry_path: Path) -> Optional[Dict[str, Any]]:
        if not self.verify_entry(entry_path):
            return None
        payload = json.loads(entry_path.read_text(encoding="utf-8"))
        return payload["data"]

    def _compute_signature(self, data: bytes) -> str:
        return hmac.new(self._key, data, hashlib.sha256).hexdigest()


class AuditSearchIndex:
    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def add_entry(self, message: AuditMessage, file_path: str) -> None:
        index_record = {
            "message_id": message.message_id,
            "event_type": message.event_type,
            "event_datetime": message.event_datetime.isoformat(),
            "event_outcome": message.event_outcome,
            "phi_accessed": message.phi_accessed,
            "phi_fields": message.phi_fields or [],
            "file_path": file_path,
            "user_ids": [p.user_id for p in message.participants],
            "patient_ids": [s.patient_id for s in message.study_objects],
        }
        with self._lock:
            self._entries.append(index_record)

    def search(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        patient_id: Optional[str] = None,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        outcome: Optional[int] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            results = []
            for entry in self._entries:
                dt = datetime.fromisoformat(entry["event_datetime"])
                if start_date and dt < start_date:
                    continue
                if end_date and dt > end_date:
                    continue
                if event_type and entry["event_type"] != event_type:
                    continue
                if user_id and user_id not in entry["user_ids"]:
                    continue
                if outcome is not None and entry["event_outcome"] != outcome:
                    continue
                if patient_id and patient_id not in entry["patient_ids"]:
                    continue
                results.append(entry)
            results.sort(key=lambda e: e["event_datetime"], reverse=True)
            return results[:limit]

    def generate_report(
        self,
        start_date: datetime,
        end_date: datetime,
        report_type: str = "summary",
    ) -> Dict[str, Any]:
        entries = self.search(start_date=start_date, end_date=end_date, limit=0)
        if report_type == "summary":
            by_event: Dict[str, int] = {}
            by_outcome: Dict[int, int] = {}
            by_user: Dict[str, int] = {}
            for e in entries:
                by_event[e["event_type"]] = by_event.get(e["event_type"], 0) + 1
                by_outcome[e["event_outcome"]] = by_outcome.get(e["event_outcome"], 0) + 1
                for uid in e["user_ids"]:
                    by_user[uid] = by_user.get(uid, 0) + 1
            return {
                "report_type": "summary",
                "total_events": len(entries),
                "by_event_type": by_event,
                "by_outcome": by_outcome,
                "by_user": by_user,
            }
        elif report_type == "phi_access":
            phi_entries = [e for e in entries if e["phi_accessed"]]
            return {
                "report_type": "phi_access",
                "total_phi_events": len(phi_entries),
                "entries": phi_entries,
            }
        elif report_type == "failures":
            failed = [e for e in entries if e["event_outcome"] != 0]
            return {
                "report_type": "failures",
                "total_failures": len(failed),
                "entries": failed,
            }
        else:
            raise ValueError(f"Unknown report_type: {report_type!r}")


class LogRetentionManager:
    def __init__(self, storage_path: Path, retention_years: int = 7) -> None:
        if not (1 <= retention_years <= 10):
            raise ValueError("retention_years must be between 1 and 10")
        self.storage_path = storage_path
        self.retention_years = retention_years

    def should_archive(self, entry_date: date) -> bool:
        # Move to cold archive after one year to keep active storage lean
        return (date.today() - entry_date).days > 365

    def should_delete(self, entry_date: date) -> bool:
        return (date.today() - entry_date).days > self.retention_years * 365

    def archive_old_entries(self, archive_path: Path) -> int:
        archive_path.mkdir(parents=True, exist_ok=True)
        moved = 0
        for day_dir in sorted(self.storage_path.glob("????????")):
            if not day_dir.is_dir():
                continue
            try:
                entry_date = datetime.strptime(day_dir.name, "%Y%m%d").date()
            except ValueError:
                continue
            if not self.should_archive(entry_date):
                continue
            dest_dir = archive_path / day_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for json_file in day_dir.glob("*.json"):
                dest = dest_dir / json_file.name
                json_file.rename(dest)
                moved += 1
            try:
                day_dir.rmdir()
            except OSError:
                pass
        return moved

    def delete_expired_entries(self) -> int:
        deleted = 0
        for day_dir in sorted(self.storage_path.glob("????????")):
            if not day_dir.is_dir():
                continue
            try:
                entry_date = datetime.strptime(day_dir.name, "%Y%m%d").date()
            except ValueError:
                continue
            if not self.should_delete(entry_date):
                continue
            for json_file in day_dir.glob("*.json"):
                # Each deletion is individually logged so the meta-audit trail
                # satisfies HIPAA's requirement to record destruction of PHI records
                logger.info("Deleting expired audit entry: %s", json_file)
                json_file.unlink()
                deleted += 1
            try:
                day_dir.rmdir()
            except OSError:
                pass
        return deleted

    def get_retention_status(self) -> Dict[str, Any]:
        all_files: List[Path] = list(self.storage_path.glob("????????/*.json"))
        total = len(all_files)
        archivable = 0
        expired = 0
        oldest: Optional[date] = None
        for json_file in all_files:
            try:
                entry_date = datetime.strptime(json_file.parent.name, "%Y%m%d").date()
            except ValueError:
                continue
            if oldest is None or entry_date < oldest:
                oldest = entry_date
            if self.should_delete(entry_date):
                expired += 1
            elif self.should_archive(entry_date):
                archivable += 1
        return {
            "total_entries": total,
            "archivable_count": archivable,
            "expired_count": expired,
            "oldest_entry_date": oldest.isoformat() if oldest else None,
            "retention_years": self.retention_years,
        }


class PACSAuditLogger:
    def __init__(
        self,
        storage_path: Union[str, Path] = "./logs/pacs_audit",
        retention_years: int = 7,
        phi_protection_enabled: bool = True,
        signing_key: Optional[bytes] = None,
        system_user_id: str = "HISTOCORE_PACS",
    ) -> None:
        self.storage_path = Path(storage_path)
        self.phi_protection_enabled = phi_protection_enabled
        self.system_user_id = system_user_id
        self._storage = TamperEvidentStorage(self.storage_path / "entries", signing_key)
        self._index = AuditSearchIndex()
        self._retention = LogRetentionManager(self.storage_path / "entries", retention_years)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_dicom_query(
        self,
        user_id: str,
        endpoint: Any,
        query_params: Dict[str, Any],
        result_count: int,
        outcome: int = 0,
    ) -> str:
        phi_fields = [f for f in ("PatientID", "PatientName") if f in query_params]
        participants = [
            AuditParticipant(
                user_id=user_id,
                user_name=user_id,
                user_role="Pathologist",
                network_access_point=getattr(endpoint, "host", None),
                is_requestor=True,
            ),
            AuditParticipant(
                user_id=getattr(endpoint, "ae_title", "UNKNOWN"),
                user_name=getattr(endpoint, "ae_title", "UNKNOWN"),
                user_role="PACS Adapter",
                network_access_point=getattr(endpoint, "host", None),
                is_requestor=False,
            ),
        ]
        message = self._create_audit_message(
            event_type="DICOM_QUERY",
            event_action="R",
            outcome=outcome,
            description=(
                f"C-FIND query by {user_id} on "
                f"{getattr(endpoint, 'ae_title', 'UNKNOWN')}: "
                f"{result_count} results"
            ),
            participants=participants,
            phi_accessed=True,
            phi_fields=phi_fields if phi_fields else ["PatientID", "PatientName"],
            additional_data={"query_params": query_params, "result_count": result_count},
        )
        return self._log_message(message)

    def log_dicom_retrieve(
        self,
        user_id: str,
        endpoint: Any,
        study_info: Any,
        file_count: int,
        outcome: int = 0,
    ) -> str:
        patient_name = study_info.patient_name
        patient_id = study_info.patient_id
        if self.phi_protection_enabled:
            patient_name = self._hash_phi(patient_name)

        participants = [
            AuditParticipant(
                user_id=user_id,
                user_name=user_id,
                user_role="Pathologist",
                network_access_point=getattr(endpoint, "host", None),
                is_requestor=True,
            ),
            AuditParticipant(
                user_id=getattr(endpoint, "ae_title", "UNKNOWN"),
                user_name=getattr(endpoint, "ae_title", "UNKNOWN"),
                user_role="PACS Adapter",
                network_access_point=getattr(endpoint, "host", None),
                is_requestor=False,
            ),
        ]
        study_objects = [
            AuditStudyObject(
                study_instance_uid=study_info.study_instance_uid,
                patient_id=patient_id,
                patient_name=patient_name,
                accession_number=getattr(study_info, "accession_number", None),
            )
        ]
        phi_fields = ["PatientID", "PatientName", "StudyInstanceUID"]
        if getattr(study_info, "accession_number", None):
            phi_fields.append("AccessionNumber")
        message = self._create_audit_message(
            event_type="DICOM_RETRIEVE",
            event_action="R",
            outcome=outcome,
            description=(
                f"C-MOVE retrieve by {user_id}: "
                f"study {study_info.study_instance_uid}, {file_count} files"
            ),
            participants=participants,
            study_objects=study_objects,
            phi_accessed=True,
            phi_fields=phi_fields,
            additional_data={"file_count": file_count},
        )
        return self._log_message(message)

    def log_dicom_store(
        self,
        user_id: str,
        endpoint: Any,
        study_instance_uid: str,
        sop_instance_uid: str,
        outcome: int = 0,
    ) -> str:
        participants = [
            AuditParticipant(
                user_id=user_id,
                user_name=user_id,
                user_role="AI System",
                is_requestor=True,
            ),
            AuditParticipant(
                user_id=getattr(endpoint, "ae_title", "UNKNOWN"),
                user_name=getattr(endpoint, "ae_title", "UNKNOWN"),
                user_role="PACS Adapter",
                network_access_point=getattr(endpoint, "host", None),
                is_requestor=False,
            ),
        ]
        message = self._create_audit_message(
            event_type="DICOM_STORE",
            event_action="C",
            outcome=outcome,
            description=(
                f"C-STORE by {user_id}: SOP {sop_instance_uid} " f"into study {study_instance_uid}"
            ),
            participants=participants,
            additional_data={
                "study_instance_uid": study_instance_uid,
                "sop_instance_uid": sop_instance_uid,
            },
        )
        return self._log_message(message)

    def log_phi_access(
        self,
        user_id: str,
        patient_id: str,
        patient_name: str,
        accessed_fields: List[str],
        reason: str,
        outcome: int = 0,
    ) -> str:
        # PHI access events are HIPAA-critical; always record real patient_id
        # but de-identify name when phi_protection is on
        stored_name = self._hash_phi(patient_name) if self.phi_protection_enabled else patient_name
        participants = [
            AuditParticipant(
                user_id=user_id,
                user_name=user_id,
                user_role="Pathologist",
                is_requestor=True,
            )
        ]
        study_objects = [
            AuditStudyObject(
                study_instance_uid="N/A",
                patient_id=patient_id,
                patient_name=stored_name,
            )
        ]
        message = self._create_audit_message(
            event_type="PHI_ACCESS",
            event_action="R",
            outcome=outcome,
            description=f"PHI access by {user_id} for patient {patient_id}: {reason}",
            participants=participants,
            study_objects=study_objects,
            phi_accessed=True,
            phi_fields=accessed_fields,
            additional_data={"reason": reason},
        )
        return self._log_message(message)

    def log_security_event(self, event: Dict[str, Any]) -> str:
        endpoint = event.get("endpoint")
        participants = [
            AuditParticipant(
                user_id=event.get("user_id", self.system_user_id),
                user_name=event.get("user_id", self.system_user_id),
                user_role="Administrator",
                network_access_point=getattr(endpoint, "host", None) if endpoint else None,
                is_requestor=True,
            )
        ]
        message = self._create_audit_message(
            event_type="SECURITY_EVENT",
            event_action="E",
            outcome=event.get("outcome", 0),
            description=event.get("details", "Security event"),
            participants=participants,
            additional_data={k: v for k, v in event.items() if k != "endpoint"},
        )
        return self._log_message(message)

    def log_system_event(self, event_type: str, description: str, outcome: int = 0) -> str:
        participants = [
            AuditParticipant(
                user_id=self.system_user_id,
                user_name=self.system_user_id,
                user_role="AI System",
                is_requestor=True,
            )
        ]
        message = self._create_audit_message(
            event_type="SYSTEM_EVENT",
            event_action="E",
            outcome=outcome,
            description=description,
            participants=participants,
            additional_data={"sub_event_type": event_type},
        )
        return self._log_message(message)

    # ------------------------------------------------------------------
    # Search and reporting
    # ------------------------------------------------------------------

    def search_logs(self, **kwargs: Any) -> List[Dict[str, Any]]:
        return self._index.search(**kwargs)

    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        report_type: str = "summary",
    ) -> Dict[str, Any]:
        report = self._index.generate_report(start_date, end_date, report_type)
        report["metadata"] = {
            "generated_at": datetime.utcnow().isoformat(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "source_system": "HistoCore PACS Integration",
        }
        return report

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------

    def verify_log_integrity(self, entry_path: Optional[str] = None) -> Dict[str, Any]:
        entries_root = self._storage.storage_path
        if entry_path is not None:
            full_path = entries_root / entry_path
            valid = self._storage.verify_entry(full_path)
            return {
                "total": 1,
                "valid": 1 if valid else 0,
                "tampered": 0 if valid else 1,
                "missing": 0,
            }

        total = valid = tampered = missing = 0
        for json_file in entries_root.glob("????????/*.json"):
            total += 1
            if not json_file.exists():
                missing += 1
            elif self._storage.verify_entry(json_file):
                valid += 1
            else:
                tampered += 1
        return {"total": total, "valid": valid, "tampered": tampered, "missing": missing}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_audit_message(
        self,
        event_type: str,
        event_action: str,
        outcome: int,
        description: str,
        participants: List[AuditParticipant],
        study_objects: Optional[List[AuditStudyObject]] = None,
        phi_accessed: bool = False,
        phi_fields: Optional[List[str]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> AuditMessage:
        # Construct a temporary instance just to resolve the EVENT_CODES mapping;
        # the field default_factory is instance-level for dataclasses.
        _codes = {
            "DICOM_QUERY": "110112",
            "DICOM_RETRIEVE": "110107",
            "DICOM_STORE": "110106",
            "PHI_ACCESS": "110103",
            "SECURITY_EVENT": "110114",
            "SYSTEM_EVENT": "110100",
        }
        msg = AuditMessage(
            message_id=str(uuid.uuid4()),
            event_id=_codes.get(event_type, "110100"),
            event_action=event_action,
            event_outcome=outcome,
            event_datetime=datetime.utcnow(),
            event_type=event_type,
            participants=participants,
            study_objects=study_objects or [],
            description=description,
            phi_accessed=phi_accessed,
            phi_fields=phi_fields,
            additional_data=additional_data,
        )
        return msg

    def _hash_phi(self, phi_value: str) -> str:
        # Truncated SHA-256 preserves enough entropy to cross-reference entries
        # without exposing the raw PHI value in log storage
        return hashlib.sha256(phi_value.encode()).hexdigest()

    def _log_message(self, message: AuditMessage) -> str:
        with self._lock:
            entry_dict = message.to_dict()
            file_path = self._storage.write_entry(entry_dict)
            self._index.add_entry(message, file_path)
        logger.debug("Audit entry written: %s (%s)", message.message_id, message.event_type)
        return message.message_id
