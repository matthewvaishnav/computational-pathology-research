"""Multi-channel clinical notification system for HistoCore PACS integration."""

import logging
import re
import smtplib
import socket
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

# Import data models
from .data_models import AnalysisResults, OperationResult, StudyInfo

logger = logging.getLogger(__name__)


@dataclass
class NotificationEvent:
    """Represents a single notification event to be dispatched across channels."""

    event_id: str
    event_type: str  # "analysis_complete", "critical_finding", "error", "system_alert"
    study_instance_uid: str
    patient_id: str
    analysis_summary: str
    result_url: Optional[str]
    priority: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    timestamp: datetime
    algorithm_name: Optional[str] = None
    confidence_score: Optional[float] = None
    findings: Optional[List[str]] = None

    def is_critical(self) -> bool:
        return self.priority == "CRITICAL"

    def format_subject(self) -> str:
        uid_short = (
            self.study_instance_uid[:8]
            if len(self.study_instance_uid) >= 8
            else self.study_instance_uid
        )
        if self.is_critical():
            return f"[CRITICAL] HistoCore AI: {uid_short}..."
        return f"[AI Result] HistoCore: {uid_short}..."

    def format_body(self) -> str:
        lines = [
            f"HistoCore AI Notification — {self.event_type}",
            f"Priority: {self.priority}",
            f"Timestamp: {self.timestamp.isoformat()}",
            "",
            f"Study UID: {self.study_instance_uid}",
            f"Patient ID: {self.patient_id}",
        ]
        if self.algorithm_name:
            lines.append(f"Algorithm: {self.algorithm_name}")
        if self.confidence_score is not None:
            lines.append(f"Confidence: {self.confidence_score:.4f}")
        lines.append("")
        lines.append(f"Summary: {self.analysis_summary}")
        if self.findings:
            lines.append("")
            lines.append("Key Findings:")
            for f_item in self.findings:
                lines.append(f"  - {f_item}")
        if self.result_url:
            lines.append("")
            lines.append(f"Results: {self.result_url}")
        return "\n".join(lines)


@dataclass
class DeliveryRecord:
    """Tracks the delivery attempt state for a single channel/recipient pair."""

    record_id: str
    event_id: str
    channel: str  # "email", "sms", "hl7"
    recipient: str
    status: str  # "pending", "sent", "failed", "retrying"
    attempts: int = 0
    max_attempts: int = 3
    sent_time: Optional[datetime] = None
    last_attempt_time: Optional[datetime] = None
    error_message: Optional[str] = None

    def can_retry(self) -> bool:
        return self.status in ("failed", "retrying") and self.attempts < self.max_attempts


class NotificationChannel(ABC):
    """Abstract base for all notification delivery channels."""

    channel_name: str

    @abstractmethod
    def send(self, event: NotificationEvent, recipient: str) -> bool:
        """Attempt delivery; returns True on success. Must not raise."""
        ...

    @abstractmethod
    def validate_recipient(self, recipient: str) -> bool:
        """Return True if recipient address/identifier is well-formed."""
        ...


class EmailNotifier(NotificationChannel):
    """Delivers notifications via SMTP email."""

    channel_name = "email"

    def __init__(
        self,
        smtp_server: str = "localhost",
        smtp_port: int = 25,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = False,
        from_address: str = "histocore@hospital.org",
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.from_address = from_address

    def send(self, event: NotificationEvent, recipient: str) -> bool:
        msg = MIMEText(event.format_body())
        msg["Subject"] = event.format_subject()
        msg["From"] = self.from_address
        msg["To"] = recipient
        try:
            if self.use_tls:
                conn = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                conn = smtplib.SMTP(self.smtp_server, self.smtp_port)
            with conn:
                if self.username and self.password and not self.use_tls:
                    conn.login(self.username, self.password)
                conn.sendmail(self.from_address, [recipient], msg.as_string())
            logger.info("Email sent to %s for event %s", recipient, event.event_id)
            return True
        except (ConnectionRefusedError, OSError) as exc:
            # No real SMTP server available — expected in simulation / test environments.
            logger.debug("Email delivery skipped (no SMTP): %s", exc)
            return False
        except Exception as exc:
            logger.warning("Email delivery failed for %s: %s", recipient, exc)
            return False

    def validate_recipient(self, recipient: str) -> bool:
        at_pos = recipient.find("@")
        if at_pos < 1:
            return False
        domain = recipient[at_pos + 1 :]
        return "." in domain


class SMSNotifier(NotificationChannel):
    """Delivers short SMS alerts via an HTTP gateway (e.g. Twilio-compatible)."""

    channel_name = "sms"

    def __init__(
        self,
        gateway_url: Optional[str] = None,
        api_key: Optional[str] = None,
        sender_id: str = "HISTOCORE",
    ):
        self.gateway_url = gateway_url
        self.api_key = api_key
        self.sender_id = sender_id

    def _build_sms_body(self, event: NotificationEvent) -> str:
        critical_flag = "[CRITICAL] " if event.is_critical() else ""
        uid_short = event.study_instance_uid[:12]
        body = f"{critical_flag}HistoCore: Study {uid_short} — {event.analysis_summary}"
        return body[:160]

    def send(self, event: NotificationEvent, recipient: str) -> bool:
        body = self._build_sms_body(event)
        if not self.gateway_url:
            # Simulation mode — log and succeed without hitting an external service.
            logger.info("SMS (sim) to %s: %s", recipient, body)
            return True
        try:
            import json as _json
            import urllib.request

            payload = _json.dumps({"to": recipient, "from": self.sender_id, "body": body}).encode()
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(
                self.gateway_url, data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                success = resp.status < 300
            if success:
                logger.info("SMS sent to %s for event %s", recipient, event.event_id)
            else:
                logger.warning("SMS gateway returned non-2xx for %s", recipient)
            return success
        except Exception as exc:
            logger.warning("SMS delivery failed for %s: %s", recipient, exc)
            return False

    def validate_recipient(self, recipient: str) -> bool:
        # Strip formatting characters before counting digits.
        digits = re.sub(r"[\s\-]", "", recipient)
        if digits.startswith("+"):
            digits = digits[1:]
        return digits.isdigit() and 7 <= len(digits) <= 15


class HL7Notifier(NotificationChannel):
    """Sends HL7 v2 ORU^R01 observation results via MLLP over TCP."""

    channel_name = "hl7"

    def __init__(
        self,
        endpoint_host: Optional[str] = None,
        endpoint_port: int = 2575,
        sending_facility: str = "HISTOCORE",
        receiving_facility: str = "HOSPITAL",
    ):
        self.endpoint_host = endpoint_host
        self.endpoint_port = endpoint_port
        self.sending_facility = sending_facility
        self.receiving_facility = receiving_facility

    def send(self, event: NotificationEvent, recipient: str) -> bool:
        message = self._build_hl7_message(event, recipient)
        if not self.endpoint_host:
            logger.info("HL7 (sim) to %s: %s", recipient, message[:120])
            return True
        return self._send_mllp(self.endpoint_host, self.endpoint_port, message)

    def validate_recipient(self, recipient: str) -> bool:
        return bool(re.match(r"^[A-Za-z0-9_\-]{1,20}$", recipient))

    def _build_hl7_message(self, event: NotificationEvent, receiving_app: str) -> str:
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        msg_id = uuid.uuid4().hex[:10].upper()
        # Truncate free-text fields to avoid segment overflow.
        summary = event.analysis_summary.replace("|", " ").replace("\n", " ")[:100]
        study_uid = event.study_instance_uid.replace("|", "")
        patient_id = event.patient_id.replace("|", "")

        msh = (
            f"MSH|^~\\&|{self.sending_facility}|{self.sending_facility}|"
            f"{receiving_app}|{self.receiving_facility}|{now}||ORU^R01|{msg_id}|P|2.3"
        )
        pid = f"PID|1||{patient_id}^^^{self.receiving_facility}||||||||||||||"
        obr = f"OBR|1||{study_uid}|AI_ANALYSIS^AI Analysis|||{now}"
        obx = (
            f"OBX|1|ST|AI_RESULT^AI Analysis Result||{summary}||||||F|||{now}|"
            f"{self.sending_facility}"
        )
        return "\r".join([msh, pid, obr, obx]) + "\r"

    def _send_mllp(self, host: str, port: int, message: str) -> bool:
        mllp_frame = b"\x0b" + message.encode("utf-8") + b"\x1c\x0d"
        try:
            with socket.create_connection((host, port), timeout=10) as sock:
                sock.sendall(mllp_frame)
                # Read ACK — minimal read; HL7 ACK may be brief.
                sock.recv(4096)
            logger.info("HL7 MLLP sent to %s:%d", host, port)
            return True
        except Exception as exc:
            logger.warning("HL7 MLLP delivery failed to %s:%d: %s", host, port, exc)
            return False


class DeliveryTracker:
    """Thread-safe store of delivery attempt records."""

    def __init__(self):
        self._records: Dict[str, DeliveryRecord] = {}
        self._event_records: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    def create_record(
        self,
        event_id: str,
        channel: str,
        recipient: str,
        max_attempts: int = 3,
    ) -> DeliveryRecord:
        record = DeliveryRecord(
            record_id=str(uuid.uuid4()),
            event_id=event_id,
            channel=channel,
            recipient=recipient,
            status="pending",
            max_attempts=max_attempts,
        )
        with self._lock:
            self._records[record.record_id] = record
            self._event_records.setdefault(event_id, []).append(record.record_id)
        return record

    def mark_sent(self, record_id: str) -> None:
        with self._lock:
            rec = self._records.get(record_id)
            if rec:
                rec.status = "sent"
                rec.sent_time = datetime.now()
                rec.last_attempt_time = rec.sent_time
                rec.attempts += 1

    def mark_failed(self, record_id: str, error_message: str) -> None:
        with self._lock:
            rec = self._records.get(record_id)
            if rec:
                rec.attempts += 1
                rec.last_attempt_time = datetime.now()
                rec.error_message = error_message
                rec.status = "retrying" if rec.attempts < rec.max_attempts else "failed"

    def get_pending_retries(self) -> List[DeliveryRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.can_retry()]

    def get_event_status(self, event_id: str) -> Dict[str, Any]:
        with self._lock:
            record_ids = self._event_records.get(event_id, [])
            records = [self._records[rid] for rid in record_ids if rid in self._records]
        sent = sum(1 for r in records if r.status == "sent")
        failed = sum(1 for r in records if r.status == "failed")
        pending = sum(1 for r in records if r.status in ("pending", "retrying"))
        return {
            "event_id": event_id,
            "total_channels": len(records),
            "sent": sent,
            "failed": failed,
            "pending": pending,
        }

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            records = list(self._records.values())
        by_channel: Dict[str, Dict[str, int]] = {}
        for rec in records:
            ch = by_channel.setdefault(rec.channel, {"sent": 0, "failed": 0, "pending": 0})
            if rec.status == "sent":
                ch["sent"] += 1
            elif rec.status == "failed":
                ch["failed"] += 1
            else:
                ch["pending"] += 1
        return {
            "total_records": len(records),
            "sent": sum(1 for r in records if r.status == "sent"),
            "failed": sum(1 for r in records if r.status == "failed"),
            "pending": sum(1 for r in records if r.status in ("pending", "retrying")),
            "by_channel": by_channel,
        }


class NotificationSystem:
    """Orchestrates multi-channel clinical alerts for AI analysis events."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.channels: Dict[str, NotificationChannel] = {}
        self.tracker = DeliveryTracker()
        self._lock = threading.Lock()

        # role → channel → [address]
        self._recipients: Dict[str, Dict[str, List[str]]] = {}

        self._critical_threshold: float = cfg.get("critical_finding_threshold", 0.9)
        self._escalation_delay_seconds: int = cfg.get("escalation_delay_seconds", 300)

        # Honour channel sub-configs if provided up front.
        if "email" in cfg and cfg["email"].get("enabled", False):
            email_cfg = cfg["email"]
            self.configure_email(
                smtp_server=email_cfg.get("smtp_server", "localhost"),
                smtp_port=email_cfg.get("smtp_port", 587),
                username=email_cfg.get("username"),
                password=email_cfg.get("password"),
                use_tls=email_cfg.get("use_tls", True),
                from_address=email_cfg.get("from_address", "histocore@hospital.org"),
            )
        if "sms" in cfg and cfg["sms"].get("enabled", False):
            sms_cfg = cfg["sms"]
            self.configure_sms(
                gateway_url=sms_cfg.get("gateway_url"),
                api_key=sms_cfg.get("api_key"),
            )
        if "hl7" in cfg and cfg["hl7"].get("enabled", False):
            hl7_cfg = cfg["hl7"]
            self.configure_hl7(
                endpoint_host=hl7_cfg.get("endpoint_host"),
                endpoint_port=hl7_cfg.get("endpoint_port", 2575),
            )

        # Bulk-load recipients from config.
        for role, addresses in cfg.get("recipients", {}).items():
            for entry in addresses:
                # entry may be a dict {"channel": ..., "address": ...} or a bare string.
                if isinstance(entry, dict):
                    self.add_recipient(role, entry["channel"], entry["address"])

    def configure_email(
        self,
        smtp_server: str,
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
        from_address: str = "histocore@hospital.org",
    ) -> None:
        self.channels["email"] = EmailNotifier(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            username=username,
            password=password,
            use_tls=use_tls,
            from_address=from_address,
        )

    def configure_sms(
        self,
        gateway_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.channels["sms"] = SMSNotifier(gateway_url=gateway_url, api_key=api_key)

    def configure_hl7(
        self,
        endpoint_host: Optional[str] = None,
        endpoint_port: int = 2575,
    ) -> None:
        self.channels["hl7"] = HL7Notifier(
            endpoint_host=endpoint_host,
            endpoint_port=endpoint_port,
        )

    def add_recipient(self, role: str, channel: str, address: str) -> None:
        """Register a recipient address for a role/channel combination.

        Validates the address format before storing — invalid addresses are logged
        and silently dropped rather than raising so callers don't have to handle
        exceptions from configuration time.
        """
        notifier = self.channels.get(channel)
        if notifier and not notifier.validate_recipient(address):
            logger.warning("Invalid %s address for role %s: %s", channel, role, address)
            return
        with self._lock:
            role_map = self._recipients.setdefault(role, {})
            role_map.setdefault(channel, [])
            if address not in role_map[channel]:
                role_map[channel].append(address)

    def notify_analysis_complete(
        self,
        analysis_results: "AnalysisResults",
        study_info: Optional["StudyInfo"] = None,
        result_url: Optional[str] = None,
    ) -> "OperationResult":
        from .data_models import OperationResult

        event = self._create_event_from_analysis(analysis_results, study_info, result_url)
        op_id = str(uuid.uuid4())
        sent, failed = self._broadcast_event(event)
        return OperationResult.success_result(
            operation_id=op_id,
            message=f"Analysis complete notification dispatched: {sent} sent, {failed} failed",
            data={"event_id": event.event_id, "sent": sent, "failed": failed},
        )

    def notify_critical_finding(
        self,
        study_instance_uid: str,
        patient_id: str,
        finding_description: str,
        confidence: float,
    ) -> "OperationResult":
        from .data_models import OperationResult

        event = NotificationEvent(
            event_id=str(uuid.uuid4()),
            event_type="critical_finding",
            study_instance_uid=study_instance_uid,
            patient_id=patient_id,
            analysis_summary=finding_description,
            result_url=None,
            priority="CRITICAL",
            timestamp=datetime.now(),
            confidence_score=confidence,
            findings=[finding_description],
        )
        op_id = str(uuid.uuid4())
        sent, failed = self._broadcast_event(event, include_admins=True)
        return OperationResult.success_result(
            operation_id=op_id,
            message=f"Critical finding notification dispatched: {sent} sent, {failed} failed",
            data={"event_id": event.event_id, "sent": sent, "failed": failed},
        )

    def notify_system_error(self, error_message: str, component: str) -> "OperationResult":
        from .data_models import OperationResult

        event = NotificationEvent(
            event_id=str(uuid.uuid4()),
            event_type="error",
            study_instance_uid="SYSTEM",
            patient_id="SYSTEM",
            analysis_summary=f"Component {component} error: {error_message}",
            result_url=None,
            priority="HIGH",
            timestamp=datetime.now(),
            findings=[f"Component: {component}", f"Error: {error_message}"],
        )
        op_id = str(uuid.uuid4())
        sent, failed = self._broadcast_to_roles(event, roles=["admin"])
        return OperationResult.success_result(
            operation_id=op_id,
            message=f"System error notification dispatched: {sent} sent, {failed} failed",
            data={"event_id": event.event_id, "sent": sent, "failed": failed},
        )

    def retry_failed_deliveries(self) -> int:
        retries = self.tracker.get_pending_retries()
        success_count = 0
        for record in retries:
            channel = self.channels.get(record.channel)
            if not channel:
                continue
            # Reconstruct a minimal stub event for re-delivery; full event data is
            # unavailable here, so we use the stored summary placeholder.
            stub_event = NotificationEvent(
                event_id=record.event_id,
                event_type="retry",
                study_instance_uid="UNKNOWN",
                patient_id="UNKNOWN",
                analysis_summary="(retry)",
                result_url=None,
                priority="HIGH",
                timestamp=datetime.now(),
            )
            ok = channel.send(stub_event, record.recipient)
            if ok:
                self.tracker.mark_sent(record.record_id)
                success_count += 1
            else:
                self.tracker.mark_failed(record.record_id, "retry failed")
        return success_count

    def get_delivery_statistics(self) -> Dict[str, Any]:
        return self.tracker.get_statistics()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_notification(
        self, event: NotificationEvent, channel_name: str, recipient: str
    ) -> bool:
        channel = self.channels.get(channel_name)
        if not channel:
            return False
        record = self.tracker.create_record(event.event_id, channel_name, recipient)
        try:
            ok = channel.send(event, recipient)
        except Exception as exc:
            # Belt-and-suspenders: channels should not raise, but guard anyway.
            logger.error("Unexpected error from channel %s: %s", channel_name, exc)
            ok = False
        if ok:
            self.tracker.mark_sent(record.record_id)
        else:
            self.tracker.mark_failed(record.record_id, "send returned False")
        return ok

    def _broadcast_event(self, event: NotificationEvent, include_admins: bool = False) -> tuple:
        """Send event to all pathologist recipients (and optionally admins) on all channels."""
        roles = ["pathologist"]
        if include_admins:
            roles.append("admin")
        return self._broadcast_to_roles(event, roles=roles)

    def _broadcast_to_roles(self, event: NotificationEvent, roles: List[str]) -> tuple:
        sent = 0
        failed = 0
        with self._lock:
            snapshot = {r: dict(self._recipients.get(r, {})) for r in roles}

        for channel_name in list(self.channels):
            # Collect all unique addresses across the requested roles for this channel.
            addresses: List[str] = []
            for role_map in snapshot.values():
                for addr in role_map.get(channel_name, []):
                    if addr not in addresses:
                        addresses.append(addr)

            # When no named recipients are registered, send a single "broadcast"
            # placeholder so delivery tracking reflects the channel was attempted.
            if not addresses:
                addresses = [f"broadcast@{channel_name}"]

            for addr in addresses:
                ok = self._send_notification(event, channel_name, addr)
                if ok:
                    sent += 1
                else:
                    failed += 1
        return sent, failed

    def _create_event_from_analysis(
        self,
        analysis_results,
        study_info,
        result_url: Optional[str],
    ) -> NotificationEvent:
        patient_id = getattr(study_info, "patient_id", None) or getattr(
            analysis_results, "patient_id", "UNKNOWN"
        )

        # Determine priority from urgency levels and confidence score.
        has_urgent = any(
            getattr(rec, "urgency_level", "") == "URGENT"
            for rec in (analysis_results.diagnostic_recommendations or [])
        )
        above_threshold = (
            analysis_results.confidence_score is not None
            and analysis_results.confidence_score >= self._critical_threshold
        )
        if has_urgent or above_threshold:
            priority = "CRITICAL"
        elif (
            analysis_results.confidence_score is not None
            and analysis_results.confidence_score >= 0.7
        ):
            priority = "HIGH"
        else:
            priority = "MEDIUM"

        findings = [
            rec.recommendation_text for rec in (analysis_results.diagnostic_recommendations or [])
        ]

        summary_parts = []
        if analysis_results.primary_diagnosis:
            summary_parts.append(f"Diagnosis: {analysis_results.primary_diagnosis}")
        summary_parts.append(
            f"Confidence: {analysis_results.confidence_score:.2f}"
            if analysis_results.confidence_score is not None
            else "Confidence: N/A"
        )
        analysis_summary = "; ".join(summary_parts) if summary_parts else "Analysis complete"

        return NotificationEvent(
            event_id=str(uuid.uuid4()),
            event_type="analysis_complete",
            study_instance_uid=analysis_results.study_instance_uid,
            patient_id=patient_id,
            analysis_summary=analysis_summary,
            result_url=result_url,
            priority=priority,
            timestamp=datetime.now(),
            algorithm_name=getattr(analysis_results, "algorithm_name", None),
            confidence_score=analysis_results.confidence_score,
            findings=findings if findings else None,
        )
