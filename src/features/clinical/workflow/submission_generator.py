"""
Regulatory Submission Package Generator

Generates complete regulatory submission packages (510(k), PMA, CE marking)
including DMR, risk analysis, V&V reports, and cybersecurity documentation.

Requirements: 20.7 (regulatory submission support)
"""

import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SubmissionGenerator:
    """
    Regulatory submission package generator

    Creates complete submission packages for FDA 510(k), PMA, and CE marking.
    """

    def __init__(self, documentation_path: str = "regulatory_docs"):
        self.documentation_path = Path(documentation_path)
        self.cybersecurity_path = self.documentation_path / "cybersecurity"
        self.cybersecurity_path.mkdir(exist_ok=True)

        logger.info(f"Initialized submission generator at {self.documentation_path}")

    def create_cybersecurity_plan(
        self,
        device_name: str,
        device_version: str,
        threat_model: Dict[str, Any],
        security_controls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Create cybersecurity plan following FDA guidance

        Args:
            device_name: Name of the device
            device_version: Version of the device
            threat_model: Cybersecurity threat model
            security_controls: List of implemented security controls

        Returns:
            Cybersecurity plan record
        """
        cybersecurity_plan = {
            "device_name": device_name,
            "device_version": device_version,
            "plan_date": datetime.datetime.now().isoformat(),
            "fda_guidance_version": "2022",
            "threat_model": threat_model,
            "security_controls": security_controls,
            "vulnerability_management": {
                "scanning_frequency": "monthly",
                "patch_management_process": "automated_with_approval",
                "incident_response_plan": "defined",
            },
            "security_monitoring": {
                "logging_enabled": True,
                "intrusion_detection": True,
                "anomaly_detection": True,
            },
        }

        # Save cybersecurity plan
        filename = f"{device_name}_{device_version}_cybersecurity_plan.json"
        filepath = self.cybersecurity_path / filename

        with open(filepath, "w") as f:
            json.dump(cybersecurity_plan, f, indent=2)

        logger.info(f"Created cybersecurity plan for {device_name} v{device_version}")
        return cybersecurity_plan

    def log_security_event(
        self,
        device_name: str,
        device_version: str,
        event_type: str,
        severity: str,
        description: str,
        mitigation_actions: List[str],
    ) -> Dict[str, Any]:
        """
        Log cybersecurity event

        Args:
            device_name: Name of the device
            device_version: Version of the device
            event_type: Type of security event
            severity: Severity level (low, medium, high, critical)
            description: Description of the event
            mitigation_actions: Actions taken to mitigate

        Returns:
            Security event record
        """
        event_record = {
            "device_name": device_name,
            "device_version": device_version,
            "event_id": hashlib.sha256(
                f"{device_name}_{datetime.datetime.now().isoformat()}".encode()
            ).hexdigest()[:16],
            "event_type": event_type,
            "severity": severity,
            "description": description,
            "timestamp": datetime.datetime.now().isoformat(),
            "mitigation_actions": mitigation_actions,
            "status": "open",
            "resolution_date": None,
        }

        # Save security event
        filename = f"security_event_{event_record['event_id']}.json"
        filepath = self.cybersecurity_path / filename

        with open(filepath, "w") as f:
            json.dump(event_record, f, indent=2)

        logger.warning(f"Logged security event {event_record['event_id']} for {device_name}")
        return event_record

    def update_security_event(
        self, event_id: str, status: str, resolution_date: str = None
    ) -> None:
        """
        Update security event status

        Args:
            event_id: ID of the security event
            status: New status (open, investigating, resolved, closed)
            resolution_date: Date of resolution if applicable
        """
        filename = f"security_event_{event_id}.json"
        filepath = self.cybersecurity_path / filename

        if not filepath.exists():
            raise ValueError(f"Security event {event_id} not found")

        with open(filepath, "r") as f:
            event_record = json.load(f)

        event_record["status"] = status
        if resolution_date:
            event_record["resolution_date"] = resolution_date

        with open(filepath, "w") as f:
            json.dump(event_record, f, indent=2)

        logger.info(f"Updated security event {event_id} status to {status}")

    def generate_submission_package(
        self,
        device_name: str,
        device_version: str,
        submission_type: str,
        dmr_data: Dict[str, Any],
        vv_report: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        output_path: str,
    ) -> str:
        """
        Generate complete regulatory submission package

        Args:
            device_name: Name of the device
            device_version: Version of the device
            submission_type: Type of submission (510k, PMA, CE)
            dmr_data: Device Master Record data
            vv_report: V&V report data
            risk_analysis: Risk analysis data
            output_path: Path to export package

        Returns:
            Path to generated package
        """
        export_path = Path(output_path)
        export_path.mkdir(exist_ok=True)

        # Create submission directory
        submission_path = export_path / "submission"
        submission_path.mkdir(exist_ok=True)

        # Export DMR
        dmr_path = submission_path / "dmr.json"
        with open(dmr_path, "w") as f:
            json.dump(dmr_data, f, indent=2)

        # Export V&V report
        vv_path = submission_path / "vv_report.json"
        with open(vv_path, "w") as f:
            json.dump(vv_report, f, indent=2)

        # Export risk analysis
        risk_path = submission_path / "risk_analysis.json"
        with open(risk_path, "w") as f:
            json.dump(risk_analysis, f, indent=2)

        # Load and export cybersecurity plan if exists
        cybersecurity_file = (
            self.cybersecurity_path / f"{device_name}_{device_version}_cybersecurity_plan.json"
        )
        if cybersecurity_file.exists():
            with open(cybersecurity_file, "r") as src:
                cybersecurity_data = json.load(src)
            cybersecurity_path = submission_path / "cybersecurity_plan.json"
            with open(cybersecurity_path, "w") as f:
                json.dump(cybersecurity_data, f, indent=2)

        # Generate submission summary
        submission_summary = {
            "submission_type": submission_type,
            "device_name": device_name,
            "device_version": device_version,
            "submission_date": datetime.datetime.now().isoformat(),
            "vv_summary": vv_report,
            "regulatory_status": (
                "ready_for_submission"
                if vv_report.get("overall_status") == "pass"
                else "needs_remediation"
            ),
            "included_documents": [
                "dmr.json",
                "vv_report.json",
                "risk_analysis.json",
            ],
        }

        if cybersecurity_file.exists():
            submission_summary["included_documents"].append("cybersecurity_plan.json")

        # Save submission summary
        summary_path = submission_path / "submission_summary.json"
        with open(summary_path, "w") as f:
            json.dump(submission_summary, f, indent=2)

        logger.info(f"Generated {submission_type} submission package at {export_path}")
        return str(export_path)

    def load_cybersecurity_plan(self, device_name: str, device_version: str) -> Dict[str, Any]:
        """
        Load cybersecurity plan from file

        Args:
            device_name: Name of the device
            device_version: Version of the device

        Returns:
            Cybersecurity plan or None if not found
        """
        filename = f"{device_name}_{device_version}_cybersecurity_plan.json"
        filepath = self.cybersecurity_path / filename

        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            return json.load(f)

    def get_security_events(self, device_name: str = None) -> List[Dict[str, Any]]:
        """
        Get all security events, optionally filtered by device

        Args:
            device_name: Optional device name to filter by

        Returns:
            List of security event records
        """
        events = []

        for filepath in self.cybersecurity_path.glob("security_event_*.json"):
            with open(filepath, "r") as f:
                event = json.load(f)
                if device_name is None or event.get("device_name") == device_name:
                    events.append(event)

        return sorted(events, key=lambda x: x["timestamp"], reverse=True)
