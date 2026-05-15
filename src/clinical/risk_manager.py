"""
Risk Management System

Implements ISO 14971 risk management including risk analysis, risk controls,
residual risk calculation, and post-market surveillance.

Requirements: 20.4 (ISO 14971 risk management, post-market surveillance)
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk management system following ISO 14971 standards

    Provides risk analysis, risk control measures, and post-market surveillance
    for medical device software.
    """

    def __init__(self, documentation_path: str = "regulatory_docs"):
        self.documentation_path = Path(documentation_path)
        self.risk_management_path = self.documentation_path / "risk_management"
        self.risk_management_path.mkdir(exist_ok=True)

        logger.info(f"Initialized risk management system at {self.risk_management_path}")

    def create_risk_analysis(
        self,
        device_name: str,
        device_version: str,
        hazards: List[Dict[str, Any]],
        risk_controls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Create risk analysis following ISO 14971

        Args:
            device_name: Name of the device
            device_version: Version of the device
            hazards: List of identified hazards
            risk_controls: List of risk control measures

        Returns:
            Risk analysis record
        """
        risk_analysis = {
            "device_name": device_name,
            "device_version": device_version,
            "analysis_date": datetime.datetime.now().isoformat(),
            "standard": "ISO 14971",
            "hazards": hazards,
            "risk_controls": risk_controls,
            "residual_risks": [],
            "risk_benefit_analysis": {},
            "post_market_surveillance_plan": {},
        }

        # Calculate residual risks after controls
        for hazard in hazards:
            residual_risk = self._calculate_residual_risk(hazard, risk_controls)
            risk_analysis["residual_risks"].append(residual_risk)

        # Save risk analysis
        filename = f"{device_name}_{device_version}_risk_analysis.json"
        filepath = self.risk_management_path / filename

        with open(filepath, "w") as f:
            json.dump(risk_analysis, f, indent=2)

        logger.info(f"Created risk analysis for {device_name} v{device_version}")
        return risk_analysis

    def _calculate_residual_risk(
        self, hazard: Dict[str, Any], risk_controls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate residual risk after applying controls"""
        # Find applicable controls for this hazard
        applicable_controls = [
            control
            for control in risk_controls
            if hazard["hazard_id"] in control.get("applicable_hazards", [])
        ]

        # Calculate risk reduction
        initial_severity = hazard.get("severity", 5)
        initial_probability = hazard.get("probability", 5)
        initial_risk = initial_severity * initial_probability

        # Apply control effectiveness
        risk_reduction_factor = 1.0
        for control in applicable_controls:
            effectiveness = control.get("effectiveness", 0.5)  # 50% reduction by default
            risk_reduction_factor *= 1.0 - effectiveness

        residual_risk_score = initial_risk * risk_reduction_factor

        return {
            "hazard_id": hazard["hazard_id"],
            "hazard_description": hazard["description"],
            "initial_risk_score": initial_risk,
            "applicable_controls": [c["control_id"] for c in applicable_controls],
            "residual_risk_score": residual_risk_score,
            "acceptability": "acceptable" if residual_risk_score < 10 else "needs_review",
        }

    def update_post_market_surveillance(
        self,
        device_name: str,
        device_version: str,
        adverse_events: List[Dict[str, Any]],
        performance_data: Dict[str, Any],
    ) -> None:
        """
        Update post-market surveillance data

        Args:
            device_name: Name of the device
            device_version: Version of the device
            adverse_events: List of reported adverse events
            performance_data: Performance monitoring data
        """
        surveillance_record = {
            "device_name": device_name,
            "device_version": device_version,
            "update_date": datetime.datetime.now().isoformat(),
            "adverse_events": adverse_events,
            "performance_data": performance_data,
            "risk_reassessment_required": self._assess_risk_reassessment_need(
                adverse_events, performance_data
            ),
        }

        # Save surveillance record
        filename = f"{device_name}_{device_version}_surveillance_{datetime.datetime.now().strftime('%Y%m%d')}.json"
        filepath = self.risk_management_path / filename

        with open(filepath, "w") as f:
            json.dump(surveillance_record, f, indent=2)

        logger.info(f"Updated post-market surveillance for {device_name} v{device_version}")

    def _assess_risk_reassessment_need(
        self, adverse_events: List[Dict[str, Any]], performance_data: Dict[str, Any]
    ) -> bool:
        """Assess if risk reassessment is needed based on surveillance data"""
        # Check for serious adverse events
        serious_events = [event for event in adverse_events if event.get("severity") == "serious"]
        if len(serious_events) > 0:
            return True

        # Check for performance degradation
        accuracy = performance_data.get("accuracy", 1.0)
        if accuracy < 0.9:  # Below 90% accuracy threshold
            return True

        return False

    def load_risk_analysis(self, device_name: str, device_version: str) -> Dict[str, Any]:
        """
        Load risk analysis from file

        Args:
            device_name: Name of the device
            device_version: Version of the device

        Returns:
            Risk analysis record or None if not found
        """
        filename = f"{device_name}_{device_version}_risk_analysis.json"
        filepath = self.risk_management_path / filename

        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            return json.load(f)

    def get_surveillance_records(
        self, device_name: str, device_version: str
    ) -> List[Dict[str, Any]]:
        """
        Get all surveillance records for a device

        Args:
            device_name: Name of the device
            device_version: Version of the device

        Returns:
            List of surveillance records
        """
        pattern = f"{device_name}_{device_version}_surveillance_*.json"
        records = []

        for filepath in self.risk_management_path.glob(pattern):
            with open(filepath, "r") as f:
                records.append(json.load(f))

        return sorted(records, key=lambda x: x["update_date"], reverse=True)
