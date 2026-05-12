"""
Verification and Validation (V&V) System

Implements software V&V testing required for FDA and regulatory submissions,
including V&V planning, test execution, traceability matrices, and reporting.

Requirements: 20.5 (V&V support, traceability matrices)
"""

import datetime
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .dmr_manager import SoftwareComponent

logger = logging.getLogger(__name__)


class VVManager:
    """
    Software Verification and Validation (V&V) system for regulatory compliance

    Supports V&V testing required for FDA and other regulatory submissions.
    """

    def __init__(self, documentation_path: str = "regulatory_docs"):
        self.documentation_path = Path(documentation_path)
        self.vv_path = self.documentation_path / "verification_validation"
        self.vv_path.mkdir(exist_ok=True)

        logger.info(f"Initialized V&V system at {self.vv_path}")

    def create_vv_plan(
        self,
        device_name: str,
        device_version: str,
        software_components: List[SoftwareComponent],
        verification_activities: List[Dict[str, Any]],
        validation_activities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Create V&V plan for software components

        Args:
            device_name: Name of the device
            device_version: Version of the device
            software_components: List of software components to verify/validate
            verification_activities: List of verification test activities
            validation_activities: List of validation test activities

        Returns:
            V&V plan record
        """
        # Convert software components to dict with enum handling
        components_dict = []
        for comp in software_components:
            comp_dict = asdict(comp)
            comp_dict["validation_status"] = comp.validation_status.value
            components_dict.append(comp_dict)

        vv_plan = {
            "device_name": device_name,
            "device_version": device_version,
            "plan_date": datetime.datetime.now().isoformat(),
            "software_components": components_dict,
            "verification_activities": verification_activities,
            "validation_activities": validation_activities,
            "traceability_matrix": self._generate_traceability_matrix(
                software_components, verification_activities, validation_activities
            ),
            "completion_criteria": {
                "verification_pass_rate": 100,  # nosec B105 - Numeric threshold, not password
                "validation_pass_rate": 100,  # nosec B105 - Numeric threshold, not password
                "coverage_threshold": 95,
            },
        }

        # Save V&V plan
        filename = f"{device_name}_{device_version}_vv_plan.json"
        filepath = self.vv_path / filename

        with open(filepath, "w") as f:
            json.dump(vv_plan, f, indent=2)

        logger.info(f"Created V&V plan for {device_name} v{device_version}")
        return vv_plan

    def _generate_traceability_matrix(
        self,
        software_components: List[SoftwareComponent],
        verification_activities: List[Dict[str, Any]],
        validation_activities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate traceability matrix linking requirements to implementation and validation"""
        matrix = {
            "requirements_to_components": {},
            "components_to_verification": {},
            "components_to_validation": {},
            "verification_to_validation": {},
        }

        # Link components to verification activities
        for component in software_components:
            component_name = component.component_name
            matrix["components_to_verification"][component_name] = []

            for activity in verification_activities:
                if component_name in activity.get("applicable_components", []):
                    matrix["components_to_verification"][component_name].append(
                        activity["activity_id"]
                    )

        # Link components to validation activities
        for component in software_components:
            component_name = component.component_name
            matrix["components_to_validation"][component_name] = []

            for activity in validation_activities:
                if component_name in activity.get("applicable_components", []):
                    matrix["components_to_validation"][component_name].append(
                        activity["activity_id"]
                    )

        return matrix

    def execute_verification_test(
        self, device_name: str, device_version: str, activity_id: str, test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Record verification test execution results

        Args:
            device_name: Name of the device
            device_version: Version of the device
            activity_id: ID of the verification activity
            test_results: Test execution results

        Returns:
            Verification test record
        """
        test_record = {
            "device_name": device_name,
            "device_version": device_version,
            "activity_id": activity_id,
            "test_type": "verification",
            "execution_date": datetime.datetime.now().isoformat(),
            "test_results": test_results,
            "pass_fail_status": test_results.get("status", "fail"),
            "defects_found": test_results.get("defects", []),
            "coverage_achieved": test_results.get("coverage", 0),
        }

        # Save test record
        filename = f"{device_name}_{device_version}_verification_{activity_id}.json"
        filepath = self.vv_path / filename

        with open(filepath, "w") as f:
            json.dump(test_record, f, indent=2)

        logger.info(f"Recorded verification test {activity_id} for {device_name} v{device_version}")
        return test_record

    def execute_validation_test(
        self, device_name: str, device_version: str, activity_id: str, test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Record validation test execution results

        Args:
            device_name: Name of the device
            device_version: Version of the device
            activity_id: ID of the validation activity
            test_results: Test execution results

        Returns:
            Validation test record
        """
        test_record = {
            "device_name": device_name,
            "device_version": device_version,
            "activity_id": activity_id,
            "test_type": "validation",
            "execution_date": datetime.datetime.now().isoformat(),
            "test_results": test_results,
            "pass_fail_status": test_results.get("status", "fail"),
            "clinical_relevance": test_results.get("clinical_relevance", ""),
            "user_acceptance": test_results.get("user_acceptance", False),
        }

        # Save test record
        filename = f"{device_name}_{device_version}_validation_{activity_id}.json"
        filepath = self.vv_path / filename

        with open(filepath, "w") as f:
            json.dump(test_record, f, indent=2)

        logger.info(f"Recorded validation test {activity_id} for {device_name} v{device_version}")
        return test_record

    def generate_vv_report(self, device_name: str, device_version: str) -> Dict[str, Any]:
        """
        Generate V&V summary report

        Args:
            device_name: Name of the device
            device_version: Version of the device

        Returns:
            V&V summary report
        """
        # Collect all verification test results
        verification_files = list(
            self.vv_path.glob(f"{device_name}_{device_version}_verification_*.json")
        )
        verification_results = []
        for file_path in verification_files:
            with open(file_path, "r") as f:
                verification_results.append(json.load(f))

        # Collect all validation test results
        validation_files = list(
            self.vv_path.glob(f"{device_name}_{device_version}_validation_*.json")
        )
        validation_results = []
        for file_path in validation_files:
            with open(file_path, "r") as f:
                validation_results.append(json.load(f))

        # Calculate summary statistics
        verification_pass_rate = (
            len([r for r in verification_results if r["pass_fail_status"] == "pass"])
            / max(len(verification_results), 1)
            * 100
        )
        validation_pass_rate = (
            len([r for r in validation_results if r["pass_fail_status"] == "pass"])
            / max(len(validation_results), 1)
            * 100
        )

        report = {
            "device_name": device_name,
            "device_version": device_version,
            "report_date": datetime.datetime.now().isoformat(),
            "verification_summary": {
                "total_tests": len(verification_results),
                "passed_tests": len(
                    [r for r in verification_results if r["pass_fail_status"] == "pass"]
                ),
                "pass_rate": verification_pass_rate,
                "defects_found": sum(len(r.get("defects_found", [])) for r in verification_results),
            },
            "validation_summary": {
                "total_tests": len(validation_results),
                "passed_tests": len(
                    [r for r in validation_results if r["pass_fail_status"] == "pass"]
                ),
                "pass_rate": validation_pass_rate,
                "user_acceptance_rate": len(
                    [r for r in validation_results if r.get("user_acceptance", False)]
                )
                / max(len(validation_results), 1)
                * 100,
            },
            "overall_status": (
                "pass" if verification_pass_rate == 100 and validation_pass_rate == 100 else "fail"
            ),
            "recommendations": [],
        }

        # Add recommendations based on results
        if verification_pass_rate < 100:
            report["recommendations"].append("Address verification test failures before release")
        if validation_pass_rate < 100:
            report["recommendations"].append(
                "Address validation test failures before clinical deployment"
            )

        # Save report
        filename = f"{device_name}_{device_version}_vv_report.json"
        filepath = self.vv_path / filename

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Generated V&V report for {device_name} v{device_version}")
        return report

    def load_vv_plan(self, device_name: str, device_version: str) -> Dict[str, Any]:
        """
        Load V&V plan from file

        Args:
            device_name: Name of the device
            device_version: Version of the device

        Returns:
            V&V plan or None if not found
        """
        filename = f"{device_name}_{device_version}_vv_plan.json"
        filepath = self.vv_path / filename

        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            return json.load(f)
