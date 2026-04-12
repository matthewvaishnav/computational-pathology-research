"""
Regulatory Compliance Infrastructure

This module provides regulatory compliance support for clinical deployment,
including documentation tracking, device master record (DMR) management,
model development documentation, version control, and regulatory submission support.

Requirements: 20.1-20.8 (regulatory compliance, documentation tracking,
risk management, V&V support, traceability matrices, post-market surveillance,
cybersecurity controls)
"""

import datetime
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RegulatoryStandard(Enum):
    """Supported regulatory standards"""

    FDA_510K = "FDA_510K"
    FDA_PMA = "FDA_PMA"
    CE_MARKING = "CE_MARKING"
    ISO_14971 = "ISO_14971"
    ISO_13485 = "ISO_13485"
    IEC_62304 = "IEC_62304"


class ValidationStatus(Enum):
    """Validation status for components"""

    NOT_VALIDATED = "not_validated"
    IN_PROGRESS = "in_progress"
    VALIDATED = "validated"
    EXPIRED = "expired"


@dataclass
class ModelDevelopmentRecord:
    """Documentation of model development for regulatory submissions"""

    model_name: str
    model_version: str
    training_data_provenance: Dict[str, Any]
    validation_protocols: List[str]
    performance_metrics: Dict[str, float]
    training_date: str
    validation_date: str
    dataset_versions: Dict[str, str]
    hyperparameters: Dict[str, Any]
    architecture_description: str
    intended_use: str
    contraindications: List[str]
    limitations: List[str]
    clinical_validation_results: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class SoftwareComponent:
    """Software component documentation for DMR"""

    component_name: str
    version: str
    description: str
    safety_classification: str  # Class A, B, or C per IEC 62304
    validation_status: ValidationStatus
    validation_date: Optional[str] = None
    dependencies: List[str] = None
    risk_analysis: Optional[str] = None
    verification_results: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class DeviceMasterRecord:
    """Device Master Record (DMR) for regulatory compliance"""

    device_name: str
    device_version: str
    manufacturer: str
    intended_use: str
    indications_for_use: str
    contraindications: List[str]
    warnings_precautions: List[str]
    system_design: Dict[str, Any]
    specifications: Dict[str, Any]
    software_components: List[SoftwareComponent]
    model_records: List[ModelDevelopmentRecord]
    validation_summary: Dict[str, Any]
    risk_management_file: str
    creation_date: str
    last_updated: str
    regulatory_standards: List[RegulatoryStandard]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert enums to strings
        data["regulatory_standards"] = [std.value for std in self.regulatory_standards]
        for component in data["software_components"]:
            component["validation_status"] = component["validation_status"].value
        return data


class RegulatoryDocumentationSystem:
    """
    System for tracking and managing regulatory documentation

    Maintains device master record (DMR), model development documentation,
    and version control for regulatory compliance.
    """

    def __init__(self, documentation_path: str = "regulatory_docs"):
        self.documentation_path = Path(documentation_path)
        self.documentation_path.mkdir(exist_ok=True)

        # Initialize DMR storage
        self.dmr_path = self.documentation_path / "dmr"
        self.dmr_path.mkdir(exist_ok=True)

        # Initialize model documentation storage
        self.model_docs_path = self.documentation_path / "model_development"
        self.model_docs_path.mkdir(exist_ok=True)

        # Initialize version control storage
        self.version_control_path = self.documentation_path / "version_control"
        self.version_control_path.mkdir(exist_ok=True)

        logger.info(f"Initialized regulatory documentation system at {self.documentation_path}")

    def create_dmr(
        self,
        device_name: str,
        device_version: str,
        manufacturer: str,
        intended_use: str,
        indications_for_use: str,
        regulatory_standards: List[RegulatoryStandard],
    ) -> DeviceMasterRecord:
        """
        Create a new Device Master Record (DMR)

        Args:
            device_name: Name of the medical device
            device_version: Version of the device
            manufacturer: Device manufacturer
            intended_use: Intended use statement
            indications_for_use: Clinical indications
            regulatory_standards: Applicable regulatory standards

        Returns:
            DeviceMasterRecord: Created DMR
        """
        current_time = datetime.datetime.now().isoformat()

        dmr = DeviceMasterRecord(
            device_name=device_name,
            device_version=device_version,
            manufacturer=manufacturer,
            intended_use=intended_use,
            indications_for_use=indications_for_use,
            contraindications=[],
            warnings_precautions=[],
            system_design={},
            specifications={},
            software_components=[],
            model_records=[],
            validation_summary={},
            risk_management_file="",
            creation_date=current_time,
            last_updated=current_time,
            regulatory_standards=regulatory_standards,
        )

        self._save_dmr(dmr)
        logger.info(f"Created DMR for {device_name} v{device_version}")
        return dmr

    def update_dmr(self, dmr: DeviceMasterRecord) -> None:
        """
        Update an existing DMR

        Args:
            dmr: Updated DMR to save
        """
        dmr.last_updated = datetime.datetime.now().isoformat()
        self._save_dmr(dmr)
        logger.info(f"Updated DMR for {dmr.device_name} v{dmr.device_version}")

    def _save_dmr(self, dmr: DeviceMasterRecord) -> None:
        """Save DMR to file"""
        filename = f"{dmr.device_name}_{dmr.device_version}_dmr.json"
        filepath = self.dmr_path / filename

        with open(filepath, "w") as f:
            json.dump(dmr.to_dict(), f, indent=2)

    def load_dmr(self, device_name: str, device_version: str) -> Optional[DeviceMasterRecord]:
        """
        Load DMR from file

        Args:
            device_name: Name of the device
            device_version: Version of the device

        Returns:
            DeviceMasterRecord or None if not found
        """
        filename = f"{device_name}_{device_version}_dmr.json"
        filepath = self.dmr_path / filename

        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            data = json.load(f)

        # Convert back from dict to dataclass
        # Handle enum conversions
        data["regulatory_standards"] = [
            RegulatoryStandard(std) for std in data["regulatory_standards"]
        ]

        # Convert software components
        components = []
        for comp_data in data["software_components"]:
            comp_data["validation_status"] = ValidationStatus(comp_data["validation_status"])
            components.append(SoftwareComponent(**comp_data))
        data["software_components"] = components

        # Convert model records
        model_records = []
        for model_data in data["model_records"]:
            model_records.append(ModelDevelopmentRecord(**model_data))
        data["model_records"] = model_records

        return DeviceMasterRecord(**data)

    def document_model_development(
        self,
        model_name: str,
        model_version: str,
        training_data_provenance: Dict[str, Any],
        validation_protocols: List[str],
        performance_metrics: Dict[str, float],
        dataset_versions: Dict[str, str],
        hyperparameters: Dict[str, Any],
        architecture_description: str,
        intended_use: str,
        contraindications: List[str],
        limitations: List[str],
    ) -> ModelDevelopmentRecord:
        """
        Document model development for regulatory submission

        Args:
            model_name: Name of the model
            model_version: Version of the model
            training_data_provenance: Documentation of training data sources
            validation_protocols: List of validation protocols used
            performance_metrics: Performance metrics achieved
            dataset_versions: Versions of datasets used
            hyperparameters: Model hyperparameters
            architecture_description: Description of model architecture
            intended_use: Intended clinical use
            contraindications: Clinical contraindications
            limitations: Known limitations

        Returns:
            ModelDevelopmentRecord: Created model documentation
        """
        current_time = datetime.datetime.now().isoformat()

        record = ModelDevelopmentRecord(
            model_name=model_name,
            model_version=model_version,
            training_data_provenance=training_data_provenance,
            validation_protocols=validation_protocols,
            performance_metrics=performance_metrics,
            training_date=current_time,
            validation_date=current_time,
            dataset_versions=dataset_versions,
            hyperparameters=hyperparameters,
            architecture_description=architecture_description,
            intended_use=intended_use,
            contraindications=contraindications,
            limitations=limitations,
        )

        # Save model documentation
        filename = f"{model_name}_{model_version}_development.json"
        filepath = self.model_docs_path / filename

        with open(filepath, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

        logger.info(f"Documented model development for {model_name} v{model_version}")
        return record

    def add_software_component(
        self,
        dmr: DeviceMasterRecord,
        component_name: str,
        version: str,
        description: str,
        safety_classification: str,
        dependencies: List[str] = None,
    ) -> SoftwareComponent:
        """
        Add software component to DMR

        Args:
            dmr: DMR to update
            component_name: Name of the software component
            version: Version of the component
            description: Description of the component
            safety_classification: Safety class (A, B, or C per IEC 62304)
            dependencies: List of dependencies

        Returns:
            SoftwareComponent: Created component
        """
        component = SoftwareComponent(
            component_name=component_name,
            version=version,
            description=description,
            safety_classification=safety_classification,
            validation_status=ValidationStatus.NOT_VALIDATED,
            dependencies=dependencies or [],
        )

        dmr.software_components.append(component)
        self.update_dmr(dmr)

        logger.info(f"Added software component {component_name} v{version} to DMR")
        return component

    def update_component_validation(
        self,
        dmr: DeviceMasterRecord,
        component_name: str,
        validation_status: ValidationStatus,
        validation_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update validation status of a software component

        Args:
            dmr: DMR containing the component
            component_name: Name of the component to update
            validation_status: New validation status
            validation_results: Validation test results
        """
        for component in dmr.software_components:
            if component.component_name == component_name:
                component.validation_status = validation_status
                component.validation_date = datetime.datetime.now().isoformat()
                if validation_results:
                    component.verification_results = validation_results
                break
        else:
            raise ValueError(f"Component {component_name} not found in DMR")

        self.update_dmr(dmr)
        logger.info(f"Updated validation status for {component_name}: {validation_status.value}")

    def generate_version_control_record(
        self,
        component_name: str,
        version: str,
        changes: List[str],
        validation_status: ValidationStatus,
        release_notes: str,
    ) -> Dict[str, Any]:
        """
        Generate version control record for software component

        Args:
            component_name: Name of the component
            version: Version number
            changes: List of changes made
            validation_status: Validation status
            release_notes: Release notes

        Returns:
            Dict containing version control record
        """
        record = {
            "component_name": component_name,
            "version": version,
            "release_date": datetime.datetime.now().isoformat(),
            "changes": changes,
            "validation_status": validation_status.value,
            "release_notes": release_notes,
            "checksum": self._calculate_checksum(component_name, version),
        }

        # Save version control record
        filename = f"{component_name}_{version}_version.json"
        filepath = self.version_control_path / filename

        with open(filepath, "w") as f:
            json.dump(record, f, indent=2)

        logger.info(f"Generated version control record for {component_name} v{version}")
        return record

    def _calculate_checksum(self, component_name: str, version: str) -> str:
        """Calculate checksum for version integrity"""
        data = f"{component_name}_{version}_{datetime.datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()

    def get_all_dmrs(self) -> List[Tuple[str, str]]:
        """
        Get list of all DMRs

        Returns:
            List of (device_name, device_version) tuples
        """
        dmrs = []
        for filepath in self.dmr_path.glob("*_dmr.json"):
            parts = filepath.stem.split("_")
            if len(parts) >= 3:
                device_name = "_".join(parts[:-2])
                device_version = parts[-2]
                dmrs.append((device_name, device_version))
        return dmrs

    def get_model_development_records(self) -> List[Tuple[str, str]]:
        """
        Get list of all model development records

        Returns:
            List of (model_name, model_version) tuples
        """
        records = []
        for filepath in self.model_docs_path.glob("*_development.json"):
            parts = filepath.stem.split("_")
            if len(parts) >= 3:
                model_name = "_".join(parts[:-2])
                model_version = parts[-2]
                records.append((model_name, model_version))
        return records

    def export_regulatory_package(
        self, device_name: str, device_version: str, output_path: str
    ) -> str:
        """
        Export complete regulatory package for submission

        Args:
            device_name: Name of the device
            device_version: Version of the device
            output_path: Path to export package

        Returns:
            Path to exported package
        """
        export_path = Path(output_path)
        export_path.mkdir(exist_ok=True)

        # Load DMR
        dmr = self.load_dmr(device_name, device_version)
        if not dmr:
            raise ValueError(f"DMR not found for {device_name} v{device_version}")

        # Export DMR
        dmr_export_path = export_path / "dmr.json"
        with open(dmr_export_path, "w") as f:
            json.dump(dmr.to_dict(), f, indent=2)

        # Export model development records
        model_docs_path = export_path / "model_development"
        model_docs_path.mkdir(exist_ok=True)

        for model_record in dmr.model_records:
            filename = f"{model_record.model_name}_{model_record.model_version}_development.json"
            model_file_path = model_docs_path / filename
            with open(model_file_path, "w") as f:
                json.dump(model_record.to_dict(), f, indent=2)

        # Export version control records
        version_control_export_path = export_path / "version_control"
        version_control_export_path.mkdir(exist_ok=True)

        for component in dmr.software_components:
            version_files = list(
                self.version_control_path.glob(f"{component.component_name}_*_version.json")
            )
            for version_file in version_files:
                target_path = version_control_export_path / version_file.name
                with open(version_file, "r") as src, open(target_path, "w") as dst:
                    dst.write(src.read())

        logger.info(f"Exported regulatory package to {export_path}")
        return str(export_path)


class RiskManagementSystem:
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


class VerificationValidationSystem:
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
                "verification_pass_rate": 100,
                "validation_pass_rate": 100,
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


class CybersecurityControlSystem:
    """
    Cybersecurity control system following FDA guidance on medical device cybersecurity

    Implements cybersecurity controls and monitoring for clinical deployment.
    """

    def __init__(self, documentation_path: str = "regulatory_docs"):
        self.documentation_path = Path(documentation_path)
        self.cybersecurity_path = self.documentation_path / "cybersecurity"
        self.cybersecurity_path.mkdir(exist_ok=True)

        logger.info(f"Initialized cybersecurity control system at {self.cybersecurity_path}")

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
        self, event_id: str, status: str, resolution_date: Optional[str] = None
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


class RegulatoryComplianceManager:
    """
    Main regulatory compliance manager integrating all regulatory systems

    Provides unified interface for regulatory compliance activities.
    """

    def __init__(self, documentation_path: str = "regulatory_docs"):
        self.documentation_system = RegulatoryDocumentationSystem(documentation_path)
        self.risk_management = RiskManagementSystem(documentation_path)
        self.vv_system = VerificationValidationSystem(documentation_path)
        self.cybersecurity = CybersecurityControlSystem(documentation_path)

        logger.info("Initialized regulatory compliance manager")

    def initialize_device_compliance(
        self,
        device_name: str,
        device_version: str,
        manufacturer: str,
        intended_use: str,
        indications_for_use: str,
        regulatory_standards: List[RegulatoryStandard],
    ) -> DeviceMasterRecord:
        """
        Initialize complete regulatory compliance for a device

        Args:
            device_name: Name of the device
            device_version: Version of the device
            manufacturer: Device manufacturer
            intended_use: Intended use statement
            indications_for_use: Clinical indications
            regulatory_standards: Applicable regulatory standards

        Returns:
            DeviceMasterRecord: Initialized DMR
        """
        # Create DMR
        dmr = self.documentation_system.create_dmr(
            device_name=device_name,
            device_version=device_version,
            manufacturer=manufacturer,
            intended_use=intended_use,
            indications_for_use=indications_for_use,
            regulatory_standards=regulatory_standards,
        )

        logger.info(f"Initialized regulatory compliance for {device_name} v{device_version}")
        return dmr

    def generate_regulatory_submission_package(
        self, device_name: str, device_version: str, submission_type: str, output_path: str
    ) -> str:
        """
        Generate complete regulatory submission package

        Args:
            device_name: Name of the device
            device_version: Version of the device
            submission_type: Type of submission (510k, PMA, CE)
            output_path: Path to export package

        Returns:
            Path to generated package
        """
        # Export documentation package
        package_path = self.documentation_system.export_regulatory_package(
            device_name, device_version, output_path
        )

        # Generate V&V report
        vv_report = self.vv_system.generate_vv_report(device_name, device_version)

        # Add submission-specific documents
        submission_path = Path(package_path) / "submission"
        submission_path.mkdir(exist_ok=True)

        submission_summary = {
            "submission_type": submission_type,
            "device_name": device_name,
            "device_version": device_version,
            "submission_date": datetime.datetime.now().isoformat(),
            "vv_summary": vv_report,
            "regulatory_status": (
                "ready_for_submission"
                if vv_report["overall_status"] == "pass"
                else "needs_remediation"
            ),
        }

        with open(submission_path / "submission_summary.json", "w") as f:
            json.dump(submission_summary, f, indent=2)

        logger.info(f"Generated {submission_type} submission package at {package_path}")
        return package_path
