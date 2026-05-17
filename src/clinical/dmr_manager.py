"""
Device Master Record (DMR) Management

Handles DMR creation, updates, software component tracking, and version control
for regulatory compliance.

Requirements: 20.1-20.3 (DMR management, documentation tracking, version control)
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


class DMRManager:
    """
    Device Master Record (DMR) Manager

    Maintains device master record, model development documentation,
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

        logger.info(f"Initialized DMR manager at {self.documentation_path}")

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

    def export_dmr_package(self, device_name: str, device_version: str, output_path: str) -> str:
        """
        Export DMR package with all related documentation

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

        logger.info(f"Exported DMR package to {export_path}")
        return str(export_path)
