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
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .dmr_manager import (
    DMRManager,
    DeviceMasterRecord,
    ModelDevelopmentRecord,
    RegulatoryStandard,
    SoftwareComponent,
    ValidationStatus,
)
from .risk_manager import RiskManager
from .submission_generator import SubmissionGenerator
from .vv_manager import VVManager

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "RegulatoryStandard",
    "ValidationStatus",
    "ModelDevelopmentRecord",
    "SoftwareComponent",
    "DeviceMasterRecord",
    "RegulatoryDocumentationSystem",
    "RiskManagementSystem",
    "VerificationValidationSystem",
    "CybersecurityControlSystem",
    "RegulatoryComplianceManager",
]


# Backward-compatible aliases
class RegulatoryDocumentationSystem(DMRManager):
    """Backward-compatible alias for DMRManager"""
    
    def export_regulatory_package(self, device_name: str, device_version: str, output_path: str) -> str:
        """Export regulatory package for device"""
        import os
        import shutil
        from pathlib import Path
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load DMR
        dmr = self.load_dmr(device_name, device_version)
        if not dmr:
            raise ValueError(f"DMR not found for {device_name} v{device_version}")
        
        # Export DMR as JSON
        dmr_path = output_dir / "dmr.json"
        with open(dmr_path, 'w') as f:
            json.dump(dmr.to_dict(), f, indent=2, default=str)
        
        # Create package structure
        package_path = output_dir / f"{device_name}_v{device_version}_regulatory_package"
        package_path.mkdir(exist_ok=True)
        
        # Copy DMR to package
        shutil.copy2(dmr_path, package_path / "dmr.json")
        
        return str(package_path)


class RiskManagementSystem(RiskManager):
    """Backward-compatible alias for RiskManager"""
    pass


class VerificationValidationSystem(VVManager):
    """Backward-compatible alias for VVManager"""
    pass


class CybersecurityControlSystem(SubmissionGenerator):
    """Backward-compatible alias for SubmissionGenerator (cybersecurity methods)"""
    pass


class RegulatoryComplianceManager:
    """
    Main regulatory compliance manager integrating all regulatory systems

    Provides unified interface for regulatory compliance activities.
    """

    def __init__(self, documentation_path: str = "regulatory_docs"):
        self.documentation_system = DMRManager(documentation_path)
        self.risk_management = RiskManager(documentation_path)
        self.vv_system = VVManager(documentation_path)
        self.cybersecurity = SubmissionGenerator(documentation_path)

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
        # Load DMR
        dmr = self.documentation_system.load_dmr(device_name, device_version)
        if not dmr:
            raise ValueError(f"DMR not found for {device_name} v{device_version}")

        # Generate V&V report
        vv_report = self.vv_system.generate_vv_report(device_name, device_version)

        # Load risk analysis
        risk_analysis = self.risk_management.load_risk_analysis(device_name, device_version)
        if not risk_analysis:
            risk_analysis = {}

        # Generate submission package
        package_path = self.cybersecurity.generate_submission_package(
            device_name=device_name,
            device_version=device_version,
            submission_type=submission_type,
            dmr_data=dmr.to_dict(),
            vv_report=vv_report,
            risk_analysis=risk_analysis,
            output_path=output_path,
        )

        logger.info(f"Generated {submission_type} submission package at {package_path}")
        return package_path
