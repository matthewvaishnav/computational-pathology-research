"""
Unit tests for regulatory compliance infrastructure

Tests the regulatory documentation system, risk management, V&V support,
and cybersecurity controls.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from regulatory import (
    CybersecurityControlSystem,
    DeviceMasterRecord,
    ModelDevelopmentRecord,
    RegulatoryComplianceManager,
    RegulatoryDocumentationSystem,
    RegulatoryStandard,
    RiskManagementSystem,
    SoftwareComponent,
    ValidationStatus,
    VerificationValidationSystem,
)


class TestRegulatoryDocumentationSystem(unittest.TestCase):
    """Test regulatory documentation system"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.doc_system = RegulatoryDocumentationSystem(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_create_dmr(self):
        """Test DMR creation"""
        dmr = self.doc_system.create_dmr(
            device_name="PathologyAI",
            device_version="1.0.0",
            manufacturer="TestCorp",
            intended_use="Diagnostic assistance for pathology",
            indications_for_use="Cancer detection in tissue samples",
            regulatory_standards=[RegulatoryStandard.FDA_510K, RegulatoryStandard.CE_MARKING],
        )

        self.assertEqual(dmr.device_name, "PathologyAI")
        self.assertEqual(dmr.device_version, "1.0.0")
        self.assertEqual(dmr.manufacturer, "TestCorp")
        self.assertIn(RegulatoryStandard.FDA_510K, dmr.regulatory_standards)
        self.assertIn(RegulatoryStandard.CE_MARKING, dmr.regulatory_standards)

    def test_load_dmr(self):
        """Test DMR loading"""
        # Create DMR
        original_dmr = self.doc_system.create_dmr(
            device_name="PathologyAI",
            device_version="1.0.0",
            manufacturer="TestCorp",
            intended_use="Diagnostic assistance",
            indications_for_use="Cancer detection",
            regulatory_standards=[RegulatoryStandard.FDA_510K],
        )

        # Load DMR
        loaded_dmr = self.doc_system.load_dmr("PathologyAI", "1.0.0")

        self.assertIsNotNone(loaded_dmr)
        self.assertEqual(loaded_dmr.device_name, original_dmr.device_name)
        self.assertEqual(loaded_dmr.device_version, original_dmr.device_version)
        self.assertEqual(loaded_dmr.manufacturer, original_dmr.manufacturer)

    def test_document_model_development(self):
        """Test model development documentation"""
        record = self.doc_system.document_model_development(
            model_name="AttentionMIL",
            model_version="2.1.0",
            training_data_provenance={"dataset": "TCGA", "version": "2023.1"},
            validation_protocols=["k-fold cross-validation", "holdout validation"],
            performance_metrics={"accuracy": 0.92, "auc": 0.96},
            dataset_versions={"training": "v1.0", "validation": "v1.0"},
            hyperparameters={"learning_rate": 0.001, "batch_size": 32},
            architecture_description="Attention-based MIL with transformer encoder",
            intended_use="Cancer classification in WSI",
            contraindications=["Poor image quality", "Artifacts"],
            limitations=["Limited to H&E stained slides"],
        )

        self.assertEqual(record.model_name, "AttentionMIL")
        self.assertEqual(record.model_version, "2.1.0")
        self.assertEqual(record.performance_metrics["accuracy"], 0.92)
        self.assertIn("k-fold cross-validation", record.validation_protocols)

    def test_add_software_component(self):
        """Test adding software component to DMR"""
        dmr = self.doc_system.create_dmr(
            device_name="PathologyAI",
            device_version="1.0.0",
            manufacturer="TestCorp",
            intended_use="Diagnostic assistance",
            indications_for_use="Cancer detection",
            regulatory_standards=[RegulatoryStandard.FDA_510K],
        )

        component = self.doc_system.add_software_component(
            dmr=dmr,
            component_name="AttentionMIL",
            version="2.1.0",
            description="Attention-based MIL classifier",
            safety_classification="B",
            dependencies=["PyTorch", "NumPy"],
        )

        self.assertEqual(component.component_name, "AttentionMIL")
        self.assertEqual(component.safety_classification, "B")
        self.assertEqual(component.validation_status, ValidationStatus.NOT_VALIDATED)
        self.assertIn("PyTorch", component.dependencies)

    def test_update_component_validation(self):
        """Test updating component validation status"""
        dmr = self.doc_system.create_dmr(
            device_name="PathologyAI",
            device_version="1.0.0",
            manufacturer="TestCorp",
            intended_use="Diagnostic assistance",
            indications_for_use="Cancer detection",
            regulatory_standards=[RegulatoryStandard.FDA_510K],
        )

        self.doc_system.add_software_component(
            dmr=dmr,
            component_name="AttentionMIL",
            version="2.1.0",
            description="Attention-based MIL classifier",
            safety_classification="B",
        )

        validation_results = {"test_coverage": 95, "defects": 0}
        self.doc_system.update_component_validation(
            dmr=dmr,
            component_name="AttentionMIL",
            validation_status=ValidationStatus.VALIDATED,
            validation_results=validation_results,
        )

        # Find the component
        component = next(c for c in dmr.software_components if c.component_name == "AttentionMIL")
        self.assertEqual(component.validation_status, ValidationStatus.VALIDATED)
        self.assertEqual(component.verification_results["test_coverage"], 95)

    def test_generate_version_control_record(self):
        """Test version control record generation"""
        record = self.doc_system.generate_version_control_record(
            component_name="AttentionMIL",
            version="2.1.0",
            changes=["Fixed attention weight normalization", "Added uncertainty quantification"],
            validation_status=ValidationStatus.VALIDATED,
            release_notes="Bug fixes and new features for clinical deployment",
        )

        self.assertEqual(record["component_name"], "AttentionMIL")
        self.assertEqual(record["version"], "2.1.0")
        self.assertIn("Fixed attention weight normalization", record["changes"])
        self.assertEqual(record["validation_status"], "validated")
        self.assertIsNotNone(record["checksum"])

    def test_export_regulatory_package(self):
        """Test regulatory package export"""
        # Create DMR with components
        dmr = self.doc_system.create_dmr(
            device_name="PathologyAI",
            device_version="1.0.0",
            manufacturer="TestCorp",
            intended_use="Diagnostic assistance",
            indications_for_use="Cancer detection",
            regulatory_standards=[RegulatoryStandard.FDA_510K],
        )

        # Add model record
        model_record = self.doc_system.document_model_development(
            model_name="AttentionMIL",
            model_version="2.1.0",
            training_data_provenance={"dataset": "TCGA"},
            validation_protocols=["cross-validation"],
            performance_metrics={"accuracy": 0.92},
            dataset_versions={"training": "v1.0"},
            hyperparameters={"lr": 0.001},
            architecture_description="MIL model",
            intended_use="Cancer classification",
            contraindications=[],
            limitations=[],
        )
        dmr.model_records.append(model_record)
        self.doc_system.update_dmr(dmr)

        # Export package
        export_path = Path(self.temp_dir) / "export"
        package_path = self.doc_system.export_regulatory_package(
            device_name="PathologyAI", device_version="1.0.0", output_path=str(export_path)
        )

        # Verify export
        self.assertTrue(Path(package_path).exists())
        self.assertTrue((Path(package_path) / "dmr.json").exists())
        self.assertTrue((Path(package_path) / "model_development").exists())


class TestRiskManagementSystem(unittest.TestCase):
    """Test risk management system"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.risk_system = RiskManagementSystem(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_create_risk_analysis(self):
        """Test risk analysis creation"""
        hazards = [
            {
                "hazard_id": "H001",
                "description": "False positive diagnosis",
                "severity": 4,
                "probability": 3,
            },
            {
                "hazard_id": "H002",
                "description": "False negative diagnosis",
                "severity": 5,
                "probability": 2,
            },
        ]

        risk_controls = [
            {
                "control_id": "C001",
                "description": "Uncertainty quantification",
                "applicable_hazards": ["H001", "H002"],
                "effectiveness": 0.7,
            }
        ]

        risk_analysis = self.risk_system.create_risk_analysis(
            device_name="PathologyAI",
            device_version="1.0.0",
            hazards=hazards,
            risk_controls=risk_controls,
        )

        self.assertEqual(risk_analysis["device_name"], "PathologyAI")
        self.assertEqual(len(risk_analysis["hazards"]), 2)
        self.assertEqual(len(risk_analysis["risk_controls"]), 1)
        self.assertEqual(len(risk_analysis["residual_risks"]), 2)

    def test_calculate_residual_risk(self):
        """Test residual risk calculation"""
        hazard = {
            "hazard_id": "H001",
            "description": "False positive",
            "severity": 4,
            "probability": 3,
        }

        risk_controls = [
            {
                "control_id": "C001",
                "description": "Uncertainty quantification",
                "applicable_hazards": ["H001"],
                "effectiveness": 0.5,
            }
        ]

        residual_risk = self.risk_system._calculate_residual_risk(hazard, risk_controls)

        self.assertEqual(residual_risk["hazard_id"], "H001")
        self.assertEqual(residual_risk["initial_risk_score"], 12)  # 4 * 3
        self.assertEqual(residual_risk["residual_risk_score"], 6.0)  # 12 * 0.5
        self.assertEqual(residual_risk["acceptability"], "acceptable")  # < 10

    def test_post_market_surveillance(self):
        """Test post-market surveillance update"""
        adverse_events = [
            {
                "event_id": "AE001",
                "description": "Missed cancer diagnosis",
                "severity": "serious",
                "date": "2024-01-15",
            }
        ]

        performance_data = {
            "accuracy": 0.88,  # Below 90% threshold
            "sensitivity": 0.85,
            "specificity": 0.91,
        }

        self.risk_system.update_post_market_surveillance(
            device_name="PathologyAI",
            device_version="1.0.0",
            adverse_events=adverse_events,
            performance_data=performance_data,
        )

        # Check that surveillance file was created
        surveillance_files = list(self.risk_system.risk_management_path.glob("*surveillance*.json"))
        self.assertTrue(len(surveillance_files) > 0)

        # Load and verify content
        with open(surveillance_files[0], "r") as f:
            surveillance_data = json.load(f)

        self.assertEqual(surveillance_data["device_name"], "PathologyAI")
        self.assertTrue(
            surveillance_data["risk_reassessment_required"]
        )  # Due to serious event and low accuracy


class TestVerificationValidationSystem(unittest.TestCase):
    """Test V&V system"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.vv_system = VerificationValidationSystem(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_create_vv_plan(self):
        """Test V&V plan creation"""
        components = [
            SoftwareComponent(
                component_name="AttentionMIL",
                version="2.1.0",
                description="MIL classifier",
                safety_classification="B",
                validation_status=ValidationStatus.NOT_VALIDATED,
            )
        ]

        verification_activities = [
            {
                "activity_id": "V001",
                "description": "Unit testing",
                "applicable_components": ["AttentionMIL"],
            }
        ]

        validation_activities = [
            {
                "activity_id": "VAL001",
                "description": "Clinical validation",
                "applicable_components": ["AttentionMIL"],
            }
        ]

        vv_plan = self.vv_system.create_vv_plan(
            device_name="PathologyAI",
            device_version="1.0.0",
            software_components=components,
            verification_activities=verification_activities,
            validation_activities=validation_activities,
        )

        self.assertEqual(vv_plan["device_name"], "PathologyAI")
        self.assertEqual(len(vv_plan["verification_activities"]), 1)
        self.assertEqual(len(vv_plan["validation_activities"]), 1)
        self.assertIn("traceability_matrix", vv_plan)

    def test_execute_verification_test(self):
        """Test verification test execution"""
        test_results = {"status": "pass", "coverage": 95, "defects": []}

        test_record = self.vv_system.execute_verification_test(
            device_name="PathologyAI",
            device_version="1.0.0",
            activity_id="V001",
            test_results=test_results,
        )

        self.assertEqual(test_record["activity_id"], "V001")
        self.assertEqual(test_record["pass_fail_status"], "pass")
        self.assertEqual(test_record["coverage_achieved"], 95)

    def test_execute_validation_test(self):
        """Test validation test execution"""
        test_results = {
            "status": "pass",
            "clinical_relevance": "High accuracy on clinical dataset",
            "user_acceptance": True,
        }

        test_record = self.vv_system.execute_validation_test(
            device_name="PathologyAI",
            device_version="1.0.0",
            activity_id="VAL001",
            test_results=test_results,
        )

        self.assertEqual(test_record["activity_id"], "VAL001")
        self.assertEqual(test_record["pass_fail_status"], "pass")
        self.assertTrue(test_record["user_acceptance"])

    def test_generate_vv_report(self):
        """Test V&V report generation"""
        # Execute some tests first
        self.vv_system.execute_verification_test(
            device_name="PathologyAI",
            device_version="1.0.0",
            activity_id="V001",
            test_results={"status": "pass", "coverage": 95},
        )

        self.vv_system.execute_validation_test(
            device_name="PathologyAI",
            device_version="1.0.0",
            activity_id="VAL001",
            test_results={"status": "pass", "user_acceptance": True},
        )

        # Generate report
        report = self.vv_system.generate_vv_report("PathologyAI", "1.0.0")

        self.assertEqual(report["device_name"], "PathologyAI")
        self.assertEqual(report["verification_summary"]["total_tests"], 1)
        self.assertEqual(report["verification_summary"]["pass_rate"], 100.0)
        self.assertEqual(report["validation_summary"]["total_tests"], 1)
        self.assertEqual(report["validation_summary"]["pass_rate"], 100.0)
        self.assertEqual(report["overall_status"], "pass")


class TestCybersecurityControlSystem(unittest.TestCase):
    """Test cybersecurity control system"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cyber_system = CybersecurityControlSystem(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_create_cybersecurity_plan(self):
        """Test cybersecurity plan creation"""
        threat_model = {
            "threats": ["Data breach", "Unauthorized access", "Malware"],
            "attack_vectors": ["Network", "Physical", "Social engineering"],
        }

        security_controls = [
            {
                "control_id": "SC001",
                "description": "Encryption at rest",
                "implementation": "AES-256",
            },
            {"control_id": "SC002", "description": "Access control", "implementation": "RBAC"},
        ]

        plan = self.cyber_system.create_cybersecurity_plan(
            device_name="PathologyAI",
            device_version="1.0.0",
            threat_model=threat_model,
            security_controls=security_controls,
        )

        self.assertEqual(plan["device_name"], "PathologyAI")
        self.assertEqual(len(plan["security_controls"]), 2)
        self.assertIn("Data breach", plan["threat_model"]["threats"])
        self.assertTrue(plan["security_monitoring"]["logging_enabled"])

    def test_log_security_event(self):
        """Test security event logging"""
        event = self.cyber_system.log_security_event(
            device_name="PathologyAI",
            device_version="1.0.0",
            event_type="unauthorized_access_attempt",
            severity="medium",
            description="Failed login attempts from unknown IP",
            mitigation_actions=["IP blocked", "User notified"],
        )

        self.assertEqual(event["event_type"], "unauthorized_access_attempt")
        self.assertEqual(event["severity"], "medium")
        self.assertEqual(event["status"], "open")
        self.assertIn("IP blocked", event["mitigation_actions"])
        self.assertIsNotNone(event["event_id"])

    def test_update_security_event(self):
        """Test security event update"""
        # Log event first
        event = self.cyber_system.log_security_event(
            device_name="PathologyAI",
            device_version="1.0.0",
            event_type="vulnerability_detected",
            severity="high",
            description="SQL injection vulnerability found",
            mitigation_actions=["Patch applied"],
        )

        # Update event
        self.cyber_system.update_security_event(
            event_id=event["event_id"], status="resolved", resolution_date="2024-01-20T10:00:00"
        )

        # Verify update
        filename = f"security_event_{event['event_id']}.json"
        filepath = self.cyber_system.cybersecurity_path / filename

        with open(filepath, "r") as f:
            updated_event = json.load(f)

        self.assertEqual(updated_event["status"], "resolved")
        self.assertEqual(updated_event["resolution_date"], "2024-01-20T10:00:00")


class TestRegulatoryComplianceManager(unittest.TestCase):
    """Test regulatory compliance manager"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.compliance_manager = RegulatoryComplianceManager(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_initialize_device_compliance(self):
        """Test device compliance initialization"""
        dmr = self.compliance_manager.initialize_device_compliance(
            device_name="PathologyAI",
            device_version="1.0.0",
            manufacturer="TestCorp",
            intended_use="Diagnostic assistance for pathology",
            indications_for_use="Cancer detection in tissue samples",
            regulatory_standards=[RegulatoryStandard.FDA_510K, RegulatoryStandard.CE_MARKING],
        )

        self.assertEqual(dmr.device_name, "PathologyAI")
        self.assertEqual(dmr.manufacturer, "TestCorp")
        self.assertIn(RegulatoryStandard.FDA_510K, dmr.regulatory_standards)

    def test_generate_regulatory_submission_package(self):
        """Test regulatory submission package generation"""
        # Initialize device compliance
        dmr = self.compliance_manager.initialize_device_compliance(
            device_name="PathologyAI",
            device_version="1.0.0",
            manufacturer="TestCorp",
            intended_use="Diagnostic assistance",
            indications_for_use="Cancer detection",
            regulatory_standards=[RegulatoryStandard.FDA_510K],
        )

        # Execute some V&V tests
        self.compliance_manager.vv_system.execute_verification_test(
            device_name="PathologyAI",
            device_version="1.0.0",
            activity_id="V001",
            test_results={"status": "pass", "coverage": 95},
        )

        # Generate submission package
        export_path = Path(self.temp_dir) / "submission_export"
        package_path = self.compliance_manager.generate_regulatory_submission_package(
            device_name="PathologyAI",
            device_version="1.0.0",
            submission_type="510k",
            output_path=str(export_path),
        )

        # Verify package structure
        self.assertTrue(Path(package_path).exists())
        self.assertTrue((Path(package_path) / "dmr.json").exists())
        self.assertTrue((Path(package_path) / "submission").exists())
        self.assertTrue((Path(package_path) / "submission" / "submission_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
