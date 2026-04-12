"""
Example usage of regulatory compliance infrastructure

This example demonstrates how to use the regulatory compliance system
for clinical deployment of computational pathology models.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import only the regulatory module directly to avoid dependency issues
from clinical.regulatory import (
    RegulatoryComplianceManager,
    RegulatoryStandard,
    ValidationStatus
)


def main():
    """Demonstrate regulatory compliance workflow"""
    
    # Initialize regulatory compliance manager
    compliance_manager = RegulatoryComplianceManager("regulatory_docs_example")
    
    print("=== Regulatory Compliance Example ===\n")
    
    # 1. Initialize device compliance
    print("1. Initializing device compliance...")
    dmr = compliance_manager.initialize_device_compliance(
        device_name="PathologyAI_Clinical",
        device_version="1.0.0",
        manufacturer="Computational Pathology Corp",
        intended_use="Computer-aided diagnostic assistance for pathology analysis of whole-slide images",
        indications_for_use="Detection and classification of cancer in H&E stained tissue samples from breast, lung, and colon biopsies",
        regulatory_standards=[
            RegulatoryStandard.FDA_510K,
            RegulatoryStandard.CE_MARKING,
            RegulatoryStandard.ISO_14971,
            RegulatoryStandard.IEC_62304
        ]
    )
    print(f"✓ Created DMR for {dmr.device_name} v{dmr.device_version}")
    
    # 2. Document model development
    print("\n2. Documenting model development...")
    model_record = compliance_manager.documentation_system.document_model_development(
        model_name="AttentionMIL_Clinical",
        model_version="2.1.0",
        training_data_provenance={
            "primary_dataset": "TCGA",
            "dataset_version": "2023.1",
            "training_samples": 15000,
            "validation_samples": 3000,
            "test_samples": 2000,
            "data_sources": ["TCGA-BRCA", "TCGA-LUAD", "TCGA-COAD"],
            "inclusion_criteria": "H&E stained slides with confirmed diagnosis",
            "exclusion_criteria": "Poor image quality, artifacts, incomplete staining"
        },
        validation_protocols=[
            "5-fold cross-validation",
            "Holdout validation on independent test set",
            "Clinical validation on prospective cohort",
            "Inter-observer agreement study"
        ],
        performance_metrics={
            "accuracy": 0.924,
            "sensitivity": 0.918,
            "specificity": 0.931,
            "auc_roc": 0.967,
            "auc_pr": 0.952,
            "f1_score": 0.921,
            "precision": 0.925,
            "recall": 0.918
        },
        dataset_versions={
            "training": "v2.1.0",
            "validation": "v2.1.0", 
            "test": "v2.1.0",
            "clinical_validation": "v1.0.0"
        },
        hyperparameters={
            "learning_rate": 0.0001,
            "batch_size": 32,
            "num_epochs": 100,
            "attention_dim": 256,
            "hidden_dim": 512,
            "dropout_rate": 0.2,
            "weight_decay": 1e-5,
            "optimizer": "AdamW"
        },
        architecture_description="Attention-based Multiple Instance Learning (MIL) model with transformer encoder for whole-slide image analysis. Uses patch-level feature extraction followed by attention-weighted aggregation for slide-level classification.",
        intended_use="Automated detection and classification of cancer in digitized H&E stained tissue slides",
        contraindications=[
            "Non-H&E stained slides",
            "Slides with significant artifacts or poor image quality",
            "Frozen section slides",
            "Slides from tissue types not included in training data"
        ],
        limitations=[
            "Limited to H&E stained slides only",
            "Performance may vary on slides from different scanners",
            "Not validated for pediatric cases",
            "Requires manual quality control for optimal performance"
        ]
    )
    print(f"✓ Documented model development for {model_record.model_name}")
    
    # 3. Add software components
    print("\n3. Adding software components...")
    
    # Core ML model component
    ml_component = compliance_manager.documentation_system.add_software_component(
        dmr=dmr,
        component_name="AttentionMIL_Core",
        version="2.1.0",
        description="Core attention-based MIL model for cancer classification",
        safety_classification="B",  # Non-life-threatening per IEC 62304
        dependencies=["PyTorch", "NumPy", "OpenCV", "Pillow"]
    )
    
    # Clinical workflow component
    workflow_component = compliance_manager.documentation_system.add_software_component(
        dmr=dmr,
        component_name="Clinical_Workflow",
        version="1.0.0",
        description="Clinical workflow integration and reporting system",
        safety_classification="B",
        dependencies=["AttentionMIL_Core", "DICOM_Adapter", "FHIR_Adapter"]
    )
    
    # Privacy and security component
    privacy_component = compliance_manager.documentation_system.add_software_component(
        dmr=dmr,
        component_name="Privacy_Security",
        version="1.0.0",
        description="HIPAA-compliant privacy and security controls",
        safety_classification="A",  # Critical for patient privacy
        dependencies=["cryptography", "audit_logger"]
    )
    
    print(f"✓ Added {len(dmr.software_components)} software components")
    
    # 4. Create risk analysis
    print("\n4. Creating risk analysis...")
    hazards = [
        {
            "hazard_id": "H001",
            "description": "False positive cancer diagnosis leading to unnecessary treatment",
            "severity": 4,  # Moderate harm
            "probability": 3,  # Occasional occurrence
            "clinical_impact": "Patient anxiety, unnecessary procedures, healthcare costs"
        },
        {
            "hazard_id": "H002", 
            "description": "False negative cancer diagnosis leading to delayed treatment",
            "severity": 5,  # Serious harm
            "probability": 2,  # Remote occurrence
            "clinical_impact": "Disease progression, reduced treatment efficacy, potential mortality"
        },
        {
            "hazard_id": "H003",
            "description": "System failure during critical diagnosis",
            "severity": 3,  # Minor harm
            "probability": 2,  # Remote occurrence
            "clinical_impact": "Diagnostic delay, workflow disruption"
        },
        {
            "hazard_id": "H004",
            "description": "Unauthorized access to patient data",
            "severity": 4,  # Moderate harm
            "probability": 2,  # Remote occurrence
            "clinical_impact": "Privacy breach, regulatory violations, patient trust loss"
        }
    ]
    
    risk_controls = [
        {
            "control_id": "C001",
            "description": "Uncertainty quantification and confidence thresholds",
            "applicable_hazards": ["H001", "H002"],
            "effectiveness": 0.7,
            "implementation": "Calibrated confidence scores with clinical decision thresholds"
        },
        {
            "control_id": "C002",
            "description": "Physician review requirement for uncertain cases",
            "applicable_hazards": ["H001", "H002"],
            "effectiveness": 0.8,
            "implementation": "Mandatory pathologist review for cases below confidence threshold"
        },
        {
            "control_id": "C003",
            "description": "System redundancy and failover mechanisms",
            "applicable_hazards": ["H003"],
            "effectiveness": 0.9,
            "implementation": "Redundant servers with automatic failover"
        },
        {
            "control_id": "C004",
            "description": "Multi-factor authentication and access controls",
            "applicable_hazards": ["H004"],
            "effectiveness": 0.85,
            "implementation": "RBAC with MFA and audit logging"
        }
    ]
    
    risk_analysis = compliance_manager.risk_management.create_risk_analysis(
        device_name="PathologyAI_Clinical",
        device_version="1.0.0",
        hazards=hazards,
        risk_controls=risk_controls
    )
    print(f"✓ Created risk analysis with {len(hazards)} hazards and {len(risk_controls)} controls")
    
    # 5. Create V&V plan
    print("\n5. Creating verification and validation plan...")
    verification_activities = [
        {
            "activity_id": "V001",
            "description": "Unit testing of core ML components",
            "applicable_components": ["AttentionMIL_Core"],
            "test_methods": ["Automated unit tests", "Code coverage analysis"],
            "acceptance_criteria": "100% pass rate, >95% code coverage"
        },
        {
            "activity_id": "V002",
            "description": "Integration testing of clinical workflow",
            "applicable_components": ["Clinical_Workflow", "AttentionMIL_Core"],
            "test_methods": ["End-to-end integration tests", "API testing"],
            "acceptance_criteria": "All integration tests pass"
        },
        {
            "activity_id": "V003",
            "description": "Security testing of privacy controls",
            "applicable_components": ["Privacy_Security"],
            "test_methods": ["Penetration testing", "Vulnerability scanning"],
            "acceptance_criteria": "No high-severity vulnerabilities"
        }
    ]
    
    validation_activities = [
        {
            "activity_id": "VAL001",
            "description": "Clinical validation on independent dataset",
            "applicable_components": ["AttentionMIL_Core", "Clinical_Workflow"],
            "test_methods": ["Retrospective clinical study", "Performance benchmarking"],
            "acceptance_criteria": "Accuracy >90%, Sensitivity >85%, Specificity >90%"
        },
        {
            "activity_id": "VAL002",
            "description": "Usability validation with pathologists",
            "applicable_components": ["Clinical_Workflow"],
            "test_methods": ["User acceptance testing", "Workflow analysis"],
            "acceptance_criteria": "User satisfaction >80%, Task completion rate >95%"
        },
        {
            "activity_id": "VAL003",
            "description": "HIPAA compliance validation",
            "applicable_components": ["Privacy_Security"],
            "test_methods": ["Privacy impact assessment", "Compliance audit"],
            "acceptance_criteria": "Full HIPAA compliance verified"
        }
    ]
    
    vv_plan = compliance_manager.vv_system.create_vv_plan(
        device_name="PathologyAI_Clinical",
        device_version="1.0.0",
        software_components=dmr.software_components,
        verification_activities=verification_activities,
        validation_activities=validation_activities
    )
    print(f"✓ Created V&V plan with {len(verification_activities)} verification and {len(validation_activities)} validation activities")
    
    # 6. Execute sample verification tests
    print("\n6. Executing sample verification tests...")
    
    # Unit testing results
    unit_test_results = {
        "status": "pass",
        "coverage": 97,
        "defects": [],
        "test_cases_executed": 245,
        "test_cases_passed": 245,
        "execution_time": "12.3 seconds"
    }
    
    compliance_manager.vv_system.execute_verification_test(
        device_name="PathologyAI_Clinical",
        device_version="1.0.0",
        activity_id="V001",
        test_results=unit_test_results
    )
    
    # Integration testing results
    integration_test_results = {
        "status": "pass",
        "coverage": 89,
        "defects": [],
        "api_tests_passed": 156,
        "api_tests_total": 156,
        "response_time_avg": "1.2 seconds"
    }
    
    compliance_manager.vv_system.execute_verification_test(
        device_name="PathologyAI_Clinical",
        device_version="1.0.0",
        activity_id="V002",
        test_results=integration_test_results
    )
    
    print("✓ Executed verification tests")
    
    # 7. Execute sample validation tests
    print("\n7. Executing sample validation tests...")
    
    # Clinical validation results
    clinical_validation_results = {
        "status": "pass",
        "clinical_relevance": "High clinical utility demonstrated",
        "user_acceptance": True,
        "performance_metrics": {
            "accuracy": 0.924,
            "sensitivity": 0.918,
            "specificity": 0.931,
            "auc": 0.967
        },
        "clinical_dataset_size": 500,
        "pathologist_agreement": 0.89
    }
    
    compliance_manager.vv_system.execute_validation_test(
        device_name="PathologyAI_Clinical",
        device_version="1.0.0",
        activity_id="VAL001",
        test_results=clinical_validation_results
    )
    
    # Usability validation results
    usability_validation_results = {
        "status": "pass",
        "clinical_relevance": "Workflow integration successful",
        "user_acceptance": True,
        "user_satisfaction_score": 4.2,  # out of 5
        "task_completion_rate": 0.97,
        "average_task_time": "3.4 minutes",
        "user_feedback": "Intuitive interface, helpful uncertainty indicators"
    }
    
    compliance_manager.vv_system.execute_validation_test(
        device_name="PathologyAI_Clinical",
        device_version="1.0.0",
        activity_id="VAL002",
        test_results=usability_validation_results
    )
    
    print("✓ Executed validation tests")
    
    # 8. Create cybersecurity plan
    print("\n8. Creating cybersecurity plan...")
    
    threat_model = {
        "threats": [
            "Unauthorized access to patient data",
            "Data breach through network vulnerabilities",
            "Malware infection affecting system integrity",
            "Insider threats and privilege abuse",
            "Denial of service attacks"
        ],
        "attack_vectors": [
            "Network-based attacks",
            "Physical access to systems",
            "Social engineering",
            "Supply chain compromises",
            "Insider threats"
        ],
        "assets": [
            "Patient health information (PHI)",
            "ML model parameters and algorithms",
            "Clinical workflow data",
            "System configuration and credentials"
        ]
    }
    
    security_controls = [
        {
            "control_id": "SC001",
            "description": "Data encryption at rest",
            "implementation": "AES-256 encryption for all stored data",
            "fda_control_category": "Data Protection"
        },
        {
            "control_id": "SC002",
            "description": "Data encryption in transit",
            "implementation": "TLS 1.3 for all network communications",
            "fda_control_category": "Data Protection"
        },
        {
            "control_id": "SC003",
            "description": "Multi-factor authentication",
            "implementation": "MFA required for all user accounts",
            "fda_control_category": "Access Control"
        },
        {
            "control_id": "SC004",
            "description": "Role-based access control",
            "implementation": "RBAC with principle of least privilege",
            "fda_control_category": "Access Control"
        },
        {
            "control_id": "SC005",
            "description": "Audit logging and monitoring",
            "implementation": "Comprehensive audit logs with real-time monitoring",
            "fda_control_category": "Monitoring and Response"
        }
    ]
    
    cybersecurity_plan = compliance_manager.cybersecurity.create_cybersecurity_plan(
        device_name="PathologyAI_Clinical",
        device_version="1.0.0",
        threat_model=threat_model,
        security_controls=security_controls
    )
    print(f"✓ Created cybersecurity plan with {len(security_controls)} security controls")
    
    # 9. Update component validation status
    print("\n9. Updating component validation status...")
    
    compliance_manager.documentation_system.update_component_validation(
        dmr=dmr,
        component_name="AttentionMIL_Core",
        validation_status=ValidationStatus.VALIDATED,
        validation_results={
            "verification_pass_rate": 100,
            "validation_pass_rate": 100,
            "clinical_performance_verified": True,
            "safety_analysis_complete": True
        }
    )
    
    compliance_manager.documentation_system.update_component_validation(
        dmr=dmr,
        component_name="Clinical_Workflow",
        validation_status=ValidationStatus.VALIDATED,
        validation_results={
            "usability_testing_complete": True,
            "integration_testing_passed": True,
            "workflow_validation_complete": True
        }
    )
    
    compliance_manager.documentation_system.update_component_validation(
        dmr=dmr,
        component_name="Privacy_Security",
        validation_status=ValidationStatus.VALIDATED,
        validation_results={
            "hipaa_compliance_verified": True,
            "security_testing_passed": True,
            "penetration_testing_complete": True
        }
    )
    
    print("✓ Updated validation status for all components")
    
    # 10. Generate V&V report
    print("\n10. Generating V&V summary report...")
    vv_report = compliance_manager.vv_system.generate_vv_report(
        device_name="PathologyAI_Clinical",
        device_version="1.0.0"
    )
    
    print(f"✓ Generated V&V report:")
    print(f"   - Verification pass rate: {vv_report['verification_summary']['pass_rate']:.1f}%")
    print(f"   - Validation pass rate: {vv_report['validation_summary']['pass_rate']:.1f}%")
    print(f"   - Overall status: {vv_report['overall_status']}")
    
    # 11. Generate regulatory submission package
    print("\n11. Generating regulatory submission package...")
    
    submission_path = compliance_manager.generate_regulatory_submission_package(
        device_name="PathologyAI_Clinical",
        device_version="1.0.0",
        submission_type="FDA_510K",
        output_path="regulatory_submission_package"
    )
    
    print(f"✓ Generated FDA 510(k) submission package at: {submission_path}")
    
    # 12. Demonstrate post-market surveillance
    print("\n12. Demonstrating post-market surveillance...")
    
    # Simulate adverse event reporting
    adverse_events = [
        {
            "event_id": "AE001",
            "description": "False negative result in lung cancer case",
            "severity": "serious",
            "date": "2024-01-15",
            "patient_outcome": "Delayed diagnosis, successful treatment after 2 weeks",
            "root_cause": "Unusual tissue morphology not well represented in training data",
            "corrective_actions": ["Enhanced training data collection", "Model retraining planned"]
        }
    ]
    
    # Simulate performance monitoring data
    performance_data = {
        "accuracy": 0.919,  # Slight decrease from validation
        "sensitivity": 0.912,
        "specificity": 0.926,
        "total_cases_processed": 2847,
        "monitoring_period": "Q1 2024",
        "performance_trend": "stable"
    }
    
    compliance_manager.risk_management.update_post_market_surveillance(
        device_name="PathologyAI_Clinical",
        device_version="1.0.0",
        adverse_events=adverse_events,
        performance_data=performance_data
    )
    
    print("✓ Updated post-market surveillance data")
    
    print("\n=== Regulatory Compliance Workflow Complete ===")
    print(f"\nAll regulatory documentation has been generated in: regulatory_docs_example/")
    print("The system is ready for regulatory submission and clinical deployment.")
    
    return compliance_manager


if __name__ == "__main__":
    compliance_manager = main()