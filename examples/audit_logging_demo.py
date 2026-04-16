#!/usr/bin/env python3
"""
Demonstration of audit logging functionality for regulatory compliance.

This script shows how to use the audit logging system to record prediction operations,
user access events, data modifications, and system errors with tamper-evident records.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from src.clinical.audit import (
    AuditLogger,
    ComplianceAuditLogger,
    create_default_audit_logger,
    AuditContextManager,
    AuditLogAnalyzer,
)


def main():
    """Demonstrate audit logging functionality."""
    print("=== Clinical Workflow Audit Logging Demo ===\n")

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary audit storage: {temp_dir}\n")

        # Create audit logger
        audit_logger = create_default_audit_logger(Path(temp_dir))
        print("✓ Created compliance audit logger with backup storage")

        # Demo 1: Log prediction operation
        print("\n1. Logging prediction operation...")

        input_data = {
            "patient_id": "PATIENT_12345",
            "image_path": "/data/slides/case_001.svs",
            "model_config": {"attention_type": "gated", "pooling": "max"},
        }

        output_data = {
            "primary_diagnosis": "benign",
            "confidence_score": 0.87,
            "probability_distribution": {"benign": 0.87, "malignant": 0.13},
            "attention_weights": [0.1, 0.3, 0.6],
        }

        event_id = audit_logger.log_prediction_operation(
            user_id="physician_001",
            session_token="session_abc123",
            input_data=input_data,
            output_data=output_data,
            model_version="v2.1.0",
            processing_time_ms=2150.5,
            ip_address="192.168.1.100",
        )

        print(f"  ✓ Logged prediction operation: {event_id}")

        # Demo 2: Log user access events
        print("\n2. Logging user access events...")

        # Successful login
        login_event = audit_logger.log_user_access(
            event_type="authentication",
            user_id="physician_001",
            session_token="session_abc123",
            resource="clinical_system",
            action="login",
            success=True,
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 (Clinical Workstation)",
            details={"authentication_method": "two_factor"},
        )
        print(f"  ✓ Logged successful login: {login_event}")

        # Failed access attempt
        failed_access = audit_logger.log_user_access(
            event_type="data_access",
            user_id="technician_002",
            session_token="session_xyz789",
            resource="patient_data",
            action="read_sensitive",
            success=False,
            ip_address="192.168.1.101",
            details={"denial_reason": "insufficient_permissions"},
        )
        print(f"  ✓ Logged failed access attempt: {failed_access}")

        # Demo 3: Log data modification
        print("\n3. Logging data modification...")

        modification_event = audit_logger.log_data_modification(
            user_id="physician_001",
            session_token="session_abc123",
            resource="patient_record_12345",
            modification_type="diagnosis_update",
            old_data_hash="sha256:abc123...",
            new_data_hash="sha256:def456...",
            ip_address="192.168.1.100",
            details={
                "fields_modified": ["primary_diagnosis", "confidence_level"],
                "reason": "additional_review_completed",
            },
        )
        print(f"  ✓ Logged data modification: {modification_event}")

        # Demo 4: Log system error
        print("\n4. Logging system error...")

        error_event = audit_logger.log_system_error(
            error_type="ModelInferenceError",
            error_message="CUDA out of memory during inference",
            stack_trace="Traceback (most recent call last):\n  File 'inference.py', line 42...",
            input_data={"batch_size": 64, "image_dimensions": "4096x4096"},
            user_id="physician_001",
            session_token="session_abc123",
            model_version="v2.1.0",
        )
        print(f"  ✓ Logged system error: {error_event}")

        # Demo 5: Log model training event
        print("\n5. Logging model training event...")

        training_event = audit_logger.log_model_training(
            dataset_version="pathology_dataset_v3.2",
            hyperparameters={
                "learning_rate": 0.0001,
                "batch_size": 32,
                "epochs": 100,
                "attention_dim": 256,
            },
            performance_metrics={
                "accuracy": 0.924,
                "auc": 0.967,
                "sensitivity": 0.891,
                "specificity": 0.943,
            },
            training_duration_minutes=480.5,
            model_version="v2.1.0",
            user_id="researcher_003",
        )
        print(f"  ✓ Logged model training: {training_event}")

        # Demo 6: Verify audit record integrity
        print("\n6. Verifying audit record integrity...")

        records = audit_logger.get_audit_records()
        print(f"  Retrieved {len(records)} audit records")

        valid_count = 0
        for record in records:
            is_valid = audit_logger.verify_record_integrity(record)
            if is_valid:
                valid_count += 1

        print(f"  ✓ {valid_count}/{len(records)} records have valid cryptographic signatures")

        # Demo 7: Generate compliance report
        print("\n7. Generating compliance report...")

        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now() + timedelta(days=1)
        report_path = Path(temp_dir) / "compliance_report.json"

        compliance_report = audit_logger.generate_compliance_report(
            start_date, end_date, report_path
        )

        print(f"  ✓ Generated compliance report: {report_path}")
        print(f"    - Total records: {compliance_report['report_metadata']['total_records']}")
        print(
            f"    - Integrity verified: {compliance_report['report_metadata']['integrity_verification_passed']}"
        )
        print(
            f"    - FDA 21 CFR Part 11 compliant: {compliance_report['regulatory_compliance']['fda_21_cfr_part_11_compliant']}"
        )
        print(
            f"    - Retention period: {compliance_report['regulatory_compliance']['retention_period_years']} years"
        )

        # Demo 8: Audit chain validation
        print("\n8. Validating audit chain...")

        validation_results = audit_logger.validate_audit_chain()
        print(f"  ✓ Chain integrity: {validation_results['chain_integrity']}")
        print(f"    - Valid signatures: {validation_results['valid_signatures']}")
        print(f"    - Invalid signatures: {validation_results['invalid_signatures']}")
        print(f"    - Validation errors: {len(validation_results['validation_errors'])}")

        # Demo 9: Export audit logs
        print("\n9. Exporting audit logs...")

        # Export as JSON
        json_export_path = Path(temp_dir) / "audit_export.json"
        json_success = audit_logger.export_audit_logs(json_export_path, format="json")
        print(f"  ✓ JSON export: {json_success} -> {json_export_path}")

        # Export as CSV
        csv_export_path = Path(temp_dir) / "audit_export.csv"
        csv_success = audit_logger.export_audit_logs(csv_export_path, format="csv")
        print(f"  ✓ CSV export: {csv_success} -> {csv_export_path}")

        # Demo 10: Audit log analysis
        print("\n10. Analyzing audit logs...")

        analyzer = AuditLogAnalyzer(audit_logger)

        # Detect anomalous patterns
        anomalies = analyzer.detect_anomalous_patterns(lookback_days=1)
        print(f"  ✓ Anomaly detection completed:")
        print(f"    - Failed operations spikes: {len(anomalies['failed_operations_spike'])}")
        print(f"    - Off-hours activity: {len(anomalies['off_hours_activity'])}")
        print(f"    - Bulk data access: {len(anomalies['bulk_data_access'])}")

        # Generate usage report
        usage_report = analyzer.generate_usage_report(start_date, end_date)
        print(f"  ✓ Usage report generated:")
        print(f"    - Total events: {usage_report['summary_statistics']['total_events']}")
        print(f"    - Unique users: {usage_report['summary_statistics']['unique_users']}")
        print(f"    - Error rate: {usage_report['system_health']['error_rate']:.2%}")

        # Demo 11: Audit statistics
        print("\n11. Audit statistics...")

        stats = audit_logger.get_audit_statistics()
        print(f"  ✓ Audit statistics:")
        print(f"    - Total records: {stats['total_records']}")
        print(f"    - Recent records (30 days): {stats['recent_records_30_days']}")
        print(f"    - Event types: {list(stats['event_type_distribution'].keys())}")
        print(f"    - Top users: {list(stats['top_users'].keys())[:3]}")

        # Demo 12: Context manager usage
        print("\n12. Using audit context manager...")

        def simulate_prediction_operation():
            """Simulate a prediction operation with automatic audit logging."""
            # This would normally contain actual ML inference code
            import time

            time.sleep(0.1)  # Simulate processing time
            return {"prediction": "benign", "confidence": 0.92}

        with AuditContextManager(
            audit_logger, "automated_prediction", "physician_001", "session_abc123", "v2.1.0"
        ) as audit_ctx:
            result = simulate_prediction_operation()
            audit_ctx.log_success({"patient_id": "PATIENT_67890", "image_type": "H&E"}, result)

        print("  ✓ Automatic audit logging with context manager completed")

        print(f"\n=== Demo completed successfully! ===")
        print(f"Final audit record count: {audit_logger.get_audit_statistics()['total_records']}")
        print(f"All records stored with tamper-evident cryptographic signatures.")
        print(f"Retention period: 7 years (2555 days) for FDA compliance.")


if __name__ == "__main__":
    main()
