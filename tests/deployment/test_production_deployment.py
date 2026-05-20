"""
Tests for Production Deployment Module
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.platform.deployment.clinical_impact import (
    ClinicalImpactTracker,
    DiagnosticAccuracyMetric,
    TurnaroundTimeMetric,
    UserSatisfactionSurvey,
)
from src.platform.deployment.production_optimization import (
    AutoScaler,
    CapacityPlanner,
    OperationalExcellence,
    PerformanceMonitor,
)
from src.platform.deployment.site_preparation import (
    HospitalSite,
    SitePreparationManager,
    TechnicalRequirements,
)


class TestSitePreparation:
    """Test site preparation functionality."""

    def test_technical_requirements_assessment(self):
        """Test technical requirements assessment."""
        manager = SitePreparationManager()

        # Add test site
        site = HospitalSite(
            site_id="TEST001",
            name="Test Hospital",
            location="Test City",
            contact_email="test@hospital.com",
            it_contact="Test Admin",
            pathology_volume=3000,
            current_systems=["Epic EMR", "Sunquest LIS"],
            integration_requirements=["Epic FHIR", "Sunquest HL7"],
            compliance_requirements=["HIPAA", "SOX"],
        )
        manager.add_hospital_site(site)

        # Assess requirements
        assessment = manager.assess_technical_requirements("TEST001")

        assert assessment["site_id"] == "TEST001"
        assert assessment["site_name"] == "Test Hospital"
        assert "recommended_specs" in assessment
        assert "cpu_cores" in assessment["recommended_specs"]
        assert "integration_points" in assessment
        assert "estimated_setup_time_days" in assessment

    def test_integration_plan_creation(self):
        """Test integration plan creation."""
        manager = SitePreparationManager()

        site = HospitalSite(
            site_id="TEST001",
            name="Test Hospital",
            location="Test City",
            contact_email="test@hospital.com",
            it_contact="Test Admin",
            pathology_volume=2000,
            current_systems=["Cerner EMR"],
            integration_requirements=["Cerner FHIR"],
            compliance_requirements=["HIPAA"],
        )
        manager.add_hospital_site(site)

        plan = manager.create_integration_plan("TEST001")

        assert plan["site_id"] == "TEST001"
        assert "integration_phases" in plan
        assert len(plan["integration_phases"]) > 0
        assert "testing_protocols" in plan
        assert "rollback_procedures" in plan

    def test_training_program_generation(self):
        """Test training program generation."""
        manager = SitePreparationManager()

        site = HospitalSite(
            site_id="TEST001",
            name="Test Hospital",
            location="Test City",
            contact_email="test@hospital.com",
            it_contact="Test Admin",
            pathology_volume=1500,
            current_systems=["Basic LIS"],
            integration_requirements=["HL7"],
            compliance_requirements=["HIPAA"],
        )
        manager.add_hospital_site(site)

        program = manager.generate_training_program("TEST001")

        assert program["site_id"] == "TEST001"
        assert "training_modules" in program
        assert len(program["training_modules"]) > 0
        assert "certification_requirements" in program
        assert "ongoing_education" in program


class TestClinicalImpact:
    """Test clinical impact tracking functionality."""

    @pytest.fixture
    def tracker(self):
        """Create temporary clinical impact tracker."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        tracker = ClinicalImpactTracker(db_path)
        yield tracker
        Path(db_path).unlink(missing_ok=True)

    def test_diagnostic_accuracy_tracking(self, tracker):
        """Test diagnostic accuracy tracking."""
        metric = DiagnosticAccuracyMetric(
            case_id="TEST001",
            ai_prediction="Malignant",
            pathologist_diagnosis="Malignant",
            ground_truth="Malignant",
            confidence_score=0.95,
            processing_time_seconds=25.5,
            timestamp=datetime.now(),
            site_id="SITE001",
        )

        tracker.track_diagnostic_accuracy(metric)

        # Verify data was stored
        with sqlite3.connect(tracker.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM diagnostic_accuracy")
            count = cursor.fetchone()[0]
            assert count == 1

    def test_turnaround_time_tracking(self, tracker):
        """Test turnaround time tracking."""
        now = datetime.now()
        metric = TurnaroundTimeMetric(
            case_id="TEST001",
            slide_received=now,
            ai_processing_start=now + timedelta(minutes=5),
            ai_processing_complete=now + timedelta(minutes=5, seconds=30),
            pathologist_review_start=now + timedelta(minutes=10),
            pathologist_review_complete=now + timedelta(minutes=25),
            report_finalized=now + timedelta(minutes=30),
            site_id="SITE001",
        )

        tracker.track_turnaround_time(metric)

        # Test property calculations
        assert metric.ai_processing_time == 0.5  # 30 seconds = 0.5 minutes
        assert metric.pathologist_review_time == 15.0  # 15 minutes
        assert metric.total_turnaround_time == 0.5  # 30 minutes = 0.5 hours

    def test_user_satisfaction_tracking(self, tracker):
        """Test user satisfaction tracking."""
        survey = UserSatisfactionSurvey(
            user_id="PATH001",
            user_role="pathologist",
            site_id="SITE001",
            timestamp=datetime.now(),
            ease_of_use=4,
            accuracy_perception=5,
            time_savings=4,
            overall_satisfaction=4,
            would_recommend=True,
            comments="Great system!",
        )

        tracker.track_user_satisfaction(survey)

        # Verify data was stored
        with sqlite3.connect(tracker.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM user_satisfaction")
            count = cursor.fetchone()[0]
            assert count == 1

    def test_diagnostic_accuracy_metrics_calculation(self, tracker):
        """Test diagnostic accuracy metrics calculation."""
        # Add test data
        for i in range(10):
            metric = DiagnosticAccuracyMetric(
                case_id=f"TEST{i:03d}",
                ai_prediction="Malignant" if i < 9 else "Benign",  # 90% accuracy
                pathologist_diagnosis="Malignant",
                ground_truth="Malignant",
                confidence_score=0.8 + (i * 0.02),
                processing_time_seconds=20 + i,
                timestamp=datetime.now(),
                site_id="SITE001",
            )
            tracker.track_diagnostic_accuracy(metric)

        metrics = tracker.calculate_diagnostic_accuracy_metrics(days_back=1)

        assert metrics["total_cases"] == 10
        assert metrics["ai_accuracy"] == 0.9  # 9/10 correct
        assert metrics["pathologist_accuracy"] == 1.0  # All correct
        assert "average_confidence" in metrics
        assert "average_processing_time" in metrics


class TestProductionOptimization:
    """Test production optimization functionality."""

    @pytest.fixture
    def monitor(self):
        """Create temporary performance monitor."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        monitor = PerformanceMonitor(db_path)
        yield monitor
        Path(db_path).unlink(missing_ok=True)

    def test_performance_monitoring_setup(self, monitor):
        """Test performance monitoring setup."""
        # Test database initialization
        with sqlite3.connect(monitor.db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='performance_metrics'
            """)
            assert cursor.fetchone() is not None

    def test_auto_scaler_setup(self, monitor):
        """Test auto-scaler setup."""
        scaler = AutoScaler(monitor)

        scale_up_called = False
        scale_down_called = False

        def scale_up():
            nonlocal scale_up_called
            scale_up_called = True

        def scale_down():
            nonlocal scale_down_called
            scale_down_called = True

        scaler.add_scaling_policy(
            component="test",
            metric_name="cpu_utilization",
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            scale_up_action=scale_up,
            scale_down_action=scale_down,
        )

        assert "test" in scaler.scaling_policies
        policy = scaler.scaling_policies["test"]
        assert policy["scale_up_threshold"] == 80.0
        assert policy["scale_down_threshold"] == 30.0

    def test_capacity_planner(self, monitor):
        """Test capacity planner."""
        planner = CapacityPlanner(monitor)

        # Add some test metrics
        with sqlite3.connect(monitor.db_path) as conn:
            for i in range(7):  # 7 days of data
                date = datetime.now() - timedelta(days=i)
                conn.execute(
                    """
                    INSERT INTO performance_metrics 
                    (timestamp, metric_name, value, unit, site_id, component)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        date.isoformat(),
                        "cpu_utilization",
                        50 + i * 2,  # Increasing trend
                        "percent",
                        "SITE001",
                        "system",
                    ),
                )

        analysis = planner.analyze_capacity_trends(days_back=7)

        assert "analysis_period_days" in analysis
        assert "projections" in analysis
        assert "recommendations" in analysis

    def test_operational_excellence_setup(self):
        """Test operational excellence setup."""
        ops = OperationalExcellence()

        # Test monitoring setup
        monitoring_config = ops.setup_monitoring_alerting()
        assert "metrics_to_monitor" in monitoring_config
        assert "alert_channels" in monitoring_config
        assert "dashboards" in monitoring_config

        # Test incident procedures
        procedures = ops.create_incident_response_procedures()
        assert "severity_levels" in procedures
        assert "response_procedures" in procedures
        assert "communication_templates" in procedures

        # Test backup configuration
        backup_config = ops.setup_backup_recovery()
        assert "backup_schedules" in backup_config
        assert "recovery_procedures" in backup_config
        assert "rto_rpo_targets" in backup_config

        # Test security configuration
        security_config = ops.implement_security_hardening()
        assert "access_controls" in security_config
        assert "network_security" in security_config
        assert "monitoring" in security_config


if __name__ == "__main__":
    pytest.main([__file__])
