"""Unit tests for historical comparison and approval workflow."""

import pytest
import json
from pathlib import Path
from datetime import datetime
from experiments.benchmark_system.historical_comparison import (
    HistoricalComparison,
    HistoricalResult,
    DeviationFlag,
)
from experiments.benchmark_system.approval import ApprovalWorkflow, QAFlag


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for tests."""
    return tmp_path


@pytest.fixture
def historical_results():
    """Sample historical results."""
    return {
        "HistoCore": HistoricalResult(
            framework="HistoCore",
            accuracy=0.85,
            loss=0.35,
            training_time=1000.0,
            gpu_memory_peak=8.5,
            timestamp="2024-01-01T00:00:00",
            version="1.0.0",
        ),
        "PathML": HistoricalResult(
            framework="PathML",
            accuracy=0.80,
            loss=0.40,
            training_time=1200.0,
            gpu_memory_peak=9.0,
            timestamp="2024-01-01T00:00:00",
            version="1.0.0",
        ),
    }


@pytest.fixture
def current_results():
    """Sample current results."""
    return {
        "HistoCore": {
            "accuracy": 0.86,
            "loss": 0.34,
            "training_time": 1050.0,
            "gpu_memory_peak": 8.7,
        },
        "PathML": {
            "accuracy": 0.75,  # 5% decrease - warning
            "loss": 0.45,  # 12.5% increase - critical
            "training_time": 1250.0,
            "gpu_memory_peak": 9.2,
        },
    }


class TestHistoricalComparison:
    """Test historical comparison functionality."""

    def test_load_historical_results(self, temp_dir, historical_results):
        """Test loading historical results from disk."""
        # Save historical results
        historical_dir = temp_dir / "historical"
        historical_dir.mkdir()
        historical_file = historical_dir / "historical_results.json"

        data = {k: v.__dict__ for k, v in historical_results.items()}
        with open(historical_file, "w") as f:
            json.dump(data, f)

        # Load results
        comparison = HistoricalComparison(historical_dir)
        loaded = comparison.load_historical_results()

        assert len(loaded) == 2
        assert "HistoCore" in loaded
        assert loaded["HistoCore"].accuracy == 0.85

    def test_load_missing_historical_results(self, temp_dir):
        """Test loading when no historical results exist."""
        comparison = HistoricalComparison(temp_dir / "nonexistent")
        loaded = comparison.load_historical_results()

        assert loaded == {}

    def test_compare_to_historical_no_deviations(self, temp_dir, historical_results):
        """Test comparison with no significant deviations."""
        current = {
            "HistoCore": {
                "accuracy": 0.85,  # Same
                "loss": 0.35,  # Same
                "training_time": 1000.0,  # Same
                "gpu_memory_peak": 8.5,  # Same
            }
        }

        comparison = HistoricalComparison(temp_dir, warning_threshold=5.0)
        flags = comparison.compare_to_historical(current, historical_results)

        assert len(flags) == 0

    def test_compare_to_historical_warnings(self, temp_dir, historical_results):
        """Test comparison with warning-level deviations."""
        current = {
            "HistoCore": {
                "accuracy": 0.81,  # 4.7% decrease - below warning
                "loss": 0.37,  # 5.7% increase - warning
                "training_time": 1000.0,
                "gpu_memory_peak": 8.5,
            }
        }

        comparison = HistoricalComparison(temp_dir, warning_threshold=5.0)
        flags = comparison.compare_to_historical(current, historical_results)

        assert len(flags) == 1
        assert flags[0].severity == "warning"
        assert flags[0].metric == "loss"

    def test_compare_to_historical_critical(self, temp_dir, historical_results, current_results):
        """Test comparison with critical-level deviations."""
        comparison = HistoricalComparison(temp_dir, warning_threshold=5.0, critical_threshold=10.0)
        flags = comparison.compare_to_historical(current_results, historical_results)

        # PathML loss increased by 12.5% - critical
        critical_flags = [f for f in flags if f.severity == "critical"]
        assert len(critical_flags) >= 1
        assert any(f.framework == "PathML" and f.metric == "loss" for f in critical_flags)

    def test_compute_deviation_accuracy(self, temp_dir):
        """Test deviation computation for accuracy (higher is better)."""
        comparison = HistoricalComparison(temp_dir)

        # Accuracy increase is good (positive deviation)
        deviation = comparison._compute_deviation(0.90, 0.85, "accuracy")
        assert deviation > 0

        # Accuracy decrease is bad (positive deviation)
        deviation = comparison._compute_deviation(0.80, 0.85, "accuracy")
        assert deviation > 0

    def test_compute_deviation_loss(self, temp_dir):
        """Test deviation computation for loss (lower is better)."""
        comparison = HistoricalComparison(temp_dir)

        # Loss increase is bad (positive deviation)
        deviation = comparison._compute_deviation(0.40, 0.35, "loss")
        assert deviation > 0

        # Loss decrease is good (negative deviation)
        deviation = comparison._compute_deviation(0.30, 0.35, "loss")
        assert deviation < 0

    def test_save_as_historical(self, temp_dir, current_results):
        """Test saving current results as historical baseline."""
        comparison = HistoricalComparison(temp_dir)
        comparison.save_as_historical(
            current_results, version="1.1.0", timestamp="2024-02-01T00:00:00"
        )

        # Verify saved
        historical_file = temp_dir / "historical_results.json"
        assert historical_file.exists()

        with open(historical_file, "r") as f:
            data = json.load(f)

        assert "HistoCore" in data
        assert data["HistoCore"]["accuracy"] == 0.86
        assert data["HistoCore"]["version"] == "1.1.0"

    def test_generate_deviation_report_empty(self, temp_dir):
        """Test deviation report with no flags."""
        comparison = HistoricalComparison(temp_dir)
        report = comparison.generate_deviation_report([])

        assert "No significant deviations" in report

    def test_generate_deviation_report_with_flags(self, temp_dir):
        """Test deviation report with flags."""
        flags = [
            DeviationFlag(
                framework="PathML",
                metric="loss",
                current_value=0.45,
                historical_value=0.40,
                deviation_percent=12.5,
                threshold_percent=10.0,
                severity="critical",
            ),
            DeviationFlag(
                framework="PathML",
                metric="accuracy",
                current_value=0.75,
                historical_value=0.80,
                deviation_percent=6.25,
                threshold_percent=5.0,
                severity="warning",
            ),
        ]

        comparison = HistoricalComparison(temp_dir)
        report = comparison.generate_deviation_report(flags)

        assert "Critical Deviations" in report
        assert "Warnings" in report
        assert "PathML" in report
        assert "loss" in report
        assert "accuracy" in report


class TestApprovalWorkflow:
    """Test approval workflow functionality."""

    def test_generate_approval_report_no_issues(self, temp_dir, current_results):
        """Test approval report with no issues."""
        workflow = ApprovalWorkflow(temp_dir)
        report = workflow.generate_approval_report(
            results=current_results, qa_flags=[], deviation_flags=[]
        )

        assert "Benchmark Results Approval Report" in report
        assert "Status: APPROVED" in report
        assert "No QA flags detected" in report

    def test_generate_approval_report_with_warnings(self, temp_dir, current_results):
        """Test approval report with warnings."""
        qa_flags = [
            QAFlag(
                category="anomaly",
                severity="warning",
                framework="PathML",
                message="Accuracy below expected range",
            )
        ]

        workflow = ApprovalWorkflow(temp_dir)
        report = workflow.generate_approval_report(
            results=current_results, qa_flags=qa_flags, deviation_flags=[]
        )

        assert "Status: REVIEW RECOMMENDED" in report
        assert "Warnings (1)" in report
        assert "PathML" in report

    def test_generate_approval_report_with_critical(self, temp_dir, current_results):
        """Test approval report with critical issues."""
        qa_flags = [
            QAFlag(
                category="validation",
                severity="critical",
                framework="PathML",
                message="Training failed to converge",
                details="Loss did not decrease",
            )
        ]

        workflow = ApprovalWorkflow(temp_dir)
        report = workflow.generate_approval_report(
            results=current_results, qa_flags=qa_flags, deviation_flags=[]
        )

        assert "Status: REQUIRES REVIEW" in report
        assert "Critical Issues (1)" in report
        assert "Training failed to converge" in report

    def test_request_approval_non_interactive(self, temp_dir, current_results):
        """Test approval request in non-interactive mode."""
        workflow = ApprovalWorkflow(temp_dir)
        report = workflow.generate_approval_report(
            results=current_results, qa_flags=[], deviation_flags=[]
        )

        approved = workflow.request_approval(report, interactive=False)

        assert not approved
        assert (temp_dir / "approval_report.md").exists()

    def test_apply_approved_results_new_file(self, temp_dir, current_results):
        """Test applying approved results to new file."""
        perf_file = temp_dir / "PERFORMANCE_COMPARISON.md"
        perf_file.write_text("# Performance Comparison\n\n<!-- BENCHMARK_RESULTS_TABLE -->\n")

        workflow = ApprovalWorkflow(temp_dir)
        workflow.apply_approved_results(current_results, perf_file)

        content = perf_file.read_text()
        assert "HistoCore" in content
        assert "PathML" in content
        assert "0.86" in content  # HistoCore accuracy

        # Verify backup created
        backup_file = temp_dir / "PERFORMANCE_COMPARISON.md.backup"
        assert backup_file.exists()

    def test_apply_approved_results_missing_file(self, temp_dir, current_results):
        """Test applying approved results when file doesn't exist."""
        perf_file = temp_dir / "PERFORMANCE_COMPARISON.md"

        workflow = ApprovalWorkflow(temp_dir)
        workflow.apply_approved_results(current_results, perf_file)

        # Should not crash, just log error
        assert not perf_file.exists()


class TestIntegration:
    """Integration tests for historical comparison and approval."""

    def test_full_workflow(self, temp_dir, historical_results, current_results):
        """Test complete workflow from comparison to approval."""
        # 1. Compare to historical
        comparison = HistoricalComparison(
            temp_dir / "historical", warning_threshold=5.0, critical_threshold=10.0
        )

        # Save historical baseline
        comparison.save_as_historical(
            {k: v.__dict__ for k, v in historical_results.items()},
            version="1.0.0",
            timestamp="2024-01-01T00:00:00",
        )

        # Compare current to historical
        deviation_flags = comparison.compare_to_historical(current_results, historical_results)

        assert len(deviation_flags) > 0

        # 2. Generate QA flags
        qa_flags = [
            QAFlag(
                category="deviation",
                severity="warning",
                framework="PathML",
                message="Performance degradation detected",
            )
        ]

        # 3. Generate approval report
        workflow = ApprovalWorkflow(temp_dir / "approval")
        report = workflow.generate_approval_report(
            results=current_results, qa_flags=qa_flags, deviation_flags=deviation_flags
        )

        assert "REVIEW RECOMMENDED" in report
        assert len(deviation_flags) > 0

        # 4. Request approval (saves report)
        workflow.request_approval(report, interactive=False)

        # Verify report saved
        report_file = temp_dir / "approval" / "approval_report.md"
        assert report_file.exists()
