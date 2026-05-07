"""
Unit tests for RegressionDetector.

Tests coverage, performance, security, and code quality regression detection.
"""

import pytest
from datetime import datetime

from src.analysis.regression_detector import (
    RegressionDetector,
    RegressionType,
    RegressionSeverity,
    Regression,
    RegressionReport
)
from src.analysis.models import (
    AnalysisResult,
    ArchitectureAnalysis,
    PerformanceAnalysis,
    CoverageAnalysis,
    CodeQualityAnalysis,
    DependencyAnalysis,
    DeploymentAnalysis,
    SecurityAnalysis,
    ScalabilityAnalysis
)


def create_baseline_result():
    """Create baseline analysis result for testing."""
    return AnalysisResult(
        timestamp=datetime.now().isoformat(),
        project_path="/test/project",
        git_commit="a" * 40,
        architecture=ArchitectureAnalysis(score=80.0),
        performance=PerformanceAnalysis(
            gpu_utilization=85.0,
            memory_usage_peak_gb=8.0,
            bottlenecks=[{"operation": "data_loading", "time_ms": 100}],
            score=75.0
        ),
        coverage=CoverageAnalysis(
            line_coverage=70.0,
            branch_coverage=65.0,
            score=70.0
        ),
        code_quality=CodeQualityAnalysis(
            average_complexity=5.0,
            duplication_percentage=10.0,
            score=75.0
        ),
        dependencies=DependencyAnalysis(
            vulnerabilities=[{"cve_id": "CVE-2023-0001"}],
            score=80.0
        ),
        deployment=DeploymentAnalysis(score=85.0),
        security=SecurityAnalysis(
            vulnerabilities=[{"type": "sql_injection", "severity": "high"}],
            hardcoded_secrets=[],
            score=70.0
        ),
        scalability=ScalabilityAnalysis(score=80.0),
        overall_score=75.0
    )


@pytest.fixture
def baseline_result():
    """Create baseline analysis result for testing."""
    return create_baseline_result()


@pytest.fixture
def detector():
    """Create regression detector with default thresholds."""
    return RegressionDetector(coverage_threshold=2.0, performance_threshold=10.0)


class TestCoverageRegressions:
    """Test coverage regression detection."""

    def test_critical_line_coverage_regression(self, detector, baseline_result):
        """Test detection of critical line coverage regression (>2% decrease)."""
        current = create_baseline_result()
        current.coverage.line_coverage = 67.0  # 3% decrease

        report = detector.detect_regressions(baseline_result, current)

        assert report.has_regressions
        assert len(report.critical_regressions) == 1
        assert report.critical_regressions[0].type == RegressionType.COVERAGE
        assert report.critical_regressions[0].metric == "line_coverage"
        assert report.should_fail_ci()

    def test_high_line_coverage_regression(self, detector, baseline_result):
        """Test detection of high severity line coverage regression (<2% decrease)."""
        current = create_baseline_result()
        current.coverage.line_coverage = 69.0  # 1% decrease

        report = detector.detect_regressions(baseline_result, current)

        assert report.has_regressions
        assert len(report.high_regressions) == 1
        assert report.high_regressions[0].type == RegressionType.COVERAGE
        assert not report.should_fail_ci()

    def test_coverage_improvement(self, detector, baseline_result):
        """Test detection of coverage improvement."""
        current = create_baseline_result()
        current.coverage.line_coverage = 75.0  # 5% increase

        report = detector.detect_regressions(baseline_result, current)

        assert not report.has_regressions
        assert len(report.improvements) >= 1
        improvement = next(i for i in report.improvements if i.metric == "line_coverage")
        assert improvement.change_percentage > 0

    def test_branch_coverage_regression(self, detector, baseline_result):
        """Test detection of branch coverage regression."""
        current = create_baseline_result()
        current.coverage.branch_coverage = 62.0  # 3% decrease

        report = detector.detect_regressions(baseline_result, current)

        assert report.has_regressions
        critical_branch = next(
            (r for r in report.critical_regressions if r.metric == "branch_coverage"),
            None
        )
        assert critical_branch is not None
        assert critical_branch.severity == RegressionSeverity.CRITICAL


class TestPerformanceRegressions:
    """Test performance regression detection."""

    def test_gpu_utilization_regression(self, detector, baseline_result):
        """Test detection of GPU utilization regression (>10% decrease)."""
        current = create_baseline_result()
        current.performance.gpu_utilization = 75.0  # ~11.8% decrease

        report = detector.detect_regressions(baseline_result, current)

        assert report.has_regressions
        gpu_regression = next(
            (r for r in report.critical_regressions if r.metric == "gpu_utilization"),
            None
        )
        assert gpu_regression is not None
        assert gpu_regression.type == RegressionType.PERFORMANCE

    def test_memory_usage_regression(self, detector, baseline_result):
        """Test detection of memory usage regression (>10% increase)."""
        current = create_baseline_result()
        current.performance.memory_usage_peak_gb = 9.0  # 12.5% increase

        report = detector.detect_regressions(baseline_result, current)

        assert report.has_regressions
        mem_regression = next(
            (r for r in report.high_regressions if r.metric == "memory_usage_peak_gb"),
            None
        )
        assert mem_regression is not None
        assert mem_regression.severity == RegressionSeverity.HIGH

    def test_new_bottleneck_detection(self, detector, baseline_result):
        """Test detection of new performance bottlenecks."""
        current = create_baseline_result()
        current.performance.bottlenecks.append({"operation": "model_forward", "time_ms": 200})

        report = detector.detect_regressions(baseline_result, current)

        assert report.has_regressions
        bottleneck_regression = next(
            (r for r in report.high_regressions if r.metric == "bottlenecks"),
            None
        )
        assert bottleneck_regression is not None
        assert "model_forward" in bottleneck_regression.description

    def test_performance_improvement(self, detector, baseline_result):
        """Test detection of performance improvement."""
        current = create_baseline_result()
        current.performance.gpu_utilization = 95.0  # ~11.8% increase

        report = detector.detect_regressions(baseline_result, current)

        improvement = next(
            (i for i in report.improvements if i.metric == "gpu_utilization"),
            None
        )
        assert improvement is not None
        assert improvement.change_percentage > 0


class TestSecurityRegressions:
    """Test security regression detection."""

    def test_new_vulnerability_detection(self, detector, baseline_result):
        """Test detection of new security vulnerabilities."""
        current = create_baseline_result()
        current.security.vulnerabilities.append({"type": "xss", "severity": "critical"})

        report = detector.detect_regressions(baseline_result, current)

        assert report.has_regressions
        assert len(report.critical_regressions) >= 1
        vuln_regression = next(
            (r for r in report.critical_regressions if r.metric == "vulnerabilities"),
            None
        )
        assert vuln_regression is not None
        assert vuln_regression.type == RegressionType.SECURITY

    def test_new_cve_detection(self, detector, baseline_result):
        """Test detection of new CVEs in dependencies."""
        current = create_baseline_result()
        current.dependencies.vulnerabilities.append({"cve_id": "CVE-2024-0001"})

        report = detector.detect_regressions(baseline_result, current)

        assert report.has_regressions
        cve_regression = next(
            (r for r in report.critical_regressions if r.metric == "dependency_cves"),
            None
        )
        assert cve_regression is not None
        assert "CVE-2024-0001" in cve_regression.description

    def test_hardcoded_secrets_detection(self, detector, baseline_result):
        """Test detection of new hardcoded secrets."""
        current = create_baseline_result()
        current.security.hardcoded_secrets.append({
            "type": "api_key",
            "severity": "critical",
            "file": "config.py",
            "line": 42,
            "description": "API key found"
        })

        report = detector.detect_regressions(baseline_result, current)

        assert report.has_regressions
        secret_regression = next(
            (r for r in report.critical_regressions if r.metric == "hardcoded_secrets"),
            None
        )
        assert secret_regression is not None
        assert secret_regression.severity == RegressionSeverity.CRITICAL

    def test_security_improvement(self, detector, baseline_result):
        """Test detection of security improvements."""
        current = create_baseline_result()
        current.security.vulnerabilities = []  # Fixed vulnerability

        report = detector.detect_regressions(baseline_result, current)

        improvement = next(
            (i for i in report.improvements if i.metric == "vulnerabilities"),
            None
        )
        assert improvement is not None


class TestCodeQualityRegressions:
    """Test code quality regression detection."""

    def test_complexity_regression(self, detector, baseline_result):
        """Test detection of complexity regression (>20% increase)."""
        current = create_baseline_result()
        current.code_quality.average_complexity = 6.5  # 30% increase

        report = detector.detect_regressions(baseline_result, current)

        assert report.has_regressions
        complexity_regression = next(
            (r for r in report.medium_regressions if r.metric == "average_complexity"),
            None
        )
        assert complexity_regression is not None
        assert complexity_regression.type == RegressionType.CODE_QUALITY

    def test_duplication_regression(self, detector, baseline_result):
        """Test detection of code duplication regression (>5% increase)."""
        current = create_baseline_result()
        current.code_quality.duplication_percentage = 16.0  # 6% increase

        report = detector.detect_regressions(baseline_result, current)

        assert report.has_regressions
        dup_regression = next(
            (r for r in report.medium_regressions if r.metric == "duplication_percentage"),
            None
        )
        assert dup_regression is not None
        assert dup_regression.severity == RegressionSeverity.MEDIUM

    def test_code_quality_improvement(self, detector, baseline_result):
        """Test detection of code quality improvements."""
        current = create_baseline_result()
        current.code_quality.average_complexity = 3.5  # 30% decrease
        current.code_quality.duplication_percentage = 4.0  # 6% decrease

        report = detector.detect_regressions(baseline_result, current)

        assert len(report.improvements) >= 2


class TestRegressionReport:
    """Test RegressionReport functionality."""

    def test_should_fail_ci_with_critical_regressions(self):
        """Test CI failure logic with critical regressions."""
        report = RegressionReport(
            has_regressions=True,
            critical_regressions=[
                Regression(
                    type=RegressionType.COVERAGE,
                    severity=RegressionSeverity.CRITICAL,
                    metric="line_coverage",
                    baseline_value=70.0,
                    current_value=67.0,
                    change_percentage=-3.0,
                    description="Coverage decreased"
                )
            ]
        )

        assert report.should_fail_ci()

    def test_should_not_fail_ci_without_critical_regressions(self):
        """Test CI passes without critical regressions."""
        report = RegressionReport(
            has_regressions=True,
            high_regressions=[
                Regression(
                    type=RegressionType.COVERAGE,
                    severity=RegressionSeverity.HIGH,
                    metric="line_coverage",
                    baseline_value=70.0,
                    current_value=69.0,
                    change_percentage=-1.0,
                    description="Minor coverage decrease"
                )
            ]
        )

        assert not report.should_fail_ci()

    def test_get_all_regressions(self):
        """Test getting all regressions sorted by severity."""
        report = RegressionReport(
            has_regressions=True,
            critical_regressions=[
                Regression(
                    type=RegressionType.SECURITY,
                    severity=RegressionSeverity.CRITICAL,
                    metric="vulnerabilities",
                    baseline_value=0,
                    current_value=1,
                    change_percentage=0,
                    description="New vulnerability"
                )
            ],
            high_regressions=[
                Regression(
                    type=RegressionType.PERFORMANCE,
                    severity=RegressionSeverity.HIGH,
                    metric="memory",
                    baseline_value=8.0,
                    current_value=9.0,
                    change_percentage=12.5,
                    description="Memory increase"
                )
            ]
        )

        all_regressions = report.get_all_regressions()
        assert len(all_regressions) == 2
        assert all_regressions[0].severity == RegressionSeverity.CRITICAL
        assert all_regressions[1].severity == RegressionSeverity.HIGH


class TestDiffReportGeneration:
    """Test diff report generation."""

    def test_generate_diff_report(self, detector, baseline_result):
        """Test generation of side-by-side comparison report."""
        current = create_baseline_result()
        current.coverage.line_coverage = 67.0
        current.performance.gpu_utilization = 75.0

        report = detector.detect_regressions(baseline_result, current)
        diff_report = detector.generate_diff_report(baseline_result, current, report)

        assert "Regression Analysis Report" in diff_report
        assert "Coverage Metrics" in diff_report
        assert "Performance Metrics" in diff_report
        assert "67.0%" in diff_report  # Current coverage
        assert "75.0%" in diff_report  # Current GPU utilization

    def test_diff_report_includes_regressions(self, detector, baseline_result):
        """Test diff report includes regression details."""
        current = create_baseline_result()
        current.coverage.line_coverage = 67.0

        report = detector.detect_regressions(baseline_result, current)
        diff_report = detector.generate_diff_report(baseline_result, current, report)

        assert "Regressions" in diff_report
        assert "CRITICAL" in diff_report or "HIGH" in diff_report


class TestCIIntegration:
    """Test CI/CD integration functionality."""

    def test_exit_code_for_critical_regression(self, detector, baseline_result):
        """Test exit code 1 for critical regressions."""
        current = create_baseline_result()
        current.coverage.line_coverage = 67.0  # Critical regression

        report = detector.detect_regressions(baseline_result, current)
        exit_code = detector.exit_code_for_ci(report)

        assert exit_code == 1

    def test_exit_code_for_no_regressions(self, detector, baseline_result):
        """Test exit code 0 for no regressions."""
        current = create_baseline_result()

        report = detector.detect_regressions(baseline_result, current)
        exit_code = detector.exit_code_for_ci(report)

        assert exit_code == 0

    def test_exit_code_for_non_critical_regressions(self, detector, baseline_result):
        """Test exit code 0 for non-critical regressions."""
        current = create_baseline_result()
        current.coverage.line_coverage = 69.0  # High severity, not critical

        report = detector.detect_regressions(baseline_result, current)
        exit_code = detector.exit_code_for_ci(report)

        assert exit_code == 0


class TestCustomThresholds:
    """Test custom threshold configuration."""

    def test_custom_coverage_threshold(self, baseline_result):
        """Test custom coverage threshold."""
        detector = RegressionDetector(coverage_threshold=5.0)  # More lenient

        current = create_baseline_result()
        current.coverage.line_coverage = 67.0  # 3% decrease

        report = detector.detect_regressions(baseline_result, current)

        # Should be HIGH, not CRITICAL with 5% threshold
        assert len(report.critical_regressions) == 0
        assert len(report.high_regressions) >= 1

    def test_custom_performance_threshold(self, baseline_result):
        """Test custom performance threshold."""
        detector = RegressionDetector(performance_threshold=20.0)  # More lenient

        current = create_baseline_result()
        current.performance.gpu_utilization = 75.0  # ~11.8% decrease

        report = detector.detect_regressions(baseline_result, current)

        # Should not be flagged with 20% threshold
        gpu_regressions = [r for r in report.get_all_regressions() if r.metric == "gpu_utilization"]
        assert len(gpu_regressions) == 0


class TestSummaryGeneration:
    """Test summary generation."""

    def test_summary_with_critical_regressions(self, detector, baseline_result):
        """Test summary includes critical regressions."""
        current = create_baseline_result()
        current.coverage.line_coverage = 67.0

        report = detector.detect_regressions(baseline_result, current)

        assert "CRITICAL" in report.summary
        assert "coverage" in report.summary.lower()

    def test_summary_with_improvements(self, detector, baseline_result):
        """Test summary includes improvements."""
        current = create_baseline_result()
        current.coverage.line_coverage = 75.0

        report = detector.detect_regressions(baseline_result, current)

        assert "improvements" in report.summary.lower()

    def test_summary_with_no_changes(self, detector, baseline_result):
        """Test summary when no changes detected."""
        current = create_baseline_result()

        report = detector.detect_regressions(baseline_result, current)

        assert "No regressions" in report.summary or "stable" in report.summary.lower()
