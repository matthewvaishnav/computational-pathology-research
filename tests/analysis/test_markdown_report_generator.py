"""
Unit tests for Markdown Report Generator.

Tests report formatting, table generation, and task list formatting.
"""

import pytest

from src.analysis.models import (
    AnalysisResult,
    ArchitectureAnalysis,
    CodeQualityAnalysis,
    CoverageAnalysis,
    DependencyAnalysis,
    DeploymentAnalysis,
    Issue,
    PerformanceAnalysis,
    Priority,
    Role,
    ScalabilityAnalysis,
    SecurityAnalysis,
    Severity,
)
from src.analysis.reporting import ReportGenerator


@pytest.fixture
def generator():
    """Create report generator instance."""
    return ReportGenerator()


@pytest.fixture
def sample_result():
    """Create sample analysis result."""
    return AnalysisResult(
        timestamp="2024-01-01T00:00:00Z",
        project_path="/path/to/project",
        git_commit="abc123",
        architecture=ArchitectureAnalysis(
            total_files=100,
            large_files=[{"path": "src/large.py", "lines": 600, "complexity": 15.0}],
            circular_dependencies=[["module_a", "module_b", "module_a"]],
            coupling_metrics={},
            solid_violations=[],
            score=65.0,
        ),
        performance=PerformanceAnalysis(
            gpu_utilization=45.0,
            bottlenecks=[{"function": "slow_func", "time_ms": 1200, "file": "src/perf.py"}],
            flame_graph_path="/tmp/flame.svg",
            memory_usage_peak_gb=8.5,
            memory_usage_avg_gb=6.2,
            score=55.0,
        ),
        coverage=CoverageAnalysis(
            line_coverage=72.5,
            branch_coverage=65.0,
            untested_critical_paths=["src/critical.py"],
            missing_property_tests=["src/transform.py"],
            flaky_tests=["test_flaky"],
            slow_tests=[],
            score=70.0,
        ),
        code_quality=CodeQualityAnalysis(
            average_complexity=5.2,
            high_complexity_functions=[
                {"name": "complex_func", "complexity": 15, "file": "src/complex.py"}
            ],
            duplication_percentage=8.5,
            documentation_coverage=60.0,
            pylint_score=7.8,
            score=68.0,
            fix_suggestions=[],
        ),
        dependencies=DependencyAnalysis(
            total_dependencies=45,
            vulnerabilities=[{"package": "requests", "cve": "CVE-2023-1234", "severity": "high"}],
            outdated_packages=["numpy==1.20.0"],
            license_issues=[],
            unused_dependencies=[],
            redundant_dependencies=[],
            security_report={},
            score=75.0,
        ),
        deployment=DeploymentAnalysis(
            dockerfile_score=80.0,
            k8s_readiness=70.0,
            ci_cd_completeness=85.0,
            monitoring_score=60.0,
            score=73.75,
        ),
        security=SecurityAnalysis(
            vulnerabilities=[
                {"title": "SQL injection", "severity": "critical", "file": "src/db.py"}
            ],
            hipaa_compliance_score=65.0,
            hardcoded_secrets=["src/config.py:API_KEY"],
            injection_risks=[],
            tls_issues=[],
            score=60.0,
        ),
        scalability=ScalabilityAnalysis(
            ddp_correctness=True,
            scaling_efficiency="sub-linear",
            memory_bottlenecks=["large_tensor"],
            communication_overhead_ms=150.0,
            score=70.0,
            recommendations={},
        ),
        overall_score=66.32,
        critical_issues=[
            Issue(
                id="issue-001",
                dimension="security",
                severity=Severity.CRITICAL,
                category="injection",
                title="SQL injection vulnerability",
                description="Unsafe SQL query construction",
                file_path="src/db.py",
                line_number=42,
                recommendation="Use parameterized queries",
                effort_hours=4.0,
                priority=Priority.P0,
                role=Role.BACKEND,
            ),
            Issue(
                id="issue-002",
                dimension="architecture",
                severity=Severity.HIGH,
                category="solid",
                title="SRP violation",
                description="Class has too many responsibilities",
                file_path="src/models/user.py",
                line_number=10,
                recommendation="Split into smaller classes",
                effort_hours=8.0,
                priority=Priority.P1,
                role=Role.BACKEND,
            ),
        ],
    )


class TestReportGenerator:
    """Test suite for ReportGenerator."""

    def test_generate_markdown_basic(self, generator, sample_result):
        """Test basic Markdown report generation."""
        report = generator.generate_markdown(sample_result)

        assert isinstance(report, str)
        assert len(report) > 0

        # Check for main sections
        assert "# HistoCore Project Optimization Analysis Report" in report
        assert "## Executive Summary" in report
        assert "## Overall Metrics" in report
        assert "## Critical Issues" in report
        assert "## Architecture Analysis" in report
        assert "## Performance Analysis" in report
        assert "## Test Coverage Analysis" in report
        assert "## Code Quality Analysis" in report
        assert "## Dependencies Analysis" in report
        assert "## Deployment Analysis" in report
        assert "## Security Analysis" in report
        assert "## Scalability Analysis" in report
        assert "## Prioritized Task List" in report
        assert "## Recommendations" in report

    def test_header_generation(self, generator, sample_result):
        """Test report header formatting."""
        report = generator.generate_markdown(sample_result)

        # Check header content
        assert "**Generated:**" in report
        assert "**Project:** /path/to/project" in report
        assert "**Git Commit:** `abc123`" in report
        assert "**Overall Score:** 66.3/100" in report

    def test_executive_summary(self, generator, sample_result):
        """Test executive summary generation."""
        report = generator.generate_markdown(sample_result)

        # Check summary content
        assert "2 critical issues" in report
        assert "1 critical" in report
        assert "1 high-severity" in report
        assert "SQL injection vulnerability" in report

    def test_overall_metrics_table(self, generator, sample_result):
        """Test overall metrics table formatting."""
        report = generator.generate_markdown(sample_result)

        # Check table structure
        assert "| Dimension | Score | Status | Key Metric |" in report
        assert "|-----------|-------|--------|------------|" in report

        # Check dimension rows
        assert "| Architecture |" in report
        assert "| Performance |" in report
        assert "| Coverage |" in report
        assert "| Code Quality |" in report
        assert "| Dependencies |" in report
        assert "| Deployment |" in report
        assert "| Security |" in report
        assert "| Scalability |" in report

    def test_critical_issues_section(self, generator, sample_result):
        """Test critical issues formatting."""
        report = generator.generate_markdown(sample_result)

        # Check issue formatting
        assert "## Critical Issues (2)" in report
        assert "### 1. SQL injection vulnerability" in report
        assert "**File:** `src/db.py` (Line 42)" in report
        assert "**Priority:** P0" in report
        assert "**Severity:** critical" in report
        assert "**Effort:** 4.0 hours" in report
        assert "**Role:** backend" in report
        assert "**Recommendation:** Use parameterized queries" in report

    def test_architecture_section(self, generator, sample_result):
        """Test architecture section formatting."""
        report = generator.generate_markdown(sample_result)

        # Check architecture content
        assert "## Architecture Analysis" in report
        assert "**Score:** 65.0/100" in report
        assert "**Total Files:** 100" in report
        assert "**Large Files (>500 lines):** 1" in report
        assert "**Circular Dependencies:** 1" in report
        assert "`src/large.py` (600 lines, complexity: 15.0)" in report
        assert "module_a → module_b → module_a" in report

    def test_performance_section(self, generator, sample_result):
        """Test performance section formatting."""
        report = generator.generate_markdown(sample_result)

        # Check performance content
        assert "## Performance Analysis" in report
        assert "**GPU Utilization:** 45.0%" in report
        assert "**Memory Peak:** 8.5 GB" in report
        assert "**Memory Average:** 6.2 GB" in report
        assert "`slow_func` in `src/perf.py` (1200.0ms)" in report
        assert "**Flame Graph:** `/tmp/flame.svg`" in report

    def test_coverage_section(self, generator, sample_result):
        """Test coverage section formatting."""
        report = generator.generate_markdown(sample_result)

        # Check coverage content
        assert "## Test Coverage Analysis" in report
        assert "**Line Coverage:** 72.5%" in report
        assert "**Branch Coverage:** 65.0%" in report
        assert "**Untested Critical Paths:** 1" in report
        assert "`src/critical.py`" in report
        assert "**Functions Needing Property Tests:**" in report
        assert "`src/transform.py`" in report
        assert "**Flaky Tests:**" in report
        assert "`test_flaky`" in report

    def test_code_quality_section(self, generator, sample_result):
        """Test code quality section formatting."""
        report = generator.generate_markdown(sample_result)

        # Check code quality content
        assert "## Code Quality Analysis" in report
        assert "**Average Complexity:** 5.2" in report
        assert "**High Complexity Functions:** 1" in report
        assert "**Code Duplication:** 8.5%" in report
        assert "**Documentation Coverage:** 60.0%" in report
        assert "**PyLint Score:** 7.8/10" in report
        assert "`complex_func` in `src/complex.py` (complexity: 15.0)" in report

    def test_dependencies_section(self, generator, sample_result):
        """Test dependencies section formatting."""
        report = generator.generate_markdown(sample_result)

        # Check dependencies content
        assert "## Dependencies Analysis" in report
        assert "**Total Dependencies:** 45" in report
        assert "**Security Vulnerabilities:** 1" in report
        assert "**Outdated Packages:** 1" in report
        assert "`requests` - high (CVE-2023-1234)" in report
        assert "`numpy==1.20.0`" in report

    def test_deployment_section(self, generator, sample_result):
        """Test deployment section formatting."""
        report = generator.generate_markdown(sample_result)

        # Check deployment content
        assert "## Deployment Analysis" in report
        assert "**Dockerfile Score:** 80.0/100" in report
        assert "**Kubernetes Readiness:** 70.0/100" in report
        assert "**CI/CD Completeness:** 85.0/100" in report
        assert "**Monitoring Score:** 60.0/100" in report

    def test_security_section(self, generator, sample_result):
        """Test security section formatting."""
        report = generator.generate_markdown(sample_result)

        # Check security content
        assert "## Security Analysis" in report
        assert "**Security Vulnerabilities:** 1" in report
        assert "**HIPAA Compliance Score:** 65.0/100" in report
        assert "**Hardcoded Secrets:** 1" in report
        assert "`SQL injection` in `src/db.py` (critical)" in report
        assert "`src/config.py:API_KEY`" in report

    def test_scalability_section(self, generator, sample_result):
        """Test scalability section formatting."""
        report = generator.generate_markdown(sample_result)

        # Check scalability content
        assert "## Scalability Analysis" in report
        assert "**DDP Implementation:** ✓ Correct" in report
        assert "**Scaling Efficiency:** sub-linear" in report
        assert "**Memory Bottlenecks:** 1" in report
        assert "**Communication Overhead:** 150.0ms" in report

    def test_task_list_formatting(self, generator, sample_result):
        """Test prioritized task list formatting."""
        report = generator.generate_markdown(sample_result)

        # Check task list structure
        assert "## Prioritized Task List" in report
        assert "### P0 - Critical (Immediate Action Required)" in report
        assert "### P1 - High Priority (This Sprint)" in report

        # Check P0 task
        assert "1. **SQL injection vulnerability** (4.0h, backend)" in report
        assert "Use parameterized queries" in report

        # Check P1 task
        assert "1. **SRP violation** (8.0h, backend)" in report
        assert "Split into smaller classes" in report

    def test_recommendations_section(self, generator, sample_result):
        """Test recommendations generation."""
        report = generator.generate_markdown(sample_result)

        # Check recommendations
        assert "## Recommendations" in report
        assert "Medium Priority" in report or "High Priority" in report

    def test_footer_generation(self, generator, sample_result):
        """Test report footer."""
        report = generator.generate_markdown(sample_result)

        # Check footer content
        assert "**Report generated by HistoCore Project Optimization Analysis System**" in report
        assert "**Timestamp:** 2024-01-01T00:00:00Z" in report

    def test_score_emoji(self, generator):
        """Test score emoji selection."""
        assert generator._get_score_emoji(90.0) == "🟢"
        assert generator._get_score_emoji(80.0) == "🟢"
        assert generator._get_score_emoji(70.0) == "🟡"
        assert generator._get_score_emoji(60.0) == "🟡"
        assert generator._get_score_emoji(50.0) == "🟠"
        assert generator._get_score_emoji(40.0) == "🟠"
        assert generator._get_score_emoji(30.0) == "🔴"
        assert generator._get_score_emoji(0.0) == "🔴"

    def test_empty_critical_issues(self, generator, sample_result):
        """Test report with no critical issues."""
        sample_result.critical_issues = []
        report = generator.generate_markdown(sample_result)

        assert "## Critical Issues" in report
        assert "✅ No critical issues found!" in report

    def test_empty_large_files(self, generator, sample_result):
        """Test report with no large files."""
        sample_result.architecture.large_files = []
        report = generator.generate_markdown(sample_result)

        assert "✅ No large files detected" in report

    def test_empty_circular_dependencies(self, generator, sample_result):
        """Test report with no circular dependencies."""
        sample_result.architecture.circular_dependencies = []
        report = generator.generate_markdown(sample_result)

        assert "✅ No circular dependencies detected" in report

    def test_empty_bottlenecks(self, generator, sample_result):
        """Test report with no performance bottlenecks."""
        sample_result.performance.bottlenecks = []
        report = generator.generate_markdown(sample_result)

        assert "✅ No significant bottlenecks detected" in report

    def test_empty_coverage_gaps(self, generator, sample_result):
        """Test report with no coverage gaps."""
        sample_result.coverage.untested_critical_paths = []
        sample_result.coverage.missing_property_tests = []
        sample_result.coverage.flaky_tests = []
        report = generator.generate_markdown(sample_result)

        assert "✅ No significant coverage gaps detected" in report

    def test_empty_high_complexity_functions(self, generator, sample_result):
        """Test report with no high complexity functions."""
        sample_result.code_quality.high_complexity_functions = []
        report = generator.generate_markdown(sample_result)

        assert "✅ No high complexity functions detected" in report

    def test_empty_vulnerabilities(self, generator, sample_result):
        """Test report with no security vulnerabilities."""
        sample_result.dependencies.vulnerabilities = []
        report = generator.generate_markdown(sample_result)

        assert "✅ No security vulnerabilities detected" in report

    def test_empty_security_issues(self, generator, sample_result):
        """Test report with no security issues."""
        sample_result.security.vulnerabilities = []
        sample_result.security.hardcoded_secrets = []
        report = generator.generate_markdown(sample_result)

        assert "✅ No major security issues detected" in report

    def test_key_metric_extraction(self, generator):
        """Test key metric extraction for each dimension."""
        assert "large files" in generator._get_key_metric("architecture", {"large_files_count": 5})
        assert "GPU utilization" in generator._get_key_metric(
            "performance", {"gpu_utilization": 45.0}
        )
        assert "line coverage" in generator._get_key_metric("coverage", {"line_coverage": 72.5})
        assert "avg complexity" in generator._get_key_metric(
            "code_quality", {"average_complexity": 5.2}
        )
        assert "vulnerabilities" in generator._get_key_metric(
            "dependencies", {"vulnerabilities_count": 3}
        )
        assert "CI/CD complete" in generator._get_key_metric(
            "deployment", {"ci_cd_completeness": 85.0}
        )
        assert "security issues" in generator._get_key_metric(
            "security", {"vulnerabilities_count": 2}
        )
        assert "DDP: ✓" in generator._get_key_metric("scalability", {"ddp_correctness": True})
        assert "DDP: ✗" in generator._get_key_metric("scalability", {"ddp_correctness": False})

    def test_markdown_special_characters(self, generator, sample_result):
        """Test handling of special Markdown characters."""
        # Add issue with special characters
        sample_result.critical_issues.append(
            Issue(
                id="issue-003",
                dimension="code_quality",
                severity=Severity.MEDIUM,
                category="naming",
                title="Variable name with * and _ characters",
                description="Description with **bold** and *italic*",
                file_path="src/test_*.py",
                recommendation="Use proper naming",
                priority=Priority.P2,
                role=Role.BACKEND,
            )
        )

        report = generator.generate_markdown(sample_result)

        # Should not break Markdown formatting
        assert "Variable name with * and _ characters" in report
        assert "src/test_*.py" in report

    def test_long_file_paths(self, generator, sample_result):
        """Test handling of long file paths."""
        sample_result.architecture.large_files = [
            {
                "path": "src/very/long/path/to/some/deeply/nested/module/file.py",
                "lines": 600,
                "complexity": 15.0,
            }
        ]

        report = generator.generate_markdown(sample_result)

        # Should include full path
        assert "src/very/long/path/to/some/deeply/nested/module/file.py" in report

    def test_multiple_circular_dependencies(self, generator, sample_result):
        """Test formatting of multiple circular dependencies."""
        sample_result.architecture.circular_dependencies = [
            ["a", "b", "c", "a"],
            ["x", "y", "x"],
            ["p", "q", "r", "s", "p"],
        ]

        report = generator.generate_markdown(sample_result)

        # Should show all cycles
        assert "a → b → c → a" in report
        assert "x → y → x" in report
        assert "p → q → r → s → p" in report

    def test_report_consistency(self, generator, sample_result):
        """Test that multiple generations produce consistent output."""
        report1 = generator.generate_markdown(sample_result)
        report2 = generator.generate_markdown(sample_result)

        assert report1 == report2
