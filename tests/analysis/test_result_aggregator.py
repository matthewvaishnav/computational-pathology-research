"""
Unit tests for Result Aggregator.

Tests merging, deduplication, scoring, and summary generation.
"""

import pytest
from src.analysis.aggregator import ResultAggregator
from src.analysis.models import (
    AnalysisResult,
    Issue,
    Severity,
    Priority,
    Role,
    ArchitectureAnalysis,
    PerformanceAnalysis,
    CoverageAnalysis,
    CodeQualityAnalysis,
    DependencyAnalysis,
    DeploymentAnalysis,
    SecurityAnalysis,
    ScalabilityAnalysis,
)


@pytest.fixture
def aggregator():
    """Create aggregator instance."""
    return ResultAggregator()


@pytest.fixture
def sample_architecture():
    """Sample architecture analysis."""
    return ArchitectureAnalysis(
        total_files=100,
        large_files=[
            {"file": "src/models/large.py", "lines": 600},
            {"file": "src/utils/big.py", "lines": 550},
        ],
        circular_dependencies=[["module_a", "module_b", "module_a"]],
        coupling_metrics={"high_coupling": ["src/core.py"]},
        solid_violations=[
            Issue(
                id="arch-001",
                dimension="architecture",
                severity=Severity.HIGH,
                category="solid",
                title="SRP violation in UserManager",
                description="Class has too many responsibilities",
                file_path="src/models/user.py",
                line_number=10,
                recommendation="Split into UserAuth and UserProfile",
                effort_hours=8.0,
                priority=Priority.P1,
                role=Role.BACKEND,
            )
        ],
        score=65.0,
    )


@pytest.fixture
def sample_performance():
    """Sample performance analysis."""
    return PerformanceAnalysis(
        gpu_utilization=45.0,
        bottlenecks=[{"function": "data_loader", "time_ms": 1200, "file": "src/data/loader.py"}],
        flame_graph_path="/tmp/flame.svg",
        memory_usage_peak_gb=8.5,
        memory_usage_avg_gb=6.2,
        score=55.0,
    )


@pytest.fixture
def sample_coverage():
    """Sample coverage analysis."""
    return CoverageAnalysis(
        line_coverage=72.5,
        branch_coverage=65.0,
        untested_critical_paths=["src/core/error_handler.py"],
        missing_property_tests=["src/utils/transform.py"],
        flaky_tests=["tests/test_integration.py::test_flaky"],
        slow_tests=[{"test": "test_slow", "duration_ms": 5000}],
        score=70.0,
    )


@pytest.fixture
def sample_code_quality():
    """Sample code quality analysis."""
    return CodeQualityAnalysis(
        average_complexity=5.2,
        high_complexity_functions=[
            {"function": "process_data", "complexity": 15, "file": "src/processor.py"}
        ],
        duplication_percentage=8.5,
        documentation_coverage=60.0,
        pylint_score=7.8,
        score=68.0,
        fix_suggestions=[{"type": "unused_import", "file": "src/main.py"}],
    )


@pytest.fixture
def sample_dependencies():
    """Sample dependency analysis."""
    return DependencyAnalysis(
        total_dependencies=45,
        vulnerabilities=[{"package": "requests", "cve": "CVE-2023-1234", "severity": "high"}],
        outdated_packages=["numpy==1.20.0"],
        license_issues=["gpl-package"],
        unused_dependencies=["unused-lib"],
        redundant_dependencies=["duplicate-lib"],
        security_report={"critical": 0, "high": 1, "medium": 2},
        score=75.0,
    )


@pytest.fixture
def sample_deployment():
    """Sample deployment analysis."""
    return DeploymentAnalysis(
        dockerfile_score=80.0,
        k8s_readiness=70.0,
        ci_cd_completeness=85.0,
        monitoring_score=60.0,
        score=73.75,
    )


@pytest.fixture
def sample_security():
    """Sample security analysis."""
    return SecurityAnalysis(
        vulnerabilities=[
            {"title": "SQL injection risk", "severity": "critical", "file": "src/db.py"}
        ],
        hipaa_compliance_score=65.0,
        hardcoded_secrets=["src/config.py:API_KEY"],
        injection_risks=[{"type": "sql", "file": "src/db.py", "line": 42}],
        tls_issues=[{"issue": "weak_cipher", "location": "src/network.py"}],
        score=60.0,
    )


@pytest.fixture
def sample_scalability():
    """Sample scalability analysis."""
    return ScalabilityAnalysis(
        ddp_correctness=True,
        scaling_efficiency="sub-linear",
        memory_bottlenecks=["large_tensor_allocation"],
        communication_overhead_ms=150.0,
        score=70.0,
        recommendations={"gpu_count": 4, "expected_speedup": 3.2},
    )


class TestResultAggregator:
    """Test suite for ResultAggregator."""

    def test_merge_results_basic(
        self,
        aggregator,
        sample_architecture,
        sample_performance,
        sample_coverage,
        sample_code_quality,
        sample_dependencies,
        sample_deployment,
        sample_security,
        sample_scalability,
    ):
        """Test basic result merging."""
        result = aggregator.merge_results(
            architecture=sample_architecture,
            performance=sample_performance,
            coverage=sample_coverage,
            code_quality=sample_code_quality,
            dependencies=sample_dependencies,
            deployment=sample_deployment,
            security=sample_security,
            scalability=sample_scalability,
            timestamp="2024-01-01T00:00:00Z",
            project_path="/path/to/project",
            git_commit="abc123",
        )

        assert isinstance(result, AnalysisResult)
        assert result.timestamp == "2024-01-01T00:00:00Z"
        assert result.project_path == "/path/to/project"
        assert result.git_commit == "abc123"
        assert result.architecture == sample_architecture
        assert result.performance == sample_performance
        assert result.coverage == sample_coverage
        assert result.code_quality == sample_code_quality
        assert result.dependencies == sample_dependencies
        assert result.deployment == sample_deployment
        assert result.security == sample_security
        assert result.scalability == sample_scalability

    def test_overall_score_calculation(
        self,
        aggregator,
        sample_architecture,
        sample_performance,
        sample_coverage,
        sample_code_quality,
        sample_dependencies,
        sample_deployment,
        sample_security,
        sample_scalability,
    ):
        """Test weighted overall score calculation."""
        result = aggregator.merge_results(
            architecture=sample_architecture,
            performance=sample_performance,
            coverage=sample_coverage,
            code_quality=sample_code_quality,
            dependencies=sample_dependencies,
            deployment=sample_deployment,
            security=sample_security,
            scalability=sample_scalability,
            timestamp="2024-01-01T00:00:00Z",
            project_path="/path/to/project",
            git_commit="abc123",
        )

        # Verify score is calculated
        assert 0 <= result.overall_score <= 100

        # Manual calculation with weights
        expected = (
            60.0 * 0.20  # security
            + 70.0 * 0.15  # coverage
            + 68.0 * 0.15  # code_quality
            + 65.0 * 0.15  # architecture
            + 55.0 * 0.10  # performance
            + 75.0 * 0.10  # dependencies
            + 73.75 * 0.10  # deployment
            + 70.0 * 0.05  # scalability
        )

        assert abs(result.overall_score - expected) < 0.1

    def test_critical_issues_extraction(
        self,
        aggregator,
        sample_architecture,
        sample_performance,
        sample_coverage,
        sample_code_quality,
        sample_dependencies,
        sample_deployment,
        sample_security,
        sample_scalability,
    ):
        """Test extraction of critical issues (P0 and P1)."""
        # Add more issues to architecture
        sample_architecture.solid_violations.extend(
            [
                Issue(
                    id="arch-002",
                    dimension="architecture",
                    severity=Severity.CRITICAL,
                    category="solid",
                    title="Critical violation",
                    description="Critical issue",
                    file_path="src/core.py",
                    priority=Priority.P0,
                    role=Role.BACKEND,
                ),
                Issue(
                    id="arch-003",
                    dimension="architecture",
                    severity=Severity.LOW,
                    category="solid",
                    title="Low priority issue",
                    description="Low priority",
                    file_path="src/utils.py",
                    priority=Priority.P3,
                    role=Role.BACKEND,
                ),
            ]
        )

        result = aggregator.merge_results(
            architecture=sample_architecture,
            performance=sample_performance,
            coverage=sample_coverage,
            code_quality=sample_code_quality,
            dependencies=sample_dependencies,
            deployment=sample_deployment,
            security=sample_security,
            scalability=sample_scalability,
            timestamp="2024-01-01T00:00:00Z",
            project_path="/path/to/project",
            git_commit="abc123",
        )

        # Should extract P0 and P1 issues only
        assert len(result.critical_issues) == 2
        assert all(issue.priority in [Priority.P0, Priority.P1] for issue in result.critical_issues)

        # P0 should come before P1
        assert result.critical_issues[0].priority == Priority.P0
        assert result.critical_issues[1].priority == Priority.P1

    def test_critical_issues_limit(
        self,
        aggregator,
        sample_architecture,
        sample_performance,
        sample_coverage,
        sample_code_quality,
        sample_dependencies,
        sample_deployment,
        sample_security,
        sample_scalability,
    ):
        """Test that critical issues are limited to top 10."""
        # Add 15 P0 issues
        sample_architecture.solid_violations = [
            Issue(
                id=f"arch-{i:03d}",
                dimension="architecture",
                severity=Severity.CRITICAL,
                category="solid",
                title=f"Issue {i}",
                description=f"Description {i}",
                file_path=f"src/file{i}.py",
                priority=Priority.P0,
                role=Role.BACKEND,
                effort_hours=float(i),
            )
            for i in range(15)
        ]

        result = aggregator.merge_results(
            architecture=sample_architecture,
            performance=sample_performance,
            coverage=sample_coverage,
            code_quality=sample_code_quality,
            dependencies=sample_dependencies,
            deployment=sample_deployment,
            security=sample_security,
            scalability=sample_scalability,
            timestamp="2024-01-01T00:00:00Z",
            project_path="/path/to/project",
            git_commit="abc123",
        )

        # Should be limited to 10
        assert len(result.critical_issues) == 10

    def test_deduplicate_issues(self, aggregator):
        """Test issue deduplication."""
        issues = [
            Issue(
                id="1",
                dimension="architecture",
                severity=Severity.HIGH,
                category="solid",
                title="Duplicate issue",
                description="Description",
                file_path="src/file.py",
                line_number=10,
                priority=Priority.P1,
                role=Role.BACKEND,
            ),
            Issue(
                id="2",
                dimension="architecture",
                severity=Severity.HIGH,
                category="solid",
                title="Duplicate issue",
                description="Different description",
                file_path="src/file.py",
                line_number=10,
                priority=Priority.P1,
                role=Role.BACKEND,
            ),
            Issue(
                id="3",
                dimension="architecture",
                severity=Severity.HIGH,
                category="solid",
                title="Different issue",
                description="Description",
                file_path="src/file.py",
                line_number=20,
                priority=Priority.P1,
                role=Role.BACKEND,
            ),
        ]

        deduplicated = aggregator._deduplicate_issues(issues)

        # Should remove one duplicate
        assert len(deduplicated) == 2
        assert deduplicated[0].id == "1"
        assert deduplicated[1].id == "3"

    def test_extract_critical_issues_sorting(self, aggregator):
        """Test critical issues are sorted by priority, severity, effort."""
        issues = [
            Issue(
                id="1",
                dimension="architecture",
                severity=Severity.MEDIUM,
                category="solid",
                title="P1 Medium",
                description="",
                file_path="src/file.py",
                priority=Priority.P1,
                role=Role.BACKEND,
                effort_hours=10.0,
            ),
            Issue(
                id="2",
                dimension="architecture",
                severity=Severity.CRITICAL,
                category="solid",
                title="P0 Critical",
                description="",
                file_path="src/file.py",
                priority=Priority.P0,
                role=Role.BACKEND,
                effort_hours=5.0,
            ),
            Issue(
                id="3",
                dimension="architecture",
                severity=Severity.HIGH,
                category="solid",
                title="P1 High",
                description="",
                file_path="src/file.py",
                priority=Priority.P1,
                role=Role.BACKEND,
                effort_hours=3.0,
            ),
            Issue(
                id="4",
                dimension="architecture",
                severity=Severity.LOW,
                category="solid",
                title="P2 Low",
                description="",
                file_path="src/file.py",
                priority=Priority.P2,
                role=Role.BACKEND,
                effort_hours=1.0,
            ),
        ]

        critical = aggregator._extract_critical_issues(issues)

        # Should only include P0 and P1
        assert len(critical) == 3

        # Should be sorted: P0 first, then P1 by severity
        assert critical[0].id == "2"  # P0 Critical
        assert critical[1].id == "3"  # P1 High
        assert critical[2].id == "1"  # P1 Medium

    def test_calculate_overall_score_missing_dimensions(self, aggregator):
        """Test score calculation when some dimensions are missing."""
        results = {
            "architecture": ArchitectureAnalysis(score=80.0),
            "performance": PerformanceAnalysis(score=60.0),
            "coverage": CoverageAnalysis(score=70.0),
            # Missing other dimensions
        }

        score = aggregator._calculate_overall_score(results)

        # Should normalize by actual weights used
        assert 0 <= score <= 100

        # Manual calculation
        expected = (
            70.0 * 0.15 + 80.0 * 0.15 + 60.0 * 0.10  # coverage  # architecture  # performance
        ) / (0.15 + 0.15 + 0.10)

        assert abs(score - expected) < 0.1

    def test_get_dimension_summary(
        self,
        aggregator,
        sample_architecture,
        sample_performance,
        sample_coverage,
        sample_code_quality,
        sample_dependencies,
        sample_deployment,
        sample_security,
        sample_scalability,
    ):
        """Test dimension summary generation."""
        result = aggregator.merge_results(
            architecture=sample_architecture,
            performance=sample_performance,
            coverage=sample_coverage,
            code_quality=sample_code_quality,
            dependencies=sample_dependencies,
            deployment=sample_deployment,
            security=sample_security,
            scalability=sample_scalability,
            timestamp="2024-01-01T00:00:00Z",
            project_path="/path/to/project",
            git_commit="abc123",
        )

        summary = aggregator.get_dimension_summary(result)

        # Verify all dimensions present
        assert "architecture" in summary
        assert "performance" in summary
        assert "coverage" in summary
        assert "code_quality" in summary
        assert "dependencies" in summary
        assert "deployment" in summary
        assert "security" in summary
        assert "scalability" in summary

        # Verify architecture summary
        arch_summary = summary["architecture"]
        assert arch_summary["score"] == 65.0
        assert arch_summary["total_files"] == 100
        assert arch_summary["large_files_count"] == 2
        assert arch_summary["circular_dependencies_count"] == 1
        assert arch_summary["solid_violations_count"] == 1
        assert arch_summary["status"] == "good"

        # Verify performance summary
        perf_summary = summary["performance"]
        assert perf_summary["score"] == 55.0
        assert perf_summary["gpu_utilization"] == 45.0
        assert perf_summary["bottlenecks_count"] == 1
        assert perf_summary["memory_peak_gb"] == 8.5
        assert perf_summary["status"] == "needs_improvement"

    def test_get_status_from_score(self, aggregator):
        """Test status classification from score."""
        assert aggregator._get_status_from_score(90.0) == "excellent"
        assert aggregator._get_status_from_score(80.0) == "excellent"
        assert aggregator._get_status_from_score(70.0) == "good"
        assert aggregator._get_status_from_score(60.0) == "good"
        assert aggregator._get_status_from_score(50.0) == "needs_improvement"
        assert aggregator._get_status_from_score(40.0) == "needs_improvement"
        assert aggregator._get_status_from_score(30.0) == "critical"
        assert aggregator._get_status_from_score(0.0) == "critical"

    def test_get_top_issues_by_dimension(
        self,
        aggregator,
        sample_architecture,
        sample_performance,
        sample_coverage,
        sample_code_quality,
        sample_dependencies,
        sample_deployment,
        sample_security,
        sample_scalability,
    ):
        """Test extraction of top issues per dimension."""
        result = aggregator.merge_results(
            architecture=sample_architecture,
            performance=sample_performance,
            coverage=sample_coverage,
            code_quality=sample_code_quality,
            dependencies=sample_dependencies,
            deployment=sample_deployment,
            security=sample_security,
            scalability=sample_scalability,
            timestamp="2024-01-01T00:00:00Z",
            project_path="/path/to/project",
            git_commit="abc123",
        )

        top_issues = aggregator.get_top_issues_by_dimension(result, limit=3)

        # Verify all dimensions present
        assert "architecture" in top_issues
        assert "performance" in top_issues
        assert "coverage" in top_issues
        assert "security" in top_issues

        # Verify architecture issues
        arch_issues = top_issues["architecture"]
        assert len(arch_issues) == 1
        assert arch_issues[0]["title"] == "SRP violation in UserManager"
        assert arch_issues[0]["severity"] == "high"

        # Verify performance issues
        perf_issues = top_issues["performance"]
        assert len(perf_issues) == 1
        assert "data_loader" in perf_issues[0]["title"]

        # Verify coverage issues
        cov_issues = top_issues["coverage"]
        assert len(cov_issues) == 1
        assert "error_handler.py" in cov_issues[0]["title"]

        # Verify security issues
        sec_issues = top_issues["security"]
        assert len(sec_issues) == 1
        assert sec_issues[0]["title"] == "SQL injection risk"

    def test_empty_results(self, aggregator):
        """Test handling of empty analysis results."""
        result = aggregator.merge_results(
            architecture=ArchitectureAnalysis(),
            performance=PerformanceAnalysis(),
            coverage=CoverageAnalysis(),
            code_quality=CodeQualityAnalysis(),
            dependencies=DependencyAnalysis(),
            deployment=DeploymentAnalysis(),
            security=SecurityAnalysis(),
            scalability=ScalabilityAnalysis(),
            timestamp="2024-01-01T00:00:00Z",
            project_path="/path/to/project",
            git_commit="abc123",
        )

        assert result.overall_score == 0.0
        assert len(result.critical_issues) == 0

    def test_dimension_weights_sum(self, aggregator):
        """Test that dimension weights sum to 1.0."""
        total_weight = sum(aggregator.dimension_weights.values())
        assert abs(total_weight - 1.0) < 0.001
