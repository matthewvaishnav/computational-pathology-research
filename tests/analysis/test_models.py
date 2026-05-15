"""
Unit tests for data models in the HistoCore Project Optimization Analysis System.

Tests dataclass initialization, validation, and edge cases for all data models.
Requirements: 11.1, 11.2
"""

import pytest
from datetime import datetime
from dataclasses import FrozenInstanceError

from src.analysis.models import (
    AnalysisResult,
    ArchitectureAnalysis,
    PerformanceAnalysis,
    CoverageAnalysis,
    CodeQualityAnalysis,
    DependencyAnalysis,
    DeploymentAnalysis,
    SecurityAnalysis,
    ScalabilityAnalysis,
    Issue,
    Task,
    OptimizationPlan,
    Severity,
    Priority,
    Role,
)


class TestEnums:
    """Test enum classes for proper values and behavior."""

    def test_severity_enum_values(self):
        """Test Severity enum has correct values."""
        assert Severity.CRITICAL == "critical"
        assert Severity.HIGH == "high"
        assert Severity.MEDIUM == "medium"
        assert Severity.LOW == "low"

        # Test all values are strings
        for severity in Severity:
            assert isinstance(severity.value, str)

    def test_priority_enum_values(self):
        """Test Priority enum has correct values."""
        assert Priority.P0 == "P0"
        assert Priority.P1 == "P1"
        assert Priority.P2 == "P2"
        assert Priority.P3 == "P3"

        # Test all values are strings
        for priority in Priority:
            assert isinstance(priority.value, str)

    def test_role_enum_values(self):
        """Test Role enum has correct values."""
        assert Role.BACKEND == "backend"
        assert Role.ML == "ml"
        assert Role.DEVOPS == "devops"
        assert Role.SECURITY == "security"
        assert Role.QA == "qa"

        # Test all values are strings
        for role in Role:
            assert isinstance(role.value, str)


class TestIssue:
    """Test Issue dataclass initialization and methods."""

    def test_issue_initialization_required_fields(self):
        """Test Issue can be initialized with required fields."""
        issue = Issue(
            id="test-001",
            dimension="architecture",
            severity=Severity.HIGH,
            category="complexity",
            title="High complexity function",
            description="Function has cyclomatic complexity > 10",
            file_path="src/models.py",
        )

        assert issue.id == "test-001"
        assert issue.dimension == "architecture"
        assert issue.severity == Severity.HIGH
        assert issue.category == "complexity"
        assert issue.title == "High complexity function"
        assert issue.description == "Function has cyclomatic complexity > 10"
        assert issue.file_path == "src/models.py"

        # Test default values
        assert issue.line_number is None
        assert issue.recommendation == ""
        assert issue.effort_hours == 0.0
        assert issue.priority == Priority.P2
        assert issue.role == Role.BACKEND
        assert issue.references == []

    def test_issue_initialization_all_fields(self):
        """Test Issue initialization with all fields."""
        issue = Issue(
            id="test-002",
            dimension="security",
            severity=Severity.CRITICAL,
            category="injection",
            title="SQL injection vulnerability",
            description="Unsafe SQL query construction",
            file_path="src/database.py",
            line_number=42,
            recommendation="Use parameterized queries",
            effort_hours=4.5,
            priority=Priority.P0,
            role=Role.SECURITY,
            references=["https://owasp.org/sql-injection"],
        )

        assert issue.line_number == 42
        assert issue.recommendation == "Use parameterized queries"
        assert issue.effort_hours == 4.5
        assert issue.priority == Priority.P0
        assert issue.role == Role.SECURITY
        assert issue.references == ["https://owasp.org/sql-injection"]

    def test_issue_to_dict(self):
        """Test Issue.to_dict() method."""
        issue = Issue(
            id="test-003",
            dimension="performance",
            severity=Severity.MEDIUM,
            category="bottleneck",
            title="Slow data loading",
            description="DataLoader is CPU-bound",
            file_path="src/data.py",
            priority=Priority.P1,
            role=Role.ML,
        )

        result = issue.to_dict()

        assert isinstance(result, dict)
        assert result["id"] == "test-003"
        assert result["dimension"] == "performance"
        assert result["severity"] == "medium"  # Enum converted to string
        assert result["priority"] == "P1"  # Enum converted to string
        assert result["role"] == "ml"  # Enum converted to string
        assert result["category"] == "bottleneck"
        assert result["title"] == "Slow data loading"
        assert result["file_path"] == "src/data.py"

    def test_issue_from_dict(self):
        """Test Issue.from_dict() class method."""
        data = {
            "id": "test-004",
            "dimension": "coverage",
            "severity": "low",
            "category": "missing_test",
            "title": "Missing unit test",
            "description": "Function lacks test coverage",
            "file_path": "src/utils.py",
            "line_number": 15,
            "recommendation": "Add unit test",
            "effort_hours": 2.0,
            "priority": "P3",
            "role": "qa",
            "references": ["https://pytest.org"],
        }

        issue = Issue.from_dict(data)

        assert issue.id == "test-004"
        assert issue.dimension == "coverage"
        assert issue.severity == Severity.LOW
        assert issue.priority == Priority.P3
        assert issue.role == Role.QA
        assert issue.line_number == 15
        assert issue.effort_hours == 2.0
        assert issue.references == ["https://pytest.org"]

    def test_issue_from_dict_invalid_enum(self):
        """Test Issue.from_dict() with invalid enum values."""
        data = {
            "id": "test-005",
            "dimension": "test",
            "severity": "invalid_severity",  # Invalid
            "category": "test",
            "title": "Test",
            "description": "Test",
            "file_path": "test.py",
        }

        with pytest.raises(ValueError):
            Issue.from_dict(data)


class TestAnalysisDataclasses:
    """Test analysis result dataclasses."""

    def test_architecture_analysis_defaults(self):
        """Test ArchitectureAnalysis default initialization."""
        analysis = ArchitectureAnalysis()

        assert analysis.total_files == 0
        assert analysis.large_files == []
        assert analysis.circular_dependencies == []
        assert analysis.coupling_metrics == {}
        assert analysis.solid_violations == []
        assert analysis.score == 0.0

    def test_architecture_analysis_with_data(self):
        """Test ArchitectureAnalysis with actual data."""
        large_files = [{"path": "src/large.py", "lines": 1000, "complexity": 25.5}]
        circular_deps = [["module_a", "module_b", "module_a"]]
        coupling = {"fan_in": 5, "fan_out": 12}
        violations = [
            Issue(
                id="arch-001",
                dimension="architecture",
                severity=Severity.HIGH,
                category="srp_violation",
                title="Single Responsibility Principle violation",
                description="Class has too many responsibilities",
                file_path="src/large_class.py",
            )
        ]

        analysis = ArchitectureAnalysis(
            total_files=100,
            large_files=large_files,
            circular_dependencies=circular_deps,
            coupling_metrics=coupling,
            solid_violations=violations,
            score=75.5,
        )

        assert analysis.total_files == 100
        assert analysis.large_files == large_files
        assert analysis.circular_dependencies == circular_deps
        assert analysis.coupling_metrics == coupling
        assert len(analysis.solid_violations) == 1
        assert analysis.solid_violations[0].id == "arch-001"
        assert analysis.score == 75.5

    def test_performance_analysis_defaults(self):
        """Test PerformanceAnalysis default initialization."""
        analysis = PerformanceAnalysis()

        assert analysis.gpu_utilization == 0.0
        assert analysis.bottlenecks == []
        assert analysis.flame_graph_path == ""
        assert analysis.memory_usage_peak_gb == 0.0
        assert analysis.memory_usage_avg_gb == 0.0
        assert analysis.score == 0.0

    def test_performance_analysis_with_data(self):
        """Test PerformanceAnalysis with actual data."""
        bottlenecks = [{"operation": "data_loading", "time_ms": 150.5, "percentage": 25.0}]

        analysis = PerformanceAnalysis(
            gpu_utilization=85.5,
            bottlenecks=bottlenecks,
            flame_graph_path="/tmp/flamegraph.svg",
            memory_usage_peak_gb=12.8,
            memory_usage_avg_gb=8.4,
            score=82.0,
        )

        assert analysis.gpu_utilization == 85.5
        assert analysis.bottlenecks == bottlenecks
        assert analysis.flame_graph_path == "/tmp/flamegraph.svg"
        assert analysis.memory_usage_peak_gb == 12.8
        assert analysis.memory_usage_avg_gb == 8.4
        assert analysis.score == 82.0

    def test_coverage_analysis_defaults(self):
        """Test CoverageAnalysis default initialization."""
        analysis = CoverageAnalysis()

        assert analysis.line_coverage == 0.0
        assert analysis.branch_coverage == 0.0
        assert analysis.untested_critical_paths == []
        assert analysis.missing_property_tests == []
        assert analysis.flaky_tests == []
        assert analysis.slow_tests == []
        assert analysis.score == 0.0

    def test_coverage_analysis_with_data(self):
        """Test CoverageAnalysis with actual data."""
        critical_paths = ["src/error_handler.py:handle_exception"]
        property_tests = ["src/transforms.py:normalize_data"]
        flaky_tests = ["tests/test_integration.py::test_flaky"]
        slow_tests = [{"name": "test_slow", "duration_ms": 5500.0}]

        analysis = CoverageAnalysis(
            line_coverage=78.5,
            branch_coverage=65.2,
            untested_critical_paths=critical_paths,
            missing_property_tests=property_tests,
            flaky_tests=flaky_tests,
            slow_tests=slow_tests,
            score=71.8,
        )

        assert analysis.line_coverage == 78.5
        assert analysis.branch_coverage == 65.2
        assert analysis.untested_critical_paths == critical_paths
        assert analysis.missing_property_tests == property_tests
        assert analysis.flaky_tests == flaky_tests
        assert analysis.slow_tests == slow_tests
        assert analysis.score == 71.8

    def test_code_quality_analysis_defaults(self):
        """Test CodeQualityAnalysis default initialization."""
        analysis = CodeQualityAnalysis()

        assert analysis.average_complexity == 0.0
        assert analysis.high_complexity_functions == []
        assert analysis.duplication_percentage == 0.0
        assert analysis.documentation_coverage == 0.0
        assert analysis.pylint_score == 0.0
        assert analysis.score == 0.0

    def test_dependency_analysis_defaults(self):
        """Test DependencyAnalysis default initialization."""
        analysis = DependencyAnalysis()

        assert analysis.total_dependencies == 0
        assert analysis.vulnerabilities == []
        assert analysis.outdated_packages == []
        assert analysis.license_issues == []
        assert analysis.score == 0.0

    def test_deployment_analysis_defaults(self):
        """Test DeploymentAnalysis default initialization."""
        analysis = DeploymentAnalysis()

        assert analysis.dockerfile_score == 0.0
        assert analysis.k8s_readiness == 0.0
        assert analysis.ci_cd_completeness == 0.0
        assert analysis.monitoring_score == 0.0
        assert analysis.score == 0.0

    def test_security_analysis_defaults(self):
        """Test SecurityAnalysis default initialization."""
        analysis = SecurityAnalysis()

        assert analysis.vulnerabilities == []
        assert analysis.hipaa_compliance_score == 0.0
        assert analysis.hardcoded_secrets == []
        assert analysis.injection_risks == []
        assert analysis.score == 0.0

    def test_scalability_analysis_defaults(self):
        """Test ScalabilityAnalysis default initialization."""
        analysis = ScalabilityAnalysis()

        assert analysis.ddp_correctness is False
        assert analysis.scaling_efficiency == "unknown"
        assert analysis.memory_bottlenecks == []
        assert analysis.communication_overhead_ms == 0.0
        assert analysis.score == 0.0

    def test_scalability_analysis_with_data(self):
        """Test ScalabilityAnalysis with actual data."""
        bottlenecks = ["gradient_sync", "all_reduce_overhead"]

        analysis = ScalabilityAnalysis(
            ddp_correctness=True,
            scaling_efficiency="linear",
            memory_bottlenecks=bottlenecks,
            communication_overhead_ms=45.2,
            score=88.5,
        )

        assert analysis.ddp_correctness is True
        assert analysis.scaling_efficiency == "linear"
        assert analysis.memory_bottlenecks == bottlenecks
        assert analysis.communication_overhead_ms == 45.2
        assert analysis.score == 88.5


class TestAnalysisResult:
    """Test AnalysisResult dataclass and methods."""

    def test_analysis_result_initialization_minimal(self):
        """Test AnalysisResult with minimal required fields."""
        timestamp = datetime.now().isoformat()

        result = AnalysisResult(
            timestamp=timestamp,
            project_path="/test/project",
            git_commit="a" * 40,
            architecture=ArchitectureAnalysis(),
            performance=PerformanceAnalysis(),
            coverage=CoverageAnalysis(),
            code_quality=CodeQualityAnalysis(),
            dependencies=DependencyAnalysis(),
            deployment=DeploymentAnalysis(),
            security=SecurityAnalysis(),
            scalability=ScalabilityAnalysis(),
        )

        assert result.timestamp == timestamp
        assert result.project_path == "/test/project"
        assert result.git_commit == "a" * 40
        assert result.overall_score == 0.0  # Default
        assert result.critical_issues == []  # Default

        # Verify all analysis objects are present
        assert isinstance(result.architecture, ArchitectureAnalysis)
        assert isinstance(result.performance, PerformanceAnalysis)
        assert isinstance(result.coverage, CoverageAnalysis)
        assert isinstance(result.code_quality, CodeQualityAnalysis)
        assert isinstance(result.dependencies, DependencyAnalysis)
        assert isinstance(result.deployment, DeploymentAnalysis)
        assert isinstance(result.security, SecurityAnalysis)
        assert isinstance(result.scalability, ScalabilityAnalysis)

    def test_analysis_result_initialization_complete(self):
        """Test AnalysisResult with all fields populated."""
        timestamp = datetime.now().isoformat()
        critical_issues = [
            Issue(
                id="critical-001",
                dimension="security",
                severity=Severity.CRITICAL,
                category="vulnerability",
                title="Critical security vulnerability",
                description="SQL injection in user input",
                file_path="src/auth.py",
                line_number=123,
                priority=Priority.P0,
                role=Role.SECURITY,
            )
        ]

        result = AnalysisResult(
            timestamp=timestamp,
            project_path="/test/project",
            git_commit="b" * 40,
            architecture=ArchitectureAnalysis(total_files=500, score=85.0),
            performance=PerformanceAnalysis(gpu_utilization=92.5, score=88.0),
            coverage=CoverageAnalysis(line_coverage=78.5, score=75.0),
            code_quality=CodeQualityAnalysis(average_complexity=8.2, score=82.0),
            dependencies=DependencyAnalysis(total_dependencies=45, score=90.0),
            deployment=DeploymentAnalysis(dockerfile_score=95.0, score=92.0),
            security=SecurityAnalysis(hipaa_compliance_score=85.0, score=80.0),
            scalability=ScalabilityAnalysis(ddp_correctness=True, score=87.0),
            overall_score=84.5,
            critical_issues=critical_issues,
        )

        assert result.overall_score == 84.5
        assert len(result.critical_issues) == 1
        assert result.critical_issues[0].id == "critical-001"
        assert result.architecture.total_files == 500
        assert result.performance.gpu_utilization == 92.5
        assert result.coverage.line_coverage == 78.5

    def test_analysis_result_get_json_schema(self):
        """Test AnalysisResult.get_json_schema() returns valid schema."""
        schema = AnalysisResult.get_json_schema()

        assert isinstance(schema, dict)
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert schema["type"] == "object"

        # Check required fields
        required_fields = [
            "timestamp",
            "project_path",
            "git_commit",
            "architecture",
            "performance",
            "coverage",
            "code_quality",
            "dependencies",
            "deployment",
            "security",
            "scalability",
            "overall_score",
        ]
        assert schema["required"] == required_fields

        # Check properties exist
        for field in required_fields:
            assert field in schema["properties"]

        # Check overall_score constraints
        score_prop = schema["properties"]["overall_score"]
        assert score_prop["type"] == "number"
        assert score_prop["minimum"] == 0
        assert score_prop["maximum"] == 100


class TestTask:
    """Test Task dataclass."""

    def test_task_initialization_required_fields(self):
        """Test Task initialization with required fields."""
        task = Task(
            id="task-001",
            title="Fix security vulnerability",
            description="Patch SQL injection in auth module",
            priority=Priority.P0,
            effort_hours=4.0,
            role=Role.SECURITY,
        )

        assert task.id == "task-001"
        assert task.title == "Fix security vulnerability"
        assert task.description == "Patch SQL injection in auth module"
        assert task.priority == Priority.P0
        assert task.effort_hours == 4.0
        assert task.role == Role.SECURITY

        # Test defaults
        assert task.dependencies == []
        assert task.success_criteria == ""
        assert task.implementation_guide == ""
        assert task.references == []

    def test_task_initialization_all_fields(self):
        """Test Task initialization with all fields."""
        task = Task(
            id="task-002",
            title="Improve test coverage",
            description="Add unit tests for data models",
            priority=Priority.P2,
            effort_hours=8.0,
            role=Role.QA,
            dependencies=["task-001"],
            success_criteria="Coverage > 80%",
            implementation_guide="Use pytest framework",
            references=["https://pytest.org"],
        )

        assert task.dependencies == ["task-001"]
        assert task.success_criteria == "Coverage > 80%"
        assert task.implementation_guide == "Use pytest framework"
        assert task.references == ["https://pytest.org"]


class TestOptimizationPlan:
    """Test OptimizationPlan dataclass."""

    def test_optimization_plan_initialization_empty(self):
        """Test OptimizationPlan with empty task list."""
        plan = OptimizationPlan(tasks=[])

        assert plan.tasks == []
        assert plan.dependencies == {}
        assert plan.total_effort_hours == 0.0
        assert plan.estimated_completion_weeks == 0.0

    def test_optimization_plan_initialization_with_tasks(self):
        """Test OptimizationPlan with tasks."""
        tasks = [
            Task(
                id="task-001",
                title="Fix vulnerability",
                description="Patch security issue",
                priority=Priority.P0,
                effort_hours=4.0,
                role=Role.SECURITY,
            ),
            Task(
                id="task-002",
                title="Add tests",
                description="Improve coverage",
                priority=Priority.P1,
                effort_hours=8.0,
                role=Role.QA,
                dependencies=["task-001"],
            ),
        ]

        dependencies = {"task-002": ["task-001"]}

        plan = OptimizationPlan(
            tasks=tasks,
            dependencies=dependencies,
            total_effort_hours=12.0,
            estimated_completion_weeks=1.5,
        )

        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "task-001"
        assert plan.tasks[1].id == "task-002"
        assert plan.dependencies == dependencies
        assert plan.total_effort_hours == 12.0
        assert plan.estimated_completion_weeks == 1.5

    def test_optimization_plan_to_gantt_chart(self):
        """Test OptimizationPlan.to_gantt_chart() placeholder."""
        plan = OptimizationPlan(tasks=[])

        result = plan.to_gantt_chart()

        # Currently returns message when no tasks
        assert isinstance(result, str)
        assert "no tasks to visualize" in result.lower()

    def test_optimization_plan_to_dict(self):
        """Test OptimizationPlan.to_dict() method."""
        task = Task(
            id="task-001",
            title="Test task",
            description="Test description",
            priority=Priority.P1,
            effort_hours=2.0,
            role=Role.BACKEND,
        )

        plan = OptimizationPlan(
            tasks=[task],
            dependencies={"task-001": []},
            total_effort_hours=2.0,
            estimated_completion_weeks=0.25,
        )

        result = plan.to_dict()

        assert isinstance(result, dict)
        assert "tasks" in result
        assert "dependencies" in result
        assert "total_effort_hours" in result
        assert "estimated_completion_weeks" in result

        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["id"] == "task-001"
        assert result["dependencies"] == {"task-001": []}
        assert result["total_effort_hours"] == 2.0
        assert result["estimated_completion_weeks"] == 0.25


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_strings_and_none_values(self):
        """Test handling of empty strings and None values."""
        # Issue with empty strings
        issue = Issue(
            id="",  # Empty string
            dimension="",
            severity=Severity.LOW,
            category="",
            title="",
            description="",
            file_path="",
            line_number=None,  # None value
            recommendation="",
            effort_hours=0.0,
            references=[],  # Empty list
        )

        assert issue.id == ""
        assert issue.line_number is None
        assert issue.references == []

    def test_large_numeric_values(self):
        """Test handling of large numeric values."""
        analysis = PerformanceAnalysis(
            gpu_utilization=99.999,
            memory_usage_peak_gb=999.99,
            memory_usage_avg_gb=500.50,
            score=100.0,
        )

        assert analysis.gpu_utilization == 99.999
        assert analysis.memory_usage_peak_gb == 999.99
        assert analysis.memory_usage_avg_gb == 500.50
        assert analysis.score == 100.0

    def test_negative_numeric_values(self):
        """Test handling of negative numeric values (should be allowed)."""
        # Some metrics might legitimately be negative or zero
        analysis = CodeQualityAnalysis(
            average_complexity=0.0,  # Zero complexity
            duplication_percentage=0.0,  # No duplication
            documentation_coverage=0.0,  # No docs
            pylint_score=-5.0,  # Negative pylint score is possible
            score=0.0,
        )

        assert analysis.average_complexity == 0.0
        assert analysis.duplication_percentage == 0.0
        assert analysis.documentation_coverage == 0.0
        assert analysis.pylint_score == -5.0
        assert analysis.score == 0.0

    def test_very_long_strings(self):
        """Test handling of very long strings."""
        long_description = "x" * 10000  # Very long description
        long_file_path = "src/" + "very_long_module_name_" * 50 + ".py"

        issue = Issue(
            id="long-test",
            dimension="test",
            severity=Severity.LOW,
            category="test",
            title="Test with long strings",
            description=long_description,
            file_path=long_file_path,
        )

        assert len(issue.description) == 10000
        assert issue.file_path.startswith("src/very_long_module_name_")
        assert issue.file_path.endswith(".py")

    def test_unicode_strings(self):
        """Test handling of Unicode strings."""
        issue = Issue(
            id="unicode-test",
            dimension="测试",  # Chinese characters
            severity=Severity.LOW,
            category="тест",  # Cyrillic characters
            title="Test with émojis 🐍🔧",  # Emojis and accents
            description="Descripción con acentos y símbolos: α, β, γ",
            file_path="src/módulo_español.py",
        )

        assert "测试" in issue.dimension
        assert "тест" in issue.category
        assert "🐍🔧" in issue.title
        assert "α, β, γ" in issue.description
        assert "módulo_español" in issue.file_path

    def test_nested_empty_structures(self):
        """Test handling of nested empty data structures."""
        analysis = ArchitectureAnalysis(
            total_files=0,
            large_files=[],  # Empty list
            circular_dependencies=[],  # Empty list
            coupling_metrics={},  # Empty dict
            solid_violations=[],  # Empty list of Issues
            score=0.0,
        )

        assert analysis.total_files == 0
        assert len(analysis.large_files) == 0
        assert len(analysis.circular_dependencies) == 0
        assert len(analysis.coupling_metrics) == 0
        assert len(analysis.solid_violations) == 0

    def test_deeply_nested_structures(self):
        """Test handling of deeply nested data structures."""
        # Complex nested structure
        nested_data = {
            "level1": {
                "level2": {"level3": {"metrics": [1, 2, 3], "flags": {"a": True, "b": False}}}
            }
        }

        analysis = ArchitectureAnalysis(coupling_metrics=nested_data, score=50.0)

        assert analysis.coupling_metrics["level1"]["level2"]["level3"]["metrics"] == [1, 2, 3]
        assert analysis.coupling_metrics["level1"]["level2"]["level3"]["flags"]["a"] is True
