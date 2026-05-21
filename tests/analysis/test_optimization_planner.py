"""
Unit tests for Optimization Planner.

Tests task prioritization, effort estimation, dependency resolution, and role assignment.
Requirements: 10.1, 10.2, 10.3, 10.4
"""

from datetime import datetime

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
from src.analysis.planner import OptimizationPlanner


class TestOptimizationPlanner:
    """Test suite for OptimizationPlanner."""

    def setup_method(self):
        """Set up test fixtures."""
        self.planner = OptimizationPlanner()
        self.sample_result = self._create_sample_analysis_result()

    def _create_sample_analysis_result(self) -> AnalysisResult:
        """Create a sample analysis result for testing."""
        return AnalysisResult(
            timestamp=datetime.now(),
            project_path="/test/project",
            git_commit="abc123def",
            architecture=ArchitectureAnalysis(
                total_files=100,
                large_files=[
                    {"path": "src/large_module.py", "lines": 800, "complexity": 45},
                    {"path": "src/another_large.py", "lines": 650, "complexity": 38},
                ],
                circular_dependencies=[["module_a", "module_b", "module_a"]],
                coupling_metrics={"fan_in": 5, "fan_out": 12},
                solid_violations=[
                    Issue(
                        id="ARCH-001",
                        dimension="architecture",
                        severity=Severity.HIGH,
                        category="SOLID",
                        title="SRP Violation",
                        description="Class has too many responsibilities",
                        file_path="src/god_class.py",
                        line_number=10,
                        recommendation="Split into smaller classes",
                    )
                ],
                score=65.0,
            ),
            performance=PerformanceAnalysis(
                gpu_utilization=55.0,
                memory_usage_peak_gb=14.5,
                bottlenecks=[
                    {"function": "data_loader", "time_ms": 1200, "percentage": 35.0},
                    {"function": "model_forward", "time_ms": 800, "percentage": 23.0},
                ],
                flame_graph_path="",
                score=60.0,
            ),
            coverage=CoverageAnalysis(
                line_coverage=72.5,
                branch_coverage=65.0,
                untested_critical_paths=[
                    "src/error_handler.py:handle_exception",
                    "src/validator.py:validate_input",
                ],
                missing_property_tests=[
                    "src/transforms.py:normalize_data",
                    "src/utils.py:merge_dicts",
                ],
                flaky_tests=["tests/test_integration.py::test_concurrent_access"],
                score=68.0,
            ),
            code_quality=CodeQualityAnalysis(
                average_complexity=9.2,
                high_complexity_functions=[
                    {"function": "process_data", "complexity": 15, "file": "src/processor.py"}
                ],
                duplication_percentage=7.5,
                documentation_coverage=65.0,
                pylint_score=8.5,
                score=70.0,
            ),
            dependencies=DependencyAnalysis(
                total_dependencies=45,
                outdated_packages=[
                    {"name": "numpy", "current": "1.20.0", "latest": "1.24.0"},
                    {"name": "pandas", "current": "1.3.0", "latest": "2.0.0"},
                ],
                vulnerabilities=[{"package": "pillow", "severity": "high", "cve": "CVE-2023-1234"}],
                unused_dependencies=[],
                license_issues=[],
                score=65.0,
            ),
            deployment=DeploymentAnalysis(
                dockerfile_score=65.0,
                k8s_readiness=70.0,
                ci_cd_completeness=75.0,
                monitoring_score=50.0,
                score=65.0,
            ),
            security=SecurityAnalysis(
                vulnerabilities=[
                    Issue(
                        id="SEC-001",
                        dimension="security",
                        severity=Severity.CRITICAL,
                        category="Security",
                        title="SQL Injection",
                        description="SQL injection vulnerability",
                        file_path="src/database.py",
                        line_number=45,
                        recommendation="Use parameterized queries",
                    )
                ],
                hardcoded_secrets=[{"file": "config.py", "line": 10, "type": "API_KEY"}],
                tls_issues=[],
                hipaa_compliance_score=70.0,
                score=60.0,
            ),
            scalability=ScalabilityAnalysis(
                ddp_correctness=False,
                memory_bottlenecks=["DataLoader has num_workers=0"],
                communication_overhead_ms=0.0,
                scaling_efficiency="unknown",
                recommendations={},
                score=50.0,
            ),
            overall_score=68.5,
            critical_issues=[
                Issue(
                    id="SEC-001",
                    dimension="security",
                    severity=Severity.CRITICAL,
                    category="Security",
                    title="SQL Injection",
                    description="SQL injection vulnerability",
                    file_path="src/database.py",
                    line_number=45,
                    recommendation="Use parameterized queries",
                )
            ],
        )

    def test_create_plan_basic(self):
        """Test basic plan creation."""
        plan = self.planner.create_plan(self.sample_result)

        # Plan should have tasks
        assert len(plan.tasks) > 0

        # Plan should have dependencies
        assert isinstance(plan.dependencies, dict)

        # Plan should have effort and timeline estimates
        assert plan.total_effort_hours > 0
        assert plan.estimated_completion_weeks > 0

    def test_priority_assignment_security_critical(self):
        """Test that security vulnerabilities get P0 priority."""
        plan = self.planner.create_plan(self.sample_result)

        # Find security tasks
        security_tasks = [t for t in plan.tasks if t.role == Role.SECURITY]

        # At least one security task should exist
        assert len(security_tasks) > 0

        # Security vulnerability tasks should be P0
        vuln_tasks = [t for t in security_tasks if "vulnerabilities" in t.description.lower()]
        if vuln_tasks:
            assert all(t.priority == Priority.P0 for t in vuln_tasks)

    def test_priority_assignment_circular_dependencies(self):
        """Test that circular dependencies get P0 priority."""
        plan = self.planner.create_plan(self.sample_result)

        # Find circular dependency tasks
        circular_tasks = [t for t in plan.tasks if "circular" in t.title.lower()]

        # Should have at least one circular dependency task
        assert len(circular_tasks) > 0

        # Should be P0 priority
        assert all(t.priority == Priority.P0 for t in circular_tasks)

    def test_priority_assignment_flaky_tests(self):
        """Test that flaky tests get P1 priority."""
        plan = self.planner.create_plan(self.sample_result)

        # Find flaky test tasks
        flaky_tasks = [t for t in plan.tasks if "flaky" in t.title.lower()]

        # Should have at least one flaky test task
        assert len(flaky_tasks) > 0

        # Should be P1 priority
        assert all(t.priority == Priority.P1 for t in flaky_tasks)

    def test_effort_estimation_security_multiplier(self):
        """Test that security tasks have effort multiplier applied."""
        # Create a simple result with only security issues
        result = AnalysisResult(
            timestamp=datetime.now(),
            project_path="/test",
            git_commit="abc123",
            architecture=ArchitectureAnalysis(
                total_files=50,
                large_files=[],
                circular_dependencies=[],
                coupling_metrics={},
                solid_violations=[],
                score=90.0,
            ),
            performance=PerformanceAnalysis(
                gpu_utilization=80.0,
                memory_usage_peak_gb=10.0,
                bottlenecks=[],
                flame_graph_path="",
                score=88.0,
            ),
            coverage=CoverageAnalysis(
                line_coverage=85.0,
                branch_coverage=80.0,
                untested_critical_paths=[],
                missing_property_tests=[],
                flaky_tests=[],
                score=85.0,
            ),
            code_quality=CodeQualityAnalysis(
                average_complexity=5.0,
                high_complexity_functions=[],
                duplication_percentage=2.0,
                documentation_coverage=85.0,
                pylint_score=9.0,
                score=85.0,
            ),
            dependencies=DependencyAnalysis(
                total_dependencies=20,
                outdated_packages=[],
                vulnerabilities=[],
                unused_dependencies=[],
                license_issues=[],
                score=85.0,
            ),
            deployment=DeploymentAnalysis(
                dockerfile_score=85.0,
                k8s_readiness=85.0,
                ci_cd_completeness=85.0,
                monitoring_score=85.0,
                score=85.0,
            ),
            security=SecurityAnalysis(
                vulnerabilities=[
                    Issue(
                        id="SEC-001",
                        dimension="security",
                        severity=Severity.HIGH,
                        category="Security",
                        title="Test Vulnerability",
                        description="Test vulnerability",
                        file_path="test.py",
                        line_number=1,
                        recommendation="Fix it",
                    )
                ],
                hardcoded_secrets=[],
                tls_issues=[],
                hipaa_compliance_score=85.0,
                score=85.0,
            ),
            scalability=ScalabilityAnalysis(
                ddp_correctness=True,
                scaling_efficiency="linear",
                memory_bottlenecks=[],
                communication_overhead_ms=20.0,
                score=87.0,
                recommendations={},
            ),
            overall_score=85.0,
            critical_issues=[],
        )

        plan = self.planner.create_plan(result)

        # Find security tasks
        security_tasks = [t for t in plan.tasks if t.role == Role.SECURITY]

        # Security tasks should have effort > base effort (multiplier applied)
        # Base effort for 1 vulnerability is 3.0 hours, with 1.3x multiplier = 3.9 hours
        if security_tasks:
            assert any(t.effort_hours > 3.0 for t in security_tasks)

    def test_dependency_resolution_security_first(self):
        """Test that non-security tasks depend on security tasks."""
        plan = self.planner.create_plan(self.sample_result)

        # Find P0 security tasks
        p0_security_tasks = [
            t.id for t in plan.tasks if t.role == Role.SECURITY and t.priority == Priority.P0
        ]

        if p0_security_tasks:
            # Find non-security tasks
            non_security_tasks = [t for t in plan.tasks if t.role != Role.SECURITY]

            # At least some non-security tasks should depend on security tasks
            has_security_dep = False
            for task in non_security_tasks:
                deps = plan.dependencies.get(task.id, [])
                if any(dep in p0_security_tasks for dep in deps):
                    has_security_dep = True
                    break

            assert has_security_dep

    def test_dependency_resolution_architecture_before_performance(self):
        """Test that performance tasks depend on architecture tasks."""
        plan = self.planner.create_plan(self.sample_result)

        # Find architecture refactoring tasks
        arch_tasks = [
            t.id
            for t in plan.tasks
            if "refactor" in t.title.lower() or "circular" in t.title.lower()
        ]

        # Find performance optimization tasks
        perf_tasks = [
            t
            for t in plan.tasks
            if "performance" in t.title.lower() or "optimize" in t.title.lower()
        ]

        if arch_tasks and perf_tasks:
            # Performance tasks should depend on architecture tasks
            for perf_task in perf_tasks:
                deps = plan.dependencies.get(perf_task.id, [])
                # At least some performance tasks should have architecture dependencies
                if deps:
                    assert any(dep in arch_tasks for dep in deps)

    def test_dependency_resolution_no_circular_dependencies(self):
        """Test that dependency graph has no circular dependencies."""
        plan = self.planner.create_plan(self.sample_result)

        # Try topological sort - should not fail
        sorted_tasks = self.planner._topological_sort(plan.tasks, plan.dependencies)

        # Should return all tasks
        assert len(sorted_tasks) == len(plan.tasks)

    def test_role_assignment_architecture_tasks(self):
        """Test that architecture tasks are assigned to BACKEND role."""
        plan = self.planner.create_plan(self.sample_result)

        # Find architecture tasks
        arch_tasks = [t for t in plan.tasks if "refactor" in t.title.lower()]

        # Should be assigned to BACKEND
        if arch_tasks:
            assert all(t.role == Role.BACKEND for t in arch_tasks)

    def test_role_assignment_performance_tasks(self):
        """Test that performance tasks are assigned to ML role."""
        plan = self.planner.create_plan(self.sample_result)

        # Find performance tasks
        perf_tasks = [
            t for t in plan.tasks if "gpu" in t.title.lower() or "performance" in t.title.lower()
        ]

        # Should be assigned to ML
        if perf_tasks:
            assert all(t.role == Role.ML for t in perf_tasks)

    def test_role_assignment_security_tasks(self):
        """Test that security tasks are assigned to SECURITY role."""
        plan = self.planner.create_plan(self.sample_result)

        # Find security tasks
        security_tasks = [
            t
            for t in plan.tasks
            if "security" in t.title.lower() or "vulnerabilities" in t.title.lower()
        ]

        # Should be assigned to SECURITY
        if security_tasks:
            assert all(t.role == Role.SECURITY for t in security_tasks)

    def test_role_assignment_testing_tasks(self):
        """Test that testing tasks are assigned to QA role."""
        plan = self.planner.create_plan(self.sample_result)

        # Find testing tasks
        test_tasks = [
            t for t in plan.tasks if "test" in t.title.lower() or "coverage" in t.title.lower()
        ]

        # Should be assigned to QA
        if test_tasks:
            assert all(t.role == Role.QA for t in test_tasks)

    def test_timeline_calculation(self):
        """Test that timeline calculation is reasonable."""
        plan = self.planner.create_plan(self.sample_result)

        # Timeline should be positive
        assert plan.estimated_completion_weeks > 0

        # Timeline should be less than total effort (due to parallelization)
        total_effort_weeks = plan.total_effort_hours / 40  # Assuming 40 hours/week
        assert plan.estimated_completion_weeks <= total_effort_weeks

    def test_implementation_guides_generated(self):
        """Test that implementation guides are generated for tasks."""
        plan = self.planner.create_plan(self.sample_result)

        # All tasks should have implementation guides
        assert all(t.implementation_guide for t in plan.tasks)

        # All tasks should have success criteria
        assert all(t.success_criteria for t in plan.tasks)

    def test_gantt_data_generation(self):
        """Test Gantt chart data generation."""
        plan = self.planner.create_plan(self.sample_result)
        gantt_data = self.planner.generate_gantt_data(plan)

        # Should have required keys
        assert "tasks" in gantt_data
        assert "total_weeks" in gantt_data
        assert "roles" in gantt_data
        assert "start_date" in gantt_data
        assert "end_date" in gantt_data

        # Should have task schedules
        assert len(gantt_data["tasks"]) == len(plan.tasks)

        # Each task schedule should have required fields
        for task_schedule in gantt_data["tasks"]:
            assert "id" in task_schedule
            assert "title" in task_schedule
            assert "role" in task_schedule
            assert "priority" in task_schedule
            assert "start_week" in task_schedule
            assert "end_week" in task_schedule
            assert "duration_weeks" in task_schedule
            assert "effort_hours" in task_schedule
            assert "start_date" in task_schedule
            assert "end_date" in task_schedule

    def test_task_generation_from_large_files(self):
        """Test that large files generate refactoring tasks."""
        plan = self.planner.create_plan(self.sample_result)

        # Should have refactoring task for large files
        refactor_tasks = [t for t in plan.tasks if "large files" in t.title.lower()]
        assert len(refactor_tasks) > 0

        # Task should mention the number of large files
        assert any("2" in t.description for t in refactor_tasks)

    def test_task_generation_from_coverage_gaps(self):
        """Test that coverage gaps generate testing tasks."""
        plan = self.planner.create_plan(self.sample_result)

        # Should have coverage improvement task
        coverage_tasks = [t for t in plan.tasks if "coverage" in t.title.lower()]
        assert len(coverage_tasks) > 0

    def test_task_generation_from_security_issues(self):
        """Test that security issues generate security tasks."""
        plan = self.planner.create_plan(self.sample_result)

        # Should have security task
        security_tasks = [t for t in plan.tasks if t.role == Role.SECURITY]
        assert len(security_tasks) > 0

        # Should have task for vulnerabilities
        vuln_tasks = [t for t in security_tasks if "vulnerabilities" in t.description.lower()]
        assert len(vuln_tasks) > 0

    def test_empty_analysis_result(self):
        """Test plan creation with minimal issues."""
        # Create result with no issues
        result = AnalysisResult(
            timestamp=datetime.now(),
            project_path="/test",
            git_commit="abc123",
            architecture=ArchitectureAnalysis(
                total_files=50,
                large_files=[],
                circular_dependencies=[],
                coupling_metrics={},
                solid_violations=[],
                score=90.0,
            ),
            performance=PerformanceAnalysis(
                gpu_utilization=85.0,
                memory_usage_peak_gb=10.0,
                bottlenecks=[],
                flame_graph_path="",
                score=90.0,
            ),
            coverage=CoverageAnalysis(
                line_coverage=90.0,
                branch_coverage=85.0,
                untested_critical_paths=[],
                missing_property_tests=[],
                flaky_tests=[],
                score=90.0,
            ),
            code_quality=CodeQualityAnalysis(
                average_complexity=5.0,
                high_complexity_functions=[],
                duplication_percentage=2.0,
                documentation_coverage=90.0,
                pylint_score=9.5,
                score=90.0,
            ),
            dependencies=DependencyAnalysis(
                total_dependencies=20,
                outdated_packages=[],
                vulnerabilities=[],
                unused_dependencies=[],
                license_issues=[],
                score=90.0,
            ),
            deployment=DeploymentAnalysis(
                dockerfile_score=90.0,
                k8s_readiness=90.0,
                ci_cd_completeness=90.0,
                monitoring_score=90.0,
                score=90.0,
            ),
            security=SecurityAnalysis(
                vulnerabilities=[],
                hardcoded_secrets=[],
                tls_issues=[],
                hipaa_compliance_score=90.0,
                score=90.0,
            ),
            scalability=ScalabilityAnalysis(
                ddp_correctness=True,
                memory_bottlenecks=[],
                communication_overhead_ms=15.0,
                scaling_efficiency="linear",
                recommendations={},
                score=90.0,
            ),
            overall_score=90.0,
            critical_issues=[],
        )

        plan = self.planner.create_plan(result)

        # Should still create a plan (even if minimal)
        assert isinstance(plan.tasks, list)
        assert plan.total_effort_hours >= 0
        assert plan.estimated_completion_weeks >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
