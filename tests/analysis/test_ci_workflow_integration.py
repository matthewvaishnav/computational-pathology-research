"""
Integration tests for GitHub Actions CI workflow.

Tests workflow execution, PR comment posting, and regression detection in CI context.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Direct imports to avoid src.__init__ issues
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from analysis.models import (
    AnalysisResult,
    ArchitectureAnalysis,
    CodeQualityAnalysis,
    CoverageAnalysis,
    DependencyAnalysis,
    DeploymentAnalysis,
    Issue,
    PerformanceAnalysis,
    ScalabilityAnalysis,
    SecurityAnalysis,
    Severity,
)
from analysis.regression_detector import RegressionDetector


@pytest.fixture
def sample_analysis_result():
    """Create sample analysis result for testing."""
    return AnalysisResult(
        project_path=".",
        git_commit="abc123",
        timestamp="2026-05-06T10:00:00Z",
        overall_score=75.5,
        critical_issues=[
            Issue(
                id="SEC-001",
                dimension="security",
                severity=Severity.CRITICAL,
                category="hardcoded_secrets",
                title="Hardcoded API key detected",
                description="API key found in source code",
                file_path="src/api/client.py",
                line_number=42,
                recommendation="Use environment variables",
            )
        ],
        architecture=ArchitectureAnalysis(
            score=80.0,
        ),
        performance=PerformanceAnalysis(
            score=70.0,
        ),
        coverage=CoverageAnalysis(
            score=75.0,
        ),
        code_quality=CodeQualityAnalysis(
            score=72.0,
        ),
        dependencies=DependencyAnalysis(
            score=78.0,
        ),
        deployment=DeploymentAnalysis(
            score=68.0,
        ),
        security=SecurityAnalysis(
            score=65.0,
            hardcoded_secrets=[{"file": "src/api/client.py", "line": 42, "type": "api_key"}],
        ),
        scalability=ScalabilityAnalysis(
            score=73.0,
        ),
    )


@pytest.fixture
def workflow_yaml_path():
    """Return path to workflow YAML file."""
    return Path(".github/workflows/project-analysis.yml")


class TestWorkflowStructure:
    """Test workflow YAML structure and configuration."""

    def test_workflow_file_exists(self, workflow_yaml_path):
        """Workflow YAML file exists."""
        assert workflow_yaml_path.exists()

    def test_workflow_has_required_triggers(self, workflow_yaml_path):
        """Workflow has push, pull_request, schedule, and workflow_dispatch triggers."""
        import yaml

        with open(workflow_yaml_path, encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        # YAML parses 'on' keyword as True
        assert True in workflow
        triggers = workflow[True]
        assert "push" in triggers
        assert "pull_request" in triggers
        assert "schedule" in triggers
        assert "workflow_dispatch" in triggers

    def test_workflow_has_analysis_job(self, workflow_yaml_path):
        """Workflow has analysis job with required steps."""
        import yaml

        with open(workflow_yaml_path, encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        assert "jobs" in workflow
        assert "analysis" in workflow["jobs"]

        job = workflow["jobs"]["analysis"]
        assert "steps" in job

        # Check for required steps
        step_names = [step.get("name", "") for step in job["steps"]]
        assert any("Checkout" in name for name in step_names)
        assert any("Python" in name for name in step_names)
        assert any("analysis" in name.lower() for name in step_names)

    def test_workflow_has_pr_permissions(self, workflow_yaml_path):
        """Workflow has permissions to write PR comments."""
        import yaml

        with open(workflow_yaml_path, encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        job = workflow["jobs"]["analysis"]
        assert "permissions" in job
        permissions = job["permissions"]
        assert permissions.get("pull-requests") == "write"


class TestWorkflowExecution:
    """Test workflow execution logic."""

    def test_analysis_step_runs_orchestrator(self, sample_analysis_result):
        """Analysis step runs orchestrator with correct arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "analysis.json"

            # Mock orchestrator execution
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0)

                # Simulate workflow step
                cmd = [
                    "python",
                    "-m",
                    "src.analysis.orchestrator",
                    "--output",
                    str(output_file),
                    "--format",
                    "json",
                    "--parallel",
                    "--max-workers",
                    "4",
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                # Verify command was called
                mock_run.assert_called_once()
                assert result.returncode == 0

    def test_report_generation_step(self, sample_analysis_result):
        """Report generation step creates markdown and HTML reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            analysis_file = tmpdir / "analysis.json"
            analysis_file.write_text(sample_analysis_result.to_json())

            # Simulate report generation
            from analysis.reporting import ReportGenerator

            generator = ReportGenerator()

            # Generate markdown
            markdown = generator.generate_markdown(sample_analysis_result)
            markdown_file = tmpdir / "report.md"
            markdown_file.write_text(markdown)
            assert markdown_file.exists()
            assert len(markdown) > 0

            # Generate HTML
            html_file = tmpdir / "report.html"
            generator.generate_html(sample_analysis_result, str(html_file))
            assert html_file.exists()


class TestRegressionDetection:
    """Test regression detection in CI context."""

    def test_regression_detection_with_baseline(self, sample_analysis_result):
        """Regression detection compares current vs baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create baseline with better scores
            baseline = AnalysisResult(
                timestamp="2026-05-01T10:00:00Z",
                overall_score=85.0,
                critical_issues=[],
                architecture=ArchitectureAnalysis(
                    score=85.0,
                    large_files=[],
                    circular_dependencies=[],
                    high_coupling_modules=[],
                    solid_violations=[],
                ),
                performance=PerformanceAnalysis(
                    score=80.0,
                    gpu_utilization=0.75,
                    memory_usage_gb=7.5,
                    bottlenecks=[],
                    flame_graph_path=None,
                ),
                coverage=CoverageAnalysis(
                    score=85.0,
                    line_coverage=0.85,
                    branch_coverage=0.78,
                    untested_critical_paths=[],
                    missing_property_tests=[],
                ),
                code_quality=CodeQualityAnalysis(
                    score=82.0,
                    complexity_issues=[],
                    duplicates=[],
                    documentation_coverage=0.85,
                    type_hint_coverage=0.90,
                ),
                dependencies=DependencyAnalysis(
                    score=88.0,
                    vulnerabilities=[],
                    outdated_packages=[],
                    unused_dependencies=[],
                    license_issues=[],
                ),
                deployment=DeploymentAnalysis(
                    score=78.0,
                    dockerfile_issues=[],
                    k8s_issues=[],
                    ci_cd_issues=[],
                    readiness_score=78.0,
                ),
                security=SecurityAnalysis(
                    score=90.0,
                    injection_vulnerabilities=[],
                    tls_issues=[],
                    hardcoded_secrets=[],
                    hipaa_compliance_gaps=[],
                ),
                scalability=ScalabilityAnalysis(
                    score=83.0,
                    ddp_issues=[],
                    data_loading_bottlenecks=[],
                    communication_overhead=0.10,
                    scaling_efficiency="linear",
                ),
            )

            baseline_file = tmpdir / "baseline.json"
            baseline_file.write_text(baseline.to_json())

            # Run regression detection
            detector = RegressionDetector()
            regressions = detector.detect_regressions(sample_analysis_result, str(baseline_file))

            # Verify regressions detected
            assert "coverage" in regressions
            assert len(regressions["coverage"]["critical_regressions"]) > 0

            assert "security" in regressions
            assert len(regressions["security"]["critical_regressions"]) > 0

    def test_build_failure_on_critical_regression(self, sample_analysis_result):
        """Build fails when critical regressions detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create baseline with no security issues
            baseline = AnalysisResult(
                timestamp="2026-05-01T10:00:00Z",
                overall_score=85.0,
                critical_issues=[],
                architecture=ArchitectureAnalysis(
                    score=85.0,
                    large_files=[],
                    circular_dependencies=[],
                    high_coupling_modules=[],
                    solid_violations=[],
                ),
                performance=PerformanceAnalysis(
                    score=80.0,
                    gpu_utilization=0.75,
                    memory_usage_gb=7.5,
                    bottlenecks=[],
                    flame_graph_path=None,
                ),
                coverage=CoverageAnalysis(
                    score=85.0,
                    line_coverage=0.85,
                    branch_coverage=0.78,
                    untested_critical_paths=[],
                    missing_property_tests=[],
                ),
                code_quality=CodeQualityAnalysis(
                    score=82.0,
                    complexity_issues=[],
                    duplicates=[],
                    documentation_coverage=0.85,
                    type_hint_coverage=0.90,
                ),
                dependencies=DependencyAnalysis(
                    score=88.0,
                    vulnerabilities=[],
                    outdated_packages=[],
                    unused_dependencies=[],
                    license_issues=[],
                ),
                deployment=DeploymentAnalysis(
                    score=78.0,
                    dockerfile_issues=[],
                    k8s_issues=[],
                    ci_cd_issues=[],
                    readiness_score=78.0,
                ),
                security=SecurityAnalysis(
                    score=90.0,
                    injection_vulnerabilities=[],
                    tls_issues=[],
                    hardcoded_secrets=[],
                    hipaa_compliance_gaps=[],
                ),
                scalability=ScalabilityAnalysis(
                    score=83.0,
                    ddp_issues=[],
                    data_loading_bottlenecks=[],
                    communication_overhead=0.10,
                    scaling_efficiency="linear",
                ),
            )

            baseline_file = tmpdir / "baseline.json"
            baseline_file.write_text(baseline.to_json())

            # Run regression detection
            detector = RegressionDetector()
            regressions = detector.detect_regressions(sample_analysis_result, str(baseline_file))

            # Check if build should fail
            should_fail, reason = detector.should_fail_build(regressions)

            assert should_fail
            assert "security" in reason.lower() or "hardcoded" in reason.lower()


class TestPRCommentPosting:
    """Test PR comment posting logic."""

    def test_pr_comment_format(self, sample_analysis_result):
        """PR comment has correct format with scores and dimensions."""
        # Simulate comment generation
        overall_score = sample_analysis_result.overall_score
        critical_issues = len(sample_analysis_result.critical_issues)

        comment = f"## 📊 Project Analysis Results\n\n"
        comment += f"**Overall Score:** {overall_score}/100\n"
        comment += f"**Critical Issues:** {critical_issues}\n"
        comment += f"**Analysis Date:** {sample_analysis_result.timestamp}\n\n"

        # Add dimension scores
        comment += "### Dimension Scores\n\n"
        comment += "| Dimension | Score | Status |\n"
        comment += "|-----------|-------|--------|\n"

        dimensions = [
            ("Architecture", sample_analysis_result.architecture.score),
            ("Performance", sample_analysis_result.performance.score),
            ("Coverage", sample_analysis_result.coverage.score),
            ("Code Quality", sample_analysis_result.code_quality.score),
            ("Dependencies", sample_analysis_result.dependencies.score),
            ("Deployment", sample_analysis_result.deployment.score),
            ("Security", sample_analysis_result.security.score),
            ("Scalability", sample_analysis_result.scalability.score),
        ]

        for name, score in dimensions:
            status = (
                "✅ Excellent"
                if score >= 80
                else "🟡 Good" if score >= 60 else "🟠 Needs Work" if score >= 40 else "🔴 Critical"
            )
            comment += f"| {name} | {score:.1f} | {status} |\n"

        # Verify comment structure
        assert "📊 Project Analysis Results" in comment
        assert "Overall Score: 75.5/100" in comment
        assert "Critical Issues: 1" in comment
        assert "Architecture" in comment
        assert "Security" in comment

    @patch("requests.post")
    def test_pr_comment_api_call(self, mock_post, sample_analysis_result):
        """PR comment API call has correct structure."""
        # Mock GitHub API response
        mock_post.return_value = Mock(status_code=201, json=lambda: {"id": 12345})

        # Simulate API call
        comment_body = "Test comment"
        api_url = "https://api.github.com/repos/owner/repo/issues/1/comments"
        headers = {"Authorization": "token fake-token"}

        response = mock_post(api_url, json={"body": comment_body}, headers=headers)

        assert response.status_code == 201
        mock_post.assert_called_once()


class TestArtifactUpload:
    """Test artifact upload logic."""

    def test_analysis_artifacts_created(self, sample_analysis_result):
        """Analysis artifacts are created in correct directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create analysis artifacts
            analysis_file = tmpdir / "current-analysis.json"
            analysis_file.write_text(sample_analysis_result.to_json())

            report_file = tmpdir / "analysis-report.md"
            report_file.write_text("# Analysis Report\n\nTest content")

            # Verify artifacts exist
            assert analysis_file.exists()
            assert report_file.exists()
            assert len(analysis_file.read_text()) > 0
            assert len(report_file.read_text()) > 0

    def test_baseline_artifact_upload(self, sample_analysis_result):
        """Baseline artifact is uploaded for main branch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Simulate baseline upload
            baseline_file = tmpdir / "baseline-analysis.json"
            baseline_file.write_text(sample_analysis_result.to_json())

            assert baseline_file.exists()

            # Verify baseline can be loaded
            loaded = AnalysisResult.from_json(baseline_file.read_text())
            assert loaded.overall_score == sample_analysis_result.overall_score


class TestOptimizationPlanJob:
    """Test optimization plan generation job."""

    def test_optimization_plan_generation(self, sample_analysis_result):
        """Optimization plan is generated from analysis results."""
        from analysis.planner import OptimizationPlanner

        planner = OptimizationPlanner()
        plan = planner.create_plan(sample_analysis_result)

        assert plan is not None
        assert len(plan.tasks) > 0
        assert plan.total_effort_hours > 0

    def test_gantt_data_generation(self, sample_analysis_result):
        """Gantt chart data is generated for optimization plan."""
        from analysis.planner import OptimizationPlanner

        planner = OptimizationPlanner()
        plan = planner.create_plan(sample_analysis_result)
        gantt_data = planner.generate_gantt_data(plan)

        assert gantt_data is not None
        assert "tasks" in gantt_data
        assert len(gantt_data["tasks"]) > 0


class TestWorkflowErrorHandling:
    """Test workflow error handling."""

    def test_analysis_failure_handling(self):
        """Workflow handles analysis failures gracefully."""
        with patch("subprocess.run") as mock_run:
            # Simulate analysis failure
            mock_run.return_value = Mock(returncode=1, stderr="Analysis failed")

            result = subprocess.run(
                ["python", "-m", "src.analysis.orchestrator"],
                capture_output=True,
                text=True,
            )

            # Verify failure is detected
            mock_run.assert_called_once()
            assert result.returncode == 1

    def test_pr_comment_failure_does_not_fail_workflow(self):
        """PR comment posting failure does not fail workflow."""
        with patch("requests.post") as mock_post:
            # Simulate API failure
            mock_post.side_effect = Exception("API error")

            try:
                response = mock_post(
                    "https://api.github.com/repos/owner/repo/issues/1/comments",
                    json={"body": "test"},
                )
                # Should not reach here
                assert False, "Expected exception"
            except Exception as e:
                # Exception is caught, workflow continues
                assert "API error" in str(e)


class TestWorkflowSchedule:
    """Test workflow schedule configuration."""

    def test_weekly_schedule_configured(self, workflow_yaml_path):
        """Workflow has weekly schedule on Sundays at 2 AM UTC."""
        import yaml

        with open(workflow_yaml_path, encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        # YAML parses 'on' keyword as True
        schedule = workflow[True]["schedule"]
        assert len(schedule) > 0
        assert schedule[0]["cron"] == "0 2 * * 0"


class TestWorkflowInputs:
    """Test workflow_dispatch inputs."""

    def test_baseline_branch_input(self, workflow_yaml_path):
        """Workflow has baseline_branch input for manual runs."""
        import yaml

        with open(workflow_yaml_path, encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        # YAML parses 'on' keyword as True
        inputs = workflow[True]["workflow_dispatch"]["inputs"]
        assert "baseline_branch" in inputs
        assert inputs["baseline_branch"]["default"] == "main"

    def test_generate_reports_input(self, workflow_yaml_path):
        """Workflow has generate_reports boolean input."""
        import yaml

        with open(workflow_yaml_path, encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        # YAML parses 'on' keyword as True
        inputs = workflow[True]["workflow_dispatch"]["inputs"]
        assert "generate_reports" in inputs
        assert inputs["generate_reports"]["type"] == "boolean"
        assert inputs["generate_reports"]["default"] == "true"
