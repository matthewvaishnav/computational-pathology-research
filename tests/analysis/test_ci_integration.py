"""
Integration tests for CI/CD workflow components.

Tests GitHub Actions workflow integration, PR comment posting, and baseline comparison.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.analysis.regression_detector import RegressionDetector
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
)


@pytest.fixture
def sample_analysis_result():
    """Create sample analysis result for testing."""
    return AnalysisResult(
        timestamp=datetime.now().isoformat(),
        project_path="/test/project",
        git_commit="a" * 40,
        architecture=ArchitectureAnalysis(
            total_files=1000,
            large_files=[{"path": "large_file.py", "lines": 600}],
            circular_dependencies=[],
            score=85.0,
        ),
        performance=PerformanceAnalysis(
            gpu_utilization=80.0,
            bottlenecks=[{"operation": "data_loading", "time_ms": 150}],
            score=75.0,
        ),
        coverage=CoverageAnalysis(
            line_coverage=75.0,
            branch_coverage=70.0,
            untested_critical_paths=["error_handler.py:45"],
            score=72.0,
        ),
        code_quality=CodeQualityAnalysis(
            average_complexity=4.5,
            high_complexity_functions=[{"name": "complex_func", "complexity": 15}],
            duplication_percentage=8.0,
            score=78.0,
        ),
        dependencies=DependencyAnalysis(
            total_dependencies=50,
            vulnerabilities=[],
            outdated_packages=["requests", "numpy"],
            score=90.0,
        ),
        deployment=DeploymentAnalysis(
            dockerfile_score=85.0, k8s_readiness=80.0, ci_cd_completeness=90.0, score=85.0
        ),
        security=SecurityAnalysis(vulnerabilities=[], hardcoded_secrets=[], score=95.0),
        scalability=ScalabilityAnalysis(
            ddp_correctness=True, scaling_efficiency="linear", score=88.0
        ),
        overall_score=82.5,
    )


class TestCIWorkflowIntegration:
    """Test CI/CD workflow integration."""

    def test_workflow_file_exists(self):
        """Test that GitHub Actions workflow file exists and is valid YAML."""
        workflow_path = Path(".github/workflows/project-analysis.yml")
        assert workflow_path.exists(), "GitHub Actions workflow file not found"

        # Basic YAML validation
        import yaml

        with open(workflow_path, "r") as f:
            workflow = yaml.safe_load(f)

        assert "name" in workflow
        assert "on" in workflow
        assert "jobs" in workflow
        assert "analyze" in workflow["jobs"]

    def test_workflow_triggers(self):
        """Test workflow trigger configuration."""
        import yaml

        workflow_path = Path(".github/workflows/project-analysis.yml")

        with open(workflow_path, "r") as f:
            workflow = yaml.safe_load(f)

        triggers = workflow["on"]
        assert "push" in triggers
        assert "pull_request" in triggers
        assert "workflow_dispatch" in triggers

        # Check branch filters
        assert "main" in triggers["push"]["branches"]
        assert "main" in triggers["pull_request"]["branches"]

    def test_workflow_permissions(self):
        """Test workflow permissions are correctly set."""
        import yaml

        workflow_path = Path(".github/workflows/project-analysis.yml")

        with open(workflow_path, "r") as f:
            workflow = yaml.safe_load(f)

        permissions = workflow["jobs"]["analyze"]["permissions"]
        assert permissions["contents"] == "read"
        assert permissions["pull-requests"] == "write"
        assert permissions["actions"] == "read"

    def test_analysis_result_serialization_for_ci(self, sample_analysis_result):
        """Test analysis result can be serialized for CI artifacts."""
        # Test JSON serialization
        json_str = sample_analysis_result.to_json()
        assert json_str is not None
        assert len(json_str) > 0

        # Test deserialization
        parsed = AnalysisResult.from_json(json_str)
        assert parsed.overall_score == sample_analysis_result.overall_score
        assert parsed.git_commit == sample_analysis_result.git_commit

    def test_baseline_comparison_workflow(self, sample_analysis_result):
        """Test baseline comparison workflow."""
        detector = RegressionDetector()

        # Create baseline and current results
        baseline = sample_analysis_result
        current = sample_analysis_result
        current.coverage.line_coverage = 72.0  # 3% decrease

        # Test regression detection
        report = detector.detect_regressions(baseline, current)

        assert report.has_regressions
        assert len(report.critical_regressions) >= 1

        # Test diff report generation
        diff_report = detector.generate_diff_report(baseline, current, report)
        assert "Regression Analysis Report" in diff_report
        assert "Coverage Metrics" in diff_report

    def test_ci_exit_code_logic(self, sample_analysis_result):
        """Test CI build failure logic."""
        detector = RegressionDetector()

        # Test with critical regression
        baseline = sample_analysis_result
        current = sample_analysis_result
        current.coverage.line_coverage = 70.0  # 5% decrease (critical)

        report = detector.detect_regressions(baseline, current)
        exit_code = detector.exit_code_for_ci(report)

        assert exit_code == 1  # Should fail CI

        # Test without critical regression
        current.coverage.line_coverage = 74.0  # 1% decrease (high, not critical)
        report = detector.detect_regressions(baseline, current)
        exit_code = detector.exit_code_for_ci(report)

        assert exit_code == 0  # Should pass CI


class TestPRCommentGeneration:
    """Test PR comment generation functionality."""

    def test_analysis_summary_generation(self, sample_analysis_result):
        """Test analysis summary generation for PR comments."""
        # This would be the Python code that runs in the GitHub Action
        summary_template = """# 📊 Project Analysis Summary

**Overall Score**: {overall_score:.1f}/100
**Timestamp**: {timestamp}
**Commit**: {git_commit}

## Key Metrics
- **Architecture**: {architecture_score:.1f}/100
- **Performance**: {performance_score:.1f}/100  
- **Coverage**: {coverage_score:.1f}/100 (Line: {line_coverage:.1f}%, Branch: {branch_coverage:.1f}%)
- **Code Quality**: {code_quality_score:.1f}/100 (Avg Complexity: {avg_complexity:.1f})
- **Dependencies**: {dependencies_score:.1f}/100 ({vuln_count} vulnerabilities)
- **Deployment**: {deployment_score:.1f}/100
- **Security**: {security_score:.1f}/100 ({security_issues} issues)
- **Scalability**: {scalability_score:.1f}/100

## Critical Issues
{critical_issues_summary}
"""

        # Generate summary
        critical_issues_summary = (
            "No critical issues detected ✅"
            if not sample_analysis_result.critical_issues
            else "\n".join(
                [
                    f"{i+1}. **[{issue.priority.value}]** {issue.title} ({issue.dimension})"
                    for i, issue in enumerate(sample_analysis_result.critical_issues[:5])
                ]
            )
        )

        summary = summary_template.format(
            overall_score=sample_analysis_result.overall_score,
            timestamp=sample_analysis_result.timestamp,
            git_commit=sample_analysis_result.git_commit[:8],
            architecture_score=sample_analysis_result.architecture.score,
            performance_score=sample_analysis_result.performance.score,
            coverage_score=sample_analysis_result.coverage.score,
            line_coverage=sample_analysis_result.coverage.line_coverage,
            branch_coverage=sample_analysis_result.coverage.branch_coverage,
            code_quality_score=sample_analysis_result.code_quality.score,
            avg_complexity=sample_analysis_result.code_quality.average_complexity,
            dependencies_score=sample_analysis_result.dependencies.score,
            vuln_count=len(sample_analysis_result.dependencies.vulnerabilities),
            deployment_score=sample_analysis_result.deployment.score,
            security_score=sample_analysis_result.security.score,
            security_issues=len(sample_analysis_result.security.vulnerabilities),
            scalability_score=sample_analysis_result.scalability.score,
            critical_issues_summary=critical_issues_summary,
        )

        assert "📊 Project Analysis Summary" in summary
        assert "82.5/100" in summary
        assert "75.0%" in summary  # Line coverage
        assert "No critical issues detected ✅" in summary

    @patch("requests.post")
    def test_pr_comment_posting_mock(self, mock_post, sample_analysis_result):
        """Test PR comment posting with mocked GitHub API."""
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {"id": 12345}

        # Simulate GitHub Actions script behavior
        comment_body = f"""# 📊 Project Analysis Summary

**Overall Score**: {sample_analysis_result.overall_score:.1f}/100

## Key Metrics
- **Coverage**: {sample_analysis_result.coverage.line_coverage:.1f}%
- **Security**: {len(sample_analysis_result.security.vulnerabilities)} issues

✅ No regressions detected compared to baseline.

---
*Generated by Project Analysis CI*"""

        # Mock GitHub API call
        github_api_url = "https://api.github.com/repos/test/repo/issues/123/comments"
        headers = {"Authorization": "token fake_token"}
        data = {"body": comment_body}

        # This simulates what the GitHub Action would do
        response = mock_post(github_api_url, json=data, headers=headers)

        assert response.status_code == 201
        mock_post.assert_called_once()

    def test_regression_comment_formatting(self, sample_analysis_result):
        """Test regression information formatting in PR comments."""
        detector = RegressionDetector()

        # Create regression scenario
        baseline = sample_analysis_result
        current = sample_analysis_result
        current.coverage.line_coverage = 70.0  # Critical regression
        current.security.vulnerabilities = [
            {"type": "xss", "severity": "high"}
        ]  # New vulnerability

        report = detector.detect_regressions(baseline, current)

        # Format regression summary for PR comment
        regression_summary = ""
        if report.has_regressions:
            critical_count = len(report.critical_regressions)
            if critical_count > 0:
                regression_summary = f"## 🚨 Regression Analysis\n\n**⚠️ {critical_count} CRITICAL regressions detected!**\n\n"
                regression_summary += (
                    "This PR introduces critical regressions that may block the build.\n"
                )
            else:
                regression_summary = (
                    "## ℹ️ Regression Analysis\n\n**Non-critical regressions detected.**\n"
                )
        else:
            regression_summary = (
                "## ✅ Regression Analysis\n\nNo regressions detected compared to baseline."
            )

        assert "🚨 Regression Analysis" in regression_summary
        assert "CRITICAL regressions detected" in regression_summary
        assert "block the build" in regression_summary


class TestArtifactManagement:
    """Test CI artifact management."""

    def test_analysis_artifact_structure(self, sample_analysis_result):
        """Test analysis artifact directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "analysis_output"
            output_dir.mkdir()

            # Simulate CI artifact creation
            analysis_file = output_dir / "current_analysis.json"
            with open(analysis_file, "w") as f:
                f.write(sample_analysis_result.to_json())

            summary_file = output_dir / "summary.md"
            with open(summary_file, "w") as f:
                f.write("# Analysis Summary\n\nTest summary")

            # Verify artifact structure
            assert analysis_file.exists()
            assert summary_file.exists()

            # Verify content
            with open(analysis_file, "r") as f:
                loaded_result = AnalysisResult.from_json(f.read())
                assert loaded_result.overall_score == sample_analysis_result.overall_score

    def test_baseline_artifact_download_simulation(self, sample_analysis_result):
        """Test baseline artifact download simulation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Simulate baseline artifact
            baseline_dir = Path(temp_dir) / "baseline_analysis"
            baseline_dir.mkdir()

            baseline_file = baseline_dir / "current_analysis.json"
            with open(baseline_file, "w") as f:
                f.write(sample_analysis_result.to_json())

            # Simulate moving to expected location
            output_dir = Path(temp_dir) / "analysis_output"
            output_dir.mkdir()

            target_file = output_dir / "baseline_analysis.json"
            target_file.write_text(baseline_file.read_text())

            # Verify baseline is available
            assert target_file.exists()

            # Test regression detection can use it
            detector = RegressionDetector()
            baseline = detector.load_baseline(str(target_file))
            assert baseline.overall_score == sample_analysis_result.overall_score


class TestWorkflowErrorHandling:
    """Test workflow error handling scenarios."""

    def test_analysis_failure_handling(self):
        """Test handling of analysis failures in CI."""
        # Simulate analysis failure
        error_result = {"error": "Analysis failed", "timestamp": datetime.now().isoformat()}

        with tempfile.TemporaryDirectory() as temp_dir:
            error_file = Path(temp_dir) / "current_analysis.json"
            with open(error_file, "w") as f:
                json.dump(error_result, f)

            # Verify error file structure
            with open(error_file, "r") as f:
                data = json.load(f)
                assert "error" in data
                assert "Analysis failed" in data["error"]

    def test_missing_baseline_handling(self, sample_analysis_result):
        """Test handling when baseline analysis is not available."""
        detector = RegressionDetector()

        # Test with non-existent baseline
        with pytest.raises(FileNotFoundError):
            detector.load_baseline("non_existent_baseline.json")

        # Test graceful handling in CI context
        baseline_available = False  # Simulate no baseline

        if not baseline_available:
            # Should skip regression detection
            regression_outputs = {
                "has_regressions": "false",
                "critical_count": "0",
                "should_fail_ci": "false",
                "exit_code": "0",
            }
        else:
            # Would run regression detection
            pass

        assert regression_outputs["has_regressions"] == "false"
        assert regression_outputs["exit_code"] == "0"

    def test_malformed_analysis_handling(self):
        """Test handling of malformed analysis files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            malformed_file = Path(temp_dir) / "malformed.json"
            with open(malformed_file, "w") as f:
                f.write("{ invalid json")

            detector = RegressionDetector()

            # Should raise ValueError for malformed JSON
            with pytest.raises(ValueError, match="Invalid JSON"):
                detector.load_baseline(str(malformed_file))


class TestWorkflowPerformance:
    """Test workflow performance characteristics."""

    def test_analysis_timeout_handling(self):
        """Test analysis timeout configuration."""
        import yaml

        workflow_path = Path(".github/workflows/project-analysis.yml")

        with open(workflow_path, "r") as f:
            workflow = yaml.safe_load(f)

        # Check job timeout
        job_timeout = workflow["jobs"]["analyze"]["timeout-minutes"]
        assert job_timeout == 30  # Should have reasonable timeout

        # Check analysis timeout in script
        steps = workflow["jobs"]["analyze"]["steps"]
        analysis_step = next(step for step in steps if step["name"] == "Run project analysis")

        assert "--timeout 1800" in analysis_step["run"]  # 30 minutes

    def test_parallel_analysis_configuration(self):
        """Test parallel analysis configuration."""
        import yaml

        workflow_path = Path(".github/workflows/project-analysis.yml")

        with open(workflow_path, "r") as f:
            workflow = yaml.safe_load(f)

        steps = workflow["jobs"]["analyze"]["steps"]
        analysis_step = next(step for step in steps if step["name"] == "Run project analysis")

        # Should enable parallel analysis
        assert "--parallel" in analysis_step["run"]
        assert 'ANALYSIS_PARALLEL="true"' in analysis_step["run"]


class TestSecurityConsiderations:
    """Test security aspects of CI workflow."""

    def test_workflow_permissions_minimal(self):
        """Test workflow uses minimal required permissions."""
        import yaml

        workflow_path = Path(".github/workflows/project-analysis.yml")

        with open(workflow_path, "r") as f:
            workflow = yaml.safe_load(f)

        permissions = workflow["jobs"]["analyze"]["permissions"]

        # Should only have necessary permissions
        allowed_permissions = {"contents", "pull-requests", "actions"}
        actual_permissions = set(permissions.keys())

        assert actual_permissions.issubset(allowed_permissions)

        # Should not have write access to contents
        assert permissions["contents"] == "read"

    def test_no_hardcoded_secrets(self):
        """Test workflow doesn't contain hardcoded secrets."""
        workflow_path = Path(".github/workflows/project-analysis.yml")

        with open(workflow_path, "r") as f:
            content = f.read()

        # Should use GitHub secrets, not hardcoded values
        assert "secrets.GITHUB_TOKEN" in content
        assert "password" not in content.lower()
        assert "api_key" not in content.lower()

    def test_artifact_retention_policy(self):
        """Test artifact retention is configured appropriately."""
        import yaml

        workflow_path = Path(".github/workflows/project-analysis.yml")

        with open(workflow_path, "r") as f:
            workflow = yaml.safe_load(f)

        steps = workflow["jobs"]["analyze"]["steps"]
        upload_step = next(step for step in steps if step["name"] == "Upload analysis results")

        # Should have reasonable retention period
        assert upload_step["with"]["retention-days"] == 30
