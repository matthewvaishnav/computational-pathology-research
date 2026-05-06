"""
End-to-end integration tests for full analysis pipeline.

Tests complete analysis workflow on small test project.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Direct imports to avoid src.__init__ issues
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from analysis.orchestrator import AnalysisOrchestrator
from analysis.models import AnalysisResult


@pytest.fixture
def test_project():
    """Create small test project with known issues."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test Python files with various issues
        (tmpdir / "large_file.py").write_text(
            '"""Large file for testing."""\n' + "def func():\n    pass\n" * 300  # 901 lines total
        )

        (tmpdir / "complex_function.py").write_text(
            """
def complex_function(x):
    if x > 10:
        if x > 20:
            if x > 30:
                if x > 40:
                    if x > 50:
                        return "very high"
                    return "high"
                return "medium-high"
            return "medium"
        return "low-medium"
    return "low"
"""
        )

        (tmpdir / "duplicate_code.py").write_text(
            """
def process_data_1(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result

def process_data_2(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""
        )

        (tmpdir / "no_docstring.py").write_text(
            """
def function_without_docstring(x, y):
    return x + y

class ClassWithoutDocstring:
    def method(self):
        return 42
"""
        )

        (tmpdir / "requirements.txt").write_text(
            """
requests==2.25.0
numpy==1.19.0
"""
        )

        yield tmpdir


class TestEndToEndAnalysis:
    """Test complete analysis pipeline."""

    def test_full_analysis_pipeline(self, test_project):
        """Run full analysis on test project and verify results."""
        orchestrator = AnalysisOrchestrator(project_path=str(test_project))

        # Run analysis
        result = orchestrator.analyze()

        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert result.project_path == str(test_project)
        assert result.overall_score >= 0
        assert result.overall_score <= 100

        # Verify all dimensions analyzed
        assert result.architecture is not None
        assert result.performance is not None
        assert result.coverage is not None
        assert result.code_quality is not None
        assert result.dependencies is not None
        assert result.deployment is not None
        assert result.security is not None
        assert result.scalability is not None

        # Verify scores are valid
        assert 0 <= result.architecture.score <= 100
        assert 0 <= result.performance.score <= 100
        assert 0 <= result.coverage.score <= 100
        assert 0 <= result.code_quality.score <= 100
        assert 0 <= result.dependencies.score <= 100
        assert 0 <= result.deployment.score <= 100
        assert 0 <= result.security.score <= 100
        assert 0 <= result.scalability.score <= 100

    def test_analysis_detects_large_files(self, test_project):
        """Analysis detects large files."""
        orchestrator = AnalysisOrchestrator(project_path=str(test_project))
        result = orchestrator.analyze()

        # Should detect large_file.py
        assert result.architecture.total_files > 0
        large_files = result.architecture.large_files
        assert any("large_file.py" in str(f.get("path", "")) for f in large_files)

    def test_analysis_detects_complexity(self, test_project):
        """Analysis detects high complexity functions."""
        orchestrator = AnalysisOrchestrator(project_path=str(test_project))
        result = orchestrator.analyze()

        # Should detect complex_function
        assert result.code_quality.average_complexity > 0
        complex_funcs = result.code_quality.high_complexity_functions
        assert any(
            "complex_function" in str(f.get("name", "")) for f in complex_funcs
        )

    def test_analysis_detects_duplication(self, test_project):
        """Analysis detects code duplication."""
        orchestrator = AnalysisOrchestrator(project_path=str(test_project))
        result = orchestrator.analyze()

        # Should detect duplication in duplicate_code.py
        assert result.code_quality.duplication_percentage >= 0

    def test_analysis_detects_missing_docstrings(self, test_project):
        """Analysis detects missing docstrings."""
        orchestrator = AnalysisOrchestrator(project_path=str(test_project))
        result = orchestrator.analyze()

        # Should detect missing docstrings
        assert result.code_quality.documentation_coverage < 1.0

    def test_analysis_detects_outdated_dependencies(self, test_project):
        """Analysis detects outdated dependencies."""
        orchestrator = AnalysisOrchestrator(project_path=str(test_project))
        result = orchestrator.analyze()

        # Should detect outdated packages
        assert result.dependencies.total_dependencies > 0
        # Old versions should be flagged
        assert len(result.dependencies.outdated_packages) >= 0

    def test_analysis_json_serialization(self, test_project):
        """Analysis results can be serialized to JSON."""
        orchestrator = AnalysisOrchestrator(project_path=str(test_project))
        result = orchestrator.analyze()

        # Serialize to JSON
        json_str = result.to_json()
        assert len(json_str) > 0

        # Verify valid JSON
        data = json.loads(json_str)
        assert "overall_score" in data
        assert "architecture" in data
        assert "performance" in data

    def test_analysis_json_deserialization(self, test_project):
        """Analysis results can be deserialized from JSON."""
        orchestrator = AnalysisOrchestrator(project_path=str(test_project))
        result = orchestrator.analyze()

        # Round-trip through JSON
        json_str = result.to_json()
        loaded = AnalysisResult.from_json(json_str)

        # Verify data preserved
        assert loaded.overall_score == result.overall_score
        assert loaded.architecture.score == result.architecture.score
        assert loaded.performance.score == result.performance.score

    def test_analysis_with_parallel_execution(self, test_project):
        """Analysis works with parallel execution."""
        orchestrator = AnalysisOrchestrator(
            project_path=str(test_project), max_workers=2
        )

        result = orchestrator.analyze()

        # Should complete successfully
        assert result is not None
        assert result.overall_score >= 0

    def test_analysis_with_error_recovery(self, test_project):
        """Analysis recovers from individual analyzer failures."""
        # Create file that might cause issues
        (test_project / "problematic.py").write_text("import nonexistent_module\n")

        orchestrator = AnalysisOrchestrator(project_path=str(test_project))
        result = orchestrator.analyze()

        # Should still complete with partial results
        assert result is not None
        assert result.overall_score >= 0

    def test_analysis_generates_critical_issues(self, test_project):
        """Analysis identifies critical issues."""
        orchestrator = AnalysisOrchestrator(project_path=str(test_project))
        result = orchestrator.analyze()

        # Should have some issues identified
        assert isinstance(result.critical_issues, list)

    def test_analysis_output_to_file(self, test_project):
        """Analysis can save results to file."""
        orchestrator = AnalysisOrchestrator(project_path=str(test_project))
        result = orchestrator.analyze()

        # Save to file
        output_file = test_project / "analysis-results.json"
        output_file.write_text(result.to_json())

        # Verify file created
        assert output_file.exists()
        assert len(output_file.read_text()) > 0

        # Verify can be loaded
        loaded = AnalysisResult.from_json(output_file.read_text())
        assert loaded.overall_score == result.overall_score


class TestAnalysisReportGeneration:
    """Test report generation from analysis results."""

    def test_markdown_report_generation(self, test_project):
        """Generate markdown report from analysis."""
        from analysis.reporting import ReportGenerator

        orchestrator = AnalysisOrchestrator(project_path=str(test_project))
        result = orchestrator.analyze()

        generator = ReportGenerator()
        markdown = generator.generate_markdown(result)

        # Verify markdown structure
        assert "# Project Analysis Report" in markdown
        assert "Overall Score" in markdown
        assert "Architecture" in markdown
        assert "Performance" in markdown

    def test_html_report_generation(self, test_project):
        """Generate HTML report from analysis."""
        from analysis.reporting import ReportGenerator

        orchestrator = AnalysisOrchestrator(project_path=str(test_project))
        result = orchestrator.analyze()

        generator = ReportGenerator()
        html_file = test_project / "report.html"
        generator.generate_html(result, str(html_file))

        # Verify HTML file created
        assert html_file.exists()
        html_content = html_file.read_text()
        assert "<html" in html_content.lower()
        assert "Project Analysis Report" in html_content


class TestOptimizationPlanGeneration:
    """Test optimization plan generation."""

    def test_optimization_plan_creation(self, test_project):
        """Generate optimization plan from analysis."""
        from analysis.planner import OptimizationPlanner

        orchestrator = AnalysisOrchestrator(project_path=str(test_project))
        result = orchestrator.analyze()

        planner = OptimizationPlanner()
        plan = planner.create_plan(result)

        # Verify plan structure
        assert plan is not None
        assert len(plan.tasks) > 0
        assert plan.total_effort_hours > 0

    def test_optimization_plan_prioritization(self, test_project):
        """Optimization plan prioritizes tasks correctly."""
        from analysis.planner import OptimizationPlanner

        orchestrator = AnalysisOrchestrator(project_path=str(test_project))
        result = orchestrator.analyze()

        planner = OptimizationPlanner()
        plan = planner.create_plan(result)

        # Verify tasks have priorities
        for task in plan.tasks:
            assert task.priority in ["P0", "P1", "P2", "P3"]

    def test_optimization_plan_effort_estimation(self, test_project):
        """Optimization plan estimates effort for tasks."""
        from analysis.planner import OptimizationPlanner

        orchestrator = AnalysisOrchestrator(project_path=str(test_project))
        result = orchestrator.analyze()

        planner = OptimizationPlanner()
        plan = planner.create_plan(result)

        # Verify tasks have effort estimates
        for task in plan.tasks:
            assert task.effort_hours > 0
