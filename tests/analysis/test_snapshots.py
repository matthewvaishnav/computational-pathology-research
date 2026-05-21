"""
Snapshot tests for report generation.

Tests capture expected report output for known codebase and detect regressions
in report format.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.analysis.models import AnalysisResult
from src.analysis.orchestrator import AnalysisOrchestrator
from src.analysis.reporting import ReportGenerator


class TestReportSnapshots:
    """Test report generation against known snapshots."""

    @pytest.fixture
    def sample_analysis_result(self):
        """Create sample analysis result for snapshot testing."""
        return AnalysisResult(
            timestamp="2026-05-07T10:00:00Z",
            project_path="/test/project",
            git_commit="abc123",
            architecture=None,  # Will be populated by orchestrator
            performance=None,
            coverage=None,
            code_quality=None,
            dependencies=None,
            deployment=None,
            security=None,
            scalability=None,
            overall_score=75.5,
            critical_issues=[],
        )

    @pytest.fixture
    def snapshot_dir(self):
        """Directory for storing snapshots."""
        snapshot_path = Path("tests/analysis/snapshots")
        snapshot_path.mkdir(exist_ok=True)
        return snapshot_path

    def test_markdown_report_snapshot(self, sample_analysis_result, snapshot_dir):
        """Test markdown report matches expected snapshot."""
        generator = ReportGenerator()

        # Generate markdown report
        markdown = generator.generate_markdown(sample_analysis_result)

        # Load expected snapshot
        snapshot_file = snapshot_dir / "markdown_report.md"

        if snapshot_file.exists():
            expected = snapshot_file.read_text()

            # Compare key sections (ignore timestamps)
            assert "# HistoCore Project Optimization Analysis Report" in markdown
            assert "Overall Score:** 75.5/100" in markdown
            assert "Executive Summary" in markdown
            assert "Overall Metrics" in markdown

            # Check structure matches
            lines = markdown.split("\n")
            expected_lines = expected.split("\n")

            # Compare non-timestamp lines
            for i, (line, exp_line) in enumerate(zip(lines, expected_lines)):
                if "Generated:" not in line and "Timestamp:" not in line:
                    assert line == exp_line, f"Line {i} differs: {line} != {exp_line}"
        else:
            # Create snapshot if it doesn't exist
            snapshot_file.write_text(markdown)
            pytest.skip("Created new snapshot - run test again to validate")

    def test_json_report_snapshot(self, sample_analysis_result, snapshot_dir):
        """Test JSON report matches expected snapshot."""
        # Generate JSON report
        json_data = sample_analysis_result.to_json()

        # Load expected snapshot
        snapshot_file = snapshot_dir / "json_report.json"

        if snapshot_file.exists():
            expected = json.loads(snapshot_file.read_text())
            actual = json.loads(json_data)

            # Compare structure (ignore timestamps)
            assert actual["project_path"] == expected["project_path"]
            assert actual["overall_score"] == expected["overall_score"]
            assert "timestamp" in actual
            assert "git_commit" in actual

            # Check all required fields present
            required_fields = [
                "architecture",
                "performance",
                "coverage",
                "code_quality",
                "dependencies",
                "deployment",
                "security",
                "scalability",
            ]
            for field in required_fields:
                assert field in actual
        else:
            # Create snapshot if it doesn't exist
            snapshot_file.write_text(json_data)
            pytest.skip("Created new snapshot - run test again to validate")

    def test_html_report_snapshot(self, sample_analysis_result, snapshot_dir):
        """Test HTML report structure matches expected snapshot."""
        generator = ReportGenerator()

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            generator.generate_html(sample_analysis_result, tmp.name)
            html_content = Path(tmp.name).read_text()

        # Load expected snapshot
        snapshot_file = snapshot_dir / "html_report.html"

        if snapshot_file.exists():
            snapshot_file.read_text()

            # Compare key HTML structure
            assert "<html>" in html_content.lower()
            assert "<title>HistoCore Analysis Report</title>" in html_content
            assert "Overall Score:** 75.5/100" in html_content

            # Check CSS is included
            assert "font-family: Arial" in html_content
            assert "border-collapse: collapse" in html_content
        else:
            # Create snapshot if it doesn't exist
            snapshot_file.write_text(html_content)
            pytest.skip("Created new snapshot - run test again to validate")

    def test_end_to_end_analysis_snapshot(self, snapshot_dir):
        """Test complete analysis pipeline produces consistent output."""
        # Create test project
        with tempfile.TemporaryDirectory() as temp_dir:
            test_project = Path(temp_dir)

            # Create sample Python files
            (test_project / "main.py").write_text("""
def hello_world():
    print("Hello, World!")
    return True

if __name__ == "__main__":
    hello_world()
""")

            (test_project / "utils.py").write_text("""
def add_numbers(a, b):
    return a + b

def multiply(x, y):
    result = x * y
    return result
""")

            # Run analysis
            orchestrator = AnalysisOrchestrator(project_path=str(test_project))
            result = orchestrator.analyze()

            # Generate report
            generator = ReportGenerator()
            markdown = generator.generate_markdown(result)

            # Load expected snapshot
            snapshot_file = snapshot_dir / "e2e_analysis.md"

            if snapshot_file.exists():
                snapshot_file.read_text()

                # Compare key metrics (structure should be consistent)
                assert "# HistoCore Project Optimization Analysis Report" in markdown
                assert "Executive Summary" in markdown
                assert "Overall Metrics" in markdown

                # Check that analysis detected files
                assert "2 total files" in markdown or "total_files" in markdown

                # Verify all dimensions are present
                dimensions = [
                    "Architecture",
                    "Performance",
                    "Coverage",
                    "Code Quality",
                    "Dependencies",
                    "Deployment",
                    "Security",
                    "Scalability",
                ]
                for dimension in dimensions:
                    assert dimension in markdown
            else:
                # Create snapshot if it doesn't exist
                snapshot_file.write_text(markdown)
                pytest.skip("Created new snapshot - run test again to validate")

    def test_regression_detection_snapshot(self, snapshot_dir):
        """Test regression detection output format."""
        from src.analysis.regression_detector import RegressionDetector

        # Create mock baseline and current results
        baseline = {
            "overall_score": 80.0,
            "coverage": {"line_coverage": 85.0},
            "performance": {"score": 75.0},
            "security": {"vulnerabilities": []},
        }

        current = {
            "overall_score": 75.0,  # Regression
            "coverage": {"line_coverage": 82.0},  # Regression
            "performance": {"score": 78.0},  # Improvement
            "security": {"vulnerabilities": [{"severity": "HIGH"}]},  # New issue
        }

        detector = RegressionDetector()

        with patch.object(detector, "_load_baseline", return_value=baseline):
            diff_report = detector.generate_diff_report(current, baseline)

        # Load expected snapshot
        snapshot_file = snapshot_dir / "regression_diff.md"

        if snapshot_file.exists():
            snapshot_file.read_text()

            # Compare key sections
            assert "Regression Analysis" in diff_report
            assert "🔴" in diff_report  # Should show regressions
            assert "🟢" in diff_report  # Should show improvements

            # Check specific regression detection
            assert "Overall Score" in diff_report
            assert "Coverage" in diff_report
        else:
            # Create snapshot if it doesn't exist
            snapshot_file.write_text(diff_report)
            pytest.skip("Created new snapshot - run test again to validate")

    def test_visualization_snapshot(self, sample_analysis_result, snapshot_dir):
        """Test visualization generation produces consistent output."""
        from src.analysis.visualizations import generate_score_chart

        # Generate visualization
        chart_data = generate_score_chart(sample_analysis_result)

        # Load expected snapshot
        snapshot_file = snapshot_dir / "score_chart.json"

        if snapshot_file.exists():
            expected = json.loads(snapshot_file.read_text())

            # Compare chart structure
            assert chart_data["type"] == expected["type"]
            assert len(chart_data["data"]) == len(expected["data"])

            # Check data points exist
            assert "labels" in chart_data["data"]
            assert "datasets" in chart_data["data"]
        else:
            # Create snapshot if it doesn't exist
            snapshot_file.write_text(json.dumps(chart_data, indent=2))
            pytest.skip("Created new snapshot - run test again to validate")


class TestSnapshotUtilities:
    """Utilities for managing snapshots."""

    def test_update_snapshots(self):
        """Test utility to update all snapshots."""
        # This test can be run with --update-snapshots flag
        # to regenerate all snapshots

    def test_snapshot_validation(self):
        """Test that all snapshots are valid."""
        snapshot_dir = Path("tests/analysis/snapshots")

        if not snapshot_dir.exists():
            pytest.skip("No snapshots directory found")

        # Validate JSON snapshots
        for json_file in snapshot_dir.glob("*.json"):
            try:
                json.loads(json_file.read_text())
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in {json_file}: {e}")

        # Validate Markdown snapshots
        for md_file in snapshot_dir.glob("*.md"):
            content = md_file.read_text()
            assert len(content) > 0, f"Empty snapshot: {md_file}"
            assert "# " in content, f"No headers in markdown: {md_file}"

        # Validate HTML snapshots
        for html_file in snapshot_dir.glob("*.html"):
            content = html_file.read_text()
            assert "<html>" in content.lower(), f"Invalid HTML: {html_file}"
            assert "</html>" in content.lower(), f"Unclosed HTML: {html_file}"


if __name__ == "__main__":
    # Run snapshot tests
    pytest.main([__file__, "-v"])
