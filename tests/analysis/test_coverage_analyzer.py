"""
Unit tests for Coverage Analyzer.

Tests coverage data parsing, critical path detection, and quality metrics calculation.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.analysis.coverage import CoverageAnalyzer
from src.analysis.models import CoverageAnalysis


class TestCoverageAnalyzer:
    """Test suite for CoverageAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.analyzer = CoverageAnalyzer(str(self.project_path))

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = CoverageAnalyzer("/path/to/project")
        assert analyzer.project_path == Path("/path/to/project").resolve()

    def test_analyze_returns_coverage_analysis(self):
        """Test that analyze() returns a CoverageAnalysis object."""
        with patch.object(self.analyzer, "_run_coverage", return_value=(75.5, 68.2)):
            with patch.object(self.analyzer, "_detect_untested_critical_paths", return_value=[]):
                with patch.object(self.analyzer, "_detect_missing_property_tests", return_value=[]):
                    with patch.object(self.analyzer, "_detect_flaky_tests", return_value=[]):
                        with patch.object(self.analyzer, "_detect_slow_tests", return_value=[]):
                            result = self.analyzer.analyze()

        assert isinstance(result, CoverageAnalysis)
        assert result.line_coverage == 75.5
        assert result.branch_coverage == 68.2
        assert result.untested_critical_paths == []
        assert result.missing_property_tests == []
        assert result.flaky_tests == []
        assert result.slow_tests == []
        assert isinstance(result.score, float)
        assert 0 <= result.score <= 100


class TestCoverageDataParsing:
    """Test coverage data parsing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.analyzer = CoverageAnalyzer(str(self.project_path))

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_coverage_with_existing_coverage_file(self):
        """Test parsing existing coverage data."""
        # Create mock .coverage file
        coverage_file = self.project_path / ".coverage"
        coverage_file.touch()

        # Mock coverage JSON data
        coverage_data = {"totals": {"percent_covered": 85.7, "percent_covered_display": 82.3}}

        # Mock subprocess.run to return success
        mock_result = Mock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            # Create mock coverage.json file
            coverage_json = self.project_path / "coverage.json"
            coverage_json.write_text(json.dumps(coverage_data))

            line_cov, branch_cov = self.analyzer._run_coverage()

        assert line_cov == 85.7
        assert branch_cov == 82.3

    def test_run_coverage_no_coverage_file(self):
        """Test behavior when no coverage file exists."""
        line_cov, branch_cov = self.analyzer._run_coverage()

        assert line_cov == 0.0
        assert branch_cov == 0.0

    def test_run_coverage_subprocess_failure(self):
        """Test handling of subprocess failure."""
        # Create mock .coverage file
        coverage_file = self.project_path / ".coverage"
        coverage_file.touch()

        # Mock subprocess.run to return failure
        mock_result = Mock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            line_cov, branch_cov = self.analyzer._run_coverage()

        assert line_cov == 0.0
        assert branch_cov == 0.0

    def test_run_coverage_timeout_handling(self):
        """Test handling of subprocess timeout."""
        # Create mock .coverage file
        coverage_file = self.project_path / ".coverage"
        coverage_file.touch()

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("coverage", 60)):
            line_cov, branch_cov = self.analyzer._run_coverage()

        assert line_cov == 0.0
        assert branch_cov == 0.0

    def test_run_coverage_json_decode_error(self):
        """Test handling of malformed JSON."""
        # Create mock .coverage file
        coverage_file = self.project_path / ".coverage"
        coverage_file.touch()

        # Mock subprocess.run to return success
        mock_result = Mock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            # Create malformed coverage.json file
            coverage_json = self.project_path / "coverage.json"
            coverage_json.write_text("invalid json {")

            line_cov, branch_cov = self.analyzer._run_coverage()

        assert line_cov == 0.0
        assert branch_cov == 0.0

    def test_run_coverage_missing_totals(self):
        """Test handling of coverage data without totals."""
        # Create mock .coverage file
        coverage_file = self.project_path / ".coverage"
        coverage_file.touch()

        # Mock coverage JSON data without totals
        coverage_data = {"files": {}}

        # Mock subprocess.run to return success
        mock_result = Mock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            # Create coverage.json file without totals
            coverage_json = self.project_path / "coverage.json"
            coverage_json.write_text(json.dumps(coverage_data))

            line_cov, branch_cov = self.analyzer._run_coverage()

        assert line_cov == 0.0
        assert branch_cov == 0.0

    def test_run_coverage_rounds_percentages(self):
        """Test that coverage percentages are properly rounded."""
        # Create mock .coverage file
        coverage_file = self.project_path / ".coverage"
        coverage_file.touch()

        # Mock coverage JSON data with precise decimals
        coverage_data = {
            "totals": {"percent_covered": 85.7654321, "percent_covered_display": 82.3456789}
        }

        # Mock subprocess.run to return success
        mock_result = Mock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            # Create mock coverage.json file
            coverage_json = self.project_path / "coverage.json"
            coverage_json.write_text(json.dumps(coverage_data))

            line_cov, branch_cov = self.analyzer._run_coverage()

        assert line_cov == 85.77  # Rounded to 2 decimal places
        assert branch_cov == 82.35  # Rounded to 2 decimal places


class TestCriticalPathDetection:
    """Test critical path detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.analyzer = CoverageAnalyzer(str(self.project_path))

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_detect_untested_critical_paths_placeholder(self):
        """Test untested critical path detection (currently placeholder)."""
        paths = self.analyzer._detect_untested_critical_paths()

        # Currently returns empty list as placeholder
        assert isinstance(paths, list)
        assert paths == []

    def test_detect_missing_property_tests_placeholder(self):
        """Test missing property test detection (currently placeholder)."""
        missing_tests = self.analyzer._detect_missing_property_tests()

        # Currently returns empty list as placeholder
        assert isinstance(missing_tests, list)
        assert missing_tests == []

    def test_detect_flaky_tests_placeholder(self):
        """Test flaky test detection (currently placeholder)."""
        flaky_tests = self.analyzer._detect_flaky_tests()

        # Currently returns empty list as placeholder
        assert isinstance(flaky_tests, list)
        assert flaky_tests == []

    def test_detect_slow_tests_placeholder(self):
        """Test slow test detection (currently placeholder)."""
        slow_tests = self.analyzer._detect_slow_tests()

        # Currently returns empty list as placeholder
        assert isinstance(slow_tests, list)
        assert slow_tests == []


class TestQualityMetricsCalculation:
    """Test quality metrics calculation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.analyzer = CoverageAnalyzer(str(self.project_path))

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_calculate_coverage_score_perfect_coverage(self):
        """Test score calculation with perfect coverage."""
        score = self.analyzer._calculate_coverage_score(
            line_cov=70.0,  # Target line coverage
            branch_cov=60.0,  # Target branch coverage
            untested_paths=[],
        )

        # Base (20) + Line (50) + Branch (30) = 100
        assert score == 100.0

    def test_calculate_coverage_score_no_coverage(self):
        """Test score calculation with no coverage."""
        score = self.analyzer._calculate_coverage_score(
            line_cov=0.0, branch_cov=0.0, untested_paths=[]
        )

        # Only base score (20)
        assert score == 20.0

    def test_calculate_coverage_score_partial_coverage(self):
        """Test score calculation with partial coverage."""
        score = self.analyzer._calculate_coverage_score(
            line_cov=35.0,  # Half of target (70%)
            branch_cov=30.0,  # Half of target (60%)
            untested_paths=[],
        )

        # Base (20) + Line (25) + Branch (15) = 60
        assert score == 60.0

    def test_calculate_coverage_score_with_untested_paths(self):
        """Test score calculation with untested critical paths."""
        score = self.analyzer._calculate_coverage_score(
            line_cov=70.0,
            branch_cov=60.0,
            untested_paths=["path1", "path2"],  # 2 paths = -10 points
        )

        # Base (20) + Line (50) + Branch (30) - Penalty (10) = 90
        assert score == 90.0

    def test_calculate_coverage_score_excessive_penalty(self):
        """Test score calculation with excessive untested paths."""
        score = self.analyzer._calculate_coverage_score(
            line_cov=70.0,
            branch_cov=60.0,
            untested_paths=["path" + str(i) for i in range(10)],  # 10 paths = -50, capped at -20
        )

        # Base (20) + Line (50) + Branch (30) - Max Penalty (20) = 80
        assert score == 80.0

    def test_calculate_coverage_score_above_target(self):
        """Test score calculation with coverage above target."""
        score = self.analyzer._calculate_coverage_score(
            line_cov=100.0, branch_cov=80.0, untested_paths=[]  # Above target  # Above target
        )

        # Line and branch scores are capped at target values
        # Base (20) + Line (50) + Branch (30) = 100
        assert score == 100.0

    def test_calculate_coverage_score_minimum_bound(self):
        """Test that score never goes below 0."""
        score = self.analyzer._calculate_coverage_score(
            line_cov=0.0,
            branch_cov=0.0,
            untested_paths=["path" + str(i) for i in range(20)],  # Excessive penalty
        )

        # Score should be bounded at 0
        assert score == 0.0

    def test_calculate_coverage_score_maximum_bound(self):
        """Test that score never goes above 100."""
        # This test ensures the max() function works correctly
        score = self.analyzer._calculate_coverage_score(
            line_cov=1000.0,  # Unrealistic high value
            branch_cov=1000.0,  # Unrealistic high value
            untested_paths=[],
        )

        # Score should be bounded at 100
        assert score == 100.0

    def test_calculate_coverage_score_rounding(self):
        """Test that score is properly rounded to 2 decimal places."""
        score = self.analyzer._calculate_coverage_score(
            line_cov=35.333, branch_cov=30.666, untested_paths=[]  # Results in fractional score
        )

        # Should be rounded to 2 decimal places
        assert isinstance(score, float)
        assert len(str(score).split(".")[-1]) <= 2


class TestSampleASTAnalysis:
    """Test critical path detection with sample AST structures."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.analyzer = CoverageAnalyzer(str(self.project_path))

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sample_python_file_creation(self):
        """Test creating sample Python files for AST analysis."""
        # Create a sample Python file with error handling
        sample_file = self.project_path / "sample.py"
        sample_code = '''
def process_data(data):
    """Process data with error handling."""
    try:
        if not data:
            raise ValueError("Data cannot be empty")
        
        result = []
        for item in data:
            if item < 0:
                raise ValueError(f"Negative value: {item}")
            result.append(item * 2)
        
        return result
    except ValueError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
'''
        sample_file.write_text(sample_code)

        # Verify file was created
        assert sample_file.exists()
        assert "try:" in sample_file.read_text()
        assert "except" in sample_file.read_text()

    def test_sample_ast_parsing_structure(self):
        """Test that we can identify AST structures for future implementation."""
        # This test demonstrates the kind of AST analysis needed
        # for critical path detection (to be implemented later)

        sample_file = self.project_path / "complex_sample.py"
        sample_code = '''
import ast

def complex_function(x, y=None):
    """Function with multiple paths."""
    if x is None:
        return None
    
    try:
        if y is None:
            y = x * 2
        
        if x < 0:
            raise ValueError("Negative input")
        
        result = x + y
        
        if result > 100:
            return min(result, 100)
        
        return result
    
    except ValueError:
        return -1
    except Exception:
        return -2
    finally:
        pass
'''
        sample_file.write_text(sample_code)

        # Parse the AST to identify critical paths
        import ast

        tree = ast.parse(sample_code)

        # Count different node types that represent critical paths
        node_counts = {}
        for node in ast.walk(tree):
            node_type = type(node).__name__
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

        # Verify we can identify critical constructs
        assert "Try" in node_counts  # Exception handling
        assert "If" in node_counts  # Conditional branches
        assert "ExceptHandler" in node_counts  # Exception handlers
        assert node_counts["If"] >= 3  # Multiple conditional paths
        assert node_counts["ExceptHandler"] >= 2  # Multiple exception handlers


class TestIntegrationWithMockData:
    """Integration tests with mock coverage data."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.analyzer = CoverageAnalyzer(str(self.project_path))

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_analysis_with_mock_data(self):
        """Test complete analysis workflow with mocked data."""
        # Mock all the detection methods
        with patch.object(self.analyzer, "_run_coverage", return_value=(65.5, 58.2)):
            with patch.object(
                self.analyzer,
                "_detect_untested_critical_paths",
                return_value=["src/module1.py:45", "src/module2.py:123"],
            ):
                with patch.object(
                    self.analyzer,
                    "_detect_missing_property_tests",
                    return_value=["test_data_transform", "test_validation"],
                ):
                    with patch.object(
                        self.analyzer, "_detect_flaky_tests", return_value=["test_flaky_network"]
                    ):
                        with patch.object(
                            self.analyzer,
                            "_detect_slow_tests",
                            return_value=[{"name": "test_slow", "duration": 12.5}],
                        ):

                            result = self.analyzer.analyze()

        # Verify all fields are populated
        assert result.line_coverage == 65.5
        assert result.branch_coverage == 58.2
        assert len(result.untested_critical_paths) == 2
        assert len(result.missing_property_tests) == 2
        assert len(result.flaky_tests) == 1
        assert len(result.slow_tests) == 1

        # Verify score calculation
        expected_score = self.analyzer._calculate_coverage_score(
            65.5, 58.2, result.untested_critical_paths
        )
        assert result.score == expected_score

    def test_analysis_with_high_quality_metrics(self):
        """Test analysis with high-quality test metrics."""
        with patch.object(self.analyzer, "_run_coverage", return_value=(85.0, 78.0)):
            with patch.object(self.analyzer, "_detect_untested_critical_paths", return_value=[]):
                with patch.object(self.analyzer, "_detect_missing_property_tests", return_value=[]):
                    with patch.object(self.analyzer, "_detect_flaky_tests", return_value=[]):
                        with patch.object(self.analyzer, "_detect_slow_tests", return_value=[]):

                            result = self.analyzer.analyze()

        # High coverage should result in high score
        assert result.score > 90.0
        assert result.line_coverage == 85.0
        assert result.branch_coverage == 78.0

    def test_analysis_with_poor_quality_metrics(self):
        """Test analysis with poor test metrics."""
        with patch.object(self.analyzer, "_run_coverage", return_value=(25.0, 18.0)):
            with patch.object(
                self.analyzer,
                "_detect_untested_critical_paths",
                return_value=["path1", "path2", "path3", "path4"],
            ):
                with patch.object(
                    self.analyzer,
                    "_detect_missing_property_tests",
                    return_value=["test1", "test2", "test3"],
                ):
                    with patch.object(
                        self.analyzer, "_detect_flaky_tests", return_value=["flaky1", "flaky2"]
                    ):
                        with patch.object(
                            self.analyzer,
                            "_detect_slow_tests",
                            return_value=[{"name": "slow1", "duration": 30.0}],
                        ):

                            result = self.analyzer.analyze()

        # Poor coverage should result in low score
        assert result.score < 50.0
        assert len(result.untested_critical_paths) == 4
        assert len(result.missing_property_tests) == 3
