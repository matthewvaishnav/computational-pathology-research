"""
Integration tests for Analysis Orchestrator.

Tests parallel execution, error recovery, and CLI interface.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.analysis.orchestrator import AnalysisOrchestrator, main
from src.analysis.models import (
    AnalysisResult,
    ArchitectureAnalysis,
    PerformanceAnalysis,
)


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)

        # Create minimal project structure
        (project_path / "src").mkdir()
        (project_path / "tests").mkdir()
        (project_path / "src" / "__init__.py").touch()
        (project_path / "tests" / "__init__.py").touch()

        # Initialize git repo
        import subprocess

        subprocess.run(["git", "init"], cwd=project_path, check=False, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=project_path,
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=project_path,
            check=False,
            capture_output=True,
        )
        subprocess.run(["git", "add", "."], cwd=project_path, check=False, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"], cwd=project_path, check=False, capture_output=True
        )

        yield project_path


def test_orchestrator_initialization(temp_project_dir):
    """Test orchestrator initializes correctly."""
    orchestrator = AnalysisOrchestrator(str(temp_project_dir))

    assert orchestrator.project_path == temp_project_dir
    assert orchestrator.max_workers > 0
    assert orchestrator.max_workers <= 8
    assert len(orchestrator.timing_data) == 0
    assert len(orchestrator.errors) == 0


def test_orchestrator_parallel_execution(temp_project_dir):
    """Test parallel execution of analyzers."""
    orchestrator = AnalysisOrchestrator(str(temp_project_dir), max_workers=4)

    result = orchestrator.analyze_project(parallel=True)

    # Verify result structure
    assert isinstance(result, AnalysisResult)
    assert result.project_path == str(temp_project_dir)
    assert result.git_commit != "unknown"
    assert 0 <= result.overall_score <= 100

    # Verify all analyzers ran
    assert isinstance(result.architecture, ArchitectureAnalysis)
    assert isinstance(result.performance, PerformanceAnalysis)

    # Verify timing data collected
    assert len(orchestrator.timing_data) == 8
    assert "architecture" in orchestrator.timing_data
    assert "performance" in orchestrator.timing_data


def test_orchestrator_sequential_execution(temp_project_dir):
    """Test sequential execution of analyzers."""
    orchestrator = AnalysisOrchestrator(str(temp_project_dir))

    result = orchestrator.analyze_project(parallel=False)

    # Verify result structure
    assert isinstance(result, AnalysisResult)
    assert result.project_path == str(temp_project_dir)

    # Verify all analyzers ran
    assert len(orchestrator.timing_data) == 8


def test_orchestrator_error_recovery(temp_project_dir):
    """Test graceful degradation when analyzer fails."""
    orchestrator = AnalysisOrchestrator(str(temp_project_dir))

    # Mock one analyzer to fail
    original_func = orchestrator._create_stub_architecture_analysis

    def failing_analyzer():
        raise RuntimeError("Simulated analyzer failure")

    orchestrator._create_stub_architecture_analysis = failing_analyzer

    # Should not raise exception, should use stub
    result = orchestrator.analyze_project(parallel=False)

    # Verify error recorded
    assert len(orchestrator.errors) > 0
    assert any("architecture" in err["analyzer"] for err in orchestrator.errors)

    # Verify result still valid (graceful degradation)
    assert isinstance(result, AnalysisResult)
    assert isinstance(result.architecture, ArchitectureAnalysis)

    # Restore original
    orchestrator._create_stub_architecture_analysis = original_func


def test_orchestrator_git_commit_detection(temp_project_dir):
    """Test git commit hash detection."""
    orchestrator = AnalysisOrchestrator(str(temp_project_dir))

    commit_hash = orchestrator._get_git_commit()

    # Should be valid git hash (40 hex chars) or 'unknown'
    assert commit_hash == "unknown" or (
        len(commit_hash) == 40 and all(c in "0123456789abcdef" for c in commit_hash)
    )


def test_orchestrator_overall_score_calculation(temp_project_dir):
    """Test overall score calculation with weighted dimensions."""
    orchestrator = AnalysisOrchestrator(str(temp_project_dir))

    # Create mock results with known scores
    results = {
        "architecture": ArchitectureAnalysis(score=80.0),
        "performance": PerformanceAnalysis(score=60.0),
        "coverage": MagicMock(score=70.0),
        "code_quality": MagicMock(score=75.0),
        "dependencies": MagicMock(score=90.0),
        "deployment": MagicMock(score=50.0),
        "security": MagicMock(score=85.0),
        "scalability": MagicMock(score=65.0),
    }

    overall_score = orchestrator._calculate_overall_score(results)

    # Verify weighted calculation
    # Security (20%) + Coverage (15%) + Code Quality (15%) + Architecture (15%) +
    # Performance (10%) + Dependencies (10%) + Deployment (10%) + Scalability (5%)
    expected = (
        85 * 0.20
        + 70 * 0.15
        + 75 * 0.15
        + 80 * 0.15
        + 60 * 0.10
        + 90 * 0.10
        + 50 * 0.10
        + 65 * 0.05
    )
    assert abs(overall_score - expected) < 0.1


def test_orchestrator_json_output(temp_project_dir):
    """Test JSON serialization of results."""
    orchestrator = AnalysisOrchestrator(str(temp_project_dir))

    result = orchestrator.analyze_project(parallel=False)

    # Serialize to JSON
    json_str = result.to_json(validate_schema=True)

    # Verify valid JSON
    data = json.loads(json_str)
    assert "timestamp" in data
    assert "project_path" in data
    assert "git_commit" in data
    assert "overall_score" in data

    # Verify all dimensions present
    assert "architecture" in data
    assert "performance" in data
    assert "coverage" in data
    assert "code_quality" in data
    assert "dependencies" in data
    assert "deployment" in data
    assert "security" in data
    assert "scalability" in data


def test_cli_argument_parsing():
    """Test CLI argument parsing."""
    with patch(
        "sys.argv",
        ["orchestrator.py", ".", "--output", "test.json", "--format", "json", "--no-parallel"],
    ):
        with patch("src.analysis.orchestrator.AnalysisOrchestrator") as mock_orchestrator:
            mock_instance = MagicMock()
            mock_orchestrator.return_value = mock_instance
            mock_instance.analyze_project.return_value = MagicMock(
                to_json=MagicMock(return_value="{}"), overall_score=75.0
            )
            mock_instance.timing_data = {"test": 1.0}
            mock_instance.errors = []

            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.write_text"):
                    try:
                        main()
                    except SystemExit:
                        pass

            # Verify orchestrator called with correct args
            mock_orchestrator.assert_called_once()
            mock_instance.analyze_project.assert_called_once_with(parallel=False)


def test_cli_invalid_project_path():
    """Test CLI with invalid project path."""
    with patch("sys.argv", ["orchestrator.py", "/nonexistent/path"]):
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1


def test_orchestrator_max_workers_limit(temp_project_dir):
    """Test max_workers respects limit."""
    # Request more workers than limit
    orchestrator = AnalysisOrchestrator(str(temp_project_dir), max_workers=100)

    # Should be capped at 8
    assert orchestrator.max_workers <= 8


def test_orchestrator_timing_data_collection(temp_project_dir):
    """Test timing data collected for all analyzers."""
    orchestrator = AnalysisOrchestrator(str(temp_project_dir))

    result = orchestrator.analyze_project(parallel=False)

    # Verify timing data for all analyzers
    expected_analyzers = [
        "architecture",
        "performance",
        "coverage",
        "code_quality",
        "dependencies",
        "deployment",
        "security",
        "scalability",
    ]

    for analyzer in expected_analyzers:
        assert analyzer in orchestrator.timing_data
        assert orchestrator.timing_data[analyzer] >= 0


def test_orchestrator_cpu_count_fallback():
    """Test CPU count detection with fallback."""
    with patch("os.cpu_count", return_value=None):
        orchestrator = AnalysisOrchestrator(".")
        assert orchestrator.max_workers == 4  # Fallback value
