"""
Unit tests for Architecture Analyzer.

Tests large file detection, circular dependency detection, coupling metrics,
and SOLID principle violation detection.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.architecture import ArchitectureAnalyzer
from src.analysis.models import ArchitectureAnalysis, Issue, Priority, Role, Severity


class TestArchitectureAnalyzer:
    """Test suite for ArchitectureAnalyzer."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)

        # Create basic project structure
        (self.project_path / "src").mkdir()
        (self.project_path / "tests").mkdir()

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def create_python_file(self, path: str, content: str, lines: int = None):
        """Create a Python file with specified content."""
        file_path = self.project_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if lines:
            # Create file with specific number of lines
            content = "\n".join([f"# Line {i+1}" for i in range(lines)])

        file_path.write_text(content)
        return file_path

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = ArchitectureAnalyzer(str(self.project_path))

        assert analyzer.project_path == self.project_path
        assert analyzer.python_files == []
        assert analyzer.import_graph == {}

    def test_discover_python_files(self):
        """Test Python file discovery."""
        # Create test files
        self.create_python_file("src/main.py", 'print("hello")')
        self.create_python_file("src/utils/helper.py", "def helper(): pass")
        self.create_python_file("tests/test_main.py", "def test(): pass")

        # Create non-Python files (should be ignored)
        (self.project_path / "README.md").write_text("# README")
        (self.project_path / "config.json").write_text("{}")

        # Create files in excluded directories (should be ignored)
        (self.project_path / ".venv").mkdir()
        self.create_python_file(".venv/lib/python.py", "import sys")
        (self.project_path / "__pycache__").mkdir()
        self.create_python_file("__pycache__/cache.py", "cached = True")

        analyzer = ArchitectureAnalyzer(str(self.project_path))
        analyzer._discover_python_files()

        # Should find 3 Python files, excluding .venv and __pycache__
        assert len(analyzer.python_files) == 3

        file_names = [f.name for f in analyzer.python_files]
        assert "main.py" in file_names
        assert "helper.py" in file_names
        assert "test_main.py" in file_names
        assert "python.py" not in file_names  # Excluded from .venv
        assert "cache.py" not in file_names  # Excluded from __pycache__

    def test_get_module_name(self):
        """Test module name conversion."""
        analyzer = ArchitectureAnalyzer(str(self.project_path))

        # Test regular module
        file_path = self.project_path / "src" / "utils" / "helper.py"
        module_name = analyzer._get_module_name(file_path)
        assert module_name == "src.utils.helper"

        # Test __init__.py file
        init_path = self.project_path / "src" / "models" / "__init__.py"
        module_name = analyzer._get_module_name(init_path)
        assert module_name == "src.models"

        # Test root level file
        root_file = self.project_path / "main.py"
        module_name = analyzer._get_module_name(root_file)
        assert module_name == "main"

    def test_build_import_graph(self):
        """Test import graph building."""
        # Create files with imports
        self.create_python_file(
            "src/main.py",
            """
import os
import sys
from src.utils import helper
from src.models.base import BaseModel
""",
        )

        self.create_python_file(
            "src/utils/helper.py",
            """
import json
from src.models import base
""",
        )

        self.create_python_file(
            "src/models/base.py",
            """
import numpy as np
""",
        )

        analyzer = ArchitectureAnalyzer(str(self.project_path))
        analyzer._discover_python_files()
        analyzer._build_import_graph()

        # Check import graph structure
        assert "src.main" in analyzer.import_graph
        assert "src.utils.helper" in analyzer.import_graph
        assert "src.models.base" in analyzer.import_graph

        # Check imports for main module
        main_imports = analyzer.import_graph["src.main"]
        assert "os" in main_imports
        assert "sys" in main_imports
        assert "src.utils" in main_imports
        assert "src.models.base" in main_imports

        # Check imports for helper module
        helper_imports = analyzer.import_graph["src.utils.helper"]
        assert "json" in helper_imports
        assert "src.models" in helper_imports

    def test_build_import_graph_syntax_error(self):
        """Test import graph building with syntax errors."""
        # Create file with syntax error
        self.create_python_file(
            "src/broken.py",
            """
import os
def broken_function(
    # Missing closing parenthesis
""",
        )

        self.create_python_file("src/good.py", "import sys")

        analyzer = ArchitectureAnalyzer(str(self.project_path))
        analyzer._discover_python_files()

        # Should not raise exception, just skip broken files
        analyzer._build_import_graph()

        # Should still process good files
        assert "src.good" in analyzer.import_graph
        assert "src.broken" not in analyzer.import_graph

    def test_detect_large_files(self):
        """Test large file detection."""
        # Create small file (should not be flagged)
        self.create_python_file("src/small.py", content="", lines=100)

        # Create large files (should be flagged)
        self.create_python_file("src/large1.py", content="", lines=600)
        self.create_python_file("src/large2.py", content="", lines=1000)

        analyzer = ArchitectureAnalyzer(str(self.project_path))
        analyzer._discover_python_files()

        with patch.object(analyzer, "_get_file_complexity", return_value=5.2):
            large_files = analyzer._detect_large_files()

        # Should find 2 large files
        assert len(large_files) == 2

        # Check sorting (largest first)
        assert large_files[0]["lines"] == 1000
        assert large_files[1]["lines"] == 600

        # Check file paths (normalize path separators)
        paths = [lf["path"].replace("\\", "/") for lf in large_files]
        assert "src/large2.py" in paths
        assert "src/large1.py" in paths
        assert "src/small.py" not in [lf["path"].replace("\\", "/") for lf in large_files]

        # Check complexity is included
        for large_file in large_files:
            assert "complexity" in large_file
            assert large_file["complexity"] == 5.2

    @patch("subprocess.run")
    def test_get_file_complexity_success(self, mock_run):
        """Test file complexity calculation success."""
        # Mock successful radon output
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = """
src/test.py
    F 1:0 test_function - A
Average complexity: A (5.2)
"""

        analyzer = ArchitectureAnalyzer(str(self.project_path))
        file_path = self.project_path / "src" / "test.py"

        complexity = analyzer._get_file_complexity(file_path)

        assert complexity == 5.2
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_get_file_complexity_failure(self, mock_run):
        """Test file complexity calculation failure."""
        # Mock failed radon execution
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""

        analyzer = ArchitectureAnalyzer(str(self.project_path))
        file_path = self.project_path / "src" / "test.py"

        complexity = analyzer._get_file_complexity(file_path)

        assert complexity == 0.0

    @patch("subprocess.run")
    def test_get_file_complexity_timeout(self, mock_run):
        """Test file complexity calculation timeout."""
        # Mock timeout exception
        mock_run.side_effect = subprocess.TimeoutExpired("radon", 30)

        analyzer = ArchitectureAnalyzer(str(self.project_path))
        file_path = self.project_path / "src" / "test.py"

        complexity = analyzer._get_file_complexity(file_path)

        assert complexity == 0.0

    def test_detect_circular_dependencies_no_cycles(self):
        """Test circular dependency detection with no cycles."""
        # Create linear dependency chain: A -> B -> C
        self.create_python_file("src/a.py", "from src.b import func_b")
        self.create_python_file("src/b.py", "from src.c import func_c")
        self.create_python_file("src/c.py", "def func_c(): pass")

        analyzer = ArchitectureAnalyzer(str(self.project_path))
        analyzer._discover_python_files()
        analyzer._build_import_graph()

        cycles = analyzer._detect_circular_dependencies()

        assert len(cycles) == 0

    def test_detect_circular_dependencies_simple_cycle(self):
        """Test circular dependency detection with simple cycle."""
        # Create circular dependency: A -> B -> A
        self.create_python_file("src/a.py", "from src.b import func_b")
        self.create_python_file("src/b.py", "from src.a import func_a")

        analyzer = ArchitectureAnalyzer(str(self.project_path))
        analyzer._discover_python_files()
        analyzer._build_import_graph()

        cycles = analyzer._detect_circular_dependencies()

        assert len(cycles) >= 1
        # Should find cycle involving src.a and src.b
        cycle_modules = set()
        for cycle in cycles:
            cycle_modules.update(cycle)
        assert "src.a" in cycle_modules
        assert "src.b" in cycle_modules

    def test_detect_circular_dependencies_complex_cycle(self):
        """Test circular dependency detection with complex cycle."""
        # Create complex cycle: A -> B -> C -> A
        self.create_python_file("src/a.py", "from src.b import func_b")
        self.create_python_file("src/b.py", "from src.c import func_c")
        self.create_python_file("src/c.py", "from src.a import func_a")

        analyzer = ArchitectureAnalyzer(str(self.project_path))
        analyzer._discover_python_files()
        analyzer._build_import_graph()

        cycles = analyzer._detect_circular_dependencies()

        assert len(cycles) >= 1
        # Should find cycle involving all three modules
        cycle_modules = set()
        for cycle in cycles:
            cycle_modules.update(cycle)
        assert "src.a" in cycle_modules
        assert "src.b" in cycle_modules
        assert "src.c" in cycle_modules

    def test_detect_circular_dependencies_external_imports(self):
        """Test that external imports don't create false cycles."""
        # Create files that import external packages
        self.create_python_file(
            "src/a.py",
            """
import os
import numpy as np
from src.b import func_b
""",
        )
        self.create_python_file(
            "src/b.py",
            """
import sys
import pandas as pd
# No import back to src.a
""",
        )

        analyzer = ArchitectureAnalyzer(str(self.project_path))
        analyzer._discover_python_files()
        analyzer._build_import_graph()

        cycles = analyzer._detect_circular_dependencies()

        # Should not find any cycles
        assert len(cycles) == 0

    def test_is_internal_module(self):
        """Test internal module detection."""
        analyzer = ArchitectureAnalyzer(str(self.project_path))

        # Set up some modules in import graph
        analyzer.import_graph = {
            "src.main": set(),
            "src.utils.helper": set(),
            "tests.test_main": set(),
        }

        # Test internal modules
        assert analyzer._is_internal_module("src.main")
        assert analyzer._is_internal_module("src.utils.helper")
        assert analyzer._is_internal_module("tests.test_main")

        # Test external modules
        assert not analyzer._is_internal_module("os")
        assert not analyzer._is_internal_module("numpy")
        assert not analyzer._is_internal_module("pandas.core")

    def test_calculate_coupling_metrics(self):
        """Test coupling metrics calculation."""
        # Set up import graph with known structure
        analyzer = ArchitectureAnalyzer(str(self.project_path))
        analyzer.import_graph = {
            "src.main": {"src.utils", "src.models", "os", "sys"},  # fan-out: 2 internal
            "src.utils": {"src.models", "json"},  # fan-out: 1 internal
            "src.models": {"numpy"},  # fan-out: 0 internal
            "src.high_coupling": {f"src.dep{i}" for i in range(15)},  # fan-out: 15 internal
        }

        # Add the dependencies to import graph so they're considered internal
        for i in range(15):
            analyzer.import_graph[f"src.dep{i}"] = set()

        metrics = analyzer._calculate_coupling_metrics()

        # Check structure
        assert "avg_fan_in" in metrics
        assert "avg_fan_out" in metrics
        assert "high_coupling_modules" in metrics
        assert "total_modules" in metrics

        # Check total modules
        assert metrics["total_modules"] == len(analyzer.import_graph)

        # Check high coupling detection
        high_coupling = metrics["high_coupling_modules"]
        assert len(high_coupling) >= 1

        # Should include the high coupling module
        high_coupling_names = [hc["module"] for hc in high_coupling]
        assert "src.high_coupling" in high_coupling_names

        # Check fan-out for high coupling module
        for hc in high_coupling:
            if hc["module"] == "src.high_coupling":
                assert hc["fan_out"] == 15

    def test_calculate_coupling_metrics_empty_graph(self):
        """Test coupling metrics with empty import graph."""
        analyzer = ArchitectureAnalyzer(str(self.project_path))
        analyzer.import_graph = {}

        metrics = analyzer._calculate_coupling_metrics()

        assert metrics["avg_fan_in"] == 0
        assert metrics["avg_fan_out"] == 0
        assert metrics["high_coupling_modules"] == []
        assert metrics["total_modules"] == 0

    def test_detect_solid_violations(self):
        """Test SOLID principle violation detection."""
        # Create file with large class (SRP violation)
        # Need to create a class with more than 500 lines between start and end
        large_class_content = '''class LargeClass:
    """A class that violates Single Responsibility Principle."""
    
    def method1(self):
        pass
    
''' + "\n".join([f"    def method{i}(self): pass" for i in range(2, 520)])  # Create 520+ lines

        self.create_python_file("src/large_class.py", large_class_content)

        # Create file with normal-sized class
        normal_class_content = '''
class NormalClass:
    """A reasonably sized class."""
    
    def method1(self):
        pass
    
    def method2(self):
        pass
'''

        self.create_python_file("src/normal_class.py", normal_class_content)

        analyzer = ArchitectureAnalyzer(str(self.project_path))
        analyzer._discover_python_files()

        violations = analyzer._detect_solid_violations()

        # Should find violation for large class
        assert len(violations) >= 1

        # Check violation details
        large_class_violation = None
        for violation in violations:
            if "LargeClass" in violation.title:
                large_class_violation = violation
                break

        assert large_class_violation is not None
        assert large_class_violation.dimension == "architecture"
        assert large_class_violation.severity == Severity.MEDIUM
        assert large_class_violation.category == "solid_srp"
        assert "LargeClass" in large_class_violation.title
        assert large_class_violation.file_path.replace("\\", "/") == "src/large_class.py"
        assert large_class_violation.line_number is not None
        assert large_class_violation.priority == Priority.P2
        assert large_class_violation.role == Role.BACKEND

    def test_detect_solid_violations_syntax_error(self):
        """Test SOLID violation detection with syntax errors."""
        # Create file with syntax error
        self.create_python_file(
            "src/broken.py",
            """
class BrokenClass:
    def broken_method(
        # Missing closing parenthesis
""",
        )

        analyzer = ArchitectureAnalyzer(str(self.project_path))
        analyzer._discover_python_files()

        # Should not raise exception
        violations = analyzer._detect_solid_violations()

        # Should return empty list or not include broken file
        assert isinstance(violations, list)

    @patch("subprocess.run")
    def test_calculate_complexity_score_success(self, mock_run):
        """Test complexity score calculation success."""
        # Mock successful radon maintainability index output
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = """
src/file1.py - A (85.2)
src/file2.py - B (72.1)
src/file3.py - A (90.5)
"""

        analyzer = ArchitectureAnalyzer(str(self.project_path))

        # Create src directory
        (self.project_path / "src").mkdir(exist_ok=True)

        score = analyzer._calculate_complexity_score()

        # Should calculate average: (85.2 + 72.1 + 90.5) / 3 = 82.6
        expected_avg = (85.2 + 72.1 + 90.5) / 3
        assert score == round(expected_avg, 2)

    @patch("subprocess.run")
    def test_calculate_complexity_score_no_src(self, mock_run):
        """Test complexity score when no src directory exists."""
        analyzer = ArchitectureAnalyzer(str(self.project_path))

        # Remove src directory
        shutil.rmtree(self.project_path / "src")

        score = analyzer._calculate_complexity_score()

        # Should return default score
        assert score == 50.0
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_calculate_complexity_score_failure(self, mock_run):
        """Test complexity score calculation failure."""
        # Mock failed radon execution
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""

        analyzer = ArchitectureAnalyzer(str(self.project_path))

        score = analyzer._calculate_complexity_score()

        # Should return default score
        assert score == 50.0

    def test_calculate_architecture_score(self):
        """Test overall architecture score calculation."""
        analyzer = ArchitectureAnalyzer(str(self.project_path))

        # Test with various inputs
        large_files = [{"path": f"file{i}.py", "lines": 600} for i in range(25)]  # 25 large files
        circular_deps = [["a", "b", "a"], ["c", "d", "c"]]  # 2 cycles
        coupling_metrics = {
            "high_coupling_modules": [{"module": "high1"}, {"module": "high2"}]
        }  # 2 high coupling
        solid_violations = [MagicMock() for _ in range(8)]  # 8 violations
        complexity_score = 80.0

        score = analyzer._calculate_architecture_score(
            large_files, circular_deps, coupling_metrics, solid_violations, complexity_score
        )

        # Expected calculation:
        # Base: 80.0 * 0.4 + 60.0 = 32.0 + 60.0 = 92.0
        # Large files penalty: min(30, 25//10*10) = min(30, 20) = -20
        # Circular deps penalty: min(20, 2*5) = min(20, 10) = -10
        # High coupling penalty: min(15, 2*5) = min(15, 10) = -10
        # SOLID violations penalty: min(10, 8*2) = min(10, 16) = -10 (capped at 10)
        # Total: 92.0 - 20 - 10 - 10 - 10 = 42.0

        expected_score = 92.0 - 20 - 10 - 10 - 10
        assert score == expected_score

    def test_calculate_architecture_score_perfect(self):
        """Test architecture score with perfect metrics."""
        analyzer = ArchitectureAnalyzer(str(self.project_path))

        # Perfect inputs
        large_files = []
        circular_deps = []
        coupling_metrics = {"high_coupling_modules": []}
        solid_violations = []
        complexity_score = 100.0

        score = analyzer._calculate_architecture_score(
            large_files, circular_deps, coupling_metrics, solid_violations, complexity_score
        )

        # Expected: 100.0 * 0.4 + 60.0 = 100.0
        assert score == 100.0

    def test_calculate_architecture_score_clamping(self):
        """Test architecture score clamping to 0-100 range."""
        analyzer = ArchitectureAnalyzer(str(self.project_path))

        # Terrible inputs that would result in negative score
        large_files = [
            {"path": f"file{i}.py", "lines": 600} for i in range(100)
        ]  # Many large files
        circular_deps = [["a", "b", "a"] for _ in range(10)]  # Many cycles
        coupling_metrics = {"high_coupling_modules": [{"module": f"high{i}"} for i in range(10)]}
        solid_violations = [MagicMock() for _ in range(20)]  # Many violations
        complexity_score = 0.0

        score = analyzer._calculate_architecture_score(
            large_files, circular_deps, coupling_metrics, solid_violations, complexity_score
        )

        # Should be clamped to 0
        assert score == 0.0

    def test_analyze_integration(self):
        """Test full analysis integration."""
        # Create test project structure
        self.create_python_file(
            "src/main.py",
            '''
import os
from src.utils import helper
from src.models.base import BaseModel

class MainClass:
    """Main application class."""
    
    def run(self):
        pass
''',
        )

        self.create_python_file(
            "src/utils/helper.py",
            """
from src.models import base

def helper_function():
    pass
""",
        )

        self.create_python_file(
            "src/models/base.py",
            '''
import numpy as np

class BaseModel:
    """Base model class."""
    
    def predict(self):
        pass
''',
        )

        # Create a large file
        self.create_python_file("src/large.py", content="", lines=600)

        analyzer = ArchitectureAnalyzer(str(self.project_path))

        with (
            patch.object(analyzer, "_get_file_complexity", return_value=5.0),
            patch.object(analyzer, "_calculate_complexity_score", return_value=75.0),
        ):

            result = analyzer.analyze()

        # Verify result structure
        assert isinstance(result, ArchitectureAnalysis)
        assert result.total_files == 4
        assert len(result.large_files) == 1
        assert result.large_files[0]["path"].replace("\\", "/") == "src/large.py"
        assert result.large_files[0]["lines"] == 600

        # Should have coupling metrics
        assert "avg_fan_in" in result.coupling_metrics
        assert "avg_fan_out" in result.coupling_metrics
        assert "total_modules" in result.coupling_metrics

        # Should have calculated score
        assert 0 <= result.score <= 100

        # SOLID violations should be a list
        assert isinstance(result.solid_violations, list)

        # Circular dependencies should be a list
        assert isinstance(result.circular_dependencies, list)


if __name__ == "__main__":
    pytest.main([__file__])
