"""
Coverage Analyzer for HistoCore Project Optimization Analysis System.

Analyzes test coverage, identifies untested paths, and measures test quality.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from .models import CoverageAnalysis

logger = logging.getLogger(__name__)


class CoverageAnalyzer:
    """Analyzes test coverage and quality."""

    def __init__(self, project_path: str):
        """
        Initialize analyzer.

        Args:
            project_path: Path to project root directory
        """
        self.project_path = Path(project_path).resolve()

    def analyze(self) -> CoverageAnalysis:
        """
        Run coverage analysis.

        Returns:
            CoverageAnalysis with metrics
        """
        logger.info("Starting coverage analysis...")

        # Run pytest with coverage
        line_cov, branch_cov = self._run_coverage()

        # Detect untested paths
        untested_paths = self._detect_untested_critical_paths()

        # Detect missing property tests
        missing_property_tests = self._detect_missing_property_tests()

        # Test quality metrics
        flaky_tests = self._detect_flaky_tests()
        slow_tests = self._detect_slow_tests()

        # Calculate score
        score = self._calculate_coverage_score(line_cov, branch_cov, untested_paths)

        return CoverageAnalysis(
            line_coverage=line_cov,
            branch_coverage=branch_cov,
            untested_critical_paths=untested_paths,
            missing_property_tests=missing_property_tests,
            flaky_tests=flaky_tests,
            slow_tests=slow_tests,
            score=score,
        )

    def _run_coverage(self) -> tuple[float, float]:
        """Run pytest with coverage and parse results."""
        try:
            # Check if .coverage file exists
            coverage_file = self.project_path / ".coverage"

            if coverage_file.exists():
                # Parse existing coverage data
                result = subprocess.run(
                    ["coverage", "json", "-o", "coverage.json"],
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )

                if result.returncode == 0:
                    coverage_json = self.project_path / "coverage.json"
                    if coverage_json.exists():
                        try:
                            data = json.loads(coverage_json.read_text())

                            line_cov = data.get("totals", {}).get("percent_covered", 0.0)
                            branch_cov = data.get("totals", {}).get("percent_covered_display", 0.0)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse coverage JSON: {e}")
                            return (0.0, 0.0)

                        # Clean up
                        coverage_json.unlink()

                        return (round(line_cov, 2), round(branch_cov, 2))

            logger.info("No coverage data found, returning 0%")
            return (0.0, 0.0)

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to run coverage: {e}")
            return (0.0, 0.0)

    def _detect_untested_critical_paths(self) -> List[str]:
        """
        Detect untested critical paths using AST analysis.

        Critical paths include:
        - Error handling blocks (try/except)
        - Security-sensitive functions (auth, crypto, validation)
        - Data transformation pipelines

        Returns:
            List of untested critical path identifiers
        """
        import ast

        untested = []

        try:
            # Parse coverage data to find uncovered lines
            coverage_file = self.project_path / ".coverage"
            if not coverage_file.exists():
                return []

            # Get coverage data
            result = subprocess.run(
                ["coverage", "json", "-o", "coverage.json"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            if result.returncode != 0:
                return []

            coverage_json = self.project_path / "coverage.json"
            if not coverage_json.exists():
                return []

            data = json.loads(coverage_json.read_text())
            files = data.get("files", {})

            # Analyze each Python file
            for filepath, file_data in files.items():
                missing_lines = file_data.get("missing_lines", [])
                if not missing_lines:
                    continue

                # Parse AST to find critical paths
                try:
                    file_path = Path(filepath)
                    if not file_path.exists():
                        continue

                    tree = ast.parse(file_path.read_text())

                    # Find try/except blocks
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ExceptHandler):
                            if hasattr(node, "lineno") and node.lineno in missing_lines:
                                untested.append(f"{filepath}:{node.lineno} (exception handler)")
                                logger.debug(
                                    f"Found untested exception handler: {filepath}:{node.lineno}"
                                )

                        # Find security-sensitive function calls
                        if isinstance(node, ast.Call):
                            if hasattr(node.func, "id"):
                                func_name = node.func.id
                                if any(
                                    keyword in func_name.lower()
                                    for keyword in [
                                        "auth",
                                        "encrypt",
                                        "decrypt",
                                        "validate",
                                        "sanitize",
                                    ]
                                ):
                                    if hasattr(node, "lineno") and node.lineno in missing_lines:
                                        untested.append(
                                            f"{filepath}:{node.lineno} (security: {func_name})"
                                        )
                                        logger.debug(
                                            f"Found untested security function: {func_name} at {filepath}:{node.lineno}"
                                        )

                except (SyntaxError, UnicodeDecodeError) as e:
                    logger.debug(f"Failed to parse {filepath}: {e}")
                    continue

            # Clean up
            coverage_json.unlink()

        except Exception as e:
            logger.debug(f"Critical path detection error: {e}")

        return untested[:20]  # Limit to top 20

    def _detect_missing_property_tests(self) -> List[str]:
        """
        Detect functions missing property-based tests.

        Scans for data transformation functions that should have
        property tests but don't have corresponding test files.

        Returns:
            List of functions missing property tests
        """
        import ast

        missing = []

        try:
            # Find all Python files in src/
            src_dir = self.project_path / "src"
            if not src_dir.exists():
                return []

            # Find test files with property tests
            tests_dir = self.project_path / "tests"
            property_test_files = set()

            if tests_dir.exists():
                for test_file in tests_dir.rglob("test_*.py"):
                    content = test_file.read_text()
                    if "@given" in content or "from hypothesis" in content:
                        property_test_files.add(test_file.stem.replace("test_", ""))

            # Scan source files for data transformation functions
            for src_file in src_dir.rglob("*.py"):
                if src_file.name.startswith("_"):
                    continue

                try:
                    tree = ast.parse(src_file.read_text())

                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check if function transforms data
                            func_name = node.name

                            # Skip private/magic methods
                            if func_name.startswith("_"):
                                continue

                            # Look for transformation keywords
                            is_transformer = any(
                                keyword in func_name.lower()
                                for keyword in [
                                    "transform",
                                    "convert",
                                    "process",
                                    "normalize",
                                    "encode",
                                    "decode",
                                    "parse",
                                    "serialize",
                                ]
                            )

                            if is_transformer:
                                # Check if module has property tests
                                module_name = src_file.stem
                                if module_name not in property_test_files:
                                    rel_path = src_file.relative_to(self.project_path)
                                    missing.append(f"{rel_path}::{func_name}")

                except (SyntaxError, UnicodeDecodeError):
                    continue

        except Exception as e:
            logger.debug(f"Property test detection error: {e}")

        return missing[:15]  # Limit to top 15

    def _detect_flaky_tests(self) -> List[str]:
        """
        Detect flaky tests by analyzing pytest output.

        Flaky tests are identified by:
        - Tests that use time.sleep() or asyncio.sleep()
        - Tests with random number generation without seeds
        - Tests that depend on external network calls

        Returns:
            List of potentially flaky test identifiers
        """
        import ast

        flaky = []

        try:
            tests_dir = self.project_path / "tests"
            if not tests_dir.exists():
                return []

            for test_file in tests_dir.rglob("test_*.py"):
                try:
                    content = test_file.read_text()
                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                            test_name = node.name

                            # Check for sleep calls
                            for child in ast.walk(node):
                                if isinstance(child, ast.Call):
                                    if hasattr(child.func, "attr"):
                                        if child.func.attr in ["sleep", "wait"]:
                                            rel_path = test_file.relative_to(self.project_path)
                                            flaky.append(f"{rel_path}::{test_name} (uses sleep)")
                                            break

                                    # Check for random without seed
                                    if hasattr(child.func, "id"):
                                        if "random" in child.func.id.lower():
                                            # Check if seed is set
                                            has_seed = any(
                                                isinstance(n, ast.Call)
                                                and hasattr(n.func, "attr")
                                                and n.func.attr == "seed"
                                                for n in ast.walk(node)
                                            )
                                            if not has_seed:
                                                rel_path = test_file.relative_to(self.project_path)
                                                flaky.append(
                                                    f"{rel_path}::{test_name} (unseeded random)"
                                                )
                                                break

                            # Check for network calls
                            if any(
                                keyword in content
                                for keyword in ["requests.", "urllib.", "http.client"]
                            ):
                                if "mock" not in content.lower() and "patch" not in content.lower():
                                    rel_path = test_file.relative_to(self.project_path)
                                    if f"{rel_path}::{test_name}" not in [
                                        f.split(" (")[0] for f in flaky
                                    ]:
                                        flaky.append(f"{rel_path}::{test_name} (network call)")

                except (SyntaxError, UnicodeDecodeError):
                    continue

        except Exception as e:
            logger.debug(f"Flaky test detection error: {e}")

        return flaky[:10]  # Limit to top 10

    def _detect_slow_tests(self) -> List[Dict[str, Any]]:
        """
        Detect slow tests by analyzing pytest timing data.

        Returns:
            List of slow tests with timing information
        """
        slow_tests = []

        try:
            # Check for pytest cache with timing data
            pytest_cache = self.project_path / ".pytest_cache"
            if not pytest_cache.exists():
                return []

            # Look for .pytest_cache/v/cache/lastfailed or nodeids
            cache_dir = pytest_cache / "v" / "cache"
            if not cache_dir.exists():
                return []

            # Parse test files to estimate complexity
            tests_dir = self.project_path / "tests"
            if not tests_dir.exists():
                return []

            import ast

            for test_file in tests_dir.rglob("test_*.py"):
                try:
                    content = test_file.read_text()
                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                            # Estimate complexity
                            complexity_score = 0

                            # Count nested loops
                            for child in ast.walk(node):
                                if isinstance(child, (ast.For, ast.While)):
                                    complexity_score += 10
                                if isinstance(child, ast.Call):
                                    complexity_score += 1

                            # Check for slow operations
                            if "subprocess" in content:
                                complexity_score += 20
                            if "torch" in content or "tensorflow" in content:
                                complexity_score += 30
                            if "train" in node.name.lower():
                                complexity_score += 25

                            # Flag if complexity > threshold
                            if complexity_score > 40:
                                rel_path = test_file.relative_to(self.project_path)
                                slow_tests.append(
                                    {
                                        "test": f"{rel_path}::{node.name}",
                                        "estimated_complexity": complexity_score,
                                        "reason": "High complexity score",
                                    }
                                )

                except (SyntaxError, UnicodeDecodeError):
                    continue

        except Exception as e:
            logger.debug(f"Slow test detection error: {e}")

        # Sort by complexity and return top 10
        slow_tests.sort(key=lambda x: x["estimated_complexity"], reverse=True)
        return slow_tests[:10]

    def _calculate_coverage_score(
        self, line_cov: float, branch_cov: float, untested_paths: List[str]
    ) -> float:
        """
        Calculate coverage score (0-100).

        Scoring:
        - Line coverage: 50% (target 70%)
        - Branch coverage: 30% (target 60%)
        - Untested critical paths penalty: -5 points per path
        - Base: 20%
        """
        score = 20.0  # Base

        # Line coverage (target 70%)
        score += 50.0 * min(1.0, line_cov / 70.0)

        # Branch coverage (target 60%)
        score += 30.0 * min(1.0, branch_cov / 60.0)

        # Critical path penalty
        score -= min(20, len(untested_paths) * 5)

        return max(0.0, min(100.0, round(score, 2)))
