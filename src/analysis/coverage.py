"""
Coverage Analyzer for HistoCore Project Optimization Analysis System.

Analyzes test coverage, identifies untested paths, and measures test quality.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any

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
            score=score
        )
    
    def _run_coverage(self) -> tuple[float, float]:
        """Run pytest with coverage and parse results."""
        try:
            # Check if .coverage file exists
            coverage_file = self.project_path / '.coverage'
            
            if coverage_file.exists():
                # Parse existing coverage data
                result = subprocess.run(
                    ['coverage', 'json', '-o', 'coverage.json'],
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False
                )
                
                if result.returncode == 0:
                    coverage_json = self.project_path / 'coverage.json'
                    if coverage_json.exists():
                        data = json.loads(coverage_json.read_text())
                        
                        line_cov = data.get('totals', {}).get('percent_covered', 0.0)
                        branch_cov = data.get('totals', {}).get('percent_covered_display', 0.0)
                        
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
            coverage_file = self.project_path / '.coverage'
            if not coverage_file.exists():
                return []
            
            # Get coverage data
            result = subprocess.run(
                ['coverage', 'json', '-o', 'coverage.json'],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
            
            if result.returncode != 0:
                return []
            
            coverage_json = self.project_path / 'coverage.json'
            if not coverage_json.exists():
                return []
            
            data = json.loads(coverage_json.read_text())
            files = data.get('files', {})
            
            # Analyze each Python file
            for filepath, file_data in files.items():
                missing_lines = file_data.get('missing_lines', [])
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
                            if hasattr(node, 'lineno') and node.lineno in missing_lines:
                                untested.append(f"{filepath}:{node.lineno} (exception handler)")
                        
                        # Find security-sensitive function calls
                        if isinstance(node, ast.Call):
                            if hasattr(node.func, 'id'):
                                func_name = node.func.id
                                if any(keyword in func_name.lower() for keyword in 
                                      ['auth', 'encrypt', 'decrypt', 'validate', 'sanitize']):
                                    if hasattr(node, 'lineno') and node.lineno in missing_lines:
                                        untested.append(f"{filepath}:{node.lineno} (security: {func_name})")
                
                except (SyntaxError, UnicodeDecodeError):
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
            src_dir = self.project_path / 'src'
            if not src_dir.exists():
                return []
            
            # Find test files with property tests
            tests_dir = self.project_path / 'tests'
            property_test_files = set()
            
            if tests_dir.exists():
                for test_file in tests_dir.rglob('test_*.py'):
                    content = test_file.read_text()
                    if '@given' in content or 'from hypothesis' in content:
                        property_test_files.add(test_file.stem.replace('test_', ''))
            
            # Scan source files for data transformation functions
            for src_file in src_dir.rglob('*.py'):
                if src_file.name.startswith('_'):
                    continue
                
                try:
                    tree = ast.parse(src_file.read_text())
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check if function transforms data
                            func_name = node.name
                            
                            # Skip private/magic methods
                            if func_name.startswith('_'):
                                continue
                            
                            # Look for transformation keywords
                            is_transformer = any(keyword in func_name.lower() for keyword in [
                                'transform', 'convert', 'process', 'normalize',
                                'encode', 'decode', 'parse', 'serialize'
                            ])
                            
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
        """Detect flaky tests (placeholder)."""
        # TODO: Analyze historical test results
        logger.info("Flaky test detection not yet implemented")
        return []
    
    def _detect_slow_tests(self) -> List[Dict[str, Any]]:
        """Detect slow tests (placeholder)."""
        # TODO: Parse pytest timing data
        logger.info("Slow test detection not yet implemented")
        return []
    
    def _calculate_coverage_score(
        self,
        line_cov: float,
        branch_cov: float,
        untested_paths: List[str]
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
