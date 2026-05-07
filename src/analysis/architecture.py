"""
Architecture Analyzer for HistoCore Project Optimization Analysis System.

Analyzes codebase architecture quality including complexity, dependencies,
coupling, and SOLID principle violations.
"""

import ast
import json
import logging
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .models import (
    ArchitectureAnalysis,
    Issue,
    Severity,
    Priority,
    Role,
)


logger = logging.getLogger(__name__)


class ArchitectureAnalyzer:
    """Analyzes architecture quality of Python codebase."""
    
    def __init__(self, project_path: str):
        """
        Initialize analyzer.
        
        Args:
            project_path: Path to project root directory
        """
        self.project_path = Path(project_path).resolve()
        self.python_files: List[Path] = []
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        
    def analyze(self) -> ArchitectureAnalysis:
        """
        Run architecture analysis.
        
        Returns:
            ArchitectureAnalysis with metrics and issues
        """
        logger.info("Starting architecture analysis...")
        
        # Discover Python files
        self._discover_python_files()
        
        # Build import graph
        self._build_import_graph()
        
        # Run sub-analyses
        large_files = self._detect_large_files()
        circular_deps = self._detect_circular_dependencies()
        coupling_metrics = self._calculate_coupling_metrics()
        solid_violations = self._detect_solid_violations()
        
        # Calculate complexity metrics using radon
        complexity_score = self._calculate_complexity_score()
        
        # Calculate overall architecture score
        score = self._calculate_architecture_score(
            large_files, circular_deps, coupling_metrics, solid_violations, complexity_score
        )
        
        return ArchitectureAnalysis(
            total_files=len(self.python_files),
            large_files=large_files,
            circular_dependencies=circular_deps,
            coupling_metrics=coupling_metrics,
            solid_violations=solid_violations,
            score=score
        )
    
    def _discover_python_files(self):
        """Discover all Python files in project."""
        logger.info(f"Discovering Python files in {self.project_path}")
        
        # Exclude common non-source directories
        exclude_dirs = {'.venv', 'venv', 'env', '__pycache__', '.git', 'build', 'dist', '.eggs'}
        
        for py_file in self.project_path.rglob('*.py'):
            # Skip if in excluded directory
            if any(excluded in py_file.parts for excluded in exclude_dirs):
                continue
            self.python_files.append(py_file)
        
        logger.info(f"Found {len(self.python_files)} Python files")
    
    def _build_import_graph(self):
        """Build import dependency graph using AST parsing."""
        logger.info("Building import graph...")
        
        for py_file in self.python_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                tree = ast.parse(content, filename=str(py_file))
                
                # Get module name relative to project root
                module_name = self._get_module_name(py_file)
                
                # Extract imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.import_graph[module_name].add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            self.import_graph[module_name].add(node.module)
            
            except (SyntaxError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to parse {py_file}: {e}")
                continue
        
        logger.info(f"Built import graph with {len(self.import_graph)} modules")
    
    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        try:
            rel_path = file_path.relative_to(self.project_path)
            parts = list(rel_path.parts[:-1]) + [rel_path.stem]
            if parts[-1] == '__init__':
                parts = parts[:-1]
            return '.'.join(parts)
        except ValueError:
            return str(file_path)
    
    def _detect_large_files(self) -> List[Dict[str, Any]]:
        """Detect files >500 lines."""
        logger.info("Detecting large files...")
        
        large_files = []
        threshold = 500
        
        for py_file in self.python_files:
            try:
                lines = py_file.read_text(encoding='utf-8').splitlines()
                line_count = len(lines)
                
                if line_count > threshold:
                    # Calculate complexity for large files
                    complexity = self._get_file_complexity(py_file)
                    
                    large_files.append({
                        'path': str(py_file.relative_to(self.project_path)),
                        'lines': line_count,
                        'complexity': complexity
                    })
            
            except (UnicodeDecodeError, OSError) as e:
                logger.warning(f"Failed to read {py_file}: {e}")
                continue
        
        # Sort by line count descending
        large_files.sort(key=lambda x: x['lines'], reverse=True)
        
        logger.info(f"Found {len(large_files)} large files (>{threshold} lines)")
        return large_files
    
    def _get_file_complexity(self, file_path: Path) -> float:
        """Get cyclomatic complexity for file using radon."""
        try:
            result = subprocess.run(
                ['radon', 'cc', str(file_path), '-s', '-a'],
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
            
            if result.returncode == 0:
                # Parse average complexity from output
                # Format: "Average complexity: A (5.2)"
                for line in result.stdout.splitlines():
                    if 'Average complexity' in line:
                        # Extract number in parentheses
                        parts = line.split('(')
                        if len(parts) > 1:
                            complexity_str = parts[1].split(')')[0]
                            return float(complexity_str)
            
            return 0.0
        
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            logger.debug(f"Failed to get complexity for {file_path}: {e}")
            return 0.0
    
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies using DFS."""
        logger.info("Detecting circular dependencies...")
        
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> bool:
            """DFS to detect cycles."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.import_graph.get(node, []):
                # Only check internal modules (not external packages)
                if not self._is_internal_module(neighbor):
                    continue
                
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    if cycle not in cycles:
                        cycles.append(cycle)
                    return True
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        # Run DFS from each node
        for node in self.import_graph:
            if node not in visited:
                dfs(node, [])
        
        logger.info(f"Found {len(cycles)} circular dependencies")
        return cycles
    
    def _is_internal_module(self, module_name: str) -> bool:
        """Check if module is internal to project."""
        # Internal modules typically start with 'src' or project name
        return module_name.startswith('src.') or module_name in self.import_graph
    
    def _calculate_coupling_metrics(self) -> Dict[str, Any]:
        """Calculate coupling metrics (fan-in, fan-out)."""
        logger.info("Calculating coupling metrics...")
        
        fan_in: Dict[str, int] = defaultdict(int)
        fan_out: Dict[str, int] = defaultdict(int)
        
        # Calculate fan-in and fan-out
        for module, imports in self.import_graph.items():
            fan_out[module] = len([imp for imp in imports if self._is_internal_module(imp)])
            
            for imported in imports:
                if self._is_internal_module(imported):
                    fan_in[imported] += 1
        
        # Find high-coupling modules
        high_coupling_threshold = 10
        high_coupling_modules = [
            {'module': mod, 'fan_out': count}
            for mod, count in fan_out.items()
            if count > high_coupling_threshold
        ]
        
        # Calculate averages
        avg_fan_in = sum(fan_in.values()) / len(fan_in) if fan_in else 0
        avg_fan_out = sum(fan_out.values()) / len(fan_out) if fan_out else 0
        
        return {
            'avg_fan_in': round(avg_fan_in, 2),
            'avg_fan_out': round(avg_fan_out, 2),
            'high_coupling_modules': high_coupling_modules[:10],  # Top 10
            'total_modules': len(self.import_graph)
        }
    
    def _detect_solid_violations(self) -> List[Issue]:
        """Detect SOLID principle violations."""
        logger.info("Detecting SOLID violations...")
        
        violations = []
        
        # Single Responsibility Principle: Large classes (>500 lines)
        for py_file in self.python_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                tree = ast.parse(content, filename=str(py_file))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Count lines in class
                        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                            class_lines = node.end_lineno - node.lineno
                            
                            if class_lines > 500:
                                violations.append(Issue(
                                    id=f"srp_{py_file.stem}_{node.name}",
                                    dimension="architecture",
                                    severity=Severity.MEDIUM,
                                    category="solid_srp",
                                    title=f"Large class violates SRP: {node.name}",
                                    description=f"Class {node.name} has {class_lines} lines, suggesting multiple responsibilities",
                                    file_path=str(py_file.relative_to(self.project_path)),
                                    line_number=node.lineno,
                                    recommendation=f"Refactor {node.name} into smaller, focused classes",
                                    effort_hours=8.0,
                                    priority=Priority.P2,
                                    role=Role.BACKEND
                                ))
            
            except (SyntaxError, UnicodeDecodeError) as e:
                logger.debug(f"Failed to parse {py_file} for SOLID violations: {e}")
                continue
        
        logger.info(f"Found {len(violations)} SOLID violations")
        return violations
    
    def _calculate_complexity_score(self) -> float:
        """Calculate overall complexity score using radon."""
        logger.info("Calculating complexity score...")
        
        try:
            # Run radon on entire src directory
            src_dir = self.project_path / 'src'
            if not src_dir.exists():
                return 50.0  # Default score if no src directory
            
            result = subprocess.run(
                ['radon', 'mi', str(src_dir), '-s'],
                capture_output=True,
                text=True,
                timeout=60,
                check=False
            )
            
            if result.returncode == 0:
                # Parse maintainability index
                # Format: "src/file.py - A (85.2)"
                scores = []
                for line in result.stdout.splitlines():
                    if ' - ' in line and '(' in line:
                        try:
                            score_str = line.split('(')[1].split(')')[0]
                            scores.append(float(score_str))
                        except (IndexError, ValueError):
                            continue
                
                if scores:
                    avg_score = sum(scores) / len(scores)
                    return round(avg_score, 2)
            
            return 50.0  # Default if parsing fails
        
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Failed to calculate complexity score: {e}")
            return 50.0
    
    def _calculate_architecture_score(
        self,
        large_files: List[Dict[str, Any]],
        circular_deps: List[List[str]],
        coupling_metrics: Dict[str, Any],
        solid_violations: List[Issue],
        complexity_score: float
    ) -> float:
        """
        Calculate overall architecture score (0-100).
        
        Scoring:
        - Complexity score: 40% (from radon maintainability index)
        - Large files penalty: -10 points per 10 large files
        - Circular dependencies penalty: -5 points per cycle
        - High coupling penalty: -5 points per high-coupling module
        - SOLID violations penalty: -2 points per violation
        """
        score = complexity_score * 0.4  # Start with 40% of complexity score
        
        # Add base score
        score += 60.0
        
        # Penalties
        score -= min(30, len(large_files) // 10 * 10)  # Max -30 for large files
        score -= min(20, len(circular_deps) * 5)  # Max -20 for circular deps
        score -= min(15, len(coupling_metrics.get('high_coupling_modules', [])) * 5)  # Max -15
        score -= min(10, len(solid_violations) * 2)  # Max -10 for SOLID violations
        
        # Clamp to 0-100
        return max(0.0, min(100.0, round(score, 2)))
