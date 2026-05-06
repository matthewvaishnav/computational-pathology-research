"""
Code Quality Scanner for HistoCore Project Optimization Analysis System.

Analyzes code quality using pylint, complexity metrics, and documentation coverage.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from src.analysis.models import CodeQualityAnalysis


logger = logging.getLogger(__name__)


class CodeQualityScanner:
    """Analyzes code quality metrics."""
    
    def __init__(self, project_path: str):
        """
        Initialize scanner.
        
        Args:
            project_path: Path to project root directory
        """
        self.project_path = Path(project_path).resolve()
        
    def analyze(self) -> CodeQualityAnalysis:
        """
        Run code quality analysis.
        
        Returns:
            CodeQualityAnalysis with metrics
        """
        logger.info("Starting code quality analysis...")
        
        # Complexity analysis
        avg_complexity, high_complexity_funcs = self._analyze_complexity()
        
        # Duplication detection
        duplication_pct = self._detect_duplication()
        
        # Documentation coverage
        doc_coverage = self._measure_documentation_coverage()
        
        # Pylint score
        pylint_score = self._run_pylint()
        
        # Calculate overall score
        score = self._calculate_quality_score(
            avg_complexity, duplication_pct, doc_coverage, pylint_score
        )
        
        return CodeQualityAnalysis(
            average_complexity=avg_complexity,
            high_complexity_functions=high_complexity_funcs,
            duplication_percentage=duplication_pct,
            documentation_coverage=doc_coverage,
            pylint_score=pylint_score,
            score=score
        )
    
    def _analyze_complexity(self) -> tuple[float, List[Dict[str, Any]]]:
        """Analyze cyclomatic complexity using radon."""
        try:
            src_dir = self.project_path / 'src'
            if not src_dir.exists():
                return (0.0, [])
            
            result = subprocess.run(
                ['radon', 'cc', str(src_dir), '-s', '-a', '--json'],
                capture_output=True,
                text=True,
                timeout=60,
                check=False
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                all_complexities = []
                high_complexity = []
                
                for file_path, functions in data.items():
                    for func in functions:
                        complexity = func.get('complexity', 0)
                        all_complexities.append(complexity)
                        
                        if complexity > 10:
                            high_complexity.append({
                                'name': func.get('name', 'unknown'),
                                'file': file_path,
                                'line': func.get('lineno', 0),
                                'complexity': complexity
                            })
                
                avg_complexity = sum(all_complexities) / len(all_complexities) if all_complexities else 0.0
                
                # Sort by complexity descending, take top 10
                high_complexity.sort(key=lambda x: x['complexity'], reverse=True)
                
                return (round(avg_complexity, 2), high_complexity[:10])
            
            return (0.0, [])
        
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to analyze complexity: {e}")
            return (0.0, [])
    
    def _detect_duplication(self) -> float:
        """Detect code duplication (placeholder)."""
        # TODO: Implement AST-based duplication detection
        logger.info("Duplication detection not yet implemented")
        return 0.0
    
    def _measure_documentation_coverage(self) -> float:
        """Measure documentation coverage (placeholder)."""
        # TODO: Scan for missing docstrings
        logger.info("Documentation coverage not yet implemented")
        return 0.0
    
    def _run_pylint(self) -> float:
        """Run pylint and get score."""
        try:
            src_dir = self.project_path / 'src'
            if not src_dir.exists():
                return 0.0
            
            result = subprocess.run(
                ['pylint', str(src_dir), '--output-format=json'],
                capture_output=True,
                text=True,
                timeout=120,
                check=False
            )
            
            # Pylint returns non-zero on issues, but still outputs JSON
            if result.stdout:
                # Parse score from stderr (pylint prints score there)
                for line in result.stderr.splitlines():
                    if 'Your code has been rated at' in line:
                        # Format: "Your code has been rated at 7.50/10"
                        parts = line.split('rated at')
                        if len(parts) > 1:
                            score_str = parts[1].split('/')[0].strip()
                            return float(score_str)
            
            return 0.0
        
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            logger.warning(f"Failed to run pylint: {e}")
            return 0.0
    
    def _calculate_quality_score(
        self,
        avg_complexity: float,
        duplication_pct: float,
        doc_coverage: float,
        pylint_score: float
    ) -> float:
        """
        Calculate code quality score (0-100).
        
        Scoring:
        - Pylint score: 40% (scale 0-10 to 0-100)
        - Complexity: 30% (lower is better, target <5)
        - Documentation: 20% (target 80%)
        - Duplication: 10% (lower is better, target <5%)
        """
        score = 0.0
        
        # Pylint score (0-10 scale)
        score += (pylint_score / 10.0) * 40.0
        
        # Complexity (target <5, penalize >10)
        if avg_complexity <= 5:
            score += 30.0
        elif avg_complexity <= 10:
            score += 30.0 * (1.0 - (avg_complexity - 5) / 5)
        else:
            score += 30.0 * max(0, 1.0 - (avg_complexity - 10) / 10)
        
        # Documentation coverage (target 80%)
        score += 20.0 * min(1.0, doc_coverage / 80.0)
        
        # Duplication (target <5%)
        if duplication_pct <= 5:
            score += 10.0
        else:
            score += 10.0 * max(0, 1.0 - (duplication_pct - 5) / 10)
        
        return max(0.0, min(100.0, round(score, 2)))
