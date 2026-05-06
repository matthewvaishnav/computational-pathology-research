"""
Code Quality Scanner for HistoCore Project Optimization Analysis System.

Analyzes code quality using pylint, complexity metrics, and documentation coverage.
"""

import ast
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

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
        
        # Automated fix suggestions
        fix_suggestions = self._generate_fix_suggestions()
        
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
            score=score,
            fix_suggestions=fix_suggestions
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
        """Detect code duplication using AST comparison."""
        try:
            src_dir = self.project_path / 'src'
            if not src_dir.exists():
                return 0.0
            
            python_files = list(src_dir.rglob('*.py'))
            if not python_files:
                return 0.0
            
            total_lines = 0
            duplicate_lines = 0
            
            # Parse all files and extract function bodies
            file_functions = {}
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        total_lines += len(content.splitlines())
                    
                    tree = ast.parse(content)
                    functions = []
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Extract function body as string
                            func_lines = content.splitlines()[node.lineno-1:node.end_lineno]
                            func_body = '\n'.join(func_lines)
                            
                            # Normalize whitespace for comparison
                            normalized = re.sub(r'\s+', ' ', func_body.strip())
                            
                            if len(normalized) > 100:  # Only check functions >100 chars
                                functions.append({
                                    'name': node.name,
                                    'body': normalized,
                                    'lines': len(func_lines),
                                    'file': str(py_file)
                                })
                    
                    file_functions[str(py_file)] = functions
                
                except (SyntaxError, UnicodeDecodeError) as e:
                    logger.warning(f"Failed to parse {py_file}: {e}")
                    continue
            
            # Compare functions for similarity
            all_functions = []
            for functions in file_functions.values():
                all_functions.extend(functions)
            
            for i, func1 in enumerate(all_functions):
                for func2 in all_functions[i+1:]:
                    similarity = self._calculate_similarity(func1['body'], func2['body'])
                    
                    if similarity > 0.8:  # >80% similar
                        duplicate_lines += min(func1['lines'], func2['lines'])
            
            return round((duplicate_lines / total_lines) * 100, 2) if total_lines > 0 else 0.0
        
        except Exception as e:
            logger.warning(f"Failed to detect duplication: {e}")
            return 0.0
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Simple token-based similarity
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _measure_documentation_coverage(self) -> float:
        """Measure documentation coverage by scanning for missing docstrings."""
        try:
            src_dir = self.project_path / 'src'
            if not src_dir.exists():
                return 0.0
            
            python_files = list(src_dir.rglob('*.py'))
            if not python_files:
                return 0.0
            
            total_functions = 0
            documented_functions = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            # Skip private functions/classes (start with _)
                            if node.name.startswith('_'):
                                continue
                            
                            total_functions += 1
                            
                            # Check if has docstring
                            if (node.body and 
                                isinstance(node.body[0], ast.Expr) and 
                                isinstance(node.body[0].value, ast.Constant) and 
                                isinstance(node.body[0].value.value, str)):
                                documented_functions += 1
                
                except (SyntaxError, UnicodeDecodeError) as e:
                    logger.warning(f"Failed to parse {py_file}: {e}")
                    continue
            
            return round((documented_functions / total_functions) * 100, 2) if total_functions > 0 else 0.0
        
        except Exception as e:
            logger.warning(f"Failed to measure documentation coverage: {e}")
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
    
    def _generate_fix_suggestions(self) -> List[Dict[str, Any]]:
        """Generate automated fix suggestions for common issues."""
        suggestions = []
        
        try:
            # Unused imports
            unused_imports = self._find_unused_imports()
            if unused_imports:
                suggestions.append({
                    'type': 'unused_imports',
                    'severity': 'low',
                    'description': 'Remove unused imports to reduce clutter',
                    'files': unused_imports,
                    'fix_command': 'autoflake --remove-all-unused-imports --in-place',
                    'estimated_effort': '5 minutes'
                })
            
            # Naming convention violations
            naming_issues = self._find_naming_violations()
            if naming_issues:
                suggestions.append({
                    'type': 'naming_conventions',
                    'severity': 'medium',
                    'description': 'Fix naming convention violations (PEP 8)',
                    'issues': naming_issues[:10],  # Top 10
                    'fix_template': 'Rename {old_name} to {suggested_name}',
                    'estimated_effort': f'{len(naming_issues) * 2} minutes'
                })
            
            # Missing docstrings
            missing_docs = self._find_missing_docstrings()
            if missing_docs:
                suggestions.append({
                    'type': 'missing_docstrings',
                    'severity': 'medium',
                    'description': 'Add docstrings to public functions and classes',
                    'functions': missing_docs[:10],  # Top 10
                    'template': '"""Brief description.\n\nArgs:\n    param: Description\n\nReturns:\n    Description\n"""',
                    'estimated_effort': f'{len(missing_docs) * 3} minutes'
                })
            
            # Complex functions needing refactoring
            complex_funcs = self._find_complex_functions()
            if complex_funcs:
                suggestions.append({
                    'type': 'complexity_refactoring',
                    'severity': 'high',
                    'description': 'Refactor high-complexity functions (>10 cyclomatic complexity)',
                    'functions': complex_funcs,
                    'strategies': [
                        'Extract smaller functions',
                        'Reduce nested conditions',
                        'Use early returns',
                        'Apply strategy pattern for complex conditionals'
                    ],
                    'estimated_effort': f'{len(complex_funcs) * 30} minutes'
                })
            
            # Calculate module quality scores
            module_scores = self._calculate_module_scores()
            low_quality_modules = [m for m in module_scores if m['score'] < 60]
            
            if low_quality_modules:
                suggestions.append({
                    'type': 'module_quality',
                    'severity': 'high',
                    'description': 'Improve low-quality modules (score <60)',
                    'modules': low_quality_modules,
                    'improvement_areas': [
                        'Reduce complexity',
                        'Add documentation',
                        'Remove duplication',
                        'Fix pylint issues'
                    ],
                    'estimated_effort': f'{len(low_quality_modules) * 60} minutes'
                })
        
        except Exception as e:
            logger.warning(f"Failed to generate fix suggestions: {e}")
        
        return suggestions
    
    def _find_unused_imports(self) -> List[str]:
        """Find files with unused imports."""
        try:
            src_dir = self.project_path / 'src'
            if not src_dir.exists():
                return []
            
            result = subprocess.run(
                ['flake8', '--select=F401', str(src_dir)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False
            )
            
            unused_files = []
            for line in result.stdout.splitlines():
                if 'F401' in line:  # unused import
                    file_path = line.split(':')[0]
                    if file_path not in unused_files:
                        unused_files.append(file_path)
            
            return unused_files[:20]  # Limit to 20 files
        
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Failed to find unused imports: {e}")
            return []
    
    def _find_naming_violations(self) -> List[Dict[str, str]]:
        """Find naming convention violations."""
        violations = []
        
        try:
            src_dir = self.project_path / 'src'
            if not src_dir.exists():
                return []
            
            python_files = list(src_dir.rglob('*.py'))
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Function names should be snake_case
                            if not re.match(r'^[a-z_][a-z0-9_]*$', node.name) and not node.name.startswith('_'):
                                suggested = re.sub(r'([A-Z])', r'_\1', node.name).lower().lstrip('_')
                                violations.append({
                                    'file': str(py_file),
                                    'line': node.lineno,
                                    'type': 'function',
                                    'old_name': node.name,
                                    'suggested_name': suggested,
                                    'issue': 'Function name should be snake_case'
                                })
                        
                        elif isinstance(node, ast.ClassDef):
                            # Class names should be PascalCase
                            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                                suggested = ''.join(word.capitalize() for word in node.name.split('_'))
                                violations.append({
                                    'file': str(py_file),
                                    'line': node.lineno,
                                    'type': 'class',
                                    'old_name': node.name,
                                    'suggested_name': suggested,
                                    'issue': 'Class name should be PascalCase'
                                })
                
                except (SyntaxError, UnicodeDecodeError) as e:
                    logger.warning(f"Failed to parse {py_file}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Failed to find naming violations: {e}")
        
        return violations[:50]  # Limit to 50 violations
    
    def _find_missing_docstrings(self) -> List[Dict[str, Any]]:
        """Find functions/classes missing docstrings."""
        missing = []
        
        try:
            src_dir = self.project_path / 'src'
            if not src_dir.exists():
                return []
            
            python_files = list(src_dir.rglob('*.py'))
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            # Skip private functions/classes
                            if node.name.startswith('_'):
                                continue
                            
                            # Check if has docstring
                            has_docstring = (
                                node.body and 
                                isinstance(node.body[0], ast.Expr) and 
                                isinstance(node.body[0].value, ast.Constant) and 
                                isinstance(node.body[0].value.value, str)
                            )
                            
                            if not has_docstring:
                                missing.append({
                                    'file': str(py_file),
                                    'line': node.lineno,
                                    'name': node.name,
                                    'type': 'class' if isinstance(node, ast.ClassDef) else 'function'
                                })
                
                except (SyntaxError, UnicodeDecodeError) as e:
                    logger.warning(f"Failed to parse {py_file}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Failed to find missing docstrings: {e}")
        
        return missing[:100]  # Limit to 100 items
    
    def _find_complex_functions(self) -> List[Dict[str, Any]]:
        """Find functions with high cyclomatic complexity."""
        try:
            src_dir = self.project_path / 'src'
            if not src_dir.exists():
                return []
            
            result = subprocess.run(
                ['radon', 'cc', str(src_dir), '-s', '--json'],
                capture_output=True,
                text=True,
                timeout=60,
                check=False
            )
            
            if result.returncode != 0:
                return []
            
            data = json.loads(result.stdout)
            complex_functions = []
            
            for file_path, functions in data.items():
                for func in functions:
                    complexity = func.get('complexity', 0)
                    if complexity > 10:
                        complex_functions.append({
                            'file': file_path,
                            'name': func.get('name', 'unknown'),
                            'line': func.get('lineno', 0),
                            'complexity': complexity,
                            'refactoring_priority': 'high' if complexity > 20 else 'medium'
                        })
            
            # Sort by complexity descending
            complex_functions.sort(key=lambda x: x['complexity'], reverse=True)
            return complex_functions[:20]  # Top 20
        
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to find complex functions: {e}")
            return []
    
    def _calculate_module_scores(self) -> List[Dict[str, Any]]:
        """Calculate quality scores for individual modules."""
        module_scores = []
        
        try:
            src_dir = self.project_path / 'src'
            if not src_dir.exists():
                return []
            
            python_files = list(src_dir.rglob('*.py'))
            
            for py_file in python_files:
                try:
                    # Get module-specific metrics
                    complexity_score = self._get_module_complexity_score(py_file)
                    doc_score = self._get_module_doc_score(py_file)
                    pylint_score = self._get_module_pylint_score(py_file)
                    
                    # Calculate overall score (weighted average)
                    overall_score = (
                        complexity_score * 0.4 +
                        doc_score * 0.3 +
                        pylint_score * 0.3
                    )
                    
                    module_scores.append({
                        'file': str(py_file),
                        'score': round(overall_score, 1),
                        'complexity_score': complexity_score,
                        'documentation_score': doc_score,
                        'pylint_score': pylint_score
                    })
                
                except Exception as e:
                    logger.warning(f"Failed to score module {py_file}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Failed to calculate module scores: {e}")
        
        return sorted(module_scores, key=lambda x: x['score'])
    
    def _get_module_complexity_score(self, file_path: Path) -> float:
        """Get complexity score for a single module (0-100)."""
        try:
            result = subprocess.run(
                ['radon', 'cc', str(file_path), '-s', '--json'],
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                file_data = data.get(str(file_path), [])
                
                if not file_data:
                    return 100.0  # No functions = perfect score
                
                complexities = [func.get('complexity', 0) for func in file_data]
                avg_complexity = sum(complexities) / len(complexities)
                
                # Score: 100 for complexity <=5, decreasing to 0 for complexity >=20
                if avg_complexity <= 5:
                    return 100.0
                elif avg_complexity >= 20:
                    return 0.0
                else:
                    return 100.0 * (1.0 - (avg_complexity - 5) / 15)
            
            return 50.0  # Default score
        
        except Exception:
            return 50.0
    
    def _get_module_doc_score(self, file_path: Path) -> float:
        """Get documentation score for a single module (0-100)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            total_items = 0
            documented_items = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if not node.name.startswith('_'):  # Skip private
                        total_items += 1
                        
                        # Check for docstring
                        if (node.body and 
                            isinstance(node.body[0], ast.Expr) and 
                            isinstance(node.body[0].value, ast.Constant) and 
                            isinstance(node.body[0].value.value, str)):
                            documented_items += 1
            
            return (documented_items / total_items) * 100 if total_items > 0 else 100.0
        
        except Exception:
            return 50.0
    
    def _get_module_pylint_score(self, file_path: Path) -> float:
        """Get pylint score for a single module (0-100)."""
        try:
            result = subprocess.run(
                ['pylint', str(file_path), '--output-format=text'],
                capture_output=True,
                text=True,
                timeout=60,
                check=False
            )
            
            # Parse score from stderr
            for line in result.stderr.splitlines():
                if 'Your code has been rated at' in line:
                    parts = line.split('rated at')
                    if len(parts) > 1:
                        score_str = parts[1].split('/')[0].strip()
                        return float(score_str) * 10  # Scale 0-10 to 0-100
            
            return 50.0  # Default score
        
        except Exception:
            return 50.0
