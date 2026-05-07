"""
Unit tests for Code Quality Scanner.

Tests complexity calculation, duplication detection, documentation coverage measurement,
and overall code quality scoring.
"""

import pytest
import tempfile
import shutil
import subprocess
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any

from src.analysis.code_quality import CodeQualityScanner
from src.analysis.models import CodeQualityAnalysis


class TestCodeQualityScanner:
    """Test suite for CodeQualityScanner."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        
        # Create basic project structure
        (self.project_path / 'src').mkdir()
        (self.project_path / 'tests').mkdir()
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_python_file(self, path: str, content: str):
        """Create a Python file with specified content."""
        file_path = self.project_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path
    
    def test_init(self):
        """Test scanner initialization."""
        scanner = CodeQualityScanner(str(self.project_path))
        
        assert scanner.project_path == self.project_path
    
    def test_init_with_relative_path(self):
        """Test scanner initialization with relative path."""
        # Change to temp directory and use relative path
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            scanner = CodeQualityScanner('.')
            
            # Should resolve to absolute path
            assert scanner.project_path.is_absolute()
            assert scanner.project_path == self.project_path
        finally:
            os.chdir(original_cwd)
    
    @patch('subprocess.run')
    def test_analyze_complexity_success(self, mock_run):
        """Test complexity analysis with successful radon execution."""
        # Mock successful radon output
        mock_radon_output = {
            "src/module1.py": [
                {
                    "type": "function",
                    "name": "simple_function",
                    "lineno": 1,
                    "complexity": 2
                },
                {
                    "type": "function", 
                    "name": "complex_function",
                    "lineno": 10,
                    "complexity": 15
                }
            ],
            "src/module2.py": [
                {
                    "type": "function",
                    "name": "another_function",
                    "lineno": 5,
                    "complexity": 8
                }
            ]
        }
        
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = json.dumps(mock_radon_output)
        
        scanner = CodeQualityScanner(str(self.project_path))
        avg_complexity, high_complexity_funcs = scanner._analyze_complexity()
        
        # Check average complexity: (2 + 15 + 8) / 3 = 8.33
        expected_avg = round((2 + 15 + 8) / 3, 2)
        assert avg_complexity == expected_avg
        
        # Check high complexity functions (complexity > 10)
        assert len(high_complexity_funcs) == 1
        high_func = high_complexity_funcs[0]
        assert high_func['name'] == 'complex_function'
        assert high_func['file'] == 'src/module1.py'
        assert high_func['line'] == 10
        assert high_func['complexity'] == 15
        
        # Verify subprocess call
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == 'radon'
        assert call_args[1] == 'cc'
        assert str(self.project_path / 'src') in call_args[2]
        assert '-s' in call_args
        assert '-a' in call_args
        assert '--json' in call_args
    
    @patch('subprocess.run')
    def test_analyze_complexity_no_src_directory(self, mock_run):
        """Test complexity analysis when src directory doesn't exist."""
        # Remove src directory
        shutil.rmtree(self.project_path / 'src')
        
        scanner = CodeQualityScanner(str(self.project_path))
        avg_complexity, high_complexity_funcs = scanner._analyze_complexity()
        
        # Should return defaults
        assert avg_complexity == 0.0
        assert high_complexity_funcs == []
        
        # Should not call subprocess
        mock_run.assert_not_called()
    
    @patch('subprocess.run')
    def test_run_pylint_success(self, mock_run):
        """Test pylint execution with successful result."""
        # Mock pylint output with score in stderr
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '[]'  # Empty JSON array
        mock_run.return_value.stderr = '''
************* Module src.test
src/test.py:1:0: C0111: Missing module docstring (missing-docstring)

-------------------------------------------------------------------
Your code has been rated at 7.50/10 (previous run: 8.00/10, -0.50)
'''
        
        scanner = CodeQualityScanner(str(self.project_path))
        pylint_score = scanner._run_pylint()
        
        assert pylint_score == 7.50
        
        # Verify subprocess call
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == 'pylint'
        assert str(self.project_path / 'src') in call_args[1]
        assert '--output-format=json' in call_args
    
    def test_calculate_quality_score_perfect(self):
        """Test quality score calculation with perfect metrics."""
        scanner = CodeQualityScanner(str(self.project_path))
        
        score = scanner._calculate_quality_score(
            avg_complexity=3.0,      # Excellent (≤5)
            duplication_pct=2.0,     # Excellent (≤5%)
            doc_coverage=90.0,       # Excellent (>80%)
            pylint_score=10.0        # Perfect
        )
        
        # Expected: 10/10*40 + 30 + 90/80*20 + 10 = 40 + 30 + 22.5 + 10 = 102.5 → 100.0 (clamped)
        assert score == 100.0
    
    def test_calculate_quality_score_poor(self):
        """Test quality score calculation with poor metrics."""
        scanner = CodeQualityScanner(str(self.project_path))
        
        score = scanner._calculate_quality_score(
            avg_complexity=25.0,     # Very poor (>10)
            duplication_pct=20.0,    # Very poor (>5%)
            doc_coverage=10.0,       # Very poor (<80%)
            pylint_score=2.0         # Poor
        )
        
        # Expected calculations:
        # Pylint: 2/10 * 40 = 8.0
        # Complexity: avg=25, penalty = max(0, 1.0 - (25-10)/10) = max(0, -0.5) = 0, so 30*0 = 0
        # Documentation: 10/80 * 20 = 2.5
        # Duplication: 20>5, penalty = max(0, 1.0 - (20-5)/10) = max(0, -0.5) = 0, so 10*0 = 0
        # Total: 8.0 + 0 + 2.5 + 0 = 10.5
        assert score == 10.5
    
    def test_analyze_integration(self):
        """Test full analysis integration."""
        # Create test files
        self.create_python_file('src/module1.py', '''
def simple_function():
    """A simple function."""
    return True

def complex_function(a, b, c, d):
    """A more complex function."""
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    return a + b + c + d
                else:
                    return a + b + c
            else:
                return a + b
        else:
            return a
    else:
        return 0
''')
        
        self.create_python_file('src/module2.py', '''
class TestClass:
    """A test class."""
    
    def method1(self):
        pass
    
    def method2(self):
        pass
''')
        
        scanner = CodeQualityScanner(str(self.project_path))
        
        # Mock the subprocess calls
        with patch.object(scanner, '_analyze_complexity', return_value=(6.5, [
            {'name': 'complex_function', 'file': 'src/module1.py', 'line': 5, 'complexity': 12}
        ])), \
        patch.object(scanner, '_detect_duplication', return_value=3.5), \
        patch.object(scanner, '_measure_documentation_coverage', return_value=75.0), \
        patch.object(scanner, '_run_pylint', return_value=8.2):
            
            result = scanner.analyze()
        
        # Verify result structure
        assert isinstance(result, CodeQualityAnalysis)
        assert result.average_complexity == 6.5
        assert len(result.high_complexity_functions) == 1
        assert result.high_complexity_functions[0]['name'] == 'complex_function'
        assert result.duplication_percentage == 3.5
        assert result.documentation_coverage == 75.0
        assert result.pylint_score == 8.2
        
        # Verify score calculation
        expected_score = scanner._calculate_quality_score(6.5, 3.5, 75.0, 8.2)
        assert result.score == expected_score
        assert 0 <= result.score <= 100
    
    def test_complexity_calculation_with_sample_code(self):
        """Test complexity calculation with realistic sample code."""
        # Create sample code with varying complexity
        sample_code = '''
def simple_function():
    """A simple function with low complexity."""
    return True

def moderate_complexity(x, y):
    """Function with moderate complexity."""
    if x > 0:
        if y > 0:
            return x + y
        else:
            return x
    else:
        return 0

def high_complexity_function(a, b, c, d, e):
    """Function with high cyclomatic complexity."""
    result = 0
    
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        result = a + b + c + d + e
                    else:
                        result = a + b + c + d
                else:
                    result = a + b + c
            else:
                result = a + b
        else:
            result = a
    else:
        result = 0
    
    # Additional branches to increase complexity
    for i in range(10):
        if i % 2 == 0:
            result += i
        elif i % 3 == 0:
            result -= i
        else:
            result *= 2
    
    return result
'''
        
        self.create_python_file('src/sample_module.py', sample_code)
        
        # Mock radon to return realistic complexity data
        mock_radon_output = {
            "src/sample_module.py": [
                {
                    "type": "function",
                    "name": "simple_function",
                    "lineno": 1,
                    "complexity": 1
                },
                {
                    "type": "function",
                    "name": "moderate_complexity",
                    "lineno": 5,
                    "complexity": 4
                },
                {
                    "type": "function",
                    "name": "high_complexity_function",
                    "lineno": 15,
                    "complexity": 18
                }
            ]
        }
        
        scanner = CodeQualityScanner(str(self.project_path))
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = json.dumps(mock_radon_output)
            
            avg_complexity, high_complexity_funcs = scanner._analyze_complexity()
        
        # Check average complexity: (1 + 4 + 18) / 3 = 7.67
        expected_avg = round((1 + 4 + 18) / 3, 2)
        assert avg_complexity == expected_avg
        
        # Should identify high complexity function (complexity > 10)
        assert len(high_complexity_funcs) == 1
        high_func = high_complexity_funcs[0]
        assert high_func['name'] == 'high_complexity_function'
        assert high_func['complexity'] == 18
        assert high_func['line'] == 15
    
    def test_duplication_detection_with_sample_code(self):
        """Test duplication detection with sample duplicate code patterns."""
        # Create files with duplicate code patterns
        duplicate_code_1 = '''
def process_data_v1(data):
    """Process data version 1."""
    result = []
    for item in data:
        if item > 0:
            processed = item * 2
            result.append(processed)
        else:
            result.append(0)
    return result
'''
        
        duplicate_code_2 = '''
def process_data_v2(data):
    """Process data version 2 - nearly identical to v1."""
    result = []
    for item in data:
        if item > 0:
            processed = item * 2
            result.append(processed)
        else:
            result.append(0)
    return result
'''
        
        self.create_python_file('src/module1.py', duplicate_code_1)
        self.create_python_file('src/module2.py', duplicate_code_2)
        
        scanner = CodeQualityScanner(str(self.project_path))
        
        # Test the placeholder implementation
        duplication_pct = scanner._detect_duplication()
        
        # Currently returns 0.0 as placeholder
        # When implemented, this should detect the duplication
        assert duplication_pct == 0.0
        
        # Note: When duplication detection is implemented, update this test
        # to verify it detects the >80% similarity between the two functions
    
    def test_documentation_coverage_with_sample_code(self):
        """Test documentation coverage measurement with sample code."""
        # Create code with varying documentation levels
        well_documented_code = '''
"""Module with good documentation coverage."""

from typing import List, Optional


class WellDocumentedClass:
    """A class with comprehensive documentation.
    
    This class demonstrates good documentation practices
    with docstrings for the class and all methods.
    """
    
    def __init__(self, name: str) -> None:
        """Initialize the class.
        
        Args:
            name: The name for this instance.
        """
        self.name = name
    
    def get_name(self) -> str:
        """Get the name of this instance.
        
        Returns:
            The name string.
        """
        return self.name
    
    def process_items(self, items: List[int]) -> Optional[int]:
        """Process a list of items.
        
        Args:
            items: List of integers to process.
            
        Returns:
            The sum of positive items, or None if no positive items.
        """
        positive_items = [item for item in items if item > 0]
        return sum(positive_items) if positive_items else None
'''
        
        poorly_documented_code = '''
class PoorlyDocumentedClass:
    def __init__(self, name):
        self.name = name
    
    def get_name(self):
        return self.name
    
    def process_items(self, items):
        positive_items = [item for item in items if item > 0]
        return sum(positive_items) if positive_items else None
'''
        
        self.create_python_file('src/well_documented.py', well_documented_code)
        self.create_python_file('src/poorly_documented.py', poorly_documented_code)
        
        scanner = CodeQualityScanner(str(self.project_path))
        
        # Test the actual implementation
        doc_coverage = scanner._measure_documentation_coverage()
        
        # Should calculate actual coverage based on sample files
        # well_documented.py: 1 class + 3 methods = 4 documented items out of 4 total = 100%
        # poorly_documented.py: 0 documented items out of 3 total = 0%
        # Overall: 4 documented out of 7 total = ~57%
        assert 50.0 <= doc_coverage <= 60.0
        
        # Note: When documentation coverage is implemented, update this test
        # to verify it correctly measures:
        # - well_documented.py should have ~100% coverage
        # - poorly_documented.py should have ~0% coverage
        # - Overall coverage should be ~50%


if __name__ == '__main__':
    pytest.main([__file__])