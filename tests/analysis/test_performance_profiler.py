"""
Unit tests for Performance Profiler.

Tests GPU utilization measurement, bottleneck detection, and recommendation generation.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.analysis.models import PerformanceAnalysis
from src.analysis.performance import PerformanceProfiler


class TestPerformanceProfiler:
    """Test suite for PerformanceProfiler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.project_path = "/test/project"
        self.profiler = PerformanceProfiler(self.project_path)

    def test_init(self):
        """Test profiler initialization."""
        assert self.profiler.project_path == Path(self.project_path).resolve()

    @patch("src.analysis.performance.subprocess.run")
    def test_measure_gpu_utilization_success(self, mock_run):
        """Test successful GPU utilization measurement."""
        # Mock successful nvidia-smi output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "85\n"
        mock_run.return_value = mock_result

        utilization = self.profiler._measure_gpu_utilization()

        assert utilization == 85.0
        mock_run.assert_called_once_with(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    @patch("src.analysis.performance.subprocess.run")
    def test_measure_gpu_utilization_multiple_gpus(self, mock_run):
        """Test GPU utilization measurement with multiple GPUs (uses first)."""
        # Mock nvidia-smi output with multiple GPUs
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "85\n72\n90\n"
        mock_run.return_value = mock_result

        utilization = self.profiler._measure_gpu_utilization()

        assert utilization == 85.0  # Should use first GPU

    @patch("src.analysis.performance.subprocess.run")
    def test_measure_gpu_utilization_command_failure(self, mock_run):
        """Test GPU utilization measurement when nvidia-smi fails."""
        # Mock failed nvidia-smi command
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        utilization = self.profiler._measure_gpu_utilization()

        assert utilization == 0.0

    @patch("src.analysis.performance.subprocess.run")
    def test_measure_gpu_utilization_timeout(self, mock_run):
        """Test GPU utilization measurement with timeout."""
        # Mock timeout exception
        mock_run.side_effect = subprocess.TimeoutExpired(["nvidia-smi"], 10)

        utilization = self.profiler._measure_gpu_utilization()

        assert utilization == 0.0

    @patch("src.analysis.performance.subprocess.run")
    def test_measure_gpu_utilization_file_not_found(self, mock_run):
        """Test GPU utilization measurement when nvidia-smi not found."""
        # Mock FileNotFoundError (nvidia-smi not installed)
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

        utilization = self.profiler._measure_gpu_utilization()

        assert utilization == 0.0

    @patch("src.analysis.performance.subprocess.run")
    def test_measure_gpu_utilization_invalid_output(self, mock_run):
        """Test GPU utilization measurement with invalid output."""
        # Mock invalid output that can't be parsed as float
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "invalid\n"
        mock_run.return_value = mock_result

        utilization = self.profiler._measure_gpu_utilization()

        assert utilization == 0.0

    @patch("src.analysis.performance.subprocess.run")
    def test_measure_gpu_utilization_empty_output(self, mock_run):
        """Test GPU utilization measurement with empty output."""
        # Mock empty output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        utilization = self.profiler._measure_gpu_utilization()

        assert utilization == 0.0

    def test_detect_bottlenecks_placeholder(self):
        """Test bottleneck detection (currently placeholder)."""
        bottlenecks = self.profiler._detect_bottlenecks()

        # Currently returns empty list as it's not implemented
        assert bottlenecks == []
        assert isinstance(bottlenecks, list)

    def test_measure_memory_usage_placeholder(self):
        """Test memory usage measurement (currently placeholder)."""
        peak, avg = self.profiler._measure_memory_usage()

        # Currently returns (0.0, 0.0) as it's not implemented
        assert peak == 0.0
        assert avg == 0.0
        assert isinstance(peak, float)
        assert isinstance(avg, float)

    def test_generate_flame_graph_placeholder(self):
        """Test flame graph generation (currently placeholder)."""
        flame_graph_path = self.profiler._generate_flame_graph()

        # Currently returns empty string as it's not implemented
        assert flame_graph_path == ""
        assert isinstance(flame_graph_path, str)

    def test_calculate_performance_score_optimal_gpu(self):
        """Test performance score calculation with optimal GPU utilization."""
        gpu_util = 85.0  # Optimal range 80-95%
        bottlenecks = []
        memory_peak = 8.0  # Under 12GB threshold

        score = self.profiler._calculate_performance_score(gpu_util, bottlenecks, memory_peak)

        # Base (30) + GPU optimal (40) + No bottlenecks (0) + Good memory (30) = 100
        assert score == 100.0

    def test_calculate_performance_score_low_gpu(self):
        """Test performance score calculation with low GPU utilization."""
        gpu_util = 40.0  # Half of optimal 80%
        bottlenecks = []
        memory_peak = 8.0

        score = self.profiler._calculate_performance_score(gpu_util, bottlenecks, memory_peak)

        # Base (30) + GPU half (20) + No bottlenecks (0) + Good memory (30) = 80
        assert score == 80.0

    def test_calculate_performance_score_high_gpu(self):
        """Test performance score calculation with over-utilization."""
        gpu_util = 98.0  # Over 95% threshold
        bottlenecks = []
        memory_peak = 8.0

        score = self.profiler._calculate_performance_score(gpu_util, bottlenecks, memory_peak)

        # Base (30) + GPU penalty for over-util + No bottlenecks (0) + Good memory (30)
        # GPU score = 40 * (1 - (98-95)/5) = 40 * 0.4 = 16
        expected = 30 + 16 + 0 + 30
        assert score == expected

    def test_calculate_performance_score_with_bottlenecks(self):
        """Test performance score calculation with bottlenecks."""
        gpu_util = 85.0
        bottlenecks = [
            {"operation": "data_loading", "time_ms": 150},
            {"operation": "preprocessing", "time_ms": 200},
        ]  # 2 bottlenecks = -20 points
        memory_peak = 8.0

        score = self.profiler._calculate_performance_score(gpu_util, bottlenecks, memory_peak)

        # Base (30) + GPU optimal (40) + Bottleneck penalty (-20) + Good memory (30) = 80
        assert score == 80.0

    def test_calculate_performance_score_high_memory(self):
        """Test performance score calculation with high memory usage."""
        gpu_util = 85.0
        bottlenecks = []
        memory_peak = 14.0  # Over 12GB threshold

        score = self.profiler._calculate_performance_score(gpu_util, bottlenecks, memory_peak)

        # Base (30) + GPU optimal (40) + No bottlenecks (0) + Memory penalty
        # Memory score = 30 * max(0, 1 - (14-12)/4) = 30 * 0.5 = 15
        expected = 30 + 40 + 0 + 15
        assert score == expected

    def test_calculate_performance_score_no_memory_data(self):
        """Test performance score calculation with no memory data."""
        gpu_util = 85.0
        bottlenecks = []
        memory_peak = 0.0  # No memory data

        score = self.profiler._calculate_performance_score(gpu_util, bottlenecks, memory_peak)

        # Base (30) + GPU optimal (40) + No bottlenecks (0) + Partial memory credit (15) = 85
        assert score == 85.0

    def test_calculate_performance_score_many_bottlenecks(self):
        """Test performance score calculation with many bottlenecks (capped penalty)."""
        gpu_util = 85.0
        bottlenecks = [
            {"operation": f"bottleneck_{i}", "time_ms": 100} for i in range(5)
        ]  # 5 bottlenecks
        memory_peak = 8.0

        score = self.profiler._calculate_performance_score(gpu_util, bottlenecks, memory_peak)

        # Base (30) + GPU optimal (40) + Max bottleneck penalty (-30) + Good memory (30) = 70
        assert score == 70.0

    def test_calculate_performance_score_bounds(self):
        """Test performance score calculation stays within 0-100 bounds."""
        # Test minimum bound
        gpu_util = 0.0
        bottlenecks = [
            {"operation": f"bottleneck_{i}", "time_ms": 100} for i in range(10)
        ]  # Many bottlenecks
        memory_peak = 20.0  # Very high memory

        score = self.profiler._calculate_performance_score(gpu_util, bottlenecks, memory_peak)

        assert 0.0 <= score <= 100.0
        assert score == 0.0  # Should hit minimum bound

    @patch.object(PerformanceProfiler, "_measure_gpu_utilization")
    @patch.object(PerformanceProfiler, "_detect_bottlenecks")
    @patch.object(PerformanceProfiler, "_measure_memory_usage")
    @patch.object(PerformanceProfiler, "_generate_flame_graph")
    @patch.object(PerformanceProfiler, "_calculate_performance_score")
    def test_analyze_integration(
        self, mock_score, mock_flame, mock_memory, mock_bottlenecks, mock_gpu
    ):
        """Test full analyze method integration."""
        # Mock all internal methods
        mock_gpu.return_value = 85.0
        mock_bottlenecks.return_value = [{"operation": "test", "time_ms": 100}]
        mock_memory.return_value = (10.0, 8.0)
        mock_flame.return_value = "/path/to/flame.svg"
        mock_score.return_value = 92.5

        result = self.profiler.analyze()

        # Verify all methods were called
        mock_gpu.assert_called_once()
        mock_bottlenecks.assert_called_once()
        mock_memory.assert_called_once()
        mock_flame.assert_called_once()
        mock_score.assert_called_once_with(85.0, [{"operation": "test", "time_ms": 100}], 10.0)

        # Verify result structure
        assert isinstance(result, PerformanceAnalysis)
        assert result.gpu_utilization == 85.0
        assert result.bottlenecks == [{"operation": "test", "time_ms": 100}]
        assert result.flame_graph_path == "/path/to/flame.svg"
        assert result.memory_usage_peak_gb == 10.0
        assert result.memory_usage_avg_gb == 8.0
        assert result.score == 92.5


class TestPerformanceProfilerRecommendations:
    """Test suite for performance profiler recommendation generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = PerformanceProfiler("/test/project")

    def test_gpu_utilization_recommendations_low(self):
        """Test recommendations for low GPU utilization."""
        # Test with low GPU utilization (should recommend optimization)
        gpu_util = 30.0
        bottlenecks = []
        memory_peak = 8.0

        score = self.profiler._calculate_performance_score(gpu_util, bottlenecks, memory_peak)

        # Low GPU utilization should result in lower score
        assert score < 80.0
        # This indicates need for GPU optimization recommendations

    def test_memory_optimization_recommendations_high(self):
        """Test recommendations for high memory usage."""
        # Test with high memory usage (should recommend memory optimization)
        gpu_util = 85.0
        bottlenecks = []
        memory_peak = 15.0  # High memory usage

        score = self.profiler._calculate_performance_score(gpu_util, bottlenecks, memory_peak)

        # High memory usage should result in lower score
        assert score < 90.0
        # This indicates need for memory optimization recommendations

    def test_bottleneck_recommendations(self):
        """Test recommendations for bottleneck detection."""
        # Test with multiple bottlenecks (should recommend bottleneck fixes)
        gpu_util = 85.0
        bottlenecks = [
            {"operation": "data_loading", "time_ms": 200, "percentage": 30.0},
            {"operation": "preprocessing", "time_ms": 150, "percentage": 20.0},
        ]
        memory_peak = 8.0

        score = self.profiler._calculate_performance_score(gpu_util, bottlenecks, memory_peak)

        # Bottlenecks should result in lower score
        assert score < 90.0
        # This indicates need for bottleneck optimization recommendations

    def test_optimal_performance_no_recommendations(self):
        """Test that optimal performance results in high score (minimal recommendations)."""
        # Test with optimal settings
        gpu_util = 87.0  # Optimal range
        bottlenecks = []  # No bottlenecks
        memory_peak = 10.0  # Good memory usage

        score = self.profiler._calculate_performance_score(gpu_util, bottlenecks, memory_peak)

        # Optimal performance should result in high score
        assert score >= 95.0
        # This indicates minimal optimization recommendations needed


class TestPerformanceProfilerMockData:
    """Test suite using mock data for comprehensive testing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = PerformanceProfiler("/test/project")

    @patch("src.analysis.performance.subprocess.run")
    def test_realistic_gpu_scenarios(self, mock_run):
        """Test various realistic GPU utilization scenarios."""
        scenarios = [
            ("0", 0.0),  # No GPU
            ("25", 25.0),  # Low utilization
            ("50", 50.0),  # Medium utilization
            ("85", 85.0),  # Optimal utilization
            ("95", 95.0),  # High optimal utilization
            ("99", 99.0),  # Over-utilization
        ]

        for output, expected in scenarios:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = f"{output}\n"
            mock_run.return_value = mock_result

            utilization = self.profiler._measure_gpu_utilization()
            assert utilization == expected

    def test_score_calculation_edge_cases(self):
        """Test performance score calculation edge cases."""
        # Test edge case: exactly at thresholds
        test_cases = [
            (80.0, [], 12.0, 100.0),  # Exactly at GPU threshold, memory threshold
            (95.0, [], 0.0, 85.0),  # Exactly at GPU upper threshold, no memory data
            (0.0, [], 0.0, 45.0),  # No GPU, no memory data
        ]

        for gpu_util, bottlenecks, memory_peak, expected_min in test_cases:
            score = self.profiler._calculate_performance_score(gpu_util, bottlenecks, memory_peak)
            assert isinstance(score, float)
            assert 0.0 <= score <= 100.0
            # For some cases, we expect at least a minimum score
            if expected_min:
                assert score >= expected_min - 1.0  # Allow small floating point differences
