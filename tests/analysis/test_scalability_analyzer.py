"""
Unit tests for Scalability Analyzer.

Tests DDP implementation verification, communication overhead estimation,
and scaling recommendation generation.
Requirements: 8.1, 8.3, 8.7
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.analysis.scalability import ScalabilityAnalyzer
from src.analysis.models import ScalabilityAnalysis


class TestScalabilityAnalyzer:
    """Test suite for ScalabilityAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.analyzer = ScalabilityAnalyzer(str(self.project_path))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = ScalabilityAnalyzer("/path/to/project")
        assert analyzer.project_path == Path("/path/to/project").resolve()
    
    def test_analyze_returns_scalability_analysis(self):
        """Test that analyze() returns a ScalabilityAnalysis object."""
        with patch.object(self.analyzer, '_verify_ddp_implementation', return_value=True):
            with patch.object(self.analyzer, '_detect_data_loading_bottlenecks', return_value=[]):
                with patch.object(self.analyzer, '_measure_communication_overhead', return_value=25.5):
                    with patch.object(self.analyzer, '_assess_large_dataset_handling', return_value=[]):
                        with patch.object(self.analyzer, '_generate_scaling_recommendations', return_value="linear"):
                            result = self.analyzer.analyze()
        
        assert isinstance(result, ScalabilityAnalysis)
        assert result.ddp_correctness is True
        assert result.scaling_efficiency == "linear"
        assert result.memory_bottlenecks == []
        assert result.communication_overhead_ms == 25.5
        assert isinstance(result.score, float)
        assert 0 <= result.score <= 100


class TestDDPImplementationVerification:
    """Test DistributedDataParallel implementation verification."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.analyzer = ScalabilityAnalyzer(str(self.project_path))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_python_file(self, path: str, content: str):
        """Create a Python file with specified content."""
        file_path = self.project_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path
    
    def test_verify_ddp_implementation_correct(self):
        """Test DDP verification with correct implementation."""
        correct_ddp_code = '''
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

def setup_ddp():
    dist.init_process_group(backend='nccl')
    model = Model()
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    return model

def train_step(model, data, target):
    optimizer = torch.optim.Adam(model.parameters())
    output = model(data)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss
'''
        
        self.create_python_file('src/training.py', correct_ddp_code)
        
        is_correct = self.analyzer._verify_ddp_implementation()
        
        # Should detect correct DDP usage
        assert is_correct is True
    
    def test_verify_ddp_implementation_incorrect(self):
        """Test DDP verification with incorrect implementation."""
        incorrect_code = '''
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

def train_step(model, data, target):
    # No DDP usage - just regular training
    optimizer = torch.optim.Adam(model.parameters())
    output = model(data)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss
'''
        
        self.create_python_file('src/training.py', incorrect_code)
        
        is_correct = self.analyzer._verify_ddp_implementation()
        
        # Should detect missing DDP usage
        assert is_correct is False
    
    def test_verify_ddp_implementation_no_files(self):
        """Test DDP verification with no Python files."""
        is_correct = self.analyzer._verify_ddp_implementation()
        
        # Should return False when no files found
        assert is_correct is False
    
    def test_verify_ddp_implementation_partial(self):
        """Test DDP verification with partial implementation."""
        partial_ddp_code = '''
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

# DDP import but no proper setup
model = Model()
# Missing: dist.init_process_group() and DDP wrapping
'''
        
        self.create_python_file('src/model.py', partial_ddp_code)
        
        is_correct = self.analyzer._verify_ddp_implementation()
        
        # Should detect incomplete DDP implementation
        assert is_correct is False


class TestDataLoadingBottleneckDetection:
    """Test data loading bottleneck detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.analyzer = ScalabilityAnalyzer(str(self.project_path))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_python_file(self, path: str, content: str):
        """Create a Python file with specified content."""
        file_path = self.project_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path
    
    def test_detect_data_loading_bottlenecks_good_config(self):
        """Test bottleneck detection with good DataLoader configuration."""
        good_dataloader_code = '''
import torch
from torch.utils.data import DataLoader, DistributedSampler

def create_dataloader(dataset, batch_size=32):
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=8,  # Good number of workers
        pin_memory=True,  # GPU optimization
        prefetch_factor=2,  # Prefetching
        persistent_workers=True  # Worker persistence
    )
    return dataloader
'''
        
        self.create_python_file('src/data_loading.py', good_dataloader_code)
        
        bottlenecks = self.analyzer._detect_data_loading_bottlenecks()
        
        # Should detect minimal bottlenecks with good configuration
        assert len(bottlenecks) == 0 or all(b['severity'] == 'low' for b in bottlenecks)
    
    def test_detect_data_loading_bottlenecks_poor_config(self):
        """Test bottleneck detection with poor DataLoader configuration."""
        poor_dataloader_code = '''
import torch
from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size=32):
    # Poor configuration - no optimization
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # No multiprocessing
        pin_memory=False,  # No GPU optimization
        # Missing: DistributedSampler, prefetching, etc.
    )
    return dataloader
'''
        
        self.create_python_file('src/data_loading.py', poor_dataloader_code)
        
        bottlenecks = self.analyzer._detect_data_loading_bottlenecks()
        
        # Should detect multiple bottlenecks
        assert len(bottlenecks) >= 1
        
        # Check for specific bottleneck types
        bottleneck_types = [b['type'] for b in bottlenecks]
        assert any('workers' in bt.lower() for bt in bottleneck_types)
    
    def test_detect_data_loading_bottlenecks_no_dataloader(self):
        """Test bottleneck detection with no DataLoader usage."""
        no_dataloader_code = '''
import torch

def simple_training():
    # No DataLoader usage
    data = torch.randn(100, 10)
    target = torch.randn(100, 1)
    return data, target
'''
        
        self.create_python_file('src/simple.py', no_dataloader_code)
        
        bottlenecks = self.analyzer._detect_data_loading_bottlenecks()
        
        # Should return empty list when no DataLoader found
        assert bottlenecks == []


class TestCommunicationOverheadMeasurement:
    """Test communication overhead measurement functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.analyzer = ScalabilityAnalyzer(str(self.project_path))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_measure_communication_overhead_placeholder(self):
        """Test communication overhead measurement (currently placeholder)."""
        overhead = self.analyzer._measure_communication_overhead()
        
        # Currently returns 0.0 as placeholder
        assert overhead == 0.0
        assert isinstance(overhead, float)
    
    def test_estimate_gradient_sync_time(self):
        """Test gradient synchronization time estimation."""
        # Mock model parameters for estimation
        mock_model_size_mb = 100.0  # 100MB model
        mock_num_gpus = 4
        
        estimated_time = self.analyzer._estimate_gradient_sync_time(mock_model_size_mb, mock_num_gpus)
        
        # Should return reasonable estimate
        assert isinstance(estimated_time, float)
        assert estimated_time >= 0.0
        
        # Larger models should take longer
        larger_model_time = self.analyzer._estimate_gradient_sync_time(200.0, mock_num_gpus)
        assert larger_model_time >= estimated_time
        
        # More GPUs should take longer (more communication)
        more_gpus_time = self.analyzer._estimate_gradient_sync_time(mock_model_size_mb, 8)
        assert more_gpus_time >= estimated_time


class TestLargeDatasetHandlingAssessment:
    """Test large dataset handling assessment functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.analyzer = ScalabilityAnalyzer(str(self.project_path))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_python_file(self, path: str, content: str):
        """Create a Python file with specified content."""
        file_path = self.project_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path
    
    def test_assess_large_dataset_handling_streaming(self):
        """Test assessment with streaming data loading."""
        streaming_code = '''
import torch
from torch.utils.data import IterableDataset

class StreamingDataset(IterableDataset):
    def __init__(self, data_path):
        self.data_path = data_path
    
    def __iter__(self):
        # Stream data without loading everything into memory
        for chunk in self.read_chunks():
            yield chunk
    
    def read_chunks(self):
        # Efficient streaming implementation
        pass

def create_streaming_loader():
    dataset = StreamingDataset('/large/dataset/path')
    return torch.utils.data.DataLoader(dataset, batch_size=32)
'''
        
        self.create_python_file('src/streaming.py', streaming_code)
        
        bottlenecks = self.analyzer._assess_large_dataset_handling()
        
        # Should detect minimal bottlenecks with streaming
        assert len(bottlenecks) == 0 or all(b['severity'] == 'low' for b in bottlenecks)
    
    def test_assess_large_dataset_handling_memory_intensive(self):
        """Test assessment with memory-intensive data loading."""
        memory_intensive_code = '''
import torch
import numpy as np

def load_entire_dataset():
    # Bad practice - loading everything into memory
    all_data = []
    for i in range(1000000):  # Very large dataset
        data = np.random.randn(1024, 1024)  # Large arrays
        all_data.append(data)
    return torch.tensor(all_data)

class MemoryIntensiveDataset:
    def __init__(self):
        self.data = load_entire_dataset()  # Load everything at once
    
    def __getitem__(self, idx):
        return self.data[idx]
'''
        
        self.create_python_file('src/memory_intensive.py', memory_intensive_code)
        
        bottlenecks = self.analyzer._assess_large_dataset_handling()
        
        # Should detect memory bottlenecks
        assert len(bottlenecks) >= 1
        
        # Check for memory-related bottlenecks
        memory_bottlenecks = [b for b in bottlenecks if 'memory' in b['type'].lower()]
        assert len(memory_bottlenecks) >= 1
    
    def test_assess_large_dataset_handling_no_large_data(self):
        """Test assessment with no large dataset handling."""
        simple_code = '''
import torch

def simple_training():
    # Small, simple dataset
    data = torch.randn(100, 10)
    return data
'''
        
        self.create_python_file('src/simple.py', simple_code)
        
        bottlenecks = self.analyzer._assess_large_dataset_handling()
        
        # Should return empty list for simple cases
        assert bottlenecks == []


class TestScalingRecommendationGeneration:
    """Test scaling recommendation generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ScalabilityAnalyzer("/test/project")
    
    def test_generate_scaling_recommendations_linear(self):
        """Test scaling recommendations for linear scaling."""
        # Mock perfect scaling conditions
        ddp_correct = True
        bottlenecks = []
        comm_overhead = 10.0  # Low overhead
        
        efficiency = self.analyzer._generate_scaling_recommendations(
            ddp_correct, bottlenecks, comm_overhead
        )
        
        # Should recommend linear scaling
        assert efficiency == "linear"
    
    def test_generate_scaling_recommendations_sublinear(self):
        """Test scaling recommendations for sub-linear scaling."""
        # Mock conditions that lead to sub-linear scaling
        ddp_correct = True
        bottlenecks = [
            {'type': 'data_loading', 'severity': 'medium'},
            {'type': 'memory_bandwidth', 'severity': 'low'}
        ]
        comm_overhead = 50.0  # Higher overhead
        
        efficiency = self.analyzer._generate_scaling_recommendations(
            ddp_correct, bottlenecks, comm_overhead
        )
        
        # Should recommend sub-linear scaling
        assert efficiency == "sub-linear"
    
    def test_generate_scaling_recommendations_poor(self):
        """Test scaling recommendations for poor scaling."""
        # Mock poor scaling conditions
        ddp_correct = False  # No DDP
        bottlenecks = [
            {'type': 'data_loading', 'severity': 'high'},
            {'type': 'memory', 'severity': 'high'},
            {'type': 'communication', 'severity': 'high'}
        ]
        comm_overhead = 100.0  # Very high overhead
        
        efficiency = self.analyzer._generate_scaling_recommendations(
            ddp_correct, bottlenecks, comm_overhead
        )
        
        # Should indicate poor scaling
        assert efficiency in ["poor", "unknown"]
    
    def test_estimate_speedup_for_gpus(self):
        """Test speedup estimation for different GPU counts."""
        base_efficiency = "linear"
        
        # Test speedup estimates
        speedup_2gpu = self.analyzer._estimate_speedup_for_gpus(2, base_efficiency)
        speedup_4gpu = self.analyzer._estimate_speedup_for_gpus(4, base_efficiency)
        speedup_8gpu = self.analyzer._estimate_speedup_for_gpus(8, base_efficiency)
        
        # Linear scaling should show proportional speedup
        assert speedup_2gpu <= 2.0
        assert speedup_4gpu <= 4.0
        assert speedup_8gpu <= 8.0
        
        # More GPUs should generally give higher speedup
        assert speedup_4gpu >= speedup_2gpu
        assert speedup_8gpu >= speedup_4gpu
    
    def test_estimate_speedup_sublinear(self):
        """Test speedup estimation for sub-linear scaling."""
        base_efficiency = "sub-linear"
        
        speedup_4gpu = self.analyzer._estimate_speedup_for_gpus(4, base_efficiency)
        speedup_8gpu = self.analyzer._estimate_speedup_for_gpus(8, base_efficiency)
        
        # Sub-linear scaling should show diminishing returns
        assert speedup_4gpu < 4.0
        assert speedup_8gpu < 8.0
        
        # Efficiency should decrease with more GPUs
        efficiency_4gpu = speedup_4gpu / 4.0
        efficiency_8gpu = speedup_8gpu / 8.0
        assert efficiency_8gpu <= efficiency_4gpu


class TestScalabilityScoreCalculation:
    """Test scalability score calculation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ScalabilityAnalyzer("/test/project")
    
    def test_calculate_scalability_score_perfect(self):
        """Test scalability score calculation with perfect scaling."""
        score = self.analyzer._calculate_scalability_score(
            ddp_correct=True,
            scaling_efficiency="linear",
            bottlenecks=[],
            comm_overhead=5.0  # Very low overhead
        )
        
        # Perfect scaling should result in high score
        assert score >= 90.0
    
    def test_calculate_scalability_score_poor(self):
        """Test scalability score calculation with poor scaling."""
        many_bottlenecks = [
            {'type': 'data_loading', 'severity': 'high'},
            {'type': 'memory', 'severity': 'high'},
            {'type': 'communication', 'severity': 'medium'}
        ]
        
        score = self.analyzer._calculate_scalability_score(
            ddp_correct=False,
            scaling_efficiency="poor",
            bottlenecks=many_bottlenecks,
            comm_overhead=100.0  # Very high overhead
        )
        
        # Poor scaling should result in low score
        assert score < 40.0
    
    def test_calculate_scalability_score_mixed(self):
        """Test scalability score calculation with mixed conditions."""
        some_bottlenecks = [
            {'type': 'data_loading', 'severity': 'medium'}
        ]
        
        score = self.analyzer._calculate_scalability_score(
            ddp_correct=True,
            scaling_efficiency="sub-linear",
            bottlenecks=some_bottlenecks,
            comm_overhead=30.0  # Moderate overhead
        )
        
        # Mixed conditions should result in moderate score
        assert 50.0 <= score <= 80.0
    
    def test_calculate_scalability_score_bounds(self):
        """Test scalability score calculation stays within bounds."""
        # Test extreme values
        score = self.analyzer._calculate_scalability_score(
            ddp_correct=False,
            scaling_efficiency="unknown",
            bottlenecks=[{'type': 'severe', 'severity': 'critical'} for _ in range(10)],
            comm_overhead=1000.0  # Unrealistic high overhead
        )
        
        # Should be bounded between 0 and 100
        assert 0.0 <= score <= 100.0


class TestIntegrationWithMockData:
    """Integration tests with mock scalability data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.analyzer = ScalabilityAnalyzer(str(self.project_path))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_analysis_with_mock_data(self):
        """Test complete analysis workflow with mocked data."""
        mock_bottlenecks = [
            {'type': 'data_loading', 'severity': 'medium', 'description': 'Low worker count'}
        ]
        
        # Mock all the analysis methods
        with patch.object(self.analyzer, '_verify_ddp_implementation', return_value=True):
            with patch.object(self.analyzer, '_detect_data_loading_bottlenecks', return_value=mock_bottlenecks):
                with patch.object(self.analyzer, '_measure_communication_overhead', return_value=25.0):
                    with patch.object(self.analyzer, '_assess_large_dataset_handling', return_value=[]):
                        with patch.object(self.analyzer, '_generate_scaling_recommendations', return_value="sub-linear"):
                            
                            result = self.analyzer.analyze()
        
        # Verify all fields are populated
        assert result.ddp_correctness is True
        assert result.scaling_efficiency == "sub-linear"
        assert len(result.memory_bottlenecks) == 1
        assert result.communication_overhead_ms == 25.0
        
        # Verify score calculation
        assert isinstance(result.score, float)
        assert 0 <= result.score <= 100
    
    def test_analysis_with_excellent_scalability(self):
        """Test analysis with excellent scalability."""
        with patch.object(self.analyzer, '_verify_ddp_implementation', return_value=True):
            with patch.object(self.analyzer, '_detect_data_loading_bottlenecks', return_value=[]):
                with patch.object(self.analyzer, '_measure_communication_overhead', return_value=5.0):
                    with patch.object(self.analyzer, '_assess_large_dataset_handling', return_value=[]):
                        with patch.object(self.analyzer, '_generate_scaling_recommendations', return_value="linear"):
                            
                            result = self.analyzer.analyze()
        
        # Excellent scalability should result in high score
        assert result.score > 85.0
    
    def test_analysis_with_poor_scalability(self):
        """Test analysis with poor scalability."""
        many_bottlenecks = [
            {'type': 'data_loading', 'severity': 'high'},
            {'type': 'memory', 'severity': 'high'},
            {'type': 'communication', 'severity': 'medium'}
        ]
        
        with patch.object(self.analyzer, '_verify_ddp_implementation', return_value=False):
            with patch.object(self.analyzer, '_detect_data_loading_bottlenecks', return_value=many_bottlenecks):
                with patch.object(self.analyzer, '_measure_communication_overhead', return_value=80.0):
                    with patch.object(self.analyzer, '_assess_large_dataset_handling', return_value=many_bottlenecks):
                        with patch.object(self.analyzer, '_generate_scaling_recommendations', return_value="poor"):
                            
                            result = self.analyzer.analyze()
        
        # Poor scalability should result in low score
        assert result.score < 50.0


if __name__ == '__main__':
    pytest.main([__file__])