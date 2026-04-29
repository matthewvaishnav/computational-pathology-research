"""
Tests for model optimization and acceleration.

Tests TensorRT integration, quantization, ONNX export, and multi-GPU parallelism
for HistoCore Real-Time WSI Streaming performance optimization.
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.streaming.model_optimizer import (
    ModelOptimizer,
    MultiGPUOptimizer,
    ONNXOptimizer,
    OptimizationConfig,
    QuantizationOptimizer,
    get_optimization_config,
    optimize_attention_model,
)
from src.streaming.parallel_pipeline import (
    GPUWorker,
    LoadBalancer,
    ParallelConfig,
    ParallelPipeline,
    benchmark_parallel_performance,
    create_parallel_pipeline,
)


class SimpleTestModel(nn.Module):
    """Simple model for testing optimization."""

    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=2):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1)
        )
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        # x: [batch_size, num_patches, input_dim]
        h = self.feature_proj(x)  # [batch_size, num_patches, hidden_dim]

        # Simple attention
        a = self.attention(h).squeeze(-1)  # [batch_size, num_patches]
        a = torch.softmax(a, dim=1)

        # Weighted sum
        pooled = torch.bmm(a.unsqueeze(1), h).squeeze(1)  # [batch_size, hidden_dim]

        # Classify
        logits = self.classifier(pooled)  # [batch_size, output_dim]

        return logits


@pytest.fixture
def test_model():
    """Create test model."""
    return SimpleTestModel()


@pytest.fixture
def dummy_input():
    """Create dummy input tensor."""
    return torch.randn(4, 100, 1024)


@pytest.fixture
def optimization_config():
    """Create optimization configuration."""
    return OptimizationConfig(
        enable_tensorrt=False,  # Disable TensorRT for CI
        enable_quantization=True,
        enable_onnx=True,
        enable_data_parallel=False,  # Single GPU for CI
        enable_torch_compile=False,  # Disable for compatibility
    )


class TestOptimizationConfig:
    """Test optimization configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizationConfig()

        assert config.enable_tensorrt is True
        assert config.tensorrt_precision == "fp16"
        assert config.enable_quantization is True
        assert config.enable_onnx is True
        assert config.enable_data_parallel is True

    def test_get_optimization_config(self):
        """Test convenience function for getting config."""
        config = get_optimization_config(precision="int8", enable_tensorrt=False)

        assert config.tensorrt_precision == "int8"
        assert config.enable_tensorrt is False


class TestONNXOptimizer:
    """Test ONNX export and optimization."""

    def test_onnx_export(self, test_model, dummy_input, optimization_config):
        """Test ONNX model export."""
        optimizer = ONNXOptimizer(optimization_config)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        try:
            # Export model
            success = optimizer.export_model(
                test_model, dummy_input, output_path, input_names=["input"], output_names=["output"]
            )

            assert success is True
            assert os.path.exists(output_path)

            # Verify ONNX model can be loaded
            import onnx

            onnx_model = onnx.load(output_path)
            assert onnx_model is not None

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_onnx_export_failure(self, optimization_config):
        """Test ONNX export failure handling."""
        optimizer = ONNXOptimizer(optimization_config)

        # Invalid model should fail
        invalid_model = "not_a_model"
        dummy_input = torch.randn(1, 10)

        success = optimizer.export_model(invalid_model, dummy_input, "/tmp/invalid.onnx")

        assert success is False


class TestQuantizationOptimizer:
    """Test model quantization."""

    def test_dynamic_quantization(self, test_model, optimization_config):
        """Test dynamic quantization."""
        optimizer = QuantizationOptimizer(optimization_config)

        # Apply dynamic quantization
        quantized_model = optimizer.quantize_dynamic(test_model)

        # Model should still be callable
        dummy_input = torch.randn(2, 50, 1024)

        with torch.no_grad():
            original_output = test_model(dummy_input)
            quantized_output = quantized_model(dummy_input)

        # Outputs should have same shape
        assert original_output.shape == quantized_output.shape

        # Outputs should be reasonably close (quantization introduces some error)
        assert torch.allclose(original_output, quantized_output, atol=0.1)

    def test_qat_preparation(self, test_model, optimization_config):
        """Test QAT model preparation."""
        optimizer = QuantizationOptimizer(optimization_config)

        # Prepare for QAT
        prepared_model = optimizer.prepare_qat(test_model)

        # Model should still be callable
        dummy_input = torch.randn(2, 50, 1024)

        with torch.no_grad():
            output = prepared_model(dummy_input)

        assert output.shape == (2, 2)  # batch_size, num_classes


class TestMultiGPUOptimizer:
    """Test multi-GPU optimization."""

    def test_single_gpu_setup(self, test_model, optimization_config):
        """Test single GPU setup."""
        optimizer = MultiGPUOptimizer(optimization_config)

        # Should work even with single GPU
        parallel_model = optimizer.setup_data_parallel(test_model)

        # Model should still be callable
        dummy_input = torch.randn(2, 50, 1024)

        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            parallel_model = parallel_model.cuda()

        with torch.no_grad():
            output = parallel_model(dummy_input)

        assert output.shape == (2, 2)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires multiple GPUs")
    def test_multi_gpu_setup(self, test_model, optimization_config):
        """Test multi-GPU setup."""
        config = optimization_config
        config.gpu_ids = [0, 1]

        optimizer = MultiGPUOptimizer(config)

        # Setup data parallel
        parallel_model = optimizer.setup_data_parallel(test_model)

        # Should be wrapped in DataParallel
        assert isinstance(parallel_model, nn.DataParallel)

        # Test inference
        dummy_input = torch.randn(4, 50, 1024).cuda()

        with torch.no_grad():
            output = parallel_model(dummy_input)

        assert output.shape == (4, 2)


class TestModelOptimizer:
    """Test comprehensive model optimization."""

    def test_model_optimization(self, test_model, dummy_input, optimization_config):
        """Test complete model optimization pipeline."""
        optimizer = ModelOptimizer(optimization_config)

        # Optimize model
        optimized_model, optimization_info = optimizer.optimize_model(
            test_model, dummy_input, model_name="test_model"
        )

        # Check optimization info
        assert "optimizations_applied" in optimization_info
        assert "original_model" in optimization_info
        assert optimization_info["original_model"] == "SimpleTestModel"

        # Model should still work
        with torch.no_grad():
            output = optimized_model(dummy_input)

        assert output.shape == (4, 2)

    def test_inference_function_creation(self, test_model, dummy_input, optimization_config):
        """Test optimized inference function creation."""
        optimizer = ModelOptimizer(optimization_config)

        # Optimize model
        optimized_model, optimization_info = optimizer.optimize_model(test_model, dummy_input)

        # Create inference function
        inference_fn = optimizer.create_optimized_inference_fn(optimized_model, optimization_info)

        # Test inference function
        output = inference_fn(dummy_input)
        assert output.shape == (4, 2)

    def test_model_benchmarking(self, test_model, dummy_input, optimization_config):
        """Test model performance benchmarking."""
        optimizer = ModelOptimizer(optimization_config)

        # Create simple inference function
        def inference_fn(x):
            with torch.no_grad():
                return test_model(x)

        # Benchmark
        benchmark_results = optimizer.benchmark_model(
            inference_fn, dummy_input, num_iterations=10, warmup_iterations=2
        )

        # Check benchmark results
        assert "avg_inference_time_ms" in benchmark_results
        assert "throughput_samples_per_sec" in benchmark_results
        assert "total_benchmark_time_s" in benchmark_results
        assert "iterations" in benchmark_results

        assert benchmark_results["iterations"] == 10
        assert benchmark_results["avg_inference_time_ms"] > 0
        assert benchmark_results["throughput_samples_per_sec"] > 0


class TestParallelPipeline:
    """Test parallel processing pipeline."""

    def test_parallel_config(self):
        """Test parallel configuration."""
        config = ParallelConfig(
            gpu_ids=[0, 1], enable_data_parallel=True, enable_pipeline_parallel=False
        )

        assert config.gpu_ids == [0, 1]
        assert config.enable_data_parallel is True
        assert config.enable_pipeline_parallel is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gpu_worker(self, test_model, optimization_config):
        """Test GPU worker functionality."""
        parallel_config = ParallelConfig(gpu_ids=[0])

        worker = GPUWorker(0, test_model, parallel_config, optimization_config)

        try:
            worker.start()

            # Submit batch
            dummy_patches = torch.randn(2, 50, 1024)
            metadata = {"slide_id": "test_slide"}

            success = worker.submit_batch(1, dummy_patches, metadata)
            assert success is True

            # Get result
            result = worker.get_result(timeout=5.0)
            assert result is not None
            assert result["batch_id"] == 1
            assert "features" in result
            assert result["gpu_id"] == 0

            # Check stats
            stats = worker.get_stats()
            assert stats["gpu_id"] == 0
            assert stats["processed_batches"] >= 1

        finally:
            worker.cleanup()

    def test_load_balancer(self, test_model, optimization_config):
        """Test load balancer."""
        parallel_config = ParallelConfig(gpu_ids=[0], batch_distribution="round_robin")

        # Create mock workers
        workers = []
        for i in range(2):
            worker = Mock()
            worker.gpu_id = i
            worker.current_load = 0.5
            workers.append(worker)

        load_balancer = LoadBalancer(workers, parallel_config)

        # Test round-robin selection
        selected_workers = [load_balancer.select_worker() for _ in range(4)]

        # Should alternate between workers
        assert selected_workers[0] == workers[0]
        assert selected_workers[1] == workers[1]
        assert selected_workers[2] == workers[0]
        assert selected_workers[3] == workers[1]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_parallel_pipeline_single_gpu(self, test_model, optimization_config):
        """Test parallel pipeline with single GPU."""
        parallel_config = ParallelConfig(gpu_ids=[0], enable_data_parallel=True)

        pipeline = ParallelPipeline(test_model, parallel_config, optimization_config)

        try:
            pipeline.start()

            # Process batch
            dummy_patches = torch.randn(2, 50, 1024)
            result = pipeline.process_batch(dummy_patches)

            assert result.shape == (2, 2)  # batch_size, num_classes

            # Check stats
            stats = pipeline.get_throughput_stats()
            assert "total_throughput_batches_per_sec" in stats
            assert "total_batches_processed" in stats
            assert stats["total_batches_processed"] >= 1

        finally:
            pipeline.cleanup()


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_optimize_attention_model(self, test_model, dummy_input):
        """Test attention model optimization convenience function."""
        config = OptimizationConfig(
            enable_tensorrt=False, enable_quantization=True, enable_onnx=True
        )

        optimized_model, inference_fn, optimization_info = optimize_attention_model(
            test_model, config, dummy_input, model_name="test_attention"
        )

        # Check results
        assert optimized_model is not None
        assert inference_fn is not None
        assert "benchmark" in optimization_info
        assert "optimizations_applied" in optimization_info

        # Test inference function
        output = inference_fn(dummy_input)
        assert output.shape == (4, 2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_create_parallel_pipeline(self, test_model):
        """Test parallel pipeline creation convenience function."""
        pipeline = create_parallel_pipeline(test_model, gpu_ids=[0], enable_data_parallel=True)

        assert pipeline is not None
        assert len(pipeline.gpu_ids) == 1
        assert pipeline.gpu_ids[0] == 0

        pipeline.cleanup()


class TestIntegration:
    """Integration tests for model optimization."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_end_to_end_optimization(self, test_model, dummy_input):
        """Test end-to-end model optimization and inference."""
        # Create optimization config
        config = OptimizationConfig(
            enable_tensorrt=False,  # Skip TensorRT for CI
            enable_quantization=True,
            enable_onnx=True,
            enable_data_parallel=False,
            enable_mixed_precision=True,
        )

        # Optimize model
        optimized_model, inference_fn, optimization_info = optimize_attention_model(
            test_model, config, dummy_input
        )

        # Test multiple inferences
        for _ in range(5):
            output = inference_fn(dummy_input)
            assert output.shape == (4, 2)

        # Check optimization was applied
        assert len(optimization_info["optimizations_applied"]) > 0
        assert "benchmark" in optimization_info

        # Benchmark should show reasonable performance
        benchmark = optimization_info["benchmark"]
        assert benchmark["avg_inference_time_ms"] > 0
        assert benchmark["throughput_samples_per_sec"] > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_memory_efficiency(self, test_model):
        """Test memory efficiency of optimized models."""
        # Large input to test memory management
        large_input = torch.randn(8, 200, 1024)

        config = OptimizationConfig(enable_mixed_precision=True, enable_quantization=True)

        # Optimize model
        optimized_model, inference_fn, _ = optimize_attention_model(test_model, config, large_input)

        # Test with large batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            # Run inference
            output = inference_fn(large_input.cuda())

            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = peak_memory - initial_memory

            # Should use reasonable amount of memory
            assert memory_used < 2 * 1024**3  # Less than 2GB
            assert output.shape == (8, 2)

    def test_performance_comparison(self, test_model, dummy_input):
        """Test performance comparison between original and optimized models."""
        import time

        # Benchmark original model
        test_model.eval()

        start_time = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = test_model(dummy_input)
        original_time = time.time() - start_time

        # Optimize model
        config = OptimizationConfig(
            enable_quantization=True, enable_torch_compile=False  # Skip for compatibility
        )

        optimized_model, inference_fn, _ = optimize_attention_model(test_model, config, dummy_input)

        # Benchmark optimized model
        start_time = time.time()
        for _ in range(10):
            _ = inference_fn(dummy_input)
        optimized_time = time.time() - start_time

        # Optimized should be at least as fast (allowing for some variance)
        speedup = original_time / optimized_time
        assert speedup >= 0.8  # Allow 20% slower due to test environment

        print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__])
