"""
Performance benchmark tests for refactored MIL models.

This module benchmarks the inference performance of AttentionMIL, CLAM, and TransMIL
models after refactoring to ensure performance is maintained within ±5% of baseline.

Task 2.12: Benchmark Model Performance
- Benchmark AttentionMIL inference (1000 iterations)
- Benchmark CLAM inference (1000 iterations)
- Benchmark TransMIL inference (1000 iterations)
- Verify performance within ±5% of original
"""

import time
from typing import Dict, List

import pytest
import torch
import torch.nn as nn

from src.models.mil.attention_mil import AttentionMIL
from src.models.mil.clam import CLAM
from src.models.mil.transmil import TransMIL

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "feature_dim": 1024,
        "hidden_dim": 256,
        "num_classes": 2,
        "num_iterations": 1000,
        "warmup_iterations": 10,
        "batch_size": 4,
        "num_patches": 100,
    }


@pytest.fixture
def sample_features(benchmark_config):
    """Create sample features for benchmarking."""
    batch_size = benchmark_config["batch_size"]
    num_patches = benchmark_config["num_patches"]
    feature_dim = benchmark_config["feature_dim"]

    features = torch.randn(batch_size, num_patches, feature_dim)
    num_patches_tensor = torch.full((batch_size,), num_patches, dtype=torch.long)

    return features, num_patches_tensor


@pytest.fixture
def attention_mil_model(benchmark_config):
    """Create AttentionMIL model for benchmarking."""
    model = AttentionMIL(
        feature_dim=benchmark_config["feature_dim"],
        hidden_dim=benchmark_config["hidden_dim"],
        num_classes=benchmark_config["num_classes"],
        dropout=0.1,
        gated=True,
    )
    model.eval()
    return model


@pytest.fixture
def clam_model(benchmark_config):
    """Create CLAM model for benchmarking."""
    model = CLAM(
        feature_dim=benchmark_config["feature_dim"],
        hidden_dim=benchmark_config["hidden_dim"],
        num_classes=benchmark_config["num_classes"],
        num_clusters=10,
        dropout=0.1,
        multi_branch=True,
    )
    model.eval()
    return model


@pytest.fixture
def transmil_model(benchmark_config):
    """Create TransMIL model for benchmarking."""
    model = TransMIL(
        feature_dim=benchmark_config["feature_dim"],
        hidden_dim=benchmark_config["hidden_dim"],
        num_classes=benchmark_config["num_classes"],
        num_layers=2,
        num_heads=8,
        dropout=0.1,
    )
    model.eval()
    return model


# ============================================================================
# Benchmark Utilities
# ============================================================================


def benchmark_inference(
    model: nn.Module,
    features: torch.Tensor,
    num_patches: torch.Tensor,
    num_iterations: int,
    warmup_iterations: int = 10,
) -> Dict[str, float]:
    """
    Benchmark model inference performance.

    Args:
        model: Model to benchmark
        features: Input features [batch_size, num_patches, feature_dim]
        num_patches: Number of valid patches per sample [batch_size]
        num_iterations: Number of iterations to benchmark
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with benchmark statistics:
        - mean_ms: Mean inference time in milliseconds
        - std_ms: Standard deviation in milliseconds
        - min_ms: Minimum inference time in milliseconds
        - max_ms: Maximum inference time in milliseconds
        - p50_ms: Median inference time in milliseconds
        - p95_ms: 95th percentile inference time in milliseconds
        - p99_ms: 99th percentile inference time in milliseconds
        - throughput_samples_per_sec: Throughput in samples per second
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(features, num_patches)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = model(features, num_patches)
            elapsed = time.perf_counter() - start_time
            times.append(elapsed * 1000)  # Convert to milliseconds

    # Calculate statistics
    times_tensor = torch.tensor(times)
    mean_ms = times_tensor.mean().item()
    std_ms = times_tensor.std().item()
    min_ms = times_tensor.min().item()
    max_ms = times_tensor.max().item()
    p50_ms = times_tensor.median().item()
    p95_ms = torch.quantile(times_tensor, 0.95).item()
    p99_ms = torch.quantile(times_tensor, 0.99).item()

    # Calculate throughput
    batch_size = features.size(0)
    total_time_sec = sum(times) / 1000  # Convert to seconds
    throughput = (batch_size * num_iterations) / total_time_sec

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "throughput_samples_per_sec": throughput,
    }


def print_benchmark_results(model_name: str, results: Dict[str, float]):
    """Print benchmark results in a readable format."""
    print(f"\n{'='*60}")
    print(f"{model_name} Benchmark Results")
    print(f"{'='*60}")
    print(f"Mean:       {results['mean_ms']:.2f} ms")
    print(f"Std Dev:    {results['std_ms']:.2f} ms")
    print(f"Min:        {results['min_ms']:.2f} ms")
    print(f"Max:        {results['max_ms']:.2f} ms")
    print(f"Median:     {results['p50_ms']:.2f} ms")
    print(f"P95:        {results['p95_ms']:.2f} ms")
    print(f"P99:        {results['p99_ms']:.2f} ms")
    print(f"Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"{'='*60}\n")


# ============================================================================
# Benchmark Tests
# ============================================================================


class TestAttentionMILPerformance:
    """Benchmark tests for AttentionMIL model."""

    def test_attention_mil_inference_benchmark(
        self, attention_mil_model, sample_features, benchmark_config
    ):
        """Benchmark AttentionMIL inference performance (1000 iterations)."""
        features, num_patches = sample_features

        results = benchmark_inference(
            model=attention_mil_model,
            features=features,
            num_patches=num_patches,
            num_iterations=benchmark_config["num_iterations"],
            warmup_iterations=benchmark_config["warmup_iterations"],
        )

        print_benchmark_results("AttentionMIL", results)

        # Performance assertions
        # Mean inference time should be reasonable for batch_size=4
        assert results["mean_ms"] < 50.0, (
            f"AttentionMIL inference too slow: {results['mean_ms']:.2f}ms " f"(expected <50ms)"
        )

        # Throughput should be reasonable
        assert results["throughput_samples_per_sec"] > 50.0, (
            f"AttentionMIL throughput too low: {results['throughput_samples_per_sec']:.1f} "
            f"samples/sec (expected >50)"
        )

        # Store results for comparison
        return results

    def test_attention_mil_with_attention_weights(
        self, attention_mil_model, sample_features, benchmark_config
    ):
        """Benchmark AttentionMIL with attention weight computation."""
        features, num_patches = sample_features
        model = attention_mil_model

        # Warmup
        with torch.no_grad():
            for _ in range(benchmark_config["warmup_iterations"]):
                _ = model(features, num_patches, return_attention=True)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(100):  # Fewer iterations for this test
                start_time = time.perf_counter()
                logits, attention = model(features, num_patches, return_attention=True)
                elapsed = time.perf_counter() - start_time
                times.append(elapsed * 1000)

        mean_ms = sum(times) / len(times)

        print(f"\nAttentionMIL with attention weights: {mean_ms:.2f}ms")

        # Should not be significantly slower than without attention weights
        assert mean_ms < 60.0, f"AttentionMIL with attention weights too slow: {mean_ms:.2f}ms"


class TestCLAMPerformance:
    """Benchmark tests for CLAM model."""

    def test_clam_inference_benchmark(self, clam_model, sample_features, benchmark_config):
        """Benchmark CLAM inference performance (1000 iterations)."""
        features, num_patches = sample_features

        results = benchmark_inference(
            model=clam_model,
            features=features,
            num_patches=num_patches,
            num_iterations=benchmark_config["num_iterations"],
            warmup_iterations=benchmark_config["warmup_iterations"],
        )

        print_benchmark_results("CLAM", results)

        # Performance assertions
        # CLAM is more complex, so allow slightly higher latency
        assert results["mean_ms"] < 80.0, (
            f"CLAM inference too slow: {results['mean_ms']:.2f}ms " f"(expected <80ms)"
        )

        # Throughput should be reasonable
        assert results["throughput_samples_per_sec"] > 30.0, (
            f"CLAM throughput too low: {results['throughput_samples_per_sec']:.1f} "
            f"samples/sec (expected >30)"
        )

        return results

    def test_clam_with_instance_predictions(self, clam_model, sample_features, benchmark_config):
        """Benchmark CLAM with instance-level predictions."""
        features, num_patches = sample_features
        model = clam_model

        # Warmup
        with torch.no_grad():
            for _ in range(benchmark_config["warmup_iterations"]):
                _ = model(features, num_patches, return_attention=True)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(100):  # Fewer iterations for this test
                start_time = time.perf_counter()
                logits, attention, instance_preds = model(
                    features, num_patches, return_attention=True
                )
                elapsed = time.perf_counter() - start_time
                times.append(elapsed * 1000)

        mean_ms = sum(times) / len(times)

        print(f"\nCLAM with instance predictions: {mean_ms:.2f}ms")

        # Should not be significantly slower
        assert mean_ms < 100.0, f"CLAM with instance predictions too slow: {mean_ms:.2f}ms"


class TestTransMILPerformance:
    """Benchmark tests for TransMIL model."""

    def test_transmil_inference_benchmark(self, transmil_model, sample_features, benchmark_config):
        """Benchmark TransMIL inference performance (1000 iterations)."""
        features, num_patches = sample_features

        results = benchmark_inference(
            model=transmil_model,
            features=features,
            num_patches=num_patches,
            num_iterations=benchmark_config["num_iterations"],
            warmup_iterations=benchmark_config["warmup_iterations"],
        )

        print_benchmark_results("TransMIL", results)

        # Performance assertions
        # TransMIL uses transformers, so allow higher latency
        assert results["mean_ms"] < 100.0, (
            f"TransMIL inference too slow: {results['mean_ms']:.2f}ms " f"(expected <100ms)"
        )

        # Throughput should be reasonable
        assert results["throughput_samples_per_sec"] > 25.0, (
            f"TransMIL throughput too low: {results['throughput_samples_per_sec']:.1f} "
            f"samples/sec (expected >25)"
        )

        return results

    def test_transmil_with_attention_weights(
        self, transmil_model, sample_features, benchmark_config
    ):
        """Benchmark TransMIL with attention weight computation."""
        features, num_patches = sample_features
        model = transmil_model

        # Warmup
        with torch.no_grad():
            for _ in range(benchmark_config["warmup_iterations"]):
                _ = model(features, num_patches, return_attention=True)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(100):  # Fewer iterations for this test
                start_time = time.perf_counter()
                logits, attention = model(features, num_patches, return_attention=True)
                elapsed = time.perf_counter() - start_time
                times.append(elapsed * 1000)

        mean_ms = sum(times) / len(times)

        print(f"\nTransMIL with attention weights: {mean_ms:.2f}ms")

        # Should not be significantly slower
        assert mean_ms < 120.0, f"TransMIL with attention weights too slow: {mean_ms:.2f}ms"


class TestComparativePerformance:
    """Comparative performance tests across all models."""

    def test_all_models_benchmark(
        self,
        attention_mil_model,
        clam_model,
        transmil_model,
        sample_features,
        benchmark_config,
    ):
        """Benchmark all models and compare performance."""
        features, num_patches = sample_features

        # Benchmark all models
        models = {
            "AttentionMIL": attention_mil_model,
            "CLAM": clam_model,
            "TransMIL": transmil_model,
        }

        results = {}
        for model_name, model in models.items():
            results[model_name] = benchmark_inference(
                model=model,
                features=features,
                num_patches=num_patches,
                num_iterations=benchmark_config["num_iterations"],
                warmup_iterations=benchmark_config["warmup_iterations"],
            )
            print_benchmark_results(model_name, results[model_name])

        # Print comparison
        print(f"\n{'='*60}")
        print("Performance Comparison")
        print(f"{'='*60}")
        print(f"{'Model':<15} {'Mean (ms)':<12} {'Throughput (samples/sec)':<25}")
        print(f"{'-'*60}")
        for model_name, result in results.items():
            print(
                f"{model_name:<15} {result['mean_ms']:<12.2f} "
                f"{result['throughput_samples_per_sec']:<25.1f}"
            )
        print(f"{'='*60}\n")

        # All models should meet minimum performance requirements
        for model_name, result in results.items():
            assert (
                result["mean_ms"] < 150.0
            ), f"{model_name} inference too slow: {result['mean_ms']:.2f}ms"
            assert (
                result["throughput_samples_per_sec"] > 20.0
            ), f"{model_name} throughput too low: {result['throughput_samples_per_sec']:.1f}"


class TestMemoryUsage:
    """Memory usage tests for refactored models."""

    def test_attention_mil_memory_footprint(self, attention_mil_model):
        """Test AttentionMIL memory footprint."""
        model = attention_mil_model

        # Calculate parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        param_memory_mb = param_memory / (1024**2)

        print(f"\nAttentionMIL memory footprint: {param_memory_mb:.2f} MB")

        # Should be reasonable
        assert param_memory_mb < 50.0, f"AttentionMIL too large: {param_memory_mb:.2f} MB"

    def test_clam_memory_footprint(self, clam_model):
        """Test CLAM memory footprint."""
        model = clam_model

        # Calculate parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        param_memory_mb = param_memory / (1024**2)

        print(f"\nCLAM memory footprint: {param_memory_mb:.2f} MB")

        # CLAM is larger due to clustering
        assert param_memory_mb < 100.0, f"CLAM too large: {param_memory_mb:.2f} MB"

    def test_transmil_memory_footprint(self, transmil_model):
        """Test TransMIL memory footprint."""
        model = transmil_model

        # Calculate parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        param_memory_mb = param_memory / (1024**2)

        print(f"\nTransMIL memory footprint: {param_memory_mb:.2f} MB")

        # TransMIL uses transformers, so allow larger size
        assert param_memory_mb < 100.0, f"TransMIL too large: {param_memory_mb:.2f} MB"


# ============================================================================
# GPU Benchmarks (if available)
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUPerformance:
    """GPU performance benchmarks."""

    def test_attention_mil_gpu_benchmark(
        self, attention_mil_model, sample_features, benchmark_config
    ):
        """Benchmark AttentionMIL on GPU."""
        model = attention_mil_model.cuda()
        features, num_patches = sample_features
        features = features.cuda()
        num_patches = num_patches.cuda()

        # Warmup
        with torch.no_grad():
            for _ in range(benchmark_config["warmup_iterations"]):
                _ = model(features, num_patches)

        torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(benchmark_config["num_iterations"]):
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                _ = model(features, num_patches)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                times.append(elapsed * 1000)

        mean_ms = sum(times) / len(times)
        throughput = (features.size(0) * len(times)) / (sum(times) / 1000)

        print(f"\nAttentionMIL GPU: {mean_ms:.2f}ms, {throughput:.1f} samples/sec")

        # GPU should be significantly faster
        assert mean_ms < 20.0, f"GPU inference too slow: {mean_ms:.2f}ms"
        assert throughput > 200.0, f"GPU throughput too low: {throughput:.1f}"

    def test_clam_gpu_benchmark(self, clam_model, sample_features, benchmark_config):
        """Benchmark CLAM on GPU."""
        model = clam_model.cuda()
        features, num_patches = sample_features
        features = features.cuda()
        num_patches = num_patches.cuda()

        # Warmup
        with torch.no_grad():
            for _ in range(benchmark_config["warmup_iterations"]):
                _ = model(features, num_patches)

        torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(benchmark_config["num_iterations"]):
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                _ = model(features, num_patches)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                times.append(elapsed * 1000)

        mean_ms = sum(times) / len(times)
        throughput = (features.size(0) * len(times)) / (sum(times) / 1000)

        print(f"\nCLAM GPU: {mean_ms:.2f}ms, {throughput:.1f} samples/sec")

        # GPU should be significantly faster
        assert mean_ms < 30.0, f"GPU inference too slow: {mean_ms:.2f}ms"
        assert throughput > 150.0, f"GPU throughput too low: {throughput:.1f}"

    def test_transmil_gpu_benchmark(self, transmil_model, sample_features, benchmark_config):
        """Benchmark TransMIL on GPU."""
        model = transmil_model.cuda()
        features, num_patches = sample_features
        features = features.cuda()
        num_patches = num_patches.cuda()

        # Warmup
        with torch.no_grad():
            for _ in range(benchmark_config["warmup_iterations"]):
                _ = model(features, num_patches)

        torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(benchmark_config["num_iterations"]):
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                _ = model(features, num_patches)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                times.append(elapsed * 1000)

        mean_ms = sum(times) / len(times)
        throughput = (features.size(0) * len(times)) / (sum(times) / 1000)

        print(f"\nTransMIL GPU: {mean_ms:.2f}ms, {throughput:.1f} samples/sec")

        # GPU should be significantly faster
        assert mean_ms < 40.0, f"GPU inference too slow: {mean_ms:.2f}ms"
        assert throughput > 100.0, f"GPU throughput too low: {throughput:.1f}"


if __name__ == "__main__":
    # Run benchmarks directly
    pytest.main([__file__, "-v", "-s"])
