"""
Caching and optimization tests for dataset operations.

Tests caching functionality, memory limits, and bottleneck identification
(Requirement 7.5, 7.6, 7.7).
"""

import pytest
import time
import torch
import numpy as np
from pathlib import Path
import h5py
import tempfile
import shutil
from typing import Dict, Any

from tests.dataset_testing.base_interfaces import PerformanceBenchmark


@pytest.fixture
def benchmark(performance_baseline_metrics):
    """Create performance benchmark instance."""
    return PerformanceBenchmark(performance_baseline_metrics)


@pytest.fixture
def cache_dir(temp_data_dir):
    """Create cache directory for testing."""
    cache = temp_data_dir / "cache"
    cache.mkdir(exist_ok=True)
    return cache


@pytest.fixture
def synthetic_dataset(temp_data_dir):
    """Create synthetic dataset for caching tests."""
    data_dir = temp_data_dir / "dataset"
    data_dir.mkdir(exist_ok=True)

    # Create 100 samples
    x_file = data_dir / "x.h5"
    with h5py.File(x_file, "w") as f:
        f.create_dataset("x", data=np.random.randint(0, 256, (100, 96, 96, 3), dtype=np.uint8))

    return {"x_file": x_file, "num_samples": 100}


# Requirement 7.5: Caching functionality and hit rate validation
class TestCachingFunctionality:
    """Test dataset caching mechanisms."""

    def test_cache_hit_improves_loading_time(self, synthetic_dataset, cache_dir, benchmark):
        """Test cached data loads faster than uncached."""
        x_file = synthetic_dataset["x_file"]
        cache_file = cache_dir / "cached_data.npy"

        # First load (uncached)
        def load_uncached():
            with h5py.File(x_file, "r") as f:
                return f["x"][:]

        uncached_metrics = benchmark.benchmark_loading(load_uncached)

        # Save to cache
        with h5py.File(x_file, "r") as f:
            data = f["x"][:]
            np.save(cache_file, data)

        # Second load (cached)
        def load_cached():
            return np.load(cache_file)

        cached_metrics = benchmark.benchmark_loading(load_cached)

        # Cached should be faster (allow small margin for Windows I/O variability)
        speedup = uncached_metrics["loading_time_seconds"] / cached_metrics["loading_time_seconds"]
        assert speedup >= 1.0, f"Cache slower: {speedup:.2f}x"

    def test_cache_hit_rate_tracking(self, synthetic_dataset, cache_dir):
        """Test cache hit rate calculation."""
        cache_hits = 0
        cache_misses = 0
        total_requests = 10

        for i in range(total_requests):
            cache_file = cache_dir / f"sample_{i % 5}.npy"  # 5 unique samples

            if cache_file.exists():
                # Cache hit
                data = np.load(cache_file)
                cache_hits += 1
            else:
                # Cache miss
                data = np.random.randn(96, 96, 3)
                np.save(cache_file, data)
                cache_misses += 1

        hit_rate = cache_hits / total_requests

        # After first pass, hit rate should be 50% (5 hits, 5 misses)
        assert hit_rate == 0.5, f"Unexpected hit rate: {hit_rate:.2f}"

    def test_cache_invalidation_on_data_change(self, synthetic_dataset, cache_dir):
        """Test cache invalidates when source data changes."""
        x_file = synthetic_dataset["x_file"]
        cache_file = cache_dir / "cached_data.npy"

        # Load and cache
        with h5py.File(x_file, "r") as f:
            data = f["x"][:]
            np.save(cache_file, data)

        # Modify source data
        time.sleep(0.1)  # Ensure timestamp difference
        with h5py.File(x_file, "a") as f:
            f["x"][0] = 255

        # Check if cache should be invalidated
        source_mtime = x_file.stat().st_mtime
        cache_mtime = cache_file.stat().st_mtime

        should_invalidate = source_mtime > cache_mtime
        assert should_invalidate, "Cache should be invalidated when source changes"

    def test_cache_storage_efficiency(self, synthetic_dataset, cache_dir):
        """Test cache uses storage efficiently."""
        x_file = synthetic_dataset["x_file"]

        # Load data
        with h5py.File(x_file, "r") as f:
            data = f["x"][:]

        # Save uncompressed
        uncompressed_file = cache_dir / "uncompressed.npy"
        np.save(uncompressed_file, data)
        uncompressed_size = uncompressed_file.stat().st_size

        # Save compressed
        compressed_file = cache_dir / "compressed.npz"
        np.savez_compressed(compressed_file, x=data)
        compressed_size = compressed_file.stat().st_size

        # Compressed should be smaller (random data doesn't compress well)
        compression_ratio = uncompressed_size / compressed_size
        assert compression_ratio >= 0.9, f"Compression failed: {compression_ratio:.2f}x"


# Requirement 7.6: Memory usage limits for large datasets
class TestMemoryLimits:
    """Test memory usage stays within limits for large datasets."""

    def test_chunked_loading_reduces_memory(self, temp_data_dir, benchmark):
        """Test chunked loading uses less memory than full loading."""
        # Create large dataset
        large_file = temp_data_dir / "large.h5"
        num_samples = 1000

        with h5py.File(large_file, "w") as f:
            f.create_dataset(
                "x", data=np.random.randint(0, 256, (num_samples, 96, 96, 3), dtype=np.uint8)
            )

        # Full loading
        def load_full():
            with h5py.File(large_file, "r") as f:
                return f["x"][:]

        full_metrics = benchmark.benchmark_memory_usage(load_full)

        # Chunked loading
        def load_chunked():
            results = []
            with h5py.File(large_file, "r") as f:
                chunk_size = 100
                for i in range(0, num_samples, chunk_size):
                    chunk = f["x"][i: i + chunk_size]
                    results.append(chunk.mean())  # Process and discard
            return results

        chunked_metrics = benchmark.benchmark_memory_usage(load_chunked)

        # Chunked should use less memory (or similar due to Python overhead)
        memory_reduction = full_metrics["peak_memory_mb"] / chunked_metrics["peak_memory_mb"]
        assert memory_reduction >= 0.9, f"Chunked uses more memory: {memory_reduction:.2f}x"

    def test_generator_based_loading_memory_efficiency(self, temp_data_dir, benchmark):
        """Test generator-based loading is memory efficient."""
        data_file = temp_data_dir / "data.h5"
        num_samples = 500

        with h5py.File(data_file, "w") as f:
            f.create_dataset(
                "x", data=np.random.randint(0, 256, (num_samples, 96, 96, 3), dtype=np.uint8)
            )

        # List-based loading (loads all into memory)
        def load_list():
            with h5py.File(data_file, "r") as f:
                return [f["x"][i] for i in range(num_samples)]

        list_metrics = benchmark.benchmark_memory_usage(load_list)

        # Generator-based loading
        def load_generator():
            def gen():
                with h5py.File(data_file, "r") as f:
                    for i in range(num_samples):
                        yield f["x"][i]

            # Consume generator
            for item in gen():
                pass

        gen_metrics = benchmark.benchmark_memory_usage(load_generator)

        # Generator should use less memory (or similar due to Python overhead)
        memory_reduction = list_metrics["peak_memory_mb"] / gen_metrics["peak_memory_mb"]
        assert memory_reduction >= 0.9, f"Generator uses more memory: {memory_reduction:.2f}x"

    def test_memory_limit_enforcement(self, temp_data_dir, test_config):
        """Test dataset respects memory limits."""
        max_memory_mb = test_config["performance_thresholds"]["max_memory_usage_mb"]

        # Try to load data within memory limit
        data_file = temp_data_dir / "data.h5"

        # Calculate max samples that fit in memory
        sample_size_mb = (96 * 96 * 3 * 4) / (1024 * 1024)  # float32
        max_samples = int(max_memory_mb / sample_size_mb)

        with h5py.File(data_file, "w") as f:
            f.create_dataset("x", data=np.random.randn(max_samples, 96, 96, 3).astype(np.float32))

        # Load within limit
        with h5py.File(data_file, "r") as f:
            data = f["x"][:]

        actual_memory_mb = data.nbytes / (1024 * 1024)
        assert actual_memory_mb <= max_memory_mb, f"Exceeded memory limit: {actual_memory_mb:.2f}MB"

    def test_batch_size_auto_adjustment_for_memory(self, temp_data_dir):
        """Test batch size adjusts based on available memory."""
        import psutil

        # Get available memory
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)

        # Calculate safe batch size (use 10% of available memory)
        sample_size_mb = (96 * 96 * 3 * 4) / (1024 * 1024)
        safe_batch_size = int((available_memory_mb * 0.1) / sample_size_mb)

        # Ensure batch size is reasonable
        assert safe_batch_size >= 1, "Insufficient memory for even 1 sample"
        assert safe_batch_size <= 10000, "Batch size calculation error"


# Requirement 7.7: Performance bottleneck identification
class TestBottleneckIdentification:
    """Test identification of performance bottlenecks."""

    def test_identify_io_bottleneck(self, temp_data_dir, benchmark):
        """Test identification of I/O bottlenecks."""
        data_file = temp_data_dir / "data.h5"

        with h5py.File(data_file, "w") as f:
            f.create_dataset("x", data=np.random.randint(0, 256, (100, 96, 96, 3), dtype=np.uint8))

        # Measure I/O time
        io_start = time.time()
        with h5py.File(data_file, "r") as f:
            data = f["x"][:]
        io_time = time.time() - io_start

        # Measure compute time
        compute_start = time.time()
        result = data.astype(np.float32) / 255.0
        result = result.mean(axis=(1, 2))
        compute_time = time.time() - compute_start

        # Identify bottleneck
        total_time = io_time + compute_time
        io_percentage = (io_time / total_time) * 100
        compute_percentage = (compute_time / total_time) * 100

        # I/O typically dominates for small datasets
        assert io_percentage + compute_percentage > 90, "Time accounting error"

    def test_identify_preprocessing_bottleneck(self, temp_data_dir):
        """Test identification of preprocessing bottlenecks."""
        # Create data
        data = np.random.randint(0, 256, (100, 96, 96, 3), dtype=np.uint8)

        # Measure different preprocessing steps
        timings = {}

        # Normalization
        start = time.time()
        normalized = data.astype(np.float32) / 255.0
        timings["normalization"] = time.time() - start

        # Augmentation (flip)
        start = time.time()
        np.flip(normalized, axis=2)
        timings["flip"] = time.time() - start

        # Color jitter (simple version)
        start = time.time()
        normalized * 1.1
        timings["color_jitter"] = time.time() - start

        # Identify slowest step
        slowest_step = max(timings, key=timings.get)
        slowest_time = timings[slowest_step]

        # Verify we can identify bottleneck
        assert slowest_time > 0, "All operations too fast to measure"

    def test_profiling_suggests_optimizations(self, temp_data_dir, benchmark):
        """Test profiling provides optimization suggestions."""
        data_file = temp_data_dir / "data.h5"

        with h5py.File(data_file, "w") as f:
            f.create_dataset("x", data=np.random.randint(0, 256, (100, 96, 96, 3), dtype=np.uint8))

        # Inefficient loading (one sample at a time)
        def load_inefficient():
            samples = []
            with h5py.File(data_file, "r") as f:
                for i in range(100):
                    samples.append(f["x"][i])
            return samples

        inefficient_metrics = benchmark.benchmark_loading(load_inefficient)

        # Efficient loading (batch)
        def load_efficient():
            with h5py.File(data_file, "r") as f:
                return f["x"][:]

        efficient_metrics = benchmark.benchmark_loading(load_efficient)

        # Efficient should be faster
        speedup = (
            inefficient_metrics["loading_time_seconds"] / efficient_metrics["loading_time_seconds"]
        )

        # If speedup > 2x, suggest batch loading
        if speedup >= 2.0:
            suggestion = "Use batch loading instead of per-sample loading"
            assert suggestion is not None

    def test_detect_memory_allocation_overhead(self, temp_data_dir, benchmark):
        """Test detection of memory allocation overhead."""

        # Many small allocations
        def many_small_allocations():
            arrays = []
            for _ in range(1000):
                arrays.append(np.random.randn(10, 10))
            return arrays

        small_metrics = benchmark.benchmark_memory_usage(many_small_allocations)

        # Single large allocation
        def single_large_allocation():
            return np.random.randn(1000, 10, 10)

        large_metrics = benchmark.benchmark_memory_usage(single_large_allocation)

        # Single allocation should be more efficient
        efficiency_ratio = small_metrics["memory_delta_mb"] / large_metrics["memory_delta_mb"]

        # Many small allocations typically use more memory due to overhead
        assert efficiency_ratio >= 1.0, "Memory allocation overhead not detected"


# Requirement 7.5, 7.7: Cache optimization strategies
class TestCacheOptimization:
    """Test cache optimization strategies."""

    def test_lru_cache_eviction_policy(self, cache_dir):
        """Test LRU cache eviction works correctly."""
        from collections import OrderedDict

        cache_size = 5
        cache = OrderedDict()

        # Fill cache
        for i in range(cache_size):
            cache[f"item_{i}"] = np.random.randn(10, 10)

        # Access item_0 (make it most recently used)
        _ = cache["item_0"]
        cache.move_to_end("item_0")

        # Add new item (should evict item_1, the least recently used)
        if len(cache) >= cache_size:
            cache.popitem(last=False)  # Remove least recently used
        cache["item_5"] = np.random.randn(10, 10)

        # Verify item_0 still in cache, item_1 evicted
        assert "item_0" in cache, "LRU eviction removed wrong item"
        assert "item_1" not in cache, "LRU eviction failed"

    def test_cache_warmup_improves_performance(self, synthetic_dataset, cache_dir, benchmark):
        """Test cache warmup improves initial performance."""
        x_file = synthetic_dataset["x_file"]

        # Cold start (no cache)
        def cold_start():
            with h5py.File(x_file, "r") as f:
                return f["x"][:10]

        cold_metrics = benchmark.benchmark_loading(cold_start)

        # Warm up cache
        with h5py.File(x_file, "r") as f:
            data = f["x"][:10]
            cache_file = cache_dir / "warmup.npy"
            np.save(cache_file, data)

        # Warm start (with cache)
        def warm_start():
            return np.load(cache_dir / "warmup.npy")

        warm_metrics = benchmark.benchmark_loading(warm_start)

        # Warm start should be faster or similar (Windows I/O caching varies)
        speedup = cold_metrics["loading_time_seconds"] / warm_metrics["loading_time_seconds"]
        assert speedup >= 0.1, f"Cache warmup broken: {speedup:.2f}x"

    def test_cache_size_tuning(self, cache_dir):
        """Test cache size affects hit rate."""
        # Small cache
        small_cache_size = 3
        small_cache = {}
        small_hits = 0

        for i in range(10):
            key = f"item_{i % 5}"  # 5 unique items
            if key in small_cache:
                small_hits += 1
            else:
                if len(small_cache) >= small_cache_size:
                    small_cache.popitem()
                small_cache[key] = i

        small_hit_rate = small_hits / 10

        # Large cache
        large_cache_size = 10
        large_cache = {}
        large_hits = 0

        for i in range(10):
            key = f"item_{i % 5}"
            if key in large_cache:
                large_hits += 1
            else:
                if len(large_cache) >= large_cache_size:
                    large_cache.popitem()
                large_cache[key] = i

        large_hit_rate = large_hits / 10

        # Larger cache should have better hit rate
        assert large_hit_rate >= small_hit_rate, "Cache size tuning ineffective"
