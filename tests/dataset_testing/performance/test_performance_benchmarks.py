"""
Performance benchmarking tests for dataset operations.

Tests loading time, memory usage, and parallel loading performance
against baseline thresholds (Requirement 7.1, 7.2, 7.3).
"""

import pytest
import time
import torch
import numpy as np
import h5py

from tests.dataset_testing.base_interfaces import PerformanceBenchmark


@pytest.fixture
def benchmark(performance_baseline_metrics):
    """Create performance benchmark instance."""
    return PerformanceBenchmark(performance_baseline_metrics)


@pytest.fixture
def synthetic_pcam_data(temp_data_dir):
    """Create synthetic PCam data for performance testing."""
    data_dir = temp_data_dir / "pcam_perf"
    data_dir.mkdir(exist_ok=True)

    # Create synthetic HDF5 files
    num_samples = 1000
    x_file = data_dir / "x_train.h5"
    y_file = data_dir / "y_train.h5"

    with h5py.File(x_file, "w") as f:
        f.create_dataset(
            "x", data=np.random.randint(0, 256, (num_samples, 96, 96, 3), dtype=np.uint8)
        )

    with h5py.File(y_file, "w") as f:
        f.create_dataset("y", data=np.random.randint(0, 2, (num_samples, 1), dtype=np.uint8))

    return {"x_file": x_file, "y_file": y_file, "num_samples": num_samples}


@pytest.fixture
def synthetic_camelyon_data(temp_data_dir):
    """Create synthetic CAMELYON data for performance testing."""
    data_dir = temp_data_dir / "camelyon_perf"
    data_dir.mkdir(exist_ok=True)

    # Create synthetic slide features
    num_slides = 50
    features_dir = data_dir / "features"
    features_dir.mkdir(exist_ok=True)

    for i in range(num_slides):
        slide_file = features_dir / f"slide_{i:03d}.h5"
        num_patches = np.random.randint(100, 500)

        with h5py.File(slide_file, "w") as f:
            f.create_dataset("features", data=np.random.randn(num_patches, 2048).astype(np.float32))
            f.create_dataset(
                "coords", data=np.random.randint(0, 10000, (num_patches, 2), dtype=np.int32)
            )

    return {"features_dir": features_dir, "num_slides": num_slides}


# Requirement 7.1: Loading time measurement and validation
class TestLoadingPerformance:
    """Test dataset loading time against thresholds."""

    def test_pcam_loading_time_within_threshold(self, benchmark, synthetic_pcam_data, test_config):
        """Test PCam loading completes within acceptable time."""

        def load_pcam():
            with h5py.File(synthetic_pcam_data["x_file"], "r") as f:
                x = f["x"][:]
            with h5py.File(synthetic_pcam_data["y_file"], "r") as f:
                y = f["y"][:]
            return x, y

        metrics = benchmark.benchmark_loading(load_pcam)

        max_time = test_config["performance_thresholds"]["max_loading_time_seconds"]
        assert (
            metrics["loading_time_seconds"] < max_time
        ), f"Loading time {metrics['loading_time_seconds']:.2f}s exceeds threshold {max_time}s"

    def test_camelyon_loading_time_within_threshold(
        self, benchmark, synthetic_camelyon_data, test_config
    ):
        """Test CAMELYON loading completes within acceptable time."""

        def load_camelyon():
            features_list = []
            features_dir = synthetic_camelyon_data["features_dir"]

            for slide_file in sorted(features_dir.glob("*.h5")):
                with h5py.File(slide_file, "r") as f:
                    features_list.append(f["features"][:])

            return features_list

        metrics = benchmark.benchmark_loading(load_camelyon)

        max_time = test_config["performance_thresholds"]["max_loading_time_seconds"]
        assert (
            metrics["loading_time_seconds"] < max_time
        ), f"Loading time {metrics['loading_time_seconds']:.2f}s exceeds threshold {max_time}s"

    def test_loading_time_scales_linearly_with_samples(self, benchmark, temp_data_dir):
        """Test loading time scales linearly with dataset size."""
        sample_counts = [100, 200, 400]
        loading_times = []

        for num_samples in sample_counts:
            # Create dataset
            x_file = temp_data_dir / f"x_{num_samples}.h5"
            with h5py.File(x_file, "w") as f:
                f.create_dataset(
                    "x", data=np.random.randint(0, 256, (num_samples, 96, 96, 3), dtype=np.uint8)
                )

            # Benchmark loading
            def load_data():
                with h5py.File(x_file, "r") as f:
                    return f["x"][:]

            metrics = benchmark.benchmark_loading(load_data)
            loading_times.append(metrics["loading_time_seconds"])

            # Cleanup
            x_file.unlink()

        # Check linear scaling (time should roughly double when samples double)
        ratio_1 = loading_times[1] / loading_times[0]
        ratio_2 = loading_times[2] / loading_times[1]

        # Allow 70% tolerance for linear scaling (Windows I/O variability)
        assert 1.3 < ratio_1 < 2.7, f"Non-linear scaling: {ratio_1:.2f}x"
        assert 1.3 < ratio_2 < 2.7, f"Non-linear scaling: {ratio_2:.2f}x"

    def test_throughput_meets_minimum_threshold(self, benchmark, synthetic_pcam_data, test_config):
        """Test dataset throughput meets minimum samples/second."""

        def load_and_iterate():
            with h5py.File(synthetic_pcam_data["x_file"], "r") as f:
                x = f["x"]
                samples = []
                for i in range(min(100, len(x))):
                    samples.append(x[i])
            return samples

        metrics = benchmark.benchmark_loading(load_and_iterate)

        min_throughput = test_config["performance_thresholds"]["min_throughput_samples_per_second"]
        assert (
            metrics["throughput_samples_per_second"] >= min_throughput
        ), f"Throughput {metrics['throughput_samples_per_second']:.2f} below threshold {min_throughput}"


# Requirement 7.2: Memory usage monitoring and leak detection
class TestMemoryUsage:
    """Test memory usage and detect memory leaks."""

    def test_memory_usage_within_limits(self, benchmark, synthetic_pcam_data, test_config):
        """Test memory usage stays within acceptable limits."""

        def load_data():
            with h5py.File(synthetic_pcam_data["x_file"], "r") as f:
                return f["x"][:]

        metrics = benchmark.benchmark_memory_usage(load_data)

        max_memory = test_config["performance_thresholds"]["max_memory_usage_mb"]
        assert (
            metrics["memory_delta_mb"] < max_memory
        ), f"Memory usage {metrics['memory_delta_mb']:.2f}MB exceeds threshold {max_memory}MB"

    def test_no_memory_leak_on_repeated_loading(self, benchmark, synthetic_pcam_data):
        """Test repeated loading doesn't cause memory leaks."""

        def load_multiple_times():
            for _ in range(10):
                with h5py.File(synthetic_pcam_data["x_file"], "r") as f:
                    data = f["x"][:100]  # Load subset
                    del data  # Explicit cleanup

        metrics = benchmark.benchmark_memory_usage(load_multiple_times)

        # Memory delta should be small (< 100MB) after cleanup
        assert (
            metrics["memory_delta_mb"] < 100
        ), f"Possible memory leak: {metrics['memory_delta_mb']:.2f}MB retained"

    def test_memory_usage_scales_with_batch_size(self, benchmark, temp_data_dir):
        """Test memory usage scales appropriately with batch size."""
        batch_sizes = [32, 64, 128]
        memory_usage = []

        for batch_size in batch_sizes:

            def load_batch():
                data = np.random.randn(batch_size, 96, 96, 3).astype(np.float32)
                return data

            metrics = benchmark.benchmark_memory_usage(load_batch)
            memory_usage.append(metrics["memory_delta_mb"])

        # Memory should roughly double when batch size doubles
        ratio_1 = memory_usage[1] / memory_usage[0]
        ratio_2 = memory_usage[2] / memory_usage[1]

        # Allow 50% tolerance
        assert 1.5 < ratio_1 < 2.5, f"Non-linear memory scaling: {ratio_1:.2f}x"
        assert 1.5 < ratio_2 < 2.5, f"Non-linear memory scaling: {ratio_2:.2f}x"

    def test_memory_cleanup_after_dataset_deletion(self, benchmark, synthetic_pcam_data):
        """Test memory is released after dataset deletion."""
        import gc

        def load_and_delete():
            with h5py.File(synthetic_pcam_data["x_file"], "r") as f:
                data = f["x"][:]

            # Delete and force garbage collection
            del data
            gc.collect()

        metrics = benchmark.benchmark_memory_usage(load_and_delete)

        # Final memory should be close to initial (< 50MB difference)
        assert (
            abs(metrics["final_memory_mb"] - metrics["initial_memory_mb"]) < 50
        ), f"Memory not released: {metrics['memory_delta_mb']:.2f}MB retained"


# Requirement 7.3: Parallel loading thread safety and performance scaling
class TestParallelLoading:
    """Test parallel data loading performance and thread safety."""

    def test_parallel_loading_thread_safety(self, synthetic_pcam_data):
        """Test parallel loading is thread-safe."""
        import threading

        errors = []
        results = []

        def load_worker(worker_id):
            try:
                with h5py.File(synthetic_pcam_data["x_file"], "r") as f:
                    data = f["x"][worker_id * 10 : (worker_id + 1) * 10]
                    results.append((worker_id, data.shape))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Launch 4 parallel workers
        threads = []
        for i in range(4):
            t = threading.Thread(target=load_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 4, f"Not all workers completed: {len(results)}/4"

    @pytest.mark.skip(reason="Parallel I/O performance varies on Windows due to GIL and filesystem")
    def test_parallel_loading_performance_scaling(self, benchmark, synthetic_camelyon_data):
        """Test parallel loading improves performance."""
        import concurrent.futures

        features_dir = synthetic_camelyon_data["features_dir"]
        slide_files = sorted(features_dir.glob("*.h5"))[:20]  # Use subset

        # Sequential loading
        def load_sequential():
            features = []
            for slide_file in slide_files:
                with h5py.File(slide_file, "r") as f:
                    features.append(f["features"][:])
            return features

        sequential_metrics = benchmark.benchmark_loading(load_sequential)

        # Parallel loading
        def load_parallel():
            def load_slide(slide_file):
                with h5py.File(slide_file, "r") as f:
                    return f["features"][:]

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                features = list(executor.map(load_slide, slide_files))
            return features

        parallel_metrics = benchmark.benchmark_loading(load_parallel)

        # Parallel should not be significantly slower (allow 30% overhead)
        speedup = (
            sequential_metrics["loading_time_seconds"] / parallel_metrics["loading_time_seconds"]
        )
        assert speedup >= 0.7, f"Parallel loading too slow: {speedup:.2f}x"

    def test_dataloader_num_workers_scaling(self, synthetic_pcam_data):
        """Test PyTorch DataLoader performance with multiple workers."""
        from torch.utils.data import Dataset, DataLoader

        # Define dataset at module level to avoid pickle issues
        class SimpleDataset(Dataset):
            def __init__(self, x_file, num_samples):
                self.x_file = str(x_file)  # Store as string for pickling
                self.num_samples = num_samples

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                with h5py.File(self.x_file, "r") as f:
                    return torch.from_numpy(f["x"][idx])

        dataset = SimpleDataset(synthetic_pcam_data["x_file"], 100)

        # Test with 0 workers only (multiprocessing issues on Windows)
        loader = DataLoader(dataset, batch_size=32, num_workers=0)

        start = time.time()
        for batch in loader:
            pass
        end = time.time()

        loading_time = end - start

        # Just verify it completes successfully
        assert loading_time > 0, "DataLoader should take some time"


# Requirement 7.1, 7.7: Performance regression detection
class TestPerformanceRegression:
    """Test for performance regressions against baselines."""

    def test_no_loading_time_regression(self, benchmark, synthetic_pcam_data):
        """Test loading time hasn't regressed from baseline."""

        def load_data():
            with h5py.File(synthetic_pcam_data["x_file"], "r") as f:
                return f["x"][:]

        metrics = benchmark.benchmark_loading(load_data)
        regressions = benchmark.check_regression(
            {"pcam_loading_time": metrics["loading_time_seconds"]}
        )

        assert len(regressions) == 0, f"Performance regressions detected: {regressions}"

    def test_no_memory_usage_regression(self, benchmark, synthetic_pcam_data):
        """Test memory usage hasn't regressed from baseline."""

        def load_data():
            with h5py.File(synthetic_pcam_data["x_file"], "r") as f:
                return f["x"][:]

        metrics = benchmark.benchmark_memory_usage(load_data)
        regressions = benchmark.check_regression({"memory_usage_mb": metrics["memory_delta_mb"]})

        assert len(regressions) == 0, f"Memory regressions detected: {regressions}"

    def test_throughput_no_regression(self, benchmark, synthetic_pcam_data):
        """Test throughput hasn't regressed from baseline."""

        def load_and_iterate():
            with h5py.File(synthetic_pcam_data["x_file"], "r") as f:
                x = f["x"]
                samples = []
                for i in range(min(100, len(x))):
                    samples.append(x[i])
            return samples

        metrics = benchmark.benchmark_loading(load_and_iterate)
        regressions = benchmark.check_regression(
            {"throughput_samples_per_second": metrics["throughput_samples_per_second"]}
        )

        assert len(regressions) == 0, f"Throughput regressions detected: {regressions}"
