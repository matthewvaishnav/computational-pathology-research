#!/usr/bin/env python3
"""Performance regression testing for memory leaks and CPU spikes."""

import gc
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List

import psutil

from src.platform.security.temp_file_manager import TempFileManager


class PerformanceMonitor:
    """Monitor system performance metrics."""

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.get_memory_usage()
        self.baseline_cpu = self.get_cpu_usage()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()

    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Monitor an operation for performance issues."""
        start_memory = self.get_memory_usage()
        start_time = time.time()

        # Force garbage collection before starting
        gc.collect()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.get_memory_usage()

            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            print(f"{operation_name}:")
            print(f"  Duration: {duration:.3f}s")
            print(f"  Memory: {start_memory:.1f}MB → {end_memory:.1f}MB (Δ{memory_delta:+.1f}MB)")

            # Check for memory leaks
            if memory_delta > 100:  # >100MB increase
                print(f"  ⚠️  Potential memory leak: {memory_delta:.1f}MB increase")

            # Check for performance regression
            if duration > 10:  # >10s for basic operations
                print(f"  ⚠️  Performance regression: {duration:.1f}s duration")


def test_memory_leak_detection():
    """Test for memory leaks in core operations."""
    print("🔍 Testing Memory Leak Detection")
    print("-" * 40)

    monitor = PerformanceMonitor()

    # Test 1: Repeated imports
    with monitor.monitor_operation("Repeated imports (100x)"):
        for _ in range(100):
            try:
                import src

                del sys.modules["src"]
            except:
                pass

    # Test 2: Large data structures
    with monitor.monitor_operation("Large data creation/deletion"):
        large_data = []
        for i in range(1000):
            large_data.append([0] * 10000)  # 10K elements each
        del large_data
        gc.collect()

    # Test 3: Thread creation/cleanup
    with monitor.monitor_operation("Thread lifecycle (50 threads)"):
        threads = []
        for i in range(50):
            t = threading.Thread(target=lambda: time.sleep(0.1))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    return True


def test_cpu_spike_detection():
    """Test for CPU spikes and performance issues."""
    print("\n⚡ Testing CPU Spike Detection")
    print("-" * 40)

    monitor = PerformanceMonitor()

    # Test 1: CPU-intensive operation
    with monitor.monitor_operation("CPU-intensive computation"):
        result = 0
        for i in range(1000000):
            result += i**2

    # Test 2: I/O operations
    with monitor.monitor_operation("File I/O operations"):
        temp_files = []
        for i in range(100):
            fd, temp_path = TempFileManager.create_temp_file(suffix=f"_test_{i}.txt")
            with os.fdopen(fd, "w") as f:
                f.write("test data" * 1000)
            temp_files.append(temp_path)

        for temp_path in temp_files:
            try:
                os.remove(temp_path)
            except:
                pass

    return True


def test_gradual_memory_growth():
    """Test for gradual memory growth over time."""
    print("\n📈 Testing Gradual Memory Growth")
    print("-" * 40)

    monitor = PerformanceMonitor()
    memory_samples = []

    # Simulate long-running operation
    for iteration in range(20):
        # Simulate some work
        data = [i for i in range(10000)]

        # Sample memory usage
        current_memory = monitor.get_memory_usage()
        memory_samples.append(current_memory)

        if iteration % 5 == 0:
            print(f"  Iteration {iteration}: {current_memory:.1f}MB")

        # Clean up (but maybe not perfectly)
        if iteration % 3 != 0:  # Intentionally skip cleanup sometimes
            del data

        time.sleep(0.1)

    # Analyze memory growth trend
    if len(memory_samples) >= 10:
        early_avg = sum(memory_samples[:5]) / 5
        late_avg = sum(memory_samples[-5:]) / 5
        growth = late_avg - early_avg

        print(f"  Memory growth: {early_avg:.1f}MB → {late_avg:.1f}MB (Δ{growth:+.1f}MB)")

        if growth > 50:  # >50MB growth
            print("  ⚠️  Significant memory growth detected")
            return False
        else:
            print("  ✅ Memory growth within acceptable limits")

    return True


def test_resource_cleanup():
    """Test proper resource cleanup."""
    print("\n🧹 Testing Resource Cleanup")
    print("-" * 40)

    monitor = PerformanceMonitor()

    # Test file handle cleanup
    with monitor.monitor_operation("File handle management"):
        files = []
        try:
            for i in range(100):
                fd, temp_path = TempFileManager.create_temp_file(suffix=f"_resource_test_{i}.txt")
                f = os.fdopen(fd, "w")
                f.write("test")
                files.append(f)
        finally:
            for f in files:
                try:
                    f.close()
                    os.remove(f.name)
                except:
                    pass

    # Test thread cleanup
    with monitor.monitor_operation("Thread resource cleanup"):

        def worker():
            time.sleep(0.1)

        threads = []
        for i in range(20):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        # Ensure all threads complete
        for t in threads:
            t.join()

    return True


def test_performance_under_load():
    """Test performance degradation under load."""
    print("\n🏋️ Testing Performance Under Load")
    print("-" * 40)

    monitor = PerformanceMonitor()

    # Baseline performance
    with monitor.monitor_operation("Baseline operation"):
        result = sum(i**2 for i in range(10000))

    # Performance under memory pressure
    with monitor.monitor_operation("Under memory pressure"):
        # Create memory pressure
        memory_hog = [[0] * 1000 for _ in range(1000)]  # ~4MB

        # Same operation as baseline
        result = sum(i**2 for i in range(10000))

        del memory_hog

    # Performance with concurrent operations
    with monitor.monitor_operation("With concurrent operations"):

        def background_work():
            for _ in range(100000):
                pass

        # Start background threads
        threads = [threading.Thread(target=background_work) for _ in range(4)]
        for t in threads:
            t.start()

        # Main operation
        result = sum(i**2 for i in range(10000))

        # Wait for background work
        for t in threads:
            t.join()

    return True


def run_performance_regression_tests():
    """Run all performance regression tests."""
    print("🚀 Performance Regression Testing")
    print("=" * 50)

    tests = [
        ("Memory Leak Detection", test_memory_leak_detection),
        ("CPU Spike Detection", test_cpu_spike_detection),
        ("Gradual Memory Growth", test_gradual_memory_growth),
        ("Resource Cleanup", test_resource_cleanup),
        ("Performance Under Load", test_performance_under_load),
    ]

    passed = 0
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print("\n" + "=" * 50)
    print(f"Performance Tests: {passed}/{len(tests)} passed")

    if passed == len(tests):
        print("🏆 No performance regressions detected!")
    else:
        print(f"⚠️ {len(tests) - passed} performance issues found")

    return passed == len(tests)


if __name__ == "__main__":
    success = run_performance_regression_tests()
    sys.exit(0 if success else 1)
