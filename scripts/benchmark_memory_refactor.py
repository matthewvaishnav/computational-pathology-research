"""Benchmark memory module refactoring performance.

Compares performance of refactored memory components to ensure
no significant overhead was introduced.
"""

import sys
from pathlib import Path
import importlib.util

def import_module_from_file(module_name, file_path):
    """Import module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import memory modules directly
src_path = Path(__file__).parent.parent / "src"
memory_path = src_path / "streaming" / "memory"

# Import exceptions first
exceptions = import_module_from_file("exceptions", src_path / "exceptions.py")

# Import memory modules
profiler_mod = import_module_from_file("profiler", memory_path / "profiler.py")
cache_mod = import_module_from_file("cache_manager", memory_path / "cache_manager.py")
optimizer_mod = import_module_from_file("batch_optimizer", memory_path / "batch_optimizer.py")

MemoryProfiler = profiler_mod.MemoryProfiler
CacheManager = cache_mod.CacheManager
BatchOptimizer = optimizer_mod.BatchOptimizer

import time
import torch
import numpy as np

def benchmark_profiler(iterations=1000):
    """Benchmark profiler overhead."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profiler = MemoryProfiler(device=device, memory_limit_gb=8.0)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        snapshot = profiler.take_snapshot()
        times.append(time.perf_counter() - start)
    
    avg_time_ms = np.mean(times) * 1000
    std_time_ms = np.std(times) * 1000
    
    return {
        "avg_ms": avg_time_ms,
        "std_ms": std_time_ms,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000
    }

def benchmark_cache(iterations=10000):
    """Benchmark cache operations."""
    cache = CacheManager(max_size_mb=100.0)
    
    # Benchmark put
    put_times = []
    for i in range(iterations):
        start = time.perf_counter()
        cache.put(f"key_{i}", np.random.rand(100))
        put_times.append(time.perf_counter() - start)
    
    # Benchmark get (hits)
    get_times = []
    for i in range(iterations):
        key = f"key_{i % 1000}"  # Ensure hits
        start = time.perf_counter()
        cache.get(key)
        get_times.append(time.perf_counter() - start)
    
    return {
        "put_avg_us": np.mean(put_times) * 1e6,
        "put_std_us": np.std(put_times) * 1e6,
        "get_avg_us": np.mean(get_times) * 1e6,
        "get_std_us": np.std(get_times) * 1e6
    }

def benchmark_optimizer(iterations=1000):
    """Benchmark batch optimizer."""
    optimizer = BatchOptimizer(
        available_memory_gb=8.0,
        min_batch_size=1,
        max_batch_size=256
    )
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        optimizer.optimize_batch_size(tile_size=224, feature_dim=512)
        times.append(time.perf_counter() - start)
    
    return {
        "avg_us": np.mean(times) * 1e6,
        "std_us": np.std(times) * 1e6
    }

def main():
    print("=" * 60)
    print("Memory Module Refactoring Benchmark")
    print("=" * 60)
    
    # Profiler
    print("\n1. MemoryProfiler Overhead")
    print("-" * 60)
    profiler_stats = benchmark_profiler(iterations=1000)
    print(f"  Average: {profiler_stats['avg_ms']:.3f} ms")
    print(f"  Std Dev: {profiler_stats['std_ms']:.3f} ms")
    print(f"  Min:     {profiler_stats['min_ms']:.3f} ms")
    print(f"  Max:     {profiler_stats['max_ms']:.3f} ms")
    print(f"  ✓ Target: <1ms (Actual: {profiler_stats['avg_ms']:.3f}ms)")
    
    # Cache
    print("\n2. CacheManager Operations")
    print("-" * 60)
    cache_stats = benchmark_cache(iterations=10000)
    print(f"  Put Average: {cache_stats['put_avg_us']:.2f} μs")
    print(f"  Put Std Dev: {cache_stats['put_std_us']:.2f} μs")
    print(f"  Get Average: {cache_stats['get_avg_us']:.2f} μs")
    print(f"  Get Std Dev: {cache_stats['get_std_us']:.2f} μs")
    print(f"  ✓ Target: <100μs (Put: {cache_stats['put_avg_us']:.2f}μs, Get: {cache_stats['get_avg_us']:.2f}μs)")
    
    # Optimizer
    print("\n3. BatchOptimizer Computation")
    print("-" * 60)
    opt_stats = benchmark_optimizer(iterations=1000)
    print(f"  Average: {opt_stats['avg_us']:.2f} μs")
    print(f"  Std Dev: {opt_stats['std_us']:.2f} μs")
    print(f"  ✓ Target: <50μs (Actual: {opt_stats['avg_us']:.2f}μs)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    profiler_ok = profiler_stats['avg_ms'] < 1.0
    cache_ok = cache_stats['put_avg_us'] < 100 and cache_stats['get_avg_us'] < 100
    opt_ok = opt_stats['avg_us'] < 50
    
    all_ok = profiler_ok and cache_ok and opt_ok
    
    print(f"  Profiler:  {'✓ PASS' if profiler_ok else '✗ FAIL'}")
    print(f"  Cache:     {'✓ PASS' if cache_ok else '✗ FAIL'}")
    print(f"  Optimizer: {'✓ PASS' if opt_ok else '✗ FAIL'}")
    print()
    print(f"  Overall:   {'✓ ALL BENCHMARKS PASS' if all_ok else '✗ SOME BENCHMARKS FAIL'}")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    exit(main())
