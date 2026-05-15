"""
Benchmark cache performance: direct pickle vs safe_pickle.

Measures serialization/deserialization overhead from safe_pickle security layer.
"""

import pickle
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security.pickle_security_control import safe_pickle


def generate_test_data(size_kb: int = 100):
    """Gen test data ~size_kb KB."""
    # Dict w/ nested lists/dicts
    data = {
        "arrays": [[i for i in range(1000)] for _ in range(size_kb // 10)],
        "metadata": {"id": 12345, "name": "test_cache_entry", "version": "1.0"},
        "nested": {f"key_{i}": {"value": i * 2, "data": [i] * 100} for i in range(100)},
    }
    return data


def benchmark_pickle_direct(data, iterations: int = 1000):
    """Benchmark direct pickle.loads/dumps."""
    times_dump = []
    times_load = []

    for _ in range(iterations):
        # Dump
        start = time.perf_counter()
        serialized = pickle.dumps(data)
        times_dump.append(time.perf_counter() - start)

        # Load
        start = time.perf_counter()
        _ = pickle.loads(serialized)
        times_load.append(time.perf_counter() - start)

    return times_dump, times_load


def benchmark_safe_pickle(data, iterations: int = 1000):
    """Benchmark safe_pickle.loads/dumps."""
    times_dump = []
    times_load = []

    for _ in range(iterations):
        # Dump
        start = time.perf_counter()
        serialized = safe_pickle.dumps(data)
        times_dump.append(time.perf_counter() - start)

        # Load
        start = time.perf_counter()
        _ = safe_pickle.loads(serialized, trusted=True)
        times_load.append(time.perf_counter() - start)

    return times_dump, times_load


def calc_stats(times):
    """Calc mean/median/p95."""
    times_sorted = sorted(times)
    n = len(times_sorted)
    return {
        "mean": sum(times_sorted) / n,
        "median": times_sorted[n // 2],
        "p95": times_sorted[int(n * 0.95)],
        "min": times_sorted[0],
        "max": times_sorted[-1],
    }


def main():
    print("Cache Security Benchmark")
    print("=" * 60)

    # Config
    data_size_kb = 100
    iterations = 1000

    print(f"\nConfig:")
    print(f"  Data size: ~{data_size_kb} KB")
    print(f"  Iterations: {iterations}")

    # Gen data
    print("\nGen test data...")
    data = generate_test_data(data_size_kb)
    print(f"  Actual size: {len(pickle.dumps(data)) / 1024:.1f} KB")

    # Benchmark direct pickle
    print("\nBenchmark direct pickle...")
    direct_dump, direct_load = benchmark_pickle_direct(data, iterations)
    direct_dump_stats = calc_stats(direct_dump)
    direct_load_stats = calc_stats(direct_load)

    # Benchmark safe_pickle
    print("Benchmark safe_pickle...")
    safe_dump, safe_load = benchmark_safe_pickle(data, iterations)
    safe_dump_stats = calc_stats(safe_dump)
    safe_load_stats = calc_stats(safe_load)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nDumps (Serialization):")
    print(f"  Direct pickle:")
    print(f"    Mean:   {direct_dump_stats['mean']*1000:.3f} ms")
    print(f"    Median: {direct_dump_stats['median']*1000:.3f} ms")
    print(f"    P95:    {direct_dump_stats['p95']*1000:.3f} ms")

    print(f"\n  safe_pickle:")
    print(f"    Mean:   {safe_dump_stats['mean']*1000:.3f} ms")
    print(f"    Median: {safe_dump_stats['median']*1000:.3f} ms")
    print(f"    P95:    {safe_dump_stats['p95']*1000:.3f} ms")

    dump_overhead = (
        (safe_dump_stats["mean"] - direct_dump_stats["mean"])
        / direct_dump_stats["mean"]
        * 100
    )
    print(f"\n  Overhead: {dump_overhead:+.2f}%")

    print("\nLoads (Deserialization):")
    print(f"  Direct pickle:")
    print(f"    Mean:   {direct_load_stats['mean']*1000:.3f} ms")
    print(f"    Median: {direct_load_stats['median']*1000:.3f} ms")
    print(f"    P95:    {direct_load_stats['p95']*1000:.3f} ms")

    print(f"\n  safe_pickle:")
    print(f"    Mean:   {safe_load_stats['mean']*1000:.3f} ms")
    print(f"    Median: {safe_load_stats['median']*1000:.3f} ms")
    print(f"    P95:    {safe_load_stats['p95']*1000:.3f} ms")

    load_overhead = (
        (safe_load_stats["mean"] - direct_load_stats["mean"])
        / direct_load_stats["mean"]
        * 100
    )
    print(f"\n  Overhead: {load_overhead:+.2f}%")

    # Total
    total_direct = direct_dump_stats["mean"] + direct_load_stats["mean"]
    total_safe = safe_dump_stats["mean"] + safe_load_stats["mean"]
    total_overhead = (total_safe - total_direct) / total_direct * 100

    print("\nTotal (Dump + Load):")
    print(f"  Direct:      {total_direct*1000:.3f} ms")
    print(f"  safe_pickle: {total_safe*1000:.3f} ms")
    print(f"  Overhead:    {total_overhead:+.2f}%")

    # Pass/fail
    print("\n" + "=" * 60)
    if total_overhead < 2.0:
        print("✅ PASS: Overhead < 2%")
    else:
        print(f"⚠️  WARN: Overhead {total_overhead:.2f}% > 2% threshold")

    print("=" * 60)


if __name__ == "__main__":
    main()
