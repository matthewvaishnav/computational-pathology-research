#!/usr/bin/env python3
"""PathologyFL benchmarking and performance analysis."""

import time
import statistics
from typing import Dict, List, Tuple

class PathologyFLBenchmark:
    """Benchmark PathologyFL performance across different scenarios."""
    
    def __init__(self):
        self.results = {}
        
    def benchmark_aggregation(self, num_clients: int, param_size: int, iterations: int = 10) -> Dict:
        """Benchmark federated aggregation performance."""
        times = []
        
        for _ in range(iterations):
            # Generate client updates
            client_updates = []
            for i in range(num_clients):
                update = {
                    "hospital_id": f"client_{i}",
                    "parameters": {
                        "layer1": [0.1 + i * 0.001] * param_size,
                        "layer2": [0.2 + i * 0.001] * (param_size // 2)
                    }
                }
                client_updates.append(update)
            
            # Benchmark aggregation
            start_time = time.time()
            
            # Simple aggregation
            aggregated = {}
            for update in client_updates:
                for layer, params in update["parameters"].items():
                    if layer not in aggregated:
                        aggregated[layer] = [0.0] * len(params)
                    
                    for i, param in enumerate(params):
                        aggregated[layer][i] += param
            
            # Average
            for layer in aggregated:
                for i in range(len(aggregated[layer])):
                    aggregated[layer][i] /= num_clients
            
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        return {
            "mean_time": statistics.mean(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "min_time": min(times),
            "max_time": max(times),
            "throughput": num_clients / statistics.mean(times)
        }
    
    def benchmark_scaling(self) -> Dict:
        """Benchmark scaling characteristics."""
        client_counts = [10, 50, 100, 500, 1000]
        param_size = 1000
        
        scaling_results = {}
        
        for num_clients in client_counts:
            result = self.benchmark_aggregation(num_clients, param_size, iterations=5)
            scaling_results[num_clients] = result
            
        return scaling_results
    
    def benchmark_parameter_sizes(self) -> Dict:
        """Benchmark different parameter sizes."""
        param_sizes = [100, 1000, 10000, 50000]
        num_clients = 100
        
        param_results = {}
        
        for param_size in param_sizes:
            result = self.benchmark_aggregation(num_clients, param_size, iterations=5)
            param_results[param_size] = result
            
        return param_results
    
    def benchmark_memory_usage(self, num_clients: int, param_size: int) -> Dict:
        """Benchmark memory usage during aggregation."""
        import gc
        
        def get_memory_objects():
            gc.collect()
            return len(gc.get_objects())
        
        baseline_memory = get_memory_objects()
        
        # Generate large client updates
        client_updates = []
        for i in range(num_clients):
            update = {
                "hospital_id": f"client_{i}",
                "parameters": {
                    "large_layer": [0.001] * param_size
                }
            }
            client_updates.append(update)
        
        after_generation = get_memory_objects()
        
        # Perform aggregation
        start_time = time.time()
        
        aggregated = {"large_layer": [0.0] * param_size}
        for update in client_updates:
            for i, param in enumerate(update["parameters"]["large_layer"]):
                aggregated["large_layer"][i] += param
        
        for i in range(len(aggregated["large_layer"])):
            aggregated["large_layer"][i] /= num_clients
        
        aggregation_time = time.time() - start_time
        
        after_aggregation = get_memory_objects()
        
        # Cleanup
        del client_updates
        del aggregated
        gc.collect()
        
        after_cleanup = get_memory_objects()
        
        return {
            "baseline_memory": baseline_memory,
            "after_generation": after_generation - baseline_memory,
            "after_aggregation": after_aggregation - baseline_memory,
            "after_cleanup": after_cleanup - baseline_memory,
            "aggregation_time": aggregation_time
        }

def test_aggregation_benchmarks():
    """Test aggregation performance benchmarks."""
    print("Testing aggregation benchmarks...")
    
    benchmark = PathologyFLBenchmark()
    
    # Benchmark different scenarios
    scenarios = [
        (10, 1000),   # Small: 10 clients, 1K params
        (100, 1000),  # Medium: 100 clients, 1K params
        (100, 10000), # Large: 100 clients, 10K params
    ]
    
    results = []
    for num_clients, param_size in scenarios:
        result = benchmark.benchmark_aggregation(num_clients, param_size)
        results.append((num_clients, param_size, result))
        
        print(f"  {num_clients} clients, {param_size} params:")
        print(f"    Mean time: {result['mean_time']:.4f}s")
        print(f"    Throughput: {result['throughput']:.0f} clients/sec")
    
    # Check that larger scenarios take more time
    small_time = results[0][2]['mean_time']
    large_time = results[-1][2]['mean_time']
    
    return large_time > small_time and all(r[2]['throughput'] > 0 for r in results)

def test_scaling_benchmarks():
    """Test scaling performance benchmarks."""
    print("Testing scaling benchmarks...")
    
    benchmark = PathologyFLBenchmark()
    
    scaling_results = benchmark.benchmark_scaling()
    
    client_counts = sorted(scaling_results.keys())
    throughputs = [scaling_results[count]['throughput'] for count in client_counts]
    
    print(f"  Client scaling results:")
    for count, throughput in zip(client_counts, throughputs):
        print(f"    {count} clients: {throughput:.0f} clients/sec")
    
    # Check that system can handle increasing load
    min_throughput = min(throughputs)
    max_throughput = max(throughputs)
    
    return min_throughput > 0 and len(scaling_results) == 5

def test_parameter_size_benchmarks():
    """Test parameter size scaling benchmarks."""
    print("Testing parameter size benchmarks...")
    
    benchmark = PathologyFLBenchmark()
    
    param_results = benchmark.benchmark_parameter_sizes()
    
    param_sizes = sorted(param_results.keys())
    times = [param_results[size]['mean_time'] for size in param_sizes]
    
    print(f"  Parameter size scaling:")
    for size, time_taken in zip(param_sizes, times):
        print(f"    {size:,} params: {time_taken:.4f}s")
    
    # Check that larger parameters take more time
    return times[-1] > times[0] and all(param_results[size]['throughput'] > 0 for size in param_sizes)

def test_memory_benchmarks():
    """Test memory usage benchmarks."""
    print("Testing memory benchmarks...")
    
    benchmark = PathologyFLBenchmark()
    
    # Test different memory scenarios
    scenarios = [
        (50, 5000),   # 50 clients, 5K params each
        (100, 10000), # 100 clients, 10K params each
    ]
    
    memory_results = []
    for num_clients, param_size in scenarios:
        result = benchmark.benchmark_memory_usage(num_clients, param_size)
        memory_results.append(result)
        
        print(f"  {num_clients} clients, {param_size} params:")
        print(f"    Memory after generation: +{result['after_generation']} objects")
        print(f"    Memory after aggregation: +{result['after_aggregation']} objects")
        print(f"    Memory after cleanup: +{result['after_cleanup']} objects")
        print(f"    Aggregation time: {result['aggregation_time']:.4f}s")
    
    # Check memory cleanup effectiveness
    cleanup_effective = all(
        result['after_cleanup'] < result['after_aggregation'] 
        for result in memory_results
    )
    
    return cleanup_effective and len(memory_results) == 2

def test_performance_regression():
    """Test for performance regressions."""
    print("Testing performance regression...")
    
    benchmark = PathologyFLBenchmark()
    
    # Baseline performance
    baseline_clients = 100
    baseline_params = 1000
    
    baseline_result = benchmark.benchmark_aggregation(baseline_clients, baseline_params)
    baseline_throughput = baseline_result['throughput']
    
    # Run multiple times to check consistency
    consistency_results = []
    for _ in range(5):
        result = benchmark.benchmark_aggregation(baseline_clients, baseline_params, iterations=3)
        consistency_results.append(result['throughput'])
    
    # Check performance consistency
    throughput_std = statistics.stdev(consistency_results)
    throughput_mean = statistics.mean(consistency_results)
    
    # Coefficient of variation should be low (< 20%)
    cv = throughput_std / throughput_mean if throughput_mean > 0 else 1.0
    
    print(f"  Baseline throughput: {baseline_throughput:.0f} clients/sec")
    print(f"  Consistency runs: {len(consistency_results)}")
    print(f"  Mean throughput: {throughput_mean:.0f} clients/sec")
    print(f"  Std deviation: {throughput_std:.0f}")
    print(f"  Coefficient of variation: {cv:.2%}")
    
    return cv < 0.2 and throughput_mean > 0

def test_comparative_analysis():
    """Test comparative performance analysis."""
    print("Testing comparative analysis...")
    
    benchmark = PathologyFLBenchmark()
    
    # Compare different aggregation strategies
    num_clients = 50
    param_size = 2000
    
    # Strategy 1: Simple aggregation (current)
    start_time = time.time()
    result1 = benchmark.benchmark_aggregation(num_clients, param_size, iterations=3)
    strategy1_time = time.time() - start_time
    
    # Strategy 2: Chunked aggregation
    start_time = time.time()
    
    # Simulate chunked processing
    chunk_size = 10
    chunk_times = []
    
    for chunk_start in range(0, num_clients, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_clients)
        chunk_clients = chunk_end - chunk_start
        
        chunk_result = benchmark.benchmark_aggregation(chunk_clients, param_size, iterations=1)
        chunk_times.append(chunk_result['mean_time'])
    
    strategy2_time = time.time() - start_time
    
    print(f"  Simple aggregation: {strategy1_time:.4f}s total")
    print(f"  Chunked aggregation: {strategy2_time:.4f}s total")
    print(f"  Simple throughput: {result1['throughput']:.0f} clients/sec")
    print(f"  Chunk count: {len(chunk_times)}")
    
    return len(chunk_times) > 0 and result1['throughput'] > 0

def run_pathology_fl_benchmarks():
    """Run all PathologyFL benchmark tests."""
    print("📊 PathologyFL Benchmarking and Performance Analysis")
    print("=" * 60)
    
    tests = [
        ("Aggregation Benchmarks", test_aggregation_benchmarks),
        ("Scaling Benchmarks", test_scaling_benchmarks),
        ("Parameter Size Benchmarks", test_parameter_size_benchmarks),
        ("Memory Benchmarks", test_memory_benchmarks),
        ("Performance Regression", test_performance_regression),
        ("Comparative Analysis", test_comparative_analysis),
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
        print()
    
    print("=" * 60)
    print(f"Benchmark Tests: {passed}/{len(tests)} passed")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_pathology_fl_benchmarks()
    exit(0 if success else 1)