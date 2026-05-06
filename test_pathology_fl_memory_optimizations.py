#!/usr/bin/env python3
"""PathologyFL memory optimization tests."""

import gc
import time
from typing import Dict, List, Any

class MemoryOptimizedPathologyFL:
    """Memory-optimized PathologyFL implementation."""
    
    def __init__(self):
        self.memory_pool = []
        self.reusable_buffers = {}
        
    def get_reusable_buffer(self, size: int, buffer_type: str = "default") -> List:
        """Get reusable buffer to avoid allocations."""
        key = f"{buffer_type}_{size}"
        
        if key not in self.reusable_buffers:
            self.reusable_buffers[key] = []
        
        if self.reusable_buffers[key]:
            buffer = self.reusable_buffers[key].pop()
            # Clear buffer
            for i in range(len(buffer)):
                buffer[i] = 0.0
            return buffer
        else:
            return [0.0] * size
    
    def return_buffer(self, buffer: List, buffer_type: str = "default"):
        """Return buffer to pool for reuse."""
        key = f"{buffer_type}_{len(buffer)}"
        
        if key not in self.reusable_buffers:
            self.reusable_buffers[key] = []
        
        if len(self.reusable_buffers[key]) < 10:  # Limit pool size
            self.reusable_buffers[key].append(buffer)
    
    def memory_efficient_aggregation(self, client_weights: List[Dict]) -> Dict:
        """Memory-efficient aggregation using buffer reuse."""
        if not client_weights:
            return {}
        
        keys = list(client_weights[0].keys())
        result = {}
        
        for key in keys:
            weight_size = len(client_weights[0][key])
            
            # Get reusable buffer
            sum_buffer = self.get_reusable_buffer(weight_size, "aggregation")
            
            # Accumulate weights
            for client in client_weights:
                client_weight = client[key]
                for i in range(weight_size):
                    sum_buffer[i] += client_weight[i]
            
            # Average
            num_clients = len(client_weights)
            for i in range(weight_size):
                sum_buffer[i] /= num_clients
            
            # Copy result (buffer will be reused)
            result[key] = sum_buffer.copy()
            
            # Return buffer to pool
            self.return_buffer(sum_buffer, "aggregation")
        
        return result
    
    def streaming_processing(self, data_generator):
        """Process data in streaming fashion to minimize memory."""
        results = []
        batch_size = 50
        batch = []
        
        for item in data_generator:
            batch.append(item)
            
            if len(batch) >= batch_size:
                # Process batch
                batch_results = self._process_batch(batch)
                results.extend(batch_results)
                
                # Clear batch
                batch.clear()
                
                # Force garbage collection periodically
                if len(results) % 500 == 0:
                    gc.collect()
        
        # Process remaining items
        if batch:
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, batch):
        """Process a batch of items."""
        return [{"id": item["id"], "processed": True} for item in batch]

def test_buffer_reuse():
    """Test buffer reuse for memory efficiency."""
    print("Testing buffer reuse...")
    
    optimizer = MemoryOptimizedPathologyFL()
    
    def get_memory_objects():
        gc.collect()
        return len(gc.get_objects())
    
    baseline_memory = get_memory_objects()
    
    # Simulate multiple aggregation rounds
    for round_num in range(10):
        # Generate client weights
        client_weights = []
        for i in range(50):
            weights = {
                "layer1": [0.1 + i * 0.001] * 1000,
                "layer2": [0.2 + i * 0.001] * 500,
            }
            client_weights.append(weights)
        
        # Aggregate
        result = optimizer.memory_efficient_aggregation(client_weights)
        
        # Clear references
        del client_weights
        del result
    
    final_memory = get_memory_objects()
    memory_growth = final_memory - baseline_memory
    
    buffer_pools = len(optimizer.reusable_buffers)
    total_buffers = sum(len(pool) for pool in optimizer.reusable_buffers.values())
    
    print(f"  Aggregation rounds: 10")
    print(f"  Memory growth: +{memory_growth} objects")
    print(f"  Buffer pools: {buffer_pools}")
    print(f"  Total pooled buffers: {total_buffers}")
    
    return memory_growth < 1000 and total_buffers > 0

def test_streaming_processing():
    """Test streaming processing for large datasets."""
    print("Testing streaming processing...")
    
    optimizer = MemoryOptimizedPathologyFL()
    
    def data_generator(size):
        """Generate data on-demand."""
        for i in range(size):
            yield {"id": f"item_{i}", "data": list(range(10))}
    
    def get_memory_objects():
        gc.collect()
        return len(gc.get_objects())
    
    baseline_memory = get_memory_objects()
    
    # Process large dataset
    large_size = 10000
    start_time = time.time()
    
    results = optimizer.streaming_processing(data_generator(large_size))
    
    processing_time = time.time() - start_time
    peak_memory = get_memory_objects()
    
    # Cleanup
    del results
    gc.collect()
    final_memory = get_memory_objects()
    
    memory_peak = peak_memory - baseline_memory
    memory_final = final_memory - baseline_memory
    
    print(f"  Dataset size: {large_size}")
    print(f"  Processing time: {processing_time:.4f}s")
    print(f"  Peak memory: +{memory_peak} objects")
    print(f"  Final memory: +{memory_final} objects")
    print(f"  Throughput: {large_size/processing_time:.0f} items/sec")
    
    return memory_peak < 2000 and memory_final < memory_peak * 0.5

def test_memory_leak_detection():
    """Test for memory leaks in repeated operations."""
    print("Testing memory leak detection...")
    
    optimizer = MemoryOptimizedPathologyFL()
    
    def get_memory_objects():
        gc.collect()
        return len(gc.get_objects())
    
    memory_samples = []
    
    # Perform repeated operations
    for iteration in range(20):
        # Generate and process data
        client_weights = []
        for i in range(100):
            weights = {
                "conv": [0.1] * 2048,
                "fc": [0.2] * 1024,
            }
            client_weights.append(weights)
        
        # Aggregate
        result = optimizer.memory_efficient_aggregation(client_weights)
        
        # Sample memory
        memory_objects = get_memory_objects()
        memory_samples.append(memory_objects)
        
        # Cleanup
        del client_weights
        del result
        
        if iteration % 5 == 0:
            gc.collect()
    
    # Analyze memory trend
    early_avg = sum(memory_samples[:5]) / 5
    late_avg = sum(memory_samples[-5:]) / 5
    memory_growth = late_avg - early_avg
    
    print(f"  Iterations: {len(memory_samples)}")
    print(f"  Early average: {early_avg:.0f} objects")
    print(f"  Late average: {late_avg:.0f} objects")
    print(f"  Memory growth: {memory_growth:+.0f} objects")
    
    # Check for significant memory growth (potential leak)
    return abs(memory_growth) < 500

def test_large_model_handling():
    """Test handling of very large models."""
    print("Testing large model handling...")
    
    optimizer = MemoryOptimizedPathologyFL()
    
    def get_memory_mb():
        """Get memory usage in MB."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    baseline_memory = get_memory_mb()
    
    # Create very large model weights
    large_client_weights = []
    for i in range(10):  # 10 clients
        weights = {
            "huge_layer1": [0.1 + i * 0.001] * 50000,  # 50K parameters
            "huge_layer2": [0.2 + i * 0.001] * 25000,  # 25K parameters
            "huge_layer3": [0.3 + i * 0.001] * 10000,  # 10K parameters
        }
        large_client_weights.append(weights)
    
    after_creation_memory = get_memory_mb()
    
    # Aggregate large model
    start_time = time.time()
    large_result = optimizer.memory_efficient_aggregation(large_client_weights)
    aggregation_time = time.time() - start_time
    
    after_aggregation_memory = get_memory_mb()
    
    # Cleanup
    del large_client_weights
    del large_result
    gc.collect()
    
    final_memory = get_memory_mb()
    
    creation_memory = after_creation_memory - baseline_memory
    aggregation_memory = after_aggregation_memory - after_creation_memory
    cleanup_memory = final_memory - baseline_memory
    
    print(f"  Model size: 85K parameters per client")
    print(f"  Clients: 10")
    print(f"  Creation memory: +{creation_memory:.1f} MB")
    print(f"  Aggregation memory: +{aggregation_memory:.1f} MB")
    print(f"  After cleanup: +{cleanup_memory:.1f} MB")
    print(f"  Aggregation time: {aggregation_time:.4f}s")
    
    return (aggregation_time < 1.0 and 
            cleanup_memory < creation_memory * 0.5)

def test_concurrent_memory_usage():
    """Test memory usage under concurrent operations."""
    print("Testing concurrent memory usage...")
    
    optimizer = MemoryOptimizedPathologyFL()
    
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    def get_memory_objects():
        gc.collect()
        return len(gc.get_objects())
    
    def worker_task(worker_id):
        """Worker task that processes data."""
        client_weights = []
        for i in range(20):
            weights = {
                f"layer_{worker_id}": [0.1 + i * 0.001] * 1000,
            }
            client_weights.append(weights)
        
        result = optimizer.memory_efficient_aggregation(client_weights)
        return len(result)
    
    baseline_memory = get_memory_objects()
    
    # Run concurrent tasks
    num_workers = 8
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_task, i) for i in range(num_workers)]
        results = [f.result() for f in futures]
    
    concurrent_time = time.time() - start_time
    peak_memory = get_memory_objects()
    
    # Cleanup
    gc.collect()
    final_memory = get_memory_objects()
    
    memory_peak = peak_memory - baseline_memory
    memory_final = final_memory - baseline_memory
    
    print(f"  Concurrent workers: {num_workers}")
    print(f"  Processing time: {concurrent_time:.4f}s")
    print(f"  Peak memory: +{memory_peak} objects")
    print(f"  Final memory: +{memory_final} objects")
    print(f"  Results: {results}")
    
    return (len(results) == num_workers and 
            all(r > 0 for r in results) and
            memory_final < memory_peak * 0.8)

def run_memory_optimization_tests():
    """Run all memory optimization tests."""
    print("🧠 PathologyFL Memory Optimization Testing")
    print("=" * 60)
    
    tests = [
        ("Buffer Reuse", test_buffer_reuse),
        ("Streaming Processing", test_streaming_processing),
        ("Memory Leak Detection", test_memory_leak_detection),
        ("Large Model Handling", test_large_model_handling),
        ("Concurrent Memory Usage", test_concurrent_memory_usage),
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
    print(f"Memory Optimization Tests: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🏆 Memory optimization perfect!")
    else:
        print(f"⚠️ {len(tests) - passed} memory issues found")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_memory_optimization_tests()
    exit(0 if success else 1)