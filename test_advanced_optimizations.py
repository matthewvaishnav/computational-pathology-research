#!/usr/bin/env python3
"""Advanced optimization and performance tuning tests."""

import time
import gc
import threading
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class OptimizationTester:
    """Test various optimization techniques."""
    
    def __init__(self):
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self) -> List[Dict[str, Any]]:
        """Generate test data for optimization tests."""
        return [
            {"id": i, "value": i * 2, "data": [j for j in range(100)]}
            for i in range(1000)
        ]

def test_memory_optimization():
    """Test memory optimization techniques."""
    print("Testing memory optimization...")
    
    # Test 1: Generator vs List
    def list_approach():
        return [i ** 2 for i in range(10000)]
    
    def generator_approach():
        return (i ** 2 for i in range(10000))
    
    # Memory usage comparison
    start_time = time.time()
    list_result = list_approach()
    list_time = time.time() - start_time
    
    start_time = time.time()
    gen_result = generator_approach()
    gen_time = time.time() - start_time
    
    # Consume generator to compare
    gen_consumed = list(gen_result)
    
    print(f"  List creation: {list_time:.4f}s")
    print(f"  Generator creation: {gen_time:.4f}s")
    print(f"  Results equal: {list_result == gen_consumed}")
    
    # Test 2: Memory pooling
    class ObjectPool:
        def __init__(self, create_func, reset_func, initial_size=10):
            self.create_func = create_func
            self.reset_func = reset_func
            self.pool = [create_func() for _ in range(initial_size)]
        
        def get(self):
            if self.pool:
                return self.pool.pop()
            return self.create_func()
        
        def put(self, obj):
            self.reset_func(obj)
            self.pool.append(obj)
    
    def create_dict():
        return {"data": [], "processed": False}
    
    def reset_dict(d):
        d["data"].clear()
        d["processed"] = False
    
    pool = ObjectPool(create_dict, reset_dict)
    
    # Test pooling performance
    start_time = time.time()
    for _ in range(1000):
        obj = pool.get()
        obj["data"] = [1, 2, 3]
        obj["processed"] = True
        pool.put(obj)
    pool_time = time.time() - start_time
    
    # Test without pooling
    start_time = time.time()
    for _ in range(1000):
        obj = create_dict()
        obj["data"] = [1, 2, 3]
        obj["processed"] = True
    no_pool_time = time.time() - start_time
    
    print(f"  With pooling: {pool_time:.4f}s")
    print(f"  Without pooling: {no_pool_time:.4f}s")
    print(f"  Pooling speedup: {no_pool_time/pool_time:.2f}x")
    
    return gen_time < list_time and pool_time < no_pool_time

def test_cpu_optimization():
    """Test CPU optimization techniques."""
    print("Testing CPU optimization...")
    
    # Test 1: List comprehension vs loop
    def loop_approach(data):
        result = []
        for item in data:
            if item % 2 == 0:
                result.append(item * 2)
        return result
    
    def comprehension_approach(data):
        return [item * 2 for item in data if item % 2 == 0]
    
    test_data = list(range(10000))
    
    start_time = time.time()
    loop_result = loop_approach(test_data)
    loop_time = time.time() - start_time
    
    start_time = time.time()
    comp_result = comprehension_approach(test_data)
    comp_time = time.time() - start_time
    
    print(f"  Loop approach: {loop_time:.4f}s")
    print(f"  Comprehension: {comp_time:.4f}s")
    print(f"  Results equal: {loop_result == comp_result}")
    
    # Test 2: String concatenation optimization
    def slow_concat(strings):
        result = ""
        for s in strings:
            result += s
        return result
    
    def fast_concat(strings):
        return "".join(strings)
    
    test_strings = [f"string_{i}" for i in range(1000)]
    
    start_time = time.time()
    slow_result = slow_concat(test_strings)
    slow_time = time.time() - start_time
    
    start_time = time.time()
    fast_result = fast_concat(test_strings)
    fast_time = time.time() - start_time
    
    print(f"  Slow concatenation: {slow_time:.4f}s")
    print(f"  Fast concatenation: {fast_time:.4f}s")
    print(f"  Join speedup: {slow_time/fast_time:.2f}x")
    
    return comp_time <= loop_time and fast_time < slow_time

def test_parallel_processing():
    """Test parallel processing optimizations."""
    print("Testing parallel processing...")
    
    def cpu_intensive_task(n):
        """CPU-intensive task for testing."""
        result = 0
        for i in range(n):
            result += i ** 2
        return result
    
    tasks = [10000] * 8
    
    # Sequential processing
    start_time = time.time()
    sequential_results = [cpu_intensive_task(n) for n in tasks]
    sequential_time = time.time() - start_time
    
    # Thread-based parallel processing
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        thread_results = list(executor.map(cpu_intensive_task, tasks))
    thread_time = time.time() - start_time
    
    # Process-based parallel processing
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        process_results = list(executor.map(cpu_intensive_task, tasks))
    process_time = time.time() - start_time
    
    print(f"  Sequential: {sequential_time:.4f}s")
    print(f"  Threading: {thread_time:.4f}s")
    print(f"  Multiprocessing: {process_time:.4f}s")
    
    results_equal = (sequential_results == thread_results == process_results)
    print(f"  Results equal: {results_equal}")
    
    # For CPU-intensive tasks, multiprocessing should be faster
    return results_equal and process_time < sequential_time

def test_caching_optimization():
    """Test caching optimization techniques."""
    print("Testing caching optimization...")
    
    # Simple cache implementation
    class SimpleCache:
        def __init__(self, max_size=100):
            self.cache = {}
            self.max_size = max_size
        
        def get(self, key):
            return self.cache.get(key)
        
        def put(self, key, value):
            if len(self.cache) >= self.max_size:
                # Remove oldest item (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = value
    
    def expensive_computation(n):
        """Simulate expensive computation."""
        time.sleep(0.001)  # 1ms delay
        return n ** 3 + n ** 2 + n
    
    cache = SimpleCache()
    
    # Test without caching
    start_time = time.time()
    no_cache_results = []
    for i in range(100):
        result = expensive_computation(i % 20)  # Repeated computations
        no_cache_results.append(result)
    no_cache_time = time.time() - start_time
    
    # Test with caching
    start_time = time.time()
    cache_results = []
    for i in range(100):
        key = i % 20
        result = cache.get(key)
        if result is None:
            result = expensive_computation(key)
            cache.put(key, result)
        cache_results.append(result)
    cache_time = time.time() - start_time
    
    print(f"  Without caching: {no_cache_time:.4f}s")
    print(f"  With caching: {cache_time:.4f}s")
    print(f"  Cache speedup: {no_cache_time/cache_time:.2f}x")
    print(f"  Results equal: {no_cache_results == cache_results}")
    
    return cache_time < no_cache_time and no_cache_results == cache_results

def test_data_structure_optimization():
    """Test data structure optimization."""
    print("Testing data structure optimization...")
    
    # Test 1: List vs Set for membership testing
    test_data = list(range(1000))
    test_list = test_data.copy()
    test_set = set(test_data)
    
    search_items = [100, 500, 999, 1500]  # Some exist, some don't
    
    # List membership testing
    start_time = time.time()
    list_results = [item in test_list for item in search_items * 100]
    list_time = time.time() - start_time
    
    # Set membership testing
    start_time = time.time()
    set_results = [item in test_set for item in search_items * 100]
    set_time = time.time() - start_time
    
    print(f"  List membership: {list_time:.4f}s")
    print(f"  Set membership: {set_time:.4f}s")
    print(f"  Set speedup: {list_time/set_time:.2f}x")
    
    # Test 2: Dictionary vs List for lookups
    lookup_data = {f"key_{i}": f"value_{i}" for i in range(1000)}
    lookup_list = [(f"key_{i}", f"value_{i}") for i in range(1000)]
    
    search_keys = ["key_100", "key_500", "key_999"]
    
    # Dictionary lookup
    start_time = time.time()
    dict_results = []
    for _ in range(100):
        for key in search_keys:
            dict_results.append(lookup_data.get(key))
    dict_time = time.time() - start_time
    
    # List lookup
    start_time = time.time()
    list_lookup_results = []
    for _ in range(100):
        for key in search_keys:
            result = None
            for k, v in lookup_list:
                if k == key:
                    result = v
                    break
            list_lookup_results.append(result)
    list_lookup_time = time.time() - start_time
    
    print(f"  Dictionary lookup: {dict_time:.4f}s")
    print(f"  List lookup: {list_lookup_time:.4f}s")
    print(f"  Dict speedup: {list_lookup_time/dict_time:.2f}x")
    
    return (set_time < list_time and dict_time < list_lookup_time and 
            list_results == set_results)

def test_algorithm_optimization():
    """Test algorithm optimization techniques."""
    print("Testing algorithm optimization...")
    
    # Test 1: Bubble sort vs Quick sort
    import random
    
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr
    
    def quick_sort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)
    
    # Generate test data
    test_data = [random.randint(1, 1000) for _ in range(500)]
    
    # Bubble sort
    start_time = time.time()
    bubble_result = bubble_sort(test_data.copy())
    bubble_time = time.time() - start_time
    
    # Quick sort
    start_time = time.time()
    quick_result = quick_sort(test_data.copy())
    quick_time = time.time() - start_time
    
    # Built-in sort (for comparison)
    start_time = time.time()
    builtin_result = sorted(test_data)
    builtin_time = time.time() - start_time
    
    print(f"  Bubble sort: {bubble_time:.4f}s")
    print(f"  Quick sort: {quick_time:.4f}s")
    print(f"  Built-in sort: {builtin_time:.4f}s")
    print(f"  Results equal: {bubble_result == quick_result == builtin_result}")
    
    return quick_time < bubble_time and bubble_result == quick_result

def run_optimization_tests():
    """Run all optimization tests."""
    print("⚡ Advanced Optimization Testing")
    print("=" * 50)
    
    tests = [
        ("Memory Optimization", test_memory_optimization),
        ("CPU Optimization", test_cpu_optimization),
        ("Parallel Processing", test_parallel_processing),
        ("Caching Optimization", test_caching_optimization),
        ("Data Structure Optimization", test_data_structure_optimization),
        ("Algorithm Optimization", test_algorithm_optimization),
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
    
    print("=" * 50)
    print(f"Optimization Tests: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🏆 All optimizations working perfectly!")
    else:
        print(f"⚠️ {len(tests) - passed} optimization issues found")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_optimization_tests()
    exit(0 if success else 1)