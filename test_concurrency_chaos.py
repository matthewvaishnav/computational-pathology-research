#!/usr/bin/env python3
"""
Concurrent Access Chaos Testing for HistoCore
"""

import threading
import time
import random
import queue
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def test_race_conditions():
    """Test for race conditions in shared resources."""
    
    print("🏃 Testing Race Conditions...")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    # Shared resource
    shared_counter = {"value": 0, "lock": threading.Lock()}
    
    def increment_counter(thread_id, iterations):
        for _ in range(iterations):
            with shared_counter["lock"]:
                old_value = shared_counter["value"]
                time.sleep(0.0001)  # Simulate processing
                shared_counter["value"] = old_value + 1
    
    try:
        # Test with multiple threads
        num_threads = 10
        iterations_per_thread = 100
        expected_total = num_threads * iterations_per_thread
        
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=increment_counter, args=(i, iterations_per_thread))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        if shared_counter["value"] == expected_total:
            results["passed"] += 1
            results["details"].append(f"✅ Race condition prevented: {shared_counter['value']}/{expected_total}")
        else:
            results["failed"] += 1
            results["details"].append(f"❌ Race condition detected: {shared_counter['value']}/{expected_total}")
    
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Race condition test failed: {str(e)}")
    
    return results

def test_deadlock_prevention():
    """Test deadlock prevention mechanisms."""
    
    print("🔒 Testing Deadlock Prevention...")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    # Two locks that could cause deadlock
    lock1 = threading.Lock()
    lock2 = threading.Lock()
    
    def worker1():
        try:
            if lock1.acquire(timeout=1.0):
                time.sleep(0.1)
                if lock2.acquire(timeout=1.0):
                    time.sleep(0.1)
                    lock2.release()
                lock1.release()
                return True
        except:
            pass
        return False
    
    def worker2():
        try:
            if lock2.acquire(timeout=1.0):
                time.sleep(0.1)
                if lock1.acquire(timeout=1.0):
                    time.sleep(0.1)
                    lock1.release()
                lock2.release()
                return True
        except:
            pass
        return False
    
    try:
        # Run potential deadlock scenario
        thread1 = threading.Thread(target=worker1)
        thread2 = threading.Thread(target=worker2)
        
        start_time = time.time()
        thread1.start()
        thread2.start()
        
        thread1.join(timeout=5.0)
        thread2.join(timeout=5.0)
        
        elapsed = time.time() - start_time
        
        if elapsed < 3.0:  # Should complete quickly with timeout
            results["passed"] += 1
            results["details"].append(f"✅ Deadlock prevented (completed in {elapsed:.2f}s)")
        else:
            results["failed"] += 1
            results["details"].append(f"❌ Potential deadlock (took {elapsed:.2f}s)")
    
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Deadlock test failed: {str(e)}")
    
    return results

def test_thread_pool_exhaustion():
    """Test thread pool exhaustion scenarios."""
    
    print("🏊 Testing Thread Pool Exhaustion...")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    def blocking_task(duration):
        time.sleep(duration)
        return f"Task completed after {duration}s"
    
    try:
        # Create small thread pool
        max_workers = 5
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit more tasks than workers
            futures = []
            for i in range(max_workers * 3):
                future = executor.submit(blocking_task, 0.1)
                futures.append(future)
            
            # Check that tasks complete despite pool exhaustion
            completed = 0
            for future in as_completed(futures, timeout=10.0):
                result = future.result()
                completed += 1
            
            if completed == len(futures):
                results["passed"] += 1
                results["details"].append(f"✅ Thread pool handled {completed} tasks with {max_workers} workers")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ Thread pool failed: {completed}/{len(futures)} completed")
    
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Thread pool test failed: {str(e)}")
    
    return results

def test_resource_contention():
    """Test resource contention scenarios."""
    
    print("⚔️ Testing Resource Contention...")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    # Shared resource queue
    resource_queue = queue.Queue(maxsize=3)
    
    # Fill queue with resources
    for i in range(3):
        resource_queue.put(f"resource_{i}")
    
    def worker(worker_id, work_duration):
        try:
            # Try to get resource
            resource = resource_queue.get(timeout=2.0)
            
            # Use resource
            time.sleep(work_duration)
            
            # Return resource
            resource_queue.put(resource)
            resource_queue.task_done()
            
            return True
        except queue.Empty:
            return False
        except Exception:
            return False
    
    try:
        # Start many workers competing for limited resources
        num_workers = 10
        work_duration = 0.1
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(num_workers):
                future = executor.submit(worker, i, work_duration)
                futures.append(future)
            
            # Count successful workers
            successful = 0
            for future in as_completed(futures, timeout=15.0):
                if future.result():
                    successful += 1
            
            if successful >= num_workers * 0.8:  # At least 80% should succeed
                results["passed"] += 1
                results["details"].append(f"✅ Resource contention handled: {successful}/{num_workers} workers succeeded")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ Resource contention failed: {successful}/{num_workers} workers succeeded")
    
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Resource contention test failed: {str(e)}")
    
    return results

def run_concurrency_chaos_tests():
    """Run all concurrency chaos tests."""
    
    print("🚀 Starting Concurrent Access Chaos Tests")
    print("=" * 50)
    
    all_results = {
        "race_conditions": test_race_conditions(),
        "deadlock_prevention": test_deadlock_prevention(),
        "thread_pool_exhaustion": test_thread_pool_exhaustion(),
        "resource_contention": test_resource_contention(),
    }
    
    # Summary
    total_passed = sum(r["passed"] for r in all_results.values())
    total_failed = sum(r["failed"] for r in all_results.values())
    
    print("=" * 50)
    print("📋 CONCURRENCY CHAOS TEST SUMMARY")
    print(f"✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    
    for test_type, results in all_results.items():
        print(f"\n📊 {test_type.upper()} TESTS:")
        print(f"  ✅ Passed: {results['passed']}")
        print(f"  ❌ Failed: {results['failed']}")
        
        for detail in results["details"]:
            print(f"    {detail}")
    
    return total_passed, total_failed

if __name__ == "__main__":
    passed, failed = run_concurrency_chaos_tests()
    sys.exit(1 if failed > 0 else 0)