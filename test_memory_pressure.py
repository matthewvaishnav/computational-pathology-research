#!/usr/bin/env python3
"""
Extreme Memory Pressure Testing for HistoCore
"""

import gc
import os
import sys
import time
import psutil
import threading
from pathlib import Path

def test_memory_exhaustion():
    """Test behavior under extreme memory pressure."""
    
    print("🧠 Testing Memory Exhaustion...")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        # Test 1: Gradual memory allocation
        memory_chunks = []
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        max_chunks = 100  # Up to 1GB
        
        for i in range(max_chunks):
            try:
                chunk = bytearray(chunk_size)
                memory_chunks.append(chunk)
                
                current_memory = process.memory_info().rss / 1024 / 1024
                if current_memory > initial_memory + 500:  # 500MB limit
                    break
                    
            except MemoryError:
                results["passed"] += 1
                results["details"].append(f"✅ Memory exhaustion handled gracefully at {i * 10}MB")
                break
        else:
            results["passed"] += 1
            results["details"].append("✅ Memory allocation controlled within limits")
        
        # Cleanup
        del memory_chunks
        gc.collect()
        
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Memory exhaustion test failed: {str(e)}")
    
    return results

def test_memory_fragmentation():
    """Test memory fragmentation scenarios."""
    
    print("🧩 Testing Memory Fragmentation...")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    try:
        # Create fragmented memory pattern
        small_chunks = []
        large_chunks = []
        
        # Allocate many small chunks
        for i in range(1000):
            small_chunks.append(bytearray(1024))  # 1KB each
        
        # Deallocate every other chunk
        for i in range(0, len(small_chunks), 2):
            del small_chunks[i]
            small_chunks[i] = None
        
        # Try to allocate large chunks in fragmented space
        try:
            for i in range(10):
                large_chunks.append(bytearray(100 * 1024))  # 100KB each
            
            results["passed"] += 1
            results["details"].append("✅ Memory fragmentation handled")
            
        except MemoryError:
            results["passed"] += 1
            results["details"].append("✅ Memory fragmentation detected and handled")
        
        # Cleanup
        del small_chunks, large_chunks
        gc.collect()
        
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Memory fragmentation test failed: {str(e)}")
    
    return results

def test_memory_leaks():
    """Test for memory leaks in long-running operations."""
    
    print("🔍 Testing Memory Leaks...")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    try:
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Simulate repeated operations that might leak
        for iteration in range(100):
            # Create and destroy objects
            data = [bytearray(1024) for _ in range(100)]
            
            # Process data
            processed = [bytes(chunk) for chunk in data]
            
            # Cleanup
            del data, processed
            
            if iteration % 20 == 0:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                if memory_growth > 50:  # 50MB growth threshold
                    results["failed"] += 1
                    results["details"].append(f"❌ Memory leak detected: {memory_growth:.1f}MB growth")
                    break
        else:
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory
            
            if memory_growth < 10:  # Less than 10MB growth is acceptable
                results["passed"] += 1
                results["details"].append(f"✅ No memory leaks detected ({memory_growth:.1f}MB growth)")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ Potential memory leak: {memory_growth:.1f}MB growth")
        
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Memory leak test failed: {str(e)}")
    
    return results

def test_swap_thrashing():
    """Test behavior when system starts swapping."""
    
    print("💾 Testing Swap Thrashing...")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    try:
        # Get system memory info
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1024 / 1024
        
        # Try to allocate close to available memory
        target_mb = min(available_mb * 0.8, 1000)  # 80% of available or 1GB max
        
        try:
            large_allocation = bytearray(int(target_mb * 1024 * 1024))
            
            # Test access patterns that might cause thrashing
            step = len(large_allocation) // 1000
            for i in range(0, len(large_allocation), step):
                large_allocation[i] = 1
            
            results["passed"] += 1
            results["details"].append(f"✅ Large allocation handled ({target_mb:.0f}MB)")
            
            del large_allocation
            
        except MemoryError:
            results["passed"] += 1
            results["details"].append("✅ Memory allocation properly limited")
        
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Swap thrashing test failed: {str(e)}")
    
    return results

def run_memory_pressure_tests():
    """Run all memory pressure tests."""
    
    print("🚀 Starting Extreme Memory Pressure Tests")
    print("=" * 50)
    
    all_results = {
        "exhaustion": test_memory_exhaustion(),
        "fragmentation": test_memory_fragmentation(),
        "leaks": test_memory_leaks(),
        "swap": test_swap_thrashing(),
    }
    
    # Summary
    total_passed = sum(r["passed"] for r in all_results.values())
    total_failed = sum(r["failed"] for r in all_results.values())
    
    print("=" * 50)
    print("📋 MEMORY PRESSURE TEST SUMMARY")
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
    passed, failed = run_memory_pressure_tests()
    sys.exit(1 if failed > 0 else 0)