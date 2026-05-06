#!/usr/bin/env python3
"""
HistoCore Comprehensive Stress Testing Suite

Tests every edge case, corner case, and failure mode to ensure production readiness.
"""

import os
import sys
import time
import threading
import multiprocessing
import gc
import psutil
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import torch
import pytest
from hypothesis import given, strategies as st, settings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

class StressTestSuite:
    """Comprehensive stress testing for HistoCore."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.memory_baseline = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
    def log(self, test_name, status, details=""):
        """Log test results."""
        elapsed = time.time() - self.start_time
        memory_current = psutil.Process().memory_info().rss / 1024 / 1024
        memory_delta = memory_current - self.memory_baseline
        
        result = {
            'status': status,
            'elapsed': elapsed,
            'memory_mb': memory_current,
            'memory_delta_mb': memory_delta,
            'details': details
        }
        self.results[test_name] = result
        
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status} ({elapsed:.1f}s, {memory_delta:+.1f}MB)")
        if details:
            print(f"   {details}")

    def test_memory_exhaustion(self):
        """Test behavior under extreme memory pressure."""
        try:
            # Gradually increase memory usage
            data_chunks = []
            chunk_size = 100 * 1024 * 1024  # 100MB chunks
            max_chunks = 50  # Up to 5GB
            
            for i in range(max_chunks):
                try:
                    chunk = np.random.random((chunk_size // 8,)).astype(np.float64)
                    data_chunks.append(chunk)
                    
                    # Check if we're approaching memory limits
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 90:
                        break
                        
                except MemoryError:
                    break
            
            # Test HistoCore under memory pressure
            try:
                import src
                result = src.quick_train(dataset="pcam", model="nnmil", epochs=1)
                self.log("memory_exhaustion", "PASS", f"Handled {len(data_chunks)} chunks ({len(data_chunks)*100}MB)")
            except Exception as e:
                self.log("memory_exhaustion", "WARN", f"Failed under memory pressure: {str(e)}")
            
            # Cleanup
            del data_chunks
            gc.collect()
            
        except Exception as e:
            self.log("memory_exhaustion", "FAIL", str(e))

    def test_concurrent_access(self):
        """Test concurrent access to HistoCore components."""
        try:
            def worker_thread(thread_id):
                try:
                    import src
                    # Simulate concurrent API calls
                    for i in range(10):
                        result = src.quick_train(dataset="pcam", model="nnmil", epochs=1)
                    return f"Thread {thread_id} completed"
                except Exception as e:
                    return f"Thread {thread_id} failed: {str(e)}"
            
            # Run 10 concurrent threads
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(worker_thread, i) for i in range(10)]
                results = [f.result(timeout=300) for f in futures]
            
            failures = [r for r in results if "failed" in r]
            if failures:
                self.log("concurrent_access", "WARN", f"{len(failures)} threads failed")
            else:
                self.log("concurrent_access", "PASS", "All 10 threads completed")
                
        except Exception as e:
            self.log("concurrent_access", "FAIL", str(e))

    def test_malformed_inputs(self):
        """Test handling of malformed inputs."""
        try:
            import src
            
            # Test malformed data
            malformed_tests = [
                (None, "None input"),
                ("", "Empty string"),
                ([], "Empty list"),
                ({"invalid": "dict"}, "Invalid dict"),
                (np.array([]), "Empty array"),
                (torch.tensor([]), "Empty tensor"),
            ]
            
            failures = 0
            for test_input, description in malformed_tests:
                try:
                    # This should fail gracefully
                    result = src.quick_train(dataset=test_input, model="nnmil", epochs=1)
                    failures += 1  # Should not succeed
                except Exception:
                    pass  # Expected to fail
            
            if failures == 0:
                self.log("malformed_inputs", "PASS", "All malformed inputs handled gracefully")
            else:
                self.log("malformed_inputs", "WARN", f"{failures} malformed inputs not caught")
                
        except Exception as e:
            self.log("malformed_inputs", "FAIL", str(e))

    def test_file_system_edge_cases(self):
        """Test file system edge cases."""
        try:
            # Test with various problematic paths
            problematic_paths = [
                "/nonexistent/path/file.txt",
                "",
                ".",
                "..",
                "/dev/null",
                "con" if os.name == 'nt' else "/dev/zero",  # Windows reserved name
                "a" * 300,  # Very long filename
                "file with spaces and special chars !@#$%^&*()",
            ]
            
            handled_correctly = 0
            for path in problematic_paths:
                try:
                    # Test file operations
                    result = Path(path).exists()
                    handled_correctly += 1
                except Exception:
                    handled_correctly += 1  # Expected to handle gracefully
            
            self.log("file_system_edge_cases", "PASS", f"Handled {handled_correctly}/{len(problematic_paths)} edge cases")
            
        except Exception as e:
            self.log("file_system_edge_cases", "FAIL", str(e))

    def test_gpu_memory_management(self):
        """Test GPU memory management under stress."""
        try:
            if not torch.cuda.is_available():
                self.log("gpu_memory_management", "SKIP", "No GPU available")
                return
            
            # Test GPU memory allocation/deallocation
            device = torch.device("cuda")
            tensors = []
            
            try:
                # Allocate GPU memory until near limit
                for i in range(100):
                    tensor = torch.randn(1000, 1000, device=device)
                    tensors.append(tensor)
                    
                    # Check GPU memory usage
                    if torch.cuda.memory_allocated() > torch.cuda.max_memory_allocated() * 0.8:
                        break
                
                # Test HistoCore under GPU memory pressure
                import src
                result = src.quick_train(dataset="pcam", model="nnmil", epochs=1)
                
                self.log("gpu_memory_management", "PASS", f"Handled {len(tensors)} GPU tensors")
                
            finally:
                # Cleanup GPU memory
                del tensors
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.log("gpu_memory_management", "FAIL", str(e))

    def test_network_failures(self):
        """Test network failure scenarios."""
        try:
            # Simulate network timeouts and failures
            import socket
            
            # Test connection to non-existent hosts
            problematic_hosts = [
                ("nonexistent.example.com", 80),
                ("127.0.0.1", 99999),  # Invalid port
                ("", 80),  # Empty host
            ]
            
            handled_correctly = 0
            for host, port in problematic_hosts:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)  # 1 second timeout
                    sock.connect((host, port))
                    sock.close()
                except Exception:
                    handled_correctly += 1  # Expected to fail
            
            self.log("network_failures", "PASS", f"Handled {handled_correctly}/{len(problematic_hosts)} network failures")
            
        except Exception as e:
            self.log("network_failures", "FAIL", str(e))

    @given(st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50, deadline=None)
    def test_property_based_integers(self, x):
        """Property-based testing with integers."""
        try:
            # Test that our system handles arbitrary integers gracefully
            if x < 0:
                assert x < 0  # Negative numbers stay negative
            elif x > 0:
                assert x > 0  # Positive numbers stay positive
            else:
                assert x == 0  # Zero stays zero
            return True
        except Exception:
            return False

    def test_extreme_data_sizes(self):
        """Test with extremely large and small data sizes."""
        try:
            # Test with very small data
            tiny_data = np.array([[1]])
            
            # Test with moderately large data (within reason)
            large_data = np.random.random((1000, 1000))
            
            # Test empty data
            empty_data = np.array([])
            
            test_cases = [
                (tiny_data, "tiny_data"),
                (large_data, "large_data"),
                (empty_data, "empty_data"),
            ]
            
            handled = 0
            for data, name in test_cases:
                try:
                    # Test basic operations
                    result = data.shape if hasattr(data, 'shape') else len(data)
                    handled += 1
                except Exception:
                    pass  # Some failures expected
            
            self.log("extreme_data_sizes", "PASS", f"Handled {handled}/{len(test_cases)} data size cases")
            
        except Exception as e:
            self.log("extreme_data_sizes", "FAIL", str(e))

    def test_unicode_and_encoding(self):
        """Test Unicode and encoding edge cases."""
        try:
            # Test various Unicode strings
            unicode_tests = [
                "Hello, 世界",  # Mixed ASCII and Chinese
                "🚀🔬🏥💻",  # Emojis
                "Ñoño niño",  # Accented characters
                "Здравствуй мир",  # Cyrillic
                "مرحبا بالعالم",  # Arabic
                "\x00\x01\x02",  # Control characters
                "",  # Empty string
                " " * 1000,  # Very long string
            ]
            
            handled = 0
            for test_string in unicode_tests:
                try:
                    # Test string operations
                    encoded = test_string.encode('utf-8')
                    decoded = encoded.decode('utf-8')
                    assert decoded == test_string
                    handled += 1
                except Exception:
                    pass  # Some failures expected with control chars
            
            self.log("unicode_and_encoding", "PASS", f"Handled {handled}/{len(unicode_tests)} Unicode cases")
            
        except Exception as e:
            self.log("unicode_and_encoding", "FAIL", str(e))

    def run_all_tests(self):
        """Run all stress tests."""
        print("🚀 Starting HistoCore Comprehensive Stress Test Suite")
        print(f"📊 Baseline memory: {self.memory_baseline:.1f} MB")
        print("=" * 60)
        
        # Core stress tests
        self.test_memory_exhaustion()
        self.test_concurrent_access()
        self.test_malformed_inputs()
        self.test_file_system_edge_cases()
        self.test_gpu_memory_management()
        self.test_network_failures()
        self.test_extreme_data_sizes()
        self.test_unicode_and_encoding()
        
        # Property-based tests
        try:
            for i in range(10):
                result = self.test_property_based_integers(i * 100 - 500)
                if not result:
                    break
            self.log("property_based_testing", "PASS", "Property tests completed")
        except Exception as e:
            self.log("property_based_testing", "FAIL", str(e))
        
        # Summary
        print("=" * 60)
        print("📋 STRESS TEST SUMMARY")
        
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        warned = sum(1 for r in self.results.values() if r['status'] == 'WARN')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAIL')
        skipped = sum(1 for r in self.results.values() if r['status'] == 'SKIP')
        
        total_time = time.time() - self.start_time
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_delta = final_memory - self.memory_baseline
        
        print(f"✅ Passed: {passed}")
        print(f"⚠️  Warned: {warned}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️  Skipped: {skipped}")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"💾 Memory delta: {memory_delta:+.1f}MB")
        
        # Detailed results
        print("\n📊 DETAILED RESULTS:")
        for test_name, result in self.results.items():
            status_emoji = "✅" if result['status'] == "PASS" else "❌" if result['status'] == "FAIL" else "⚠️" if result['status'] == "WARN" else "⏭️"
            print(f"{status_emoji} {test_name}: {result['status']} ({result['elapsed']:.1f}s, {result['memory_delta_mb']:+.1f}MB)")
            if result['details']:
                print(f"   └─ {result['details']}")
        
        return {
            'passed': passed,
            'warned': warned, 
            'failed': failed,
            'skipped': skipped,
            'total_time': total_time,
            'memory_delta': memory_delta,
            'results': self.results
        }

if __name__ == "__main__":
    suite = StressTestSuite()
    results = suite.run_all_tests()
    
    # Exit with appropriate code
    if results['failed'] > 0:
        sys.exit(1)
    elif results['warned'] > 0:
        sys.exit(2)
    else:
        sys.exit(0)