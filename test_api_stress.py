#!/usr/bin/env python3
"""API endpoint stress testing."""

import time
import json
import threading
from typing import Dict, List, Any
from unittest.mock import MagicMock

class MockAPIServer:
    """Mock API server for testing."""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.active_connections = 0
        self.max_connections = 100
        
    def handle_request(self, endpoint: str, method: str, data: Dict = None) -> Dict:
        """Handle API request."""
        start_time = time.time()
        
        try:
            self.active_connections += 1
            self.request_count += 1
            
            # Simulate connection limit
            if self.active_connections > self.max_connections:
                raise Exception("Too many connections")
            
            # Simulate processing time
            time.sleep(0.001)  # 1ms processing time
            
            # Route request
            if endpoint == "/health":
                response = {"status": "healthy", "timestamp": time.time()}
            elif endpoint == "/analyze" and method == "POST":
                if not data or "image" not in data:
                    raise ValueError("Missing image data")
                response = {"result": "analysis_complete", "confidence": 0.95}
            elif endpoint == "/patients" and method == "GET":
                response = {"patients": [{"id": i, "name": f"Patient {i}"} for i in range(10)]}
            else:
                raise ValueError(f"Unknown endpoint: {method} {endpoint}")
            
            return {"status": 200, "data": response}
            
        except Exception as e:
            self.error_count += 1
            return {"status": 500, "error": str(e)}
        
        finally:
            self.active_connections -= 1
            elapsed = time.time() - start_time
            self.response_times.append(elapsed)

def test_concurrent_requests():
    """Test concurrent API requests."""
    print("Testing concurrent requests...")
    
    server = MockAPIServer()
    
    def make_request(endpoint, method, data=None):
        """Make API request."""
        return server.handle_request(endpoint, method, data)
    
    # Test concurrent health checks
    threads = []
    results = []
    
    def health_check():
        result = make_request("/health", "GET")
        results.append(result)
    
    # Start 50 concurrent requests
    for _ in range(50):
        thread = threading.Thread(target=health_check)
        threads.append(thread)
        thread.start()
    
    # Wait for all requests
    for thread in threads:
        thread.join()
    
    successful_requests = sum(1 for r in results if r["status"] == 200)
    
    print(f"  Concurrent requests: {len(threads)}")
    print(f"  Successful responses: {successful_requests}")
    print(f"  Error rate: {(len(results) - successful_requests) / len(results) * 100:.1f}%")
    
    return successful_requests >= len(threads) * 0.9  # 90% success rate

def test_rate_limiting():
    """Test API rate limiting."""
    print("Testing rate limiting...")
    
    class RateLimitedServer(MockAPIServer):
        def __init__(self, max_requests_per_second=10):
            super().__init__()
            self.max_rps = max_requests_per_second
            self.request_timestamps = []
        
        def handle_request(self, endpoint: str, method: str, data: Dict = None) -> Dict:
            current_time = time.time()
            
            # Clean old timestamps
            self.request_timestamps = [t for t in self.request_timestamps if current_time - t < 1.0]
            
            # Check rate limit
            if len(self.request_timestamps) >= self.max_rps:
                self.error_count += 1
                return {"status": 429, "error": "Rate limit exceeded"}
            
            self.request_timestamps.append(current_time)
            return super().handle_request(endpoint, method, data)
    
    server = RateLimitedServer(max_requests_per_second=5)
    
    # Make rapid requests
    results = []
    for i in range(20):
        result = server.handle_request("/health", "GET")
        results.append(result)
        time.sleep(0.05)  # 50ms between requests (20 RPS)
    
    rate_limited = sum(1 for r in results if r["status"] == 429)
    successful = sum(1 for r in results if r["status"] == 200)
    
    print(f"  Total requests: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Rate limited: {rate_limited}")
    
    return rate_limited > 0 and successful > 0

def test_payload_size_limits():
    """Test API payload size limits."""
    print("Testing payload size limits...")
    
    class PayloadLimitedServer(MockAPIServer):
        def __init__(self, max_payload_size=1024):  # 1KB limit
            super().__init__()
            self.max_payload_size = max_payload_size
        
        def handle_request(self, endpoint: str, method: str, data: Dict = None) -> Dict:
            if data:
                payload_size = len(json.dumps(data).encode('utf-8'))
                if payload_size > self.max_payload_size:
                    self.error_count += 1
                    return {"status": 413, "error": "Payload too large"}
            
            return super().handle_request(endpoint, method, data)
    
    server = PayloadLimitedServer(max_payload_size=100)  # 100 bytes
    
    # Test small payload
    small_data = {"image": "small_image_data"}
    small_result = server.handle_request("/analyze", "POST", small_data)
    
    # Test large payload
    large_data = {"image": "x" * 1000}  # 1000 character string
    large_result = server.handle_request("/analyze", "POST", large_data)
    
    small_accepted = small_result["status"] == 200
    large_rejected = large_result["status"] == 413
    
    print(f"  Small payload accepted: {small_accepted}")
    print(f"  Large payload rejected: {large_rejected}")
    
    return small_accepted and large_rejected

def test_response_time_monitoring():
    """Test API response time monitoring."""
    print("Testing response time monitoring...")
    
    server = MockAPIServer()
    
    # Make multiple requests
    for _ in range(100):
        server.handle_request("/health", "GET")
    
    if server.response_times:
        avg_response_time = sum(server.response_times) / len(server.response_times)
        max_response_time = max(server.response_times)
        min_response_time = min(server.response_times)
        
        print(f"  Requests: {len(server.response_times)}")
        print(f"  Avg response time: {avg_response_time*1000:.2f}ms")
        print(f"  Max response time: {max_response_time*1000:.2f}ms")
        print(f"  Min response time: {min_response_time*1000:.2f}ms")
        
        # Check if response times are reasonable
        reasonable_avg = avg_response_time < 0.1  # Less than 100ms
        reasonable_max = max_response_time < 0.5  # Less than 500ms
        
        return reasonable_avg and reasonable_max
    
    return False

def test_error_handling():
    """Test API error handling."""
    print("Testing error handling...")
    
    server = MockAPIServer()
    
    test_cases = [
        ("/nonexistent", "GET", None, 500),
        ("/analyze", "POST", None, 500),  # Missing data
        ("/analyze", "POST", {"wrong": "data"}, 500),  # Wrong data format
        ("/health", "GET", None, 200),  # Valid request
    ]
    
    results = []
    for endpoint, method, data, expected_status in test_cases:
        result = server.handle_request(endpoint, method, data)
        results.append((result["status"], expected_status))
    
    correct_responses = sum(1 for actual, expected in results if actual == expected)
    
    print(f"  Test cases: {len(test_cases)}")
    print(f"  Correct responses: {correct_responses}")
    
    return correct_responses == len(test_cases)

def test_connection_pooling():
    """Test connection pooling behavior."""
    print("Testing connection pooling...")
    
    server = MockAPIServer()
    server.max_connections = 10  # Low limit for testing
    
    def long_request():
        """Simulate long-running request."""
        time.sleep(0.1)  # 100ms request
        return server.handle_request("/health", "GET")
    
    # Start many concurrent long requests
    threads = []
    results = []
    
    def make_long_request():
        result = long_request()
        results.append(result)
    
    # Start 20 concurrent requests (more than connection limit)
    for _ in range(20):
        thread = threading.Thread(target=make_long_request)
        threads.append(thread)
        thread.start()
    
    # Wait for all requests
    for thread in threads:
        thread.join()
    
    successful = sum(1 for r in results if r["status"] == 200)
    connection_errors = sum(1 for r in results if "Too many connections" in r.get("error", ""))
    
    print(f"  Concurrent requests: {len(threads)}")
    print(f"  Successful: {successful}")
    print(f"  Connection errors: {connection_errors}")
    
    # Should have some connection errors due to limit
    return connection_errors > 0 and successful > 0

def test_memory_usage_under_load():
    """Test memory usage under API load."""
    print("Testing memory usage under load...")
    
    import gc
    
    def get_memory_usage():
        """Get approximate memory usage."""
        gc.collect()
        return len(gc.get_objects())
    
    server = MockAPIServer()
    
    # Baseline memory
    baseline_memory = get_memory_usage()
    
    # Generate load
    for _ in range(1000):
        server.handle_request("/health", "GET")
    
    # Check memory after load
    after_load_memory = get_memory_usage()
    
    # Force garbage collection
    gc.collect()
    after_gc_memory = get_memory_usage()
    
    memory_growth = after_load_memory - baseline_memory
    memory_after_gc = after_gc_memory - baseline_memory
    
    print(f"  Baseline memory objects: {baseline_memory}")
    print(f"  After load: {after_load_memory} (+{memory_growth})")
    print(f"  After GC: {after_gc_memory} (+{memory_after_gc})")
    
    # Memory should not grow excessively
    reasonable_growth = memory_growth < 1000
    gc_effective = memory_after_gc < memory_growth
    
    return reasonable_growth and gc_effective

def run_api_stress_tests():
    """Run all API stress tests."""
    print("🌐 API Endpoint Stress Testing")
    print("=" * 50)
    
    tests = [
        ("Concurrent Requests", test_concurrent_requests),
        ("Rate Limiting", test_rate_limiting),
        ("Payload Size Limits", test_payload_size_limits),
        ("Response Time Monitoring", test_response_time_monitoring),
        ("Error Handling", test_error_handling),
        ("Connection Pooling", test_connection_pooling),
        ("Memory Usage Under Load", test_memory_usage_under_load),
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
    print(f"API Stress Tests: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🏆 Excellent API stress handling!")
    else:
        print(f"⚠️ {len(tests) - passed} API stress issues found")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_api_stress_tests()
    exit(0 if success else 1)