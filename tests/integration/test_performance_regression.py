#!/usr/bin/env python3
"""
Performance Regression Testing

Tests system performance benchmarks and detects performance regressions
across different system components including inference, API, and database operations.
"""

import asyncio
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psutil
import requests


class PerformanceRegressionTests:
    """Performance regression testing suite."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize performance regression tests."""
        self.base_url = base_url
        self.session = requests.Session()

        # Performance thresholds (in seconds unless specified)
        self.thresholds = {
            "api_response_time": 2.0,
            "inference_time": 30.0,
            "database_query_time": 1.0,
            "memory_usage_mb": 2048,
            "cpu_usage_percent": 80.0,
            "concurrent_requests": 10,
            "throughput_requests_per_second": 5.0,
        }

        # Results storage
        self.results = {}

        print(f"🏃 Performance Regression Tests initialized")
        print(f"📊 Performance thresholds: {self.thresholds}")

    def measure_system_resources(self) -> Dict[str, float]:
        """Measure current system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent

            # Network I/O
            network = psutil.net_io_counters()

            return {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
            }

        except Exception as e:
            print(f"❌ System resource measurement error: {e}")
            return {}

    def create_test_payload(self, size_kb: int = 100) -> bytes:
        """Create test payload of specified size."""
        # Create synthetic image data
        image_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Convert to bytes and pad to desired size
        base_size = image_data.nbytes
        padding_size = max(0, (size_kb * 1024) - base_size)
        padding = b"0" * padding_size

        return image_data.tobytes() + padding

    def test_api_response_times(self) -> Dict[str, float]:
        """Test API endpoint response times."""
        print("\n⚡ Testing API Response Times...")

        endpoints = [
            ("/health", "GET", None),
            ("/api/v1/system/status", "GET", None),
            ("/api/v1/cases", "GET", None),
            ("/metrics", "GET", None),
        ]

        results = {}

        for endpoint, method, payload in endpoints:
            times = []

            for i in range(5):  # 5 measurements per endpoint
                start_time = time.time()

                try:
                    if method == "GET":
                        response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)
                    elif method == "POST":
                        response = self.session.post(
                            f"{self.base_url}{endpoint}", json=payload, timeout=10
                        )

                    end_time = time.time()
                    response_time = end_time - start_time

                    if response.status_code < 400:
                        times.append(response_time)
                    else:
                        print(f"⚠️ {endpoint} returned {response.status_code}")

                except Exception as e:
                    print(f"❌ {endpoint} error: {e}")
                    continue

            if times:
                avg_time = statistics.mean(times)
                max_time = max(times)
                min_time = min(times)

                results[endpoint] = {
                    "avg_time": avg_time,
                    "max_time": max_time,
                    "min_time": min_time,
                    "measurements": len(times),
                }

                status = "✅" if avg_time < self.thresholds["api_response_time"] else "❌"
                print(f"{status} {endpoint}: {avg_time:.3f}s avg (max: {max_time:.3f}s)")
            else:
                print(f"❌ {endpoint}: No successful measurements")

        return results

    def test_inference_performance(self) -> Dict[str, float]:
        """Test model inference performance."""
        print("\n🧠 Testing Inference Performance...")

        # Create test image
        test_payload = self.create_test_payload(500)  # 500KB test image

        inference_times = []
        memory_usage = []

        for i in range(3):  # 3 inference tests
            # Measure memory before
            memory_before = self.measure_system_resources().get("memory_mb", 0)

            start_time = time.time()

            try:
                files = {"file": ("test_image.png", test_payload, "image/png")}
                response = self.session.post(
                    f"{self.base_url}/api/v1/analyze/upload", files=files, timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    analysis_id = result.get("analysis_id")

                    # Poll for completion
                    while True:
                        status_response = self.session.get(
                            f"{self.base_url}/api/v1/analyze/{analysis_id}", timeout=10
                        )

                        if status_response.status_code == 200:
                            status_result = status_response.json()
                            if status_result.get("status") == "completed":
                                break
                            elif status_result.get("status") == "failed":
                                raise Exception("Analysis failed")

                        time.sleep(1)

                    end_time = time.time()
                    inference_time = end_time - start_time
                    inference_times.append(inference_time)

                    # Measure memory after
                    memory_after = self.measure_system_resources().get("memory_mb", 0)
                    memory_usage.append(memory_after - memory_before)

                    print(f"✅ Inference {i+1}: {inference_time:.2f}s")

                else:
                    print(f"❌ Inference {i+1} failed: {response.status_code}")

            except Exception as e:
                print(f"❌ Inference {i+1} error: {e}")
                continue

        results = {}
        if inference_times:
            avg_inference_time = statistics.mean(inference_times)
            max_inference_time = max(inference_times)
            avg_memory_usage = statistics.mean(memory_usage) if memory_usage else 0

            results = {
                "avg_inference_time": avg_inference_time,
                "max_inference_time": max_inference_time,
                "avg_memory_usage_mb": avg_memory_usage,
                "successful_inferences": len(inference_times),
            }

            status = "✅" if avg_inference_time < self.thresholds["inference_time"] else "❌"
            print(f"{status} Average inference time: {avg_inference_time:.2f}s")
            print(f"📊 Memory usage per inference: {avg_memory_usage:.1f}MB")

        return results

    def test_concurrent_load(self) -> Dict[str, float]:
        """Test system performance under concurrent load."""
        print("\n🔄 Testing Concurrent Load Performance...")

        num_concurrent = self.thresholds["concurrent_requests"]
        test_payload = self.create_test_payload(100)  # 100KB per request

        def make_request(request_id: int) -> Tuple[int, float, bool]:
            """Make a single request and return timing info."""
            start_time = time.time()

            try:
                files = {"file": (f"test_{request_id}.png", test_payload, "image/png")}
                response = self.session.post(
                    f"{self.base_url}/api/v1/analyze/upload", files=files, timeout=30
                )

                end_time = time.time()
                response_time = end_time - start_time
                success = response.status_code == 200

                return request_id, response_time, success

            except Exception as e:
                end_time = time.time()
                response_time = end_time - start_time
                return request_id, response_time, False

        # Measure system resources before load test
        resources_before = self.measure_system_resources()

        # Execute concurrent requests
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_concurrent)]

            results_list = []
            for future in as_completed(futures):
                request_id, response_time, success = future.result()
                results_list.append((request_id, response_time, success))

        end_time = time.time()
        total_time = end_time - start_time

        # Measure system resources after load test
        resources_after = self.measure_system_resources()

        # Analyze results
        successful_requests = [r for r in results_list if r[2]]
        failed_requests = [r for r in results_list if not r[2]]

        response_times = [r[1] for r in successful_requests]

        results = {
            "total_requests": num_concurrent,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "total_time": total_time,
            "throughput_rps": len(successful_requests) / total_time if total_time > 0 else 0,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "cpu_usage_increase": resources_after.get("cpu_percent", 0)
            - resources_before.get("cpu_percent", 0),
            "memory_usage_increase_mb": resources_after.get("memory_mb", 0)
            - resources_before.get("memory_mb", 0),
        }

        success_rate = len(successful_requests) / num_concurrent * 100
        throughput_ok = (
            results["throughput_rps"] >= self.thresholds["throughput_requests_per_second"]
        )

        print(f"📊 Concurrent Load Results:")
        print(f"   Success Rate: {success_rate:.1f}% ({len(successful_requests)}/{num_concurrent})")
        print(
            f"   Throughput: {results['throughput_rps']:.2f} req/s {'✅' if throughput_ok else '❌'}"
        )
        print(f"   Avg Response Time: {results['avg_response_time']:.2f}s")
        print(f"   CPU Usage Increase: {results['cpu_usage_increase']:.1f}%")
        print(f"   Memory Usage Increase: {results['memory_usage_increase_mb']:.1f}MB")

        return results

    def test_database_performance(self) -> Dict[str, float]:
        """Test database query performance."""
        print("\n🗄️ Testing Database Performance...")

        database_tests = [
            ("health_check", "/api/v1/system/db-health"),
            ("case_list", "/api/v1/cases?limit=100"),
            ("analytics", "/api/v1/analytics/dashboard"),
            ("user_list", "/api/v1/admin/users?limit=50"),
        ]

        results = {}

        for test_name, endpoint in database_tests:
            query_times = []

            for i in range(5):  # 5 measurements per query
                start_time = time.time()

                try:
                    response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)
                    end_time = time.time()
                    query_time = end_time - start_time

                    if response.status_code == 200:
                        query_times.append(query_time)
                    else:
                        print(f"⚠️ {test_name} returned {response.status_code}")

                except Exception as e:
                    print(f"❌ {test_name} error: {e}")
                    continue

            if query_times:
                avg_time = statistics.mean(query_times)
                max_time = max(query_times)

                results[test_name] = {
                    "avg_query_time": avg_time,
                    "max_query_time": max_time,
                    "measurements": len(query_times),
                }

                status = "✅" if avg_time < self.thresholds["database_query_time"] else "❌"
                print(f"{status} {test_name}: {avg_time:.3f}s avg (max: {max_time:.3f}s)")

        return results

    def test_memory_leak_detection(self) -> Dict[str, float]:
        """Test for memory leaks during extended operation."""
        print("\n🔍 Testing Memory Leak Detection...")

        initial_memory = self.measure_system_resources().get("memory_mb", 0)
        memory_samples = [initial_memory]

        test_payload = self.create_test_payload(50)  # 50KB test payload

        # Perform 20 operations and monitor memory
        for i in range(20):
            try:
                files = {"file": (f"leak_test_{i}.png", test_payload, "image/png")}
                response = self.session.post(
                    f"{self.base_url}/api/v1/analyze/upload", files=files, timeout=30
                )

                # Sample memory every 5 operations
                if i % 5 == 0:
                    current_memory = self.measure_system_resources().get("memory_mb", 0)
                    memory_samples.append(current_memory)
                    print(f"📊 Operation {i+1}: {current_memory:.1f}MB")

                time.sleep(0.5)  # Brief pause between operations

            except Exception as e:
                print(f"⚠️ Memory leak test operation {i+1} failed: {e}")
                continue

        final_memory = self.measure_system_resources().get("memory_mb", 0)
        memory_samples.append(final_memory)

        # Analyze memory trend
        memory_increase = final_memory - initial_memory
        memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]  # Linear trend

        results = {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": memory_increase,
            "memory_trend_mb_per_operation": memory_trend,
            "operations_tested": 20,
            "memory_samples": len(memory_samples),
        }

        # Memory leak detection thresholds
        leak_detected = memory_increase > 100 or memory_trend > 5  # 100MB increase or 5MB/op trend

        status = "❌" if leak_detected else "✅"
        print(f"{status} Memory Analysis:")
        print(f"   Initial: {initial_memory:.1f}MB")
        print(f"   Final: {final_memory:.1f}MB")
        print(f"   Increase: {memory_increase:.1f}MB")
        print(f"   Trend: {memory_trend:.2f}MB per operation")

        if leak_detected:
            print("⚠️ Potential memory leak detected!")

        return results

    def run_all_performance_tests(self) -> Dict[str, Dict]:
        """Run all performance regression tests."""
        print("🚀 Starting Performance Regression Tests")
        print("=" * 60)

        # System info
        system_info = self.measure_system_resources()
        print(f"💻 System Status:")
        print(f"   CPU: {system_info.get('cpu_percent', 0):.1f}%")
        print(
            f"   Memory: {system_info.get('memory_mb', 0):.1f}MB ({system_info.get('memory_percent', 0):.1f}%)"
        )
        print(f"   Disk: {system_info.get('disk_percent', 0):.1f}%")

        tests = [
            ("API Response Times", self.test_api_response_times),
            ("Inference Performance", self.test_inference_performance),
            ("Concurrent Load", self.test_concurrent_load),
            ("Database Performance", self.test_database_performance),
            ("Memory Leak Detection", self.test_memory_leak_detection),
        ]

        all_results = {}

        for test_name, test_method in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")

            try:
                test_results = test_method()
                all_results[test_name] = test_results
                print(f"✅ {test_name}: Completed")

            except Exception as e:
                print(f"💥 {test_name}: ERROR - {e}")
                all_results[test_name] = {"error": str(e)}

        # Performance summary
        print("\n" + "=" * 60)
        print("🎯 PERFORMANCE REGRESSION SUMMARY")
        print("=" * 60)

        # Check for regressions
        regressions = []

        # API response time regressions
        api_results = all_results.get("API Response Times", {})
        for endpoint, metrics in api_results.items():
            if (
                isinstance(metrics, dict)
                and metrics.get("avg_time", 0) > self.thresholds["api_response_time"]
            ):
                regressions.append(
                    f"API {endpoint}: {metrics['avg_time']:.3f}s > {self.thresholds['api_response_time']}s"
                )

        # Inference time regressions
        inference_results = all_results.get("Inference Performance", {})
        if inference_results.get("avg_inference_time", 0) > self.thresholds["inference_time"]:
            regressions.append(
                f"Inference: {inference_results['avg_inference_time']:.2f}s > {self.thresholds['inference_time']}s"
            )

        # Throughput regressions
        load_results = all_results.get("Concurrent Load", {})
        if (
            load_results.get("throughput_rps", 0)
            < self.thresholds["throughput_requests_per_second"]
        ):
            regressions.append(
                f"Throughput: {load_results['throughput_rps']:.2f} < {self.thresholds['throughput_requests_per_second']} req/s"
            )

        if regressions:
            print("❌ PERFORMANCE REGRESSIONS DETECTED:")
            for regression in regressions:
                print(f"   • {regression}")
        else:
            print("✅ NO PERFORMANCE REGRESSIONS DETECTED")

        # Save results to file
        results_file = Path("tests/integration/performance_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(
                {
                    "timestamp": time.time(),
                    "system_info": system_info,
                    "thresholds": self.thresholds,
                    "results": all_results,
                    "regressions": regressions,
                },
                f,
                indent=2,
            )

        print(f"📊 Results saved to: {results_file}")

        return all_results


if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    test_suite = PerformanceRegressionTests(base_url=base_url)
    results = test_suite.run_all_performance_tests()

    # Exit with error code if regressions detected
    has_regressions = any("error" in str(result) for result in results.values())
    sys.exit(1 if has_regressions else 0)
