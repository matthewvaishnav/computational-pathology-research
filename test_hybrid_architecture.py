#!/usr/bin/env python3
"""
Hybrid Architecture Integration Test
Python ML + Rust FL Coordinator + Go Hospital Service
"""

import json
import time
import requests
import subprocess
import threading
from typing import Dict, Any
from rust_python_client import RustPathologyFLClient, AggregationRequest, ModelUpdate, SlideQuality

class HybridArchitectureTest:
    """Test the complete hybrid architecture."""
    
    def __init__(self):
        self.rust_client = RustPathologyFLClient()
        self.go_service_url = "http://localhost:8081"
        self.services_running = False
    
    def check_go_service(self) -> bool:
        """Check if Go hospital service is running."""
        try:
            response = requests.get(f"{self.go_service_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def register_hospital_with_go_service(self, hospital_data: Dict[str, Any]) -> bool:
        """Register hospital with Go service."""
        try:
            response = requests.post(
                f"{self.go_service_url}/api/hospitals/register",
                json=hospital_data,
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to register hospital: {e}")
            return False
    
    def get_hospitals_from_go_service(self) -> Dict[str, Any]:
        """Get hospital list from Go service."""
        try:
            response = requests.get(f"{self.go_service_url}/api/hospitals", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}
    
    def test_go_hospital_service(self) -> bool:
        """Test Go hospital registry service."""
        print("🏥 Testing Go Hospital Registry Service")
        print("-" * 40)
        
        if not self.check_go_service():
            print("❌ Go hospital service not running")
            print("💡 Start with: cd go_hospital_service && go run main.go")
            return False
        
        print("✅ Go hospital service is running")
        
        # Test hospital registration
        test_hospital = {
            "hospital_id": "test_hospital",
            "hospital_type": "teaching_hospital",
            "annual_cases": 8000,
            "cancer_specialties": ["breast", "lung"],
            "diagnostic_accuracy": 0.91,
            "years_experience": 12
        }
        
        if self.register_hospital_with_go_service(test_hospital):
            print("✅ Hospital registration successful")
        else:
            print("❌ Hospital registration failed")
            return False
        
        # Test hospital listing
        hospitals = self.get_hospitals_from_go_service()
        if hospitals and hospitals.get("count", 0) > 0:
            print(f"✅ Hospital listing successful ({hospitals['count']} hospitals)")
        else:
            print("❌ Hospital listing failed")
            return False
        
        return True
    
    def test_rust_coordinator(self) -> bool:
        """Test Rust FL coordinator."""
        print("\n🦀 Testing Rust FL Coordinator")
        print("-" * 40)
        
        try:
            # Create test model updates
            model_updates = [
                ModelUpdate(
                    hospital_id="mayo_clinic",
                    parameters={
                        "layer1.weight": [0.1, 0.2, 0.3],
                        "layer1.bias": [0.01, 0.02]
                    },
                    quality_metrics=SlideQuality(0.9, 0.85, 0.92, 0.1)
                ),
                ModelUpdate(
                    hospital_id="community_hospital",
                    parameters={
                        "layer1.weight": [0.15, 0.25, 0.35],
                        "layer1.bias": [0.015, 0.025]
                    },
                    quality_metrics=SlideQuality(0.78, 0.72, 0.81, 0.25)
                )
            ]
            
            request = AggregationRequest(
                round_number=1,
                cancer_type="breast",
                model_updates=model_updates
            )
            
            response = self.rust_client.send_aggregation_request(request)
            
            print("✅ Rust coordinator communication successful")
            print(f"   Aggregated {len(response['aggregated_parameters'])} parameter sets")
            print(f"   Hospital weights: {len(response['hospital_weights'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Rust coordinator test failed: {e}")
            print("💡 Start with: cd rust_coordinator && cargo run")
            return False
    
    def test_end_to_end_integration(self) -> bool:
        """Test complete end-to-end integration."""
        print("\n🔗 Testing End-to-End Integration")
        print("-" * 40)
        
        # Get hospitals from Go service
        hospitals_data = self.get_hospitals_from_go_service()
        if not hospitals_data or hospitals_data.get("count", 0) == 0:
            print("❌ No hospitals available from Go service")
            return False
        
        hospitals = hospitals_data["hospitals"]
        print(f"✅ Retrieved {len(hospitals)} hospitals from Go service")
        
        # Create model updates for each hospital
        model_updates = []
        for hospital in hospitals[:2]:  # Limit to first 2 for testing
            hospital_id = hospital["hospital_id"]
            
            # Create mock model parameters
            parameters = {
                "layer1.weight": [0.1 + i * 0.01 for i in range(100)],
                "layer1.bias": [0.01 + i * 0.001 for i in range(10)],
            }
            
            # Create quality metrics based on hospital type
            if hospital["hospital_type"] == "cancer_center":
                quality = SlideQuality(0.92, 0.88, 0.94, 0.08)
            else:
                quality = SlideQuality(0.78, 0.72, 0.81, 0.25)
            
            model_updates.append(ModelUpdate(
                hospital_id=hospital_id,
                parameters=parameters,
                quality_metrics=quality
            ))
        
        # Send to Rust coordinator for aggregation
        request = AggregationRequest(
            round_number=1,
            cancer_type="breast",
            model_updates=model_updates
        )
        
        try:
            response = self.rust_client.send_aggregation_request(request)
            
            print("✅ End-to-end integration successful")
            print(f"   Processed {len(model_updates)} hospital updates")
            print(f"   Aggregated {len(response['aggregated_parameters'])} parameter sets")
            
            # Show hospital weights
            weights = response['hospital_weights']
            for hospital_id, weight in weights.items():
                print(f"   {hospital_id}: {weight:.3f} weight")
            
            return True
            
        except Exception as e:
            print(f"❌ End-to-end integration failed: {e}")
            return False
    
    def benchmark_hybrid_performance(self) -> bool:
        """Benchmark hybrid architecture performance."""
        print("\n⚡ Benchmarking Hybrid Architecture")
        print("-" * 40)
        
        # Test different scales
        scales = [10, 50, 100]
        
        for scale in scales:
            print(f"\nTesting {scale} hospitals...")
            
            # Create model updates
            model_updates = []
            for i in range(scale):
                hospital_id = f"hospital_{i}"
                
                parameters = {
                    "layer1.weight": [0.1 + j * 0.001 for j in range(1000)],
                    "layer1.bias": [0.01 + j * 0.0001 for j in range(100)],
                }
                
                quality = SlideQuality(
                    image_sharpness=0.7 + (i % 30) / 100,
                    stain_consistency=0.6 + (i % 40) / 100,
                    label_confidence=0.8 + (i % 20) / 100,
                    artifact_level=(i % 25) / 100
                )
                
                model_updates.append(ModelUpdate(
                    hospital_id=hospital_id,
                    parameters=parameters,
                    quality_metrics=quality
                ))
            
            # Benchmark aggregation
            request = AggregationRequest(
                round_number=1,
                cancer_type="breast",
                model_updates=model_updates
            )
            
            try:
                start_time = time.time()
                response = self.rust_client.send_aggregation_request(request)
                elapsed = time.time() - start_time
                
                throughput = scale / elapsed
                
                print(f"  ✅ {scale} hospitals: {elapsed:.4f}s ({throughput:.0f} hospitals/sec)")
                
            except Exception as e:
                print(f"  ❌ {scale} hospitals: Failed - {e}")
                return False
        
        return True
    
    def run_all_tests(self) -> bool:
        """Run all hybrid architecture tests."""
        print("🚀 Hybrid Architecture Integration Testing")
        print("=" * 60)
        print("🐍 Python ML + 🦀 Rust FL + 🐹 Go Services")
        print("=" * 60)
        
        # Test individual services
        go_test = self.test_go_hospital_service()
        rust_test = self.test_rust_coordinator()
        
        if not (go_test and rust_test):
            print("\n❌ Individual service tests failed")
            print("💡 Make sure all services are running:")
            print("   - Go: cd go_hospital_service && go run main.go")
            print("   - Rust: cd rust_coordinator && cargo run")
            return False
        
        # Test integration
        integration_test = self.test_end_to_end_integration()
        
        if not integration_test:
            print("\n❌ Integration test failed")
            return False
        
        # Benchmark performance
        performance_test = self.benchmark_hybrid_performance()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 HYBRID ARCHITECTURE TEST RESULTS")
        print("=" * 60)
        
        tests = [
            ("Go Hospital Service", go_test),
            ("Rust FL Coordinator", rust_test),
            ("End-to-End Integration", integration_test),
            ("Performance Benchmark", performance_test)
        ]
        
        passed = 0
        for test_name, result in tests:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
            if result:
                passed += 1
        
        print(f"\nHybrid Tests: {passed}/{len(tests)} passed")
        
        if passed == len(tests):
            print("\n🏆 Hybrid architecture fully operational!")
            print("🚀 Production-grade multi-language system ready")
            print("⚡ Python ML + Rust performance + Go microservices")
        else:
            print(f"\n⚠️ {len(tests) - passed} tests need attention")
        
        return passed == len(tests)

def main():
    """Run hybrid architecture tests."""
    tester = HybridArchitectureTest()
    success = tester.run_all_tests()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()