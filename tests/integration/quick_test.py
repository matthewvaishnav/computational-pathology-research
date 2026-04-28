#!/usr/bin/env python3
"""
Quick Integration Test

A simplified test runner for rapid validation during development.
Runs essential tests only to provide fast feedback.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from test_full_workflow import IntegrationTestSuite


def quick_health_check(base_url: str = "http://localhost:8000") -> bool:
    """Quick health check to verify API is accessible."""
    
    print("🔍 Quick Health Check")
    print("=" * 30)
    
    suite = IntegrationTestSuite(base_url=base_url)
    
    # Wait for service
    if not suite.wait_for_service("/health", timeout=30):
        print("❌ API not accessible")
        return False
    
    # Run essential tests
    tests = [
        ("Health Check", suite.test_health_check),
        ("API Documentation", suite.test_api_documentation),
        ("Database Operations", suite.test_database_operations),
        ("Monitoring Endpoints", suite.test_monitoring_endpoints)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_method in tests:
        try:
            result = test_method()
            if result:
                passed += 1
                print(f"✅ {test_name}")
            else:
                print(f"❌ {test_name}")
        except Exception as e:
            print(f"💥 {test_name}: {e}")
    
    success_rate = (passed / total) * 100
    
    print(f"\n📊 Quick Test Results: {passed}/{total} passed ({success_rate:.1f}%)")
    
    if success_rate >= 75:
        print("✅ System appears healthy - ready for full testing")
        return True
    else:
        print("❌ System issues detected - check logs before full testing")
        return False


def quick_inference_test(base_url: str = "http://localhost:8000") -> bool:
    """Quick inference test with a single image."""
    
    print("\n🧠 Quick Inference Test")
    print("=" * 30)
    
    suite = IntegrationTestSuite(base_url=base_url)
    
    try:
        # Create a small test image
        test_image = suite.create_test_image(224, 224)
        
        start_time = time.time()
        
        # Upload for analysis
        files = {'file': ('quick_test.png', test_image, 'image/png')}
        response = suite.session.post(
            f"{base_url}/api/v1/analyze/upload",
            files=files,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code}")
            return False
        
        result = response.json()
        analysis_id = result.get('analysis_id')
        
        if not analysis_id:
            print("❌ No analysis ID returned")
            return False
        
        # Poll for completion (max 60 seconds)
        max_wait = 60
        poll_start = time.time()
        
        while time.time() - poll_start < max_wait:
            response = suite.session.get(
                f"{base_url}/api/v1/analyze/{analysis_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                status = result.get('status')
                
                if status == 'completed':
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    confidence = result.get('confidence_score', 0)
                    prediction = result.get('prediction_class', 'unknown')
                    
                    print(f"✅ Inference completed in {total_time:.2f}s")
                    print(f"📊 Prediction: {prediction} (confidence: {confidence:.3f})")
                    
                    if total_time < 30:  # 30 second threshold
                        print("✅ Performance within acceptable range")
                        return True
                    else:
                        print("⚠️ Performance slower than expected")
                        return False
                
                elif status == 'failed':
                    print(f"❌ Analysis failed: {result}")
                    return False
                
                else:
                    print(f"⏳ Status: {status}")
                    time.sleep(2)
            
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return False
        
        print("❌ Inference timeout")
        return False
        
    except Exception as e:
        print(f"❌ Inference test error: {e}")
        return False


def main():
    """Run quick integration tests."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Integration Test")
    parser.add_argument(
        '--base-url',
        default='http://localhost:8000',
        help='Base URL for API server'
    )
    parser.add_argument(
        '--inference',
        action='store_true',
        help='Include inference test (slower)'
    )
    
    args = parser.parse_args()
    
    print("🚀 Quick Integration Test Suite")
    print("=" * 50)
    print(f"🎯 Target: {args.base_url}")
    
    # Run health check
    health_ok = quick_health_check(args.base_url)
    
    # Run inference test if requested and health check passed
    inference_ok = True
    if args.inference and health_ok:
        inference_ok = quick_inference_test(args.base_url)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 QUICK TEST SUMMARY")
    print("=" * 50)
    
    print(f"🔍 Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    
    if args.inference:
        print(f"🧠 Inference Test: {'✅ PASS' if inference_ok else '❌ FAIL'}")
    
    overall_success = health_ok and inference_ok
    
    if overall_success:
        print("\n✅ QUICK TESTS PASSED - System ready for development/testing")
        exit_code = 0
    else:
        print("\n❌ QUICK TESTS FAILED - Check system before proceeding")
        exit_code = 1
    
    print("=" * 50)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()