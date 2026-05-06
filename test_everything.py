#!/usr/bin/env python3
"""Comprehensive test runner for all HistoCore stress tests."""

import sys
import time
import subprocess
from pathlib import Path

def run_test_suite(test_file: str, description: str) -> bool:
    """Run a test suite and return success status."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=300)
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description}: PASSED ({elapsed:.1f}s)")
            return True
        else:
            print(f"❌ {description}: FAILED ({elapsed:.1f}s)")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description}: TIMEOUT (>300s)")
        return False
    except Exception as e:
        print(f"💥 {description}: ERROR - {e}")
        return False

def main():
    """Run all comprehensive stress tests."""
    print("🚀 HistoCore Comprehensive Stress Testing Suite")
    print("=" * 60)
    print("Testing every edge case, corner case, and production scenario")
    print("=" * 60)
    
    # Define all test suites
    test_suites = [
        ("basic_stress_test.py", "Basic System Stress Tests"),
        ("test_cli_edge_cases.py", "CLI Edge Case Testing"),
        ("test_property_based.py", "Property-Based Testing"),
        ("test_training_stress.py", "Training Pipeline Stress"),
        ("test_federated_network_failures.py", "Federated Learning Network Failures"),
        ("test_pacs_dicom_edge_cases.py", "PACS DICOM Edge Cases"),
        ("test_security_penetration.py", "Security Penetration Testing"),
        ("test_memory_pressure.py", "Memory Pressure Testing"),
        ("test_concurrency_chaos.py", "Concurrency Chaos Testing"),
        ("test_filesystem_edge.py", "File System Edge Cases"),
        ("test_malicious_fuzzing.py", "Malicious Input Fuzzing"),
        ("test_performance_regression.py", "Performance Regression Testing"),
        ("test_cross_platform.py", "Cross-Platform Compatibility"),
        ("test_database_corruption.py", "Database Corruption Recovery"),
        ("test_production_load.py", "Production Load Simulation"),
        ("test_hybrid_architecture.py", "Hybrid Architecture Integration"),
    ]
    
    # Run all test suites
    passed = 0
    total_start = time.time()
    
    for test_file, description in test_suites:
        if Path(test_file).exists():
            if run_test_suite(test_file, description):
                passed += 1
        else:
            print(f"⚠️ {description}: SKIPPED (file not found)")
    
    total_elapsed = time.time() - total_start
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE STRESS TEST RESULTS")
    print("=" * 60)
    
    print(f"Total Test Suites: {len(test_suites)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(test_suites) - passed}")
    print(f"Success Rate: {passed/len(test_suites)*100:.1f}%")
    print(f"Total Time: {total_elapsed/60:.1f} minutes")
    
    if passed == len(test_suites):
        print("\n🏆 ALL STRESS TESTS PASSED!")
        print("🚀 HistoCore is production-ready and bulletproof!")
        print("💪 Ready for high-paying job applications!")
    elif passed >= len(test_suites) * 0.9:
        print(f"\n🥈 EXCELLENT: {passed}/{len(test_suites)} tests passed!")
        print("🚀 HistoCore is highly robust with minor issues to address")
    elif passed >= len(test_suites) * 0.8:
        print(f"\n🥉 GOOD: {passed}/{len(test_suites)} tests passed!")
        print("⚠️ Some stress test failures need attention")
    else:
        print(f"\n❌ NEEDS WORK: {passed}/{len(test_suites)} tests passed!")
        print("🔧 Significant issues found - requires debugging")
    
    return passed == len(test_suites)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)