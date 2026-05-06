#!/usr/bin/env python3
"""
PathologyFL Edge Case Stress Test - Test extreme scenarios
"""

from pathology_fl_demo import PathologyFLDemo, HospitalMetadata, SlideQuality, HospitalType, CancerType

def test_pathology_fl_edge_cases():
    """Test PathologyFL under extreme conditions."""
    
    print("🔥 PathologyFL Edge Case Stress Test")
    print("=" * 50)
    
    demo = PathologyFLDemo()
    results = {"passed": 0, "failed": 0, "details": []}
    
    # Test 1: Extreme hospital configurations
    try:
        # Massive cancer center
        mega_center = HospitalMetadata(
            hospital_id="mega_center",
            hospital_type=HospitalType.CANCER_CENTER,
            annual_cases=100000,  # Extreme volume
            cancer_specialties=[CancerType.BREAST, CancerType.LUNG, CancerType.PROSTATE],
            diagnostic_accuracy=0.99,  # Near perfect
            years_experience=50  # Extreme experience
        )
        
        weight = demo.calculate_expertise_weight(mega_center, CancerType.BREAST)
        if 0 < weight < 50:  # Should be high but bounded
            results["passed"] += 1
            results["details"].append(f"✅ Mega center weight: {weight:.3f} (bounded)")
        else:
            results["failed"] += 1
            results["details"].append(f"❌ Mega center weight: {weight:.3f} (unbounded)")
    
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Mega center test failed: {str(e)}")
    
    # Test 2: Minimal hospital
    try:
        tiny_clinic = HospitalMetadata(
            hospital_id="tiny_clinic",
            hospital_type=HospitalType.RURAL_HOSPITAL,
            annual_cases=1,  # Minimal volume
            cancer_specialties=[],  # No specialties
            diagnostic_accuracy=0.5,  # Poor accuracy
            years_experience=0  # No experience
        )
        
        weight = demo.calculate_expertise_weight(tiny_clinic, CancerType.BREAST)
        if weight > 0:  # Should still be positive
            results["passed"] += 1
            results["details"].append(f"✅ Tiny clinic weight: {weight:.3f} (positive)")
        else:
            results["failed"] += 1
            results["details"].append(f"❌ Tiny clinic weight: {weight:.3f} (non-positive)")
    
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Tiny clinic test failed: {str(e)}")
    
    # Test 3: Perfect slide quality
    try:
        perfect_quality = SlideQuality(1.0, 1.0, 1.0, 0.0)
        weight = demo.calculate_quality_weight(perfect_quality)
        
        if 0.9 <= weight <= 1.0:
            results["passed"] += 1
            results["details"].append(f"✅ Perfect quality weight: {weight:.3f}")
        else:
            results["failed"] += 1
            results["details"].append(f"❌ Perfect quality weight: {weight:.3f} (out of range)")
    
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Perfect quality test failed: {str(e)}")
    
    # Test 4: Terrible slide quality
    try:
        terrible_quality = SlideQuality(0.0, 0.0, 0.0, 1.0)
        weight = demo.calculate_quality_weight(terrible_quality)
        
        if 0.0 <= weight <= 0.3:
            results["passed"] += 1
            results["details"].append(f"✅ Terrible quality weight: {weight:.3f}")
        else:
            results["failed"] += 1
            results["details"].append(f"❌ Terrible quality weight: {weight:.3f} (too high)")
    
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Terrible quality test failed: {str(e)}")
    
    # Test 5: All cancer types for all hospital types
    try:
        test_count = 0
        success_count = 0
        
        for hospital_type in HospitalType:
            for cancer_type in CancerType:
                test_hospital = HospitalMetadata(
                    hospital_id=f"test_{hospital_type.value}",
                    hospital_type=hospital_type,
                    annual_cases=5000,
                    cancer_specialties=[CancerType.GENERAL],
                    diagnostic_accuracy=0.85,
                    years_experience=10
                )
                
                weight = demo.calculate_expertise_weight(test_hospital, cancer_type)
                test_count += 1
                
                if weight > 0:
                    success_count += 1
        
        if success_count == test_count:
            results["passed"] += 1
            results["details"].append(f"✅ All combinations work: {success_count}/{test_count}")
        else:
            results["failed"] += 1
            results["details"].append(f"❌ Some combinations failed: {success_count}/{test_count}")
    
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Combination test failed: {str(e)}")
    
    # Test 6: Weight consistency
    try:
        hospital = HospitalMetadata(
            hospital_id="consistency_test",
            hospital_type=HospitalType.TEACHING_HOSPITAL,
            annual_cases=8000,
            cancer_specialties=[CancerType.BREAST],
            diagnostic_accuracy=0.92,
            years_experience=15
        )
        
        # Calculate weight multiple times
        weights = [demo.calculate_expertise_weight(hospital, CancerType.BREAST) for _ in range(5)]
        
        # All weights should be identical (deterministic)
        if all(w == weights[0] for w in weights):
            results["passed"] += 1
            results["details"].append(f"✅ Weight consistency: {weights[0]:.3f}")
        else:
            results["failed"] += 1
            results["details"].append(f"❌ Weight inconsistency: {weights}")
    
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Consistency test failed: {str(e)}")
    
    # Test 7: Specialty vs non-specialty comparison
    try:
        specialist = HospitalMetadata(
            hospital_id="specialist",
            hospital_type=HospitalType.CANCER_CENTER,
            annual_cases=10000,
            cancer_specialties=[CancerType.BREAST, CancerType.LUNG],
            diagnostic_accuracy=0.94,
            years_experience=18
        )
        
        breast_weight = demo.calculate_expertise_weight(specialist, CancerType.BREAST)
        prostate_weight = demo.calculate_expertise_weight(specialist, CancerType.PROSTATE)
        
        if breast_weight > prostate_weight:
            results["passed"] += 1
            results["details"].append(f"✅ Specialty bonus works: {breast_weight:.3f} > {prostate_weight:.3f}")
        else:
            results["failed"] += 1
            results["details"].append(f"❌ Specialty bonus failed: {breast_weight:.3f} <= {prostate_weight:.3f}")
    
    except Exception as e:
        results["failed"] += 1
        results["details"].append(f"❌ Specialty test failed: {str(e)}")
    
    # Summary
    print("\n📊 Edge Case Test Results:")
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    
    print("\n📋 Detailed Results:")
    for detail in results["details"]:
        print(f"  {detail}")
    
    if results["failed"] == 0:
        print("\n🏆 PathologyFL handles all edge cases correctly!")
        print("✅ Ready for production deployment with extreme scenarios")
    else:
        print(f"\n⚠️  {results['failed']} edge cases need attention")
        print("🔧 PathologyFL needs hardening for extreme scenarios")
    
    return results["failed"] == 0

def test_pathology_fl_scalability():
    """Test PathologyFL scalability with many hospitals."""
    
    print("\n🚀 PathologyFL Scalability Test")
    print("=" * 50)
    
    demo = PathologyFLDemo()
    
    # Create many hospitals
    hospitals = []
    for i in range(100):
        hospital = HospitalMetadata(
            hospital_id=f"hospital_{i}",
            hospital_type=list(HospitalType)[i % 4],
            annual_cases=1000 + (i * 100),
            cancer_specialties=[list(CancerType)[i % 4]],
            diagnostic_accuracy=0.7 + (i % 30) / 100,
            years_experience=1 + (i % 25)
        )
        hospitals.append(hospital)
    
    # Test weight calculation for all hospitals
    print(f"📊 Testing {len(hospitals)} hospitals...")
    
    total_weight = 0
    min_weight = float('inf')
    max_weight = 0
    
    for hospital in hospitals:
        weight = demo.calculate_expertise_weight(hospital, CancerType.BREAST)
        total_weight += weight
        min_weight = min(min_weight, weight)
        max_weight = max(max_weight, weight)
    
    avg_weight = total_weight / len(hospitals)
    
    print(f"  Total weight: {total_weight:.3f}")
    print(f"  Average weight: {avg_weight:.3f}")
    print(f"  Min weight: {min_weight:.3f}")
    print(f"  Max weight: {max_weight:.3f}")
    print(f"  Weight range: {max_weight - min_weight:.3f}")
    
    # Verify reasonable distribution
    if min_weight > 0 and max_weight < 20 and avg_weight > 0.5:
        print("✅ Scalability test PASSED - weights are well distributed")
        return True
    else:
        print("❌ Scalability test FAILED - weight distribution issues")
        return False

def main():
    """Run all PathologyFL stress tests."""
    
    print("🔥 PathologyFL Comprehensive Stress Testing")
    print("=" * 60)
    
    # Run edge case tests
    edge_case_success = test_pathology_fl_edge_cases()
    
    # Run scalability tests
    scalability_success = test_pathology_fl_scalability()
    
    # Overall result
    print("\n" + "=" * 60)
    print("🏁 Final Results:")
    
    if edge_case_success and scalability_success:
        print("✅ ALL STRESS TESTS PASSED!")
        print("🏆 PathologyFL is production-ready for extreme scenarios")
        print("🚀 Ready for deployment with 100+ hospitals")
        return True
    else:
        print("❌ Some stress tests failed")
        print("🔧 PathologyFL needs additional hardening")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)