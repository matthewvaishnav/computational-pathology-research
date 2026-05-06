#!/usr/bin/env python3
"""
PathologyFL Test Suite - Comprehensive testing of hierarchical medical FL
"""

import unittest
from pathology_fl_demo import PathologyFLDemo, HospitalMetadata, SlideQuality, HospitalType, CancerType

class TestPathologyFL(unittest.TestCase):
    """Test PathologyFL core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.demo = PathologyFLDemo()
        
        # Test hospitals
        self.cancer_center = HospitalMetadata(
            hospital_id="cancer_center",
            hospital_type=HospitalType.CANCER_CENTER,
            annual_cases=15000,
            cancer_specialties=[CancerType.BREAST, CancerType.LUNG],
            diagnostic_accuracy=0.96,
            years_experience=20
        )
        
        self.rural_hospital = HospitalMetadata(
            hospital_id="rural_hospital",
            hospital_type=HospitalType.RURAL_HOSPITAL,
            annual_cases=800,
            cancer_specialties=[CancerType.GENERAL],
            diagnostic_accuracy=0.82,
            years_experience=5
        )
        
        # Test slide qualities
        self.high_quality = SlideQuality(0.9, 0.85, 0.92, 0.1)
        self.low_quality = SlideQuality(0.6, 0.55, 0.65, 0.4)
    
    def test_expertise_weighting_hierarchy(self):
        """Test that cancer centers get higher weights than rural hospitals."""
        
        cancer_weight = self.demo.calculate_expertise_weight(
            self.cancer_center, CancerType.BREAST
        )
        rural_weight = self.demo.calculate_expertise_weight(
            self.rural_hospital, CancerType.BREAST
        )
        
        self.assertGreater(cancer_weight, rural_weight,
                          "Cancer center should have higher expertise weight")
        self.assertGreater(cancer_weight, 5.0, "Cancer center weight should be substantial")
        self.assertLess(rural_weight, 2.0, "Rural hospital weight should be limited")
    
    def test_specialty_bonus(self):
        """Test specialty bonus for relevant cancer types."""
        
        # Breast specialist for breast cancer
        breast_weight = self.demo.calculate_expertise_weight(
            self.cancer_center, CancerType.BREAST
        )
        
        # Same hospital for general pathology
        general_weight = self.demo.calculate_expertise_weight(
            self.cancer_center, CancerType.GENERAL
        )
        
        self.assertGreater(breast_weight, general_weight,
                          "Specialist should get bonus for their specialty")
    
    def test_quality_weighting(self):
        """Test slide quality affects weighting."""
        
        high_weight = self.demo.calculate_quality_weight(self.high_quality)
        low_weight = self.demo.calculate_quality_weight(self.low_quality)
        
        self.assertGreater(high_weight, low_weight,
                          "High quality slides should get higher weight")
        self.assertGreater(high_weight, 0.8, "High quality should be substantial")
        self.assertLess(low_weight, 0.7, "Low quality should be penalized")
    
    def test_volume_scaling(self):
        """Test volume scaling works correctly."""
        
        high_volume = HospitalMetadata(
            hospital_id="high_volume",
            hospital_type=HospitalType.COMMUNITY_HOSPITAL,
            annual_cases=10000,
            cancer_specialties=[CancerType.GENERAL],
            diagnostic_accuracy=0.85,
            years_experience=10
        )
        
        low_volume = HospitalMetadata(
            hospital_id="low_volume", 
            hospital_type=HospitalType.COMMUNITY_HOSPITAL,
            annual_cases=1000,
            cancer_specialties=[CancerType.GENERAL],
            diagnostic_accuracy=0.85,
            years_experience=10
        )
        
        high_weight = self.demo.calculate_expertise_weight(high_volume, CancerType.GENERAL)
        low_weight = self.demo.calculate_expertise_weight(low_volume, CancerType.GENERAL)
        
        self.assertGreater(high_weight, low_weight,
                          "Higher volume should increase weight")
    
    def test_accuracy_factor(self):
        """Test diagnostic accuracy affects weighting."""
        
        high_accuracy = HospitalMetadata(
            hospital_id="high_acc",
            hospital_type=HospitalType.COMMUNITY_HOSPITAL,
            annual_cases=5000,
            cancer_specialties=[CancerType.GENERAL],
            diagnostic_accuracy=0.95,
            years_experience=10
        )
        
        low_accuracy = HospitalMetadata(
            hospital_id="low_acc",
            hospital_type=HospitalType.COMMUNITY_HOSPITAL, 
            annual_cases=5000,
            cancer_specialties=[CancerType.GENERAL],
            diagnostic_accuracy=0.75,
            years_experience=10
        )
        
        high_weight = self.demo.calculate_expertise_weight(high_accuracy, CancerType.GENERAL)
        low_weight = self.demo.calculate_expertise_weight(low_accuracy, CancerType.GENERAL)
        
        self.assertGreater(high_weight, low_weight,
                          "Higher accuracy should increase weight")
    
    def test_experience_factor(self):
        """Test years of experience affects weighting."""
        
        experienced = HospitalMetadata(
            hospital_id="experienced",
            hospital_type=HospitalType.COMMUNITY_HOSPITAL,
            annual_cases=5000,
            cancer_specialties=[CancerType.GENERAL],
            diagnostic_accuracy=0.85,
            years_experience=20
        )
        
        inexperienced = HospitalMetadata(
            hospital_id="inexperienced",
            hospital_type=HospitalType.COMMUNITY_HOSPITAL,
            annual_cases=5000, 
            cancer_specialties=[CancerType.GENERAL],
            diagnostic_accuracy=0.85,
            years_experience=2
        )
        
        exp_weight = self.demo.calculate_expertise_weight(experienced, CancerType.GENERAL)
        inexp_weight = self.demo.calculate_expertise_weight(inexperienced, CancerType.GENERAL)
        
        self.assertGreater(exp_weight, inexp_weight,
                          "More experience should increase weight")
    
    def test_quality_components(self):
        """Test individual quality components."""
        
        # Test sharpness impact
        sharp = SlideQuality(0.9, 0.8, 0.8, 0.2)
        blurry = SlideQuality(0.5, 0.8, 0.8, 0.2)
        
        sharp_weight = self.demo.calculate_quality_weight(sharp)
        blurry_weight = self.demo.calculate_quality_weight(blurry)
        
        self.assertGreater(sharp_weight, blurry_weight, "Sharper images should score higher")
        
        # Test artifact impact
        clean = SlideQuality(0.8, 0.8, 0.8, 0.1)
        artifacts = SlideQuality(0.8, 0.8, 0.8, 0.5)
        
        clean_weight = self.demo.calculate_quality_weight(clean)
        artifact_weight = self.demo.calculate_quality_weight(artifacts)
        
        self.assertGreater(clean_weight, artifact_weight, "Clean slides should score higher")
    
    def test_weight_bounds(self):
        """Test weights are within reasonable bounds."""
        
        # Test expertise weights
        for hospital_type in HospitalType:
            for cancer_type in CancerType:
                metadata = HospitalMetadata(
                    hospital_id="test",
                    hospital_type=hospital_type,
                    annual_cases=5000,
                    cancer_specialties=[CancerType.GENERAL],
                    diagnostic_accuracy=0.85,
                    years_experience=10
                )
                
                weight = self.demo.calculate_expertise_weight(metadata, cancer_type)
                self.assertGreater(weight, 0, f"Weight should be positive for {hospital_type}")
                self.assertLess(weight, 20, f"Weight should be reasonable for {hospital_type}")
        
        # Test quality weights
        for sharpness in [0.1, 0.5, 0.9]:
            for consistency in [0.1, 0.5, 0.9]:
                quality = SlideQuality(sharpness, consistency, 0.8, 0.2)
                weight = self.demo.calculate_quality_weight(quality)
                self.assertGreaterEqual(weight, 0, "Quality weight should be non-negative")
                self.assertLessEqual(weight, 1, "Quality weight should not exceed 1")
    
    def test_cancer_type_differences(self):
        """Test different cancer types produce different weights."""
        
        breast_specialist = HospitalMetadata(
            hospital_id="breast_specialist",
            hospital_type=HospitalType.CANCER_CENTER,
            annual_cases=10000,
            cancer_specialties=[CancerType.BREAST],
            diagnostic_accuracy=0.94,
            years_experience=15
        )
        
        breast_weight = self.demo.calculate_expertise_weight(breast_specialist, CancerType.BREAST)
        lung_weight = self.demo.calculate_expertise_weight(breast_specialist, CancerType.LUNG)
        
        self.assertGreater(breast_weight, lung_weight,
                          "Specialist should get higher weight for their specialty")

def run_pathology_fl_tests():
    """Run comprehensive PathologyFL tests."""
    
    print("🧪 Running PathologyFL Test Suite")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPathologyFL)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    if result.wasSuccessful():
        print("\n✅ All PathologyFL tests PASSED!")
        print("🏆 PathologyFL is ready for production deployment!")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} tests failed")
        print("🔧 PathologyFL needs fixes before deployment")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_pathology_fl_tests()
    exit(0 if success else 1)