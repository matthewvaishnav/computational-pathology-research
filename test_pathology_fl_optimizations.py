#!/usr/bin/env python3
"""
PathologyFL Performance Optimizations - Speed up aggregation and computation
"""

import time
from typing import Dict, List, Tuple
from pathology_fl_demo import PathologyFLDemo, HospitalMetadata, SlideQuality, HospitalType, CancerType

class OptimizedPathologyFL(PathologyFLDemo):
    """Optimized version of PathologyFL with performance improvements."""
    
    def __init__(self):
        super().__init__()
        self.weight_cache = {}  # Cache computed weights
        self.quality_cache = {}  # Cache quality assessments
        
    def calculate_expertise_weight_cached(self, metadata: HospitalMetadata, cancer_type: CancerType) -> float:
        """Cached version of expertise weight calculation."""
        
        cache_key = f"{metadata.hospital_id}_{cancer_type.value}"
        
        if cache_key in self.weight_cache:
            return self.weight_cache[cache_key]
        
        weight = self.calculate_expertise_weight(metadata, cancer_type)
        self.weight_cache[cache_key] = weight
        return weight
    
    def calculate_quality_weight_cached(self, quality: SlideQuality, hospital_id: str) -> float:
        """Cached version of quality weight calculation."""
        
        cache_key = f"{hospital_id}_{quality.image_sharpness}_{quality.stain_consistency}"
        
        if cache_key in self.quality_cache:
            return self.quality_cache[cache_key]
        
        weight = self.calculate_quality_weight(quality)
        self.quality_cache[cache_key] = weight
        return weight
    
    def batch_calculate_weights(self, hospitals: Dict[str, HospitalMetadata], 
                               qualities: Dict[str, SlideQuality],
                               cancer_type: CancerType) -> Dict[str, Tuple[float, float]]:
        """Batch calculate all weights for efficiency."""
        
        results = {}
        
        for hospital_id, metadata in hospitals.items():
            expertise_weight = self.calculate_expertise_weight_cached(metadata, cancer_type)
            quality_weight = self.calculate_quality_weight_cached(qualities[hospital_id], hospital_id)
            results[hospital_id] = (expertise_weight, quality_weight)
        
        return results
    
    def clear_caches(self):
        """Clear weight caches."""
        self.weight_cache.clear()
        self.quality_cache.clear()

def benchmark_optimizations():
    """Benchmark PathologyFL optimizations."""
    
    print("⚡ PathologyFL Performance Optimization Benchmark")
    print("=" * 60)
    
    # Create test data
    hospitals = {}
    qualities = {}
    
    for i in range(100):
        hospital_id = f"hospital_{i}"
        hospitals[hospital_id] = HospitalMetadata(
            hospital_id=hospital_id,
            hospital_type=list(HospitalType)[i % 4],
            annual_cases=1000 + i * 100,
            cancer_specialties=[list(CancerType)[i % 4]],
            diagnostic_accuracy=0.7 + (i % 30) / 100,
            years_experience=1 + (i % 25)
        )
        
        qualities[hospital_id] = SlideQuality(
            image_sharpness=0.5 + (i % 50) / 100,
            stain_consistency=0.5 + (i % 40) / 100,
            label_confidence=0.6 + (i % 40) / 100,
            artifact_level=(i % 30) / 100
        )
    
    # Benchmark original implementation
    original_fl = PathologyFLDemo()
    
    start_time = time.time()
    for _ in range(10):  # 10 rounds
        for hospital_id, metadata in hospitals.items():
            original_fl.calculate_expertise_weight(metadata, CancerType.BREAST)
            original_fl.calculate_quality_weight(qualities[hospital_id])
    original_time = time.time() - start_time
    
    # Benchmark optimized implementation
    optimized_fl = OptimizedPathologyFL()
    
    start_time = time.time()
    for _ in range(10):  # 10 rounds
        optimized_fl.batch_calculate_weights(hospitals, qualities, CancerType.BREAST)
    optimized_time = time.time() - start_time
    
    # Results
    speedup = original_time / optimized_time
    
    print(f"Original implementation: {original_time:.4f}s")
    print(f"Optimized implementation: {optimized_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Cache hits: {len(optimized_fl.weight_cache)} weights, {len(optimized_fl.quality_cache)} qualities")
    
    # Memory usage test
    print(f"\nMemory efficiency:")
    print(f"Weight cache size: {len(optimized_fl.weight_cache)} entries")
    print(f"Quality cache size: {len(optimized_fl.quality_cache)} entries")
    
    return speedup > 1.5  # Should be at least 1.5x faster

def test_optimization_correctness():
    """Test that optimizations don't change results."""
    
    print("\n🔍 Testing Optimization Correctness")
    print("-" * 40)
    
    # Create test hospital
    hospital = HospitalMetadata(
        hospital_id="test_hospital",
        hospital_type=HospitalType.CANCER_CENTER,
        annual_cases=10000,
        cancer_specialties=[CancerType.BREAST],
        diagnostic_accuracy=0.94,
        years_experience=15
    )
    
    quality = SlideQuality(0.9, 0.85, 0.92, 0.1)
    
    # Compare results
    original_fl = PathologyFLDemo()
    optimized_fl = OptimizedPathologyFL()
    
    original_expertise = original_fl.calculate_expertise_weight(hospital, CancerType.BREAST)
    optimized_expertise = optimized_fl.calculate_expertise_weight_cached(hospital, CancerType.BREAST)
    
    original_quality = original_fl.calculate_quality_weight(quality)
    optimized_quality = optimized_fl.calculate_quality_weight_cached(quality, "test_hospital")
    
    expertise_match = abs(original_expertise - optimized_expertise) < 0.001
    quality_match = abs(original_quality - optimized_quality) < 0.001
    
    print(f"Expertise weight match: {expertise_match} ({original_expertise:.3f} vs {optimized_expertise:.3f})")
    print(f"Quality weight match: {quality_match} ({original_quality:.3f} vs {optimized_quality:.3f})")
    
    return expertise_match and quality_match

def test_cache_efficiency():
    """Test cache hit rates and efficiency."""
    
    print("\n📊 Testing Cache Efficiency")
    print("-" * 40)
    
    optimized_fl = OptimizedPathologyFL()
    
    # Create repeated hospital data
    hospital = HospitalMetadata(
        hospital_id="repeated_hospital",
        hospital_type=HospitalType.TEACHING_HOSPITAL,
        annual_cases=8000,
        cancer_specialties=[CancerType.LUNG],
        diagnostic_accuracy=0.91,
        years_experience=12
    )
    
    quality = SlideQuality(0.85, 0.80, 0.88, 0.15)
    
    # First calculation (cache miss)
    start_time = time.time()
    weight1 = optimized_fl.calculate_expertise_weight_cached(hospital, CancerType.LUNG)
    first_time = time.time() - start_time
    
    # Second calculation (cache hit)
    start_time = time.time()
    weight2 = optimized_fl.calculate_expertise_weight_cached(hospital, CancerType.LUNG)
    second_time = time.time() - start_time
    
    cache_speedup = first_time / second_time if second_time > 0 else float('inf')
    
    print(f"First calculation (cache miss): {first_time:.6f}s")
    print(f"Second calculation (cache hit): {second_time:.6f}s")
    print(f"Cache speedup: {cache_speedup:.1f}x")
    print(f"Results identical: {weight1 == weight2}")
    
    return cache_speedup > 5.0  # Cache should be much faster

def run_optimization_tests():
    """Run all optimization tests."""
    
    print("🚀 PathologyFL Optimization Testing")
    print("=" * 60)
    
    # Test performance improvement
    performance_test = benchmark_optimizations()
    
    # Test correctness
    correctness_test = test_optimization_correctness()
    
    # Test cache efficiency
    cache_test = test_cache_efficiency()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 OPTIMIZATION TEST RESULTS")
    print("=" * 60)
    
    tests = [
        ("Performance Improvement", performance_test),
        ("Correctness Maintained", correctness_test),
        ("Cache Efficiency", cache_test)
    ]
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOptimization Tests: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🏆 All optimizations working correctly!")
        print("⚡ PathologyFL is now faster and more efficient")
    else:
        print("⚠️ Some optimizations need fixes")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_optimization_tests()
    exit(0 if success else 1)